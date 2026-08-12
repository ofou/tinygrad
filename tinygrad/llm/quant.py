"""Packed GGUF QuantLinear; Metal T<=16 lowers via QUANT_GEMV_LOWER (3-buf uchar+REDUCE)."""
from __future__ import annotations
import functools
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from tinygrad.helpers import prod, getenv
from tinygrad.llm.gguf import ggml_data_to_tensor, quant_nbytes, _GGML_QUANT
from tinygrad.uop.ops import resolve

_Q4_K, _Q5_K, _Q6_K = 12, 13, 14
_KQUANT_TYPES = {_Q4_K, _Q5_K, _Q6_K}
_QK_K = 256

def is_ggml_quant(ggml_type: int) -> bool: return ggml_type in _GGML_QUANT

def dequant_weight(qweight: Tensor, out_features: int, in_features: int, ggml_type: int) -> Tensor:
  w = ggml_data_to_tensor(qweight, out_features * in_features, ggml_type).reshape(out_features, in_features)
  return w.cast(dtypes.float16) if getenv("HALF", 1) else w

def _batch_rows(x: Tensor) -> int | None:
  """Leading batch product (T for decode/verify). None if unresolved."""
  try:
    t = prod(x.shape[:-1])
    if not isinstance(t, int): t = int(resolve(t))
    return t if t >= 1 else None
  except Exception: return None

def _is_decode_row(x: Tensor) -> bool:
  return _batch_rows(x) == 1

def _device_is_metal(x: Tensor) -> bool:
  dev = x.device if isinstance(x.device, str) else (x.device[0] if isinstance(x.device, tuple) else Device.DEFAULT)
  return str(dev).upper().startswith("METAL")

def _metal_kquant_packed(x: Tensor, ggml_type: int, in_features: int, *, max_t: int | None = None) -> bool:
  """Packed Metal path for T=1 decode and small T>1 (DFlash verify)."""
  if max_t is None: max_t = getenv("QUANT_GEMV_MAX_T", 16)
  t = _batch_rows(x)
  return (ggml_type in _KQUANT_TYPES and in_features % _QK_K == 0
          and t is not None and t <= max_t and _device_is_metal(x))

# Back-compat alias used by older call sites / tests.
def _metal_kquant_decode(x: Tensor, ggml_type: int, in_features: int) -> bool:
  return _metal_kquant_packed(x, ggml_type, in_features)

def _mul_mat_q_metal(qweight: Tensor, x: Tensor, n: int, k: int, ggml_type: int) -> Tensor:
  """Deprecated FUSED_KQUANT_GEMV=1 escape hatch. Prefer QUANT_GEMV_LOWER (~13.9 tok/s)."""
  from tinygrad.codegen.quant_gemv import program_uop
  if k % _QK_K: raise ValueError(f"K={k} not divisible by {_QK_K}")
  x_flat = x.reshape(-1, k).contiguous()
  x_half = x_flat.dtype == dtypes.float16 and bool(getenv("KQUANT_X_HALF", 1))
  out_half = bool(getenv("HALF", 1))
  x1 = x_flat[0] if x_half else x_flat[0].cast(dtypes.float32)
  out = Tensor.empty(n, dtype=dtypes.float16 if out_half else dtypes.float32, device=qweight.device)
  out = Tensor.custom_kernel(out, x1, qweight.flatten(),
                             fxn=functools.partial(program_uop, ggml_type=ggml_type, N=n, K=k,
                                                   x_half=x_half, out_half=out_half))[0]
  return out.reshape(*x.shape[:-1], n)

def mul_mat(qweight: Tensor, x: Tensor, *, ggml_type: int, out_features: int, in_features: int) -> Tensor:
  # Deprecated hatch (default 0, T=1 only). Leave off unless bisecting a lowering regression.
  if getenv("FUSED_KQUANT_GEMV", 0) and _metal_kquant_packed(x, ggml_type, in_features, max_t=1):
    return _mul_mat_q_metal(qweight, x, out_features, in_features, ggml_type)
  # QUANT_GEMV_CONTIG default 1: bufferize x/y so QUANT_GEMV_LOWER matches for T<=MAX_T.
  # Set CONTIG=0 with QUANT_GEMV_SCHED_BARRIER=1 to A/B the schedule path.
  lower = _metal_kquant_packed(x, ggml_type, in_features) and getenv("QUANT_GEMV_LOWER", 1)
  contig = lower and getenv("QUANT_GEMV_CONTIG", 1)
  if contig: x = x.contiguous()
  y = x.linear(dequant_weight(qweight, out_features, in_features, ggml_type).transpose())
  if contig: return y.contiguous()
  return y

class QuantLinear:
  def __init__(self, in_features: int, out_features: int, ggml_type: int, bias: bool = False):
    if bias: raise NotImplementedError("QuantLinear bias not supported")
    if ggml_type not in _GGML_QUANT: raise ValueError(f"QuantLinear unsupported ggml_type {ggml_type}")
    ne, _nb = _GGML_QUANT[ggml_type]
    n_elems = out_features * in_features
    if n_elems % ne: raise ValueError(f"shape ({out_features},{in_features}) not aligned to quant block {ne}")
    self.in_features, self.out_features, self.ggml_type = in_features, out_features, ggml_type
    self.qweight = Tensor.zeros(quant_nbytes(n_elems, ggml_type), dtype=dtypes.uint8)
    self.bias = None

  def dequant(self) -> Tensor:
    return dequant_weight(self.qweight, self.out_features, self.in_features, self.ggml_type)

  def __call__(self, x: Tensor) -> Tensor:
    y = mul_mat(self.qweight, x, ggml_type=self.ggml_type,
                out_features=self.out_features, in_features=self.in_features)
    return y if self.bias is None else y + self.bias

def replace_linear_with_quant(model, name: str, qweight: Tensor, ggml_type: int, shape: tuple[int, ...]) -> QuantLinear:
  parts = name.split(".")
  parent = model
  for p in parts[:-1]:
    parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
  old = getattr(parent, parts[-1])
  out_f, in_f = shape
  ql = QuantLinear(in_f, out_f, ggml_type, bias=getattr(old, "bias", None) is not None)
  ql.qweight = qweight.flatten() if ql.qweight.shape != qweight.shape else qweight
  setattr(parent, parts[-1], ql)
  return ql
