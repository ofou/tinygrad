"""Packed GGUF quantized linear layers for llm inference.

Keeps Q4_K / Q5_K / Q6_K (and other ggml block quants) resident as uint8 Parameters
and dequants inside the matmul graph, instead of materializing full f16 weights
(REALIZE=1) or leaving every Linear.weight as a view into one giant GGUF buffer.

This is an incremental path toward llama.cpp fused k-quant kernels: same numerical
dequant (via ggml_data_to_tensor), better residency / locality, no 56GB f16 blowup.
Full Metal GGML-Q4_K GEMV shaders / DFlash are still future work.
"""
from __future__ import annotations
from tinygrad import Tensor, getenv
from tinygrad.dtype import dtypes
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_QUANT, _GGML_NATIVE

class QuantLinear:
  """Linear whose weight is a packed ggml quant block buffer (qweight)."""
  def __init__(self, in_features:int, out_features:int, ggml_type:int, bias:bool=False):
    if bias: raise NotImplementedError("QuantLinear bias not supported")
    if ggml_type not in _GGML_QUANT: raise ValueError(f"QuantLinear unsupported ggml_type {ggml_type}")
    ne, nb = _GGML_QUANT[ggml_type]
    n_elems = out_features * in_features
    if n_elems % ne != 0: raise ValueError(f"shape ({out_features},{in_features}) not aligned to quant block {ne}")
    self.in_features, self.out_features, self.ggml_type = in_features, out_features, ggml_type
    self.qweight = Tensor.zeros((n_elems // ne) * nb, dtype=dtypes.uint8)
    self.bias = None

  def dequant(self) -> Tensor:
    w = ggml_data_to_tensor(self.qweight, self.out_features * self.in_features, self.ggml_type)
    w = w.reshape(self.out_features, self.in_features)
    return w.cast(dtypes.float16) if getenv("HALF", 1) else w

  def __call__(self, x:Tensor) -> Tensor:
    return x.linear(self.dequant().transpose(), self.bias)

def is_ggml_quant(ggml_type:int) -> bool:
  return ggml_type in _GGML_QUANT

def quant_nbytes(n_elems:int, ggml_type:int) -> int:
  if ggml_type in _GGML_NATIVE: return n_elems * _GGML_NATIVE[ggml_type].itemsize
  ne, nb = _GGML_QUANT[ggml_type]
  if n_elems % ne != 0: raise ValueError(f"{n_elems} not divisible by block nelems {ne}")
  return (n_elems // ne) * nb

def replace_linear_with_quant(model, name:str, qweight:Tensor, ggml_type:int, shape:tuple[int, ...]) -> QuantLinear:
  """Replace model.<name> Linear with a QuantLinear holding packed qweight. name like 'blk.0.attn_q'."""
  parts = name.split(".")
  parent = model
  for p in parts[:-1]:
    parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
  attr = parts[-1]
  old = getattr(parent, attr)
  out_f, in_f = shape
  ql = QuantLinear(in_f, out_f, ggml_type, bias=getattr(old, "bias", None) is not None)
  if ql.qweight.shape != qweight.shape:
    qweight = qweight.flatten()
  ql.qweight = qweight
  setattr(parent, attr, ql)
  return ql
