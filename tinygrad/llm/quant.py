"""Packed GGUF quantized linear layers for llm inference.

Keeps Q4_K / Q5_K / Q6_K (and other ggml block quants) resident as uint8 Parameters.
Decode (B*S==1) on Metal uses hand-tuned k-quant GEMV kernels that dequant into registers
and accumulate against x — never materializing the full (N,K) f16 weight matrix.

Prefill / non-Metal falls back to ggml_data_to_tensor fused into matmul (TinyGrad scheduler).
Still short of llama.cpp ggml-metal mul_mv + DFlash end-to-end parity.
"""
from __future__ import annotations
import functools
from tinygrad import Tensor, Device, getenv
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target, prod
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_QUANT, _GGML_NATIVE
from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo, resolve

_Q4_K, _Q5_K, _Q6_K = 12, 13, 14
_KQUANT_TYPES = {_Q4_K, _Q5_K, _Q6_K}
_QK_K = 256
_NSG = 4   # simdgroups per threadgroup
_NR0 = 2   # rows per simdgroup for Q4_K/Q5_K (share x loads)
_BLOCK_BYTES = {_Q4_K: 144, _Q5_K: 176, _Q6_K: 210}

_SCALE_MIN_K4 = r'''
    uint8_t sc[8], m[8];
    {
      const device uint8_t* s = b.scales;
      sc[0]=s[0]&63; sc[1]=s[1]&63; sc[2]=s[2]&63; sc[3]=s[3]&63;
      sc[4]=(s[8]&0xF)|((s[0]>>6)<<4); sc[5]=(s[9]&0xF)|((s[1]>>6)<<4);
      sc[6]=(s[10]&0xF)|((s[2]>>6)<<4); sc[7]=(s[11]&0xF)|((s[3]>>6)<<4);
      m[0]=s[4]&63; m[1]=s[5]&63; m[2]=s[6]&63; m[3]=s[7]&63;
      m[4]=(s[8]>>4)|((s[4]>>6)<<4); m[5]=(s[9]>>4)|((s[5]>>6)<<4);
      m[6]=(s[10]>>4)|((s[6]>>6)<<4); m[7]=(s[11]>>4)|((s[7]>>6)<<4);
    }
'''

def _metal_q4k_src(N:int, nblk:int) -> str:
  return f'''
#include <metal_stdlib>
using namespace metal;
struct block_q4_K {{ half d; half dmin; uint8_t scales[12]; uint8_t qs[128]; }};
kernel void kquant_gemv(
  device float* data0, const device float* data1, const device block_q4_K* data2,
  uint3 tgpig [[threadgroup_position_in_grid]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]]
) {{
  constexpr uint N = {N}u, nblk = {nblk}u, NSG = {_NSG}u, NR0 = {_NR0}u;
  const uint first_row = (tgpig.x * NSG + sgitg) * NR0;
  if (first_row >= N) return;
  const short it = tiisg;
  float sumf0 = 0.f, sumf1 = 0.f;
  for (uint ib = 0; ib < nblk; ib++) {{
    float xv0 = data1[ib*256 + 0*32 + it];
    float xv1 = data1[ib*256 + 1*32 + it];
    float xv2 = data1[ib*256 + 2*32 + it];
    float xv3 = data1[ib*256 + 3*32 + it];
    float xv4 = data1[ib*256 + 4*32 + it];
    float xv5 = data1[ib*256 + 5*32 + it];
    float xv6 = data1[ib*256 + 6*32 + it];
    float xv7 = data1[ib*256 + 7*32 + it];
    for (uint r = 0; r < NR0; r++) {{
      const uint row = first_row + r;
      if (row >= N) continue;
      device const block_q4_K& b = data2[row * nblk + ib];
      const float d = float(b.d), dmin = float(b.dmin);
      {_SCALE_MIN_K4}
      float acc = 0.f;
      {{
        const uint8_t q0 = b.qs[0*32 + it];
        const uint8_t q1 = b.qs[1*32 + it];
        const uint8_t q2 = b.qs[2*32 + it];
        const uint8_t q3 = b.qs[3*32 + it];
        acc += (d*float(sc[0])*float(q0&0xF) - dmin*float(m[0])) * xv0;
        acc += (d*float(sc[1])*float(q0>>4)  - dmin*float(m[1])) * xv1;
        acc += (d*float(sc[2])*float(q1&0xF) - dmin*float(m[2])) * xv2;
        acc += (d*float(sc[3])*float(q1>>4)  - dmin*float(m[3])) * xv3;
        acc += (d*float(sc[4])*float(q2&0xF) - dmin*float(m[4])) * xv4;
        acc += (d*float(sc[5])*float(q2>>4)  - dmin*float(m[5])) * xv5;
        acc += (d*float(sc[6])*float(q3&0xF) - dmin*float(m[6])) * xv6;
        acc += (d*float(sc[7])*float(q3>>4)  - dmin*float(m[7])) * xv7;
      }}
      if (r == 0) sumf0 += acc; else sumf1 += acc;
    }}
  }}
  float t0 = simd_sum(sumf0);
  if (tiisg == 0) data0[first_row] = t0;
  if (first_row + 1 < N) {{
    float t1 = simd_sum(sumf1);
    if (tiisg == 0) data0[first_row + 1] = t1;
  }}
}}
'''

def _metal_q5k_src(N:int, nblk:int) -> str:
  return f'''
#include <metal_stdlib>
using namespace metal;
struct block_q5_K {{ half d; half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; }};
kernel void kquant_gemv(
  device float* data0, const device float* data1, const device block_q5_K* data2,
  uint3 tgpig [[threadgroup_position_in_grid]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]]
) {{
  constexpr uint N = {N}u, nblk = {nblk}u, NSG = {_NSG}u, NR0 = {_NR0}u;
  const uint first_row = (tgpig.x * NSG + sgitg) * NR0;
  if (first_row >= N) return;
  const short it = tiisg;
  float sumf0 = 0.f, sumf1 = 0.f;
  for (uint ib = 0; ib < nblk; ib++) {{
    float xv[8];
    for (int g = 0; g < 8; g++) xv[g] = data1[ib*256 + g*32 + it];
    for (uint r = 0; r < NR0; r++) {{
      const uint row = first_row + r;
      if (row >= N) continue;
      device const block_q5_K& b = data2[row * nblk + ib];
      const float d = float(b.d), dmin = float(b.dmin);
      {_SCALE_MIN_K4}
      const uint8_t qh = b.qh[it];
      float acc = 0.f;
      for (int g = 0; g < 8; g++) {{
        const uint8_t qbyte = b.qs[(g >> 1) * 32 + it];
        float q = float((g & 1) ? (qbyte >> 4) : (qbyte & 0xF));
        if (qh & (1u << g)) q += 16.f;
        acc += (d * float(sc[g]) * q - dmin * float(m[g])) * xv[g];
      }}
      if (r == 0) sumf0 += acc; else sumf1 += acc;
    }}
  }}
  float t0 = simd_sum(sumf0);
  if (tiisg == 0) data0[first_row] = t0;
  if (first_row + 1 < N) {{
    float t1 = simd_sum(sumf1);
    if (tiisg == 0) data0[first_row + 1] = t1;
  }}
}}
'''

def _metal_q6k_src(N:int, nblk:int) -> str:
  return f'''
#include <metal_stdlib>
using namespace metal;
struct block_q6_K {{ uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; }};
kernel void kquant_gemv(
  device float* data0, const device float* data1, const device block_q6_K* data2,
  uint3 tgpig [[threadgroup_position_in_grid]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]]
) {{
  constexpr uint N = {N}u, nblk = {nblk}u, NSG = {_NSG}u;
  const uint row = tgpig.x * NSG + sgitg;
  if (row >= N) return;
  const short l = tiisg;
  float sumf = 0.f;
  device const block_q6_K* row_w = data2 + row * nblk;
  for (uint ib = 0; ib < nblk; ib++) {{
    device const block_q6_K& b = row_w[ib];
    const float d = float(b.d);
    device const float* xb = data1 + ib * 256;
    for (int part = 0; part < 2; part++) {{
      const int ql_off = part * 64;
      const int qh_off = part * 32;
      const int sc_off = part * 8;
      const int y_off = part * 128;
      const int is = l / 16;
      const uint8_t qh = b.qh[qh_off + l];
      const int8_t q1 = (int8_t)((b.ql[ql_off + l] & 0xF) | (((qh >> 0) & 3) << 4)) - 32;
      const int8_t q2 = (int8_t)((b.ql[ql_off + l + 32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32;
      const int8_t q3 = (int8_t)((b.ql[ql_off + l] >> 4) | (((qh >> 4) & 3) << 4)) - 32;
      const int8_t q4 = (int8_t)((b.ql[ql_off + l + 32] >> 4) | (((qh >> 6) & 3) << 4)) - 32;
      sumf += d * float(b.scales[sc_off + is + 0]) * float(q1) * xb[y_off + l + 0];
      sumf += d * float(b.scales[sc_off + is + 2]) * float(q2) * xb[y_off + l + 32];
      sumf += d * float(b.scales[sc_off + is + 4]) * float(q3) * xb[y_off + l + 64];
      sumf += d * float(b.scales[sc_off + is + 6]) * float(q4) * xb[y_off + l + 96];
    }}
  }}
  const float total = simd_sum(sumf);
  if (tiisg == 0) data0[row] = total;
}}
'''

def _metal_src(ggml_type:int, N:int, nblk:int) -> str:
  if ggml_type == _Q4_K: return _metal_q4k_src(N, nblk)
  if ggml_type == _Q5_K: return _metal_q5k_src(N, nblk)
  if ggml_type == _Q6_K: return _metal_q6k_src(N, nblk)
  raise ValueError(f"no metal gemv for ggml_type {ggml_type}")

def _rows_per_tg(ggml_type:int) -> int:
  return _NSG * (_NR0 if ggml_type in (_Q4_K, _Q5_K) else 1)

@functools.cache
def _compiled_lib(ggml_type:int, N:int, nblk:int) -> bytes:
  from tinygrad.runtime.ops_metal import MetalCompiler
  return MetalCompiler().compile(_metal_src(ggml_type, N, nblk))

def _kquant_gemv_program(out:UOp, x:UOp, qweight:UOp, *, ggml_type:int, N:int, K:int) -> UOp:
  nblk = K // _QK_K
  rpt = _rows_per_tg(ggml_type)
  out_f, x_f, qw_f = out.flatten(), x.flatten(), qweight.flatten()
  gidx0 = UOp.special((N + rpt - 1) // rpt, "gidx0")
  lidx0 = UOp.special(32 * _NSG, "lidx0")
  i = (gidx0 * (32 * _NSG) + lidx0) % N
  store = out_f[i].store(x_f[i % K].load() * 0.0 + qw_f[0].load().cast(dtypes.float32) * 0.0)
  sink = UOp.sink(store, gidx0, lidx0, arg=KernelInfo(name="kquant_gemv", opts_to_apply=()))
  pi = ProgramInfo(name="kquant_gemv",
                   global_size=((N + rpt - 1) // rpt, 1, 1),
                   local_size=(32 * _NSG, 1, 1),
                   globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target("METAL"))
  src = _metal_src(ggml_type, N, nblk)
  lib = _compiled_lib(ggml_type, N, nblk)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=()), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)), arg=pi)

def metal_kquant_gemv(qweight:Tensor, x:Tensor, n:int, k:int, ggml_type:int) -> Tensor:
  """y = x @ W.T for a single decode row, W packed as ggml k-quant blocks."""
  if k % _QK_K != 0: raise ValueError(f"K={k} not divisible by {_QK_K}")
  if ggml_type not in _KQUANT_TYPES: raise ValueError(f"unsupported ggml_type {ggml_type}")
  x_flat = x.reshape(-1, k).cast(dtypes.float32).contiguous()
  x1 = x_flat[0]
  out = Tensor.empty(n, dtype=dtypes.float32, device=qweight.device)
  qw = qweight.flatten()
  out = Tensor.custom_kernel(out, x1, qw,
                             fxn=functools.partial(_kquant_gemv_program, ggml_type=ggml_type, N=n, K=k))[0]
  y = out.reshape(*x.shape[:-1], n)
  return y.cast(dtypes.float16) if getenv("HALF", 1) else y

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

  def _use_metal_gemv(self, x:Tensor) -> bool:
    if not getenv("FUSED_KQUANT_GEMV", 1): return False
    if self.ggml_type not in _KQUANT_TYPES: return False
    if self.in_features % _QK_K != 0: return False
    dev = x.device if isinstance(x.device, str) else (x.device[0] if isinstance(x.device, tuple) else Device.DEFAULT)
    if not str(dev).upper().startswith("METAL"): return False
    try:
      lead = prod(x.shape[:-1])
      return bool(resolve(lead == 1))
    except Exception:
      return False

  def __call__(self, x:Tensor) -> Tensor:
    if self._use_metal_gemv(x):
      return metal_kquant_gemv(self.qweight, x, self.out_features, self.in_features, self.ggml_type)
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
