"""Packed ggml k-quant GEMV for Metal (llama.cpp mul_mv).

Primary entry: try_packed_kquant_gemv from do_to_program (QUANT_GEMV_LOWER).
Optional: program_uop via Tensor.custom_kernel when FUSED_KQUANT_GEMV=1.
"""
from __future__ import annotations
from typing import cast
import functools
from tinygrad.dtype import dtypes
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.renderer import Renderer
from tinygrad.helpers import Target, getenv, prod
from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo, ParamArg, AxisType

_Q4_K, _Q5_K, _Q6_K = 12, 13, 14
_QK_K = 256
_NSG = getenv("KQUANT_NSG", 2)
_NR0 = {_Q4_K: getenv("KQUANT_NR0_Q4", 2), _Q5_K: getenv("KQUANT_NR0_Q5", 1), _Q6_K: getenv("KQUANT_NR0_Q6", 2)}


_BLOCK = {
  _Q4_K: "struct block_q4_K { half d; half dmin; uint8_t scales[12]; uint8_t qs[128]; };",
  _Q5_K: "struct block_q5_K { half d; half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };",
  _Q6_K: "struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; };",
}
_BLOCK_T = {_Q4_K: "block_q4_K", _Q5_K: "block_q5_K", _Q6_K: "block_q6_K"}
_SCALE_MIN = r"""
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
"""

def _nr0(ggml_type: int) -> int:
  n = int(_NR0[ggml_type])
  if n < 1 or n > 4: raise ValueError(f"KQUANT_NR0 for type {ggml_type} must be 1..4, got {n}")
  return n

def _rows_per_tg(ggml_type: int) -> int: return _NSG * _nr0(ggml_type)

def _ycast(expr: str, x_half: bool) -> str:
  return f"float({expr})" if x_half else expr

def _metal_src_q4(N: int, nblk: int, x_half: bool, out_half: bool) -> str:
  x_ty = "half" if x_half else "float"
  out_ty = "half" if out_half else "float"
  def y(e: str) -> str: return _ycast(e, x_half)
  return f"""
#include <metal_stdlib>
using namespace metal;
{_BLOCK[_Q4_K]}
kernel void kquant_gemv(
  device {out_ty}* data0, const device {x_ty}* data1, const device block_q4_K* data2,
  uint3 tgpig [[threadgroup_position_in_grid]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]]
) {{
  constexpr uint N = {N}u, nblk = {nblk}u, NSG = {_NSG}u, NR0 = {_nr0(_Q4_K)}u;
  constexpr uint16_t kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
  const short ix = tiisg / 8;
  const short it = tiisg % 8;
  const short iq = it / 4;
  const short ir = it % 4;
  const uint first_row = (tgpig.x * NSG + sgitg) * NR0;
  if (first_row >= N) return;
  float yl[16], yh[16];
  float sumf[4] = {{0.f, 0.f, 0.f, 0.f}};
  uint16_t sc16[4];
  thread const uint8_t* sc8 = (thread const uint8_t*)sc16;
  for (uint ib = ix; ib < nblk; ib += 4u) {{
    float4 sumy = {{0.f, 0.f, 0.f, 0.f}};
    const uint base = ib * 256u + 64u * iq + 8u * ir;
    for (short i = 0; i < 8; ++i) {{
      yl[i+0] = {y('data1[base + i + 0]')}; sumy[0] += yl[i+0];
      yl[i+8] = {y('data1[base + i + 32]')}; sumy[1] += yl[i+8];
      yh[i+0] = {y('data1[base + i + 128]')}; sumy[2] += yh[i+0];
      yh[i+8] = {y('data1[base + i + 160]')}; sumy[3] += yh[i+8];
    }}
    for (uint r = 0; r < NR0; r++) {{
      const uint row = first_row + r;
      if (row >= N) continue;
      device const block_q4_K& blk = data2[row * nblk + ib];
      device const uint16_t* sc = (device const uint16_t*)blk.scales + iq;
      device const uint16_t* q1 = (device const uint16_t*)blk.qs + 16 * iq + 4 * ir;
      device const uint16_t* q2 = q1 + 32;
      const float d = float(blk.d), dmin = float(blk.dmin);
      sc16[0] = sc[0] & kmask1;
      sc16[1] = sc[2] & kmask1;
      sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
      sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
      float4 acc1 = {{0.f, 0.f, 0.f, 0.f}};
      float4 acc2 = {{0.f, 0.f, 0.f, 0.f}};
      for (short i = 0; i < 4; ++i) {{
        acc1[0] += yl[2*i + 0] * float(q1[i] & 0x000F);
        acc1[1] += yl[2*i + 1] * float(q1[i] & 0x0F00);
        acc1[2] += yl[2*i + 8] * float(q1[i] & 0x00F0);
        acc1[3] += yl[2*i + 9] * float(q1[i] & 0xF000);
        acc2[0] += yh[2*i + 0] * float(q2[i] & 0x000F);
        acc2[1] += yh[2*i + 1] * float(q2[i] & 0x0F00);
        acc2[2] += yh[2*i + 8] * float(q2[i] & 0x00F0);
        acc2[3] += yh[2*i + 9] * float(q2[i] & 0xF000);
      }}
      sumf[r] += d * ((acc1[0] + (1.f/256.f) * acc1[1]) * float(sc8[0]) +
                      (acc1[2] + (1.f/256.f) * acc1[3]) * float(sc8[1]) * (1.f/16.f) +
                      (acc2[0] + (1.f/256.f) * acc2[1]) * float(sc8[4]) +
                      (acc2[2] + (1.f/256.f) * acc2[3]) * float(sc8[5]) * (1.f/16.f))
               - dmin * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                         sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
    }}
  }}
  for (uint r = 0; r < NR0; r++) {{
    const uint row = first_row + r;
    if (row >= N) break;
    float t = simd_sum(sumf[r]);
    if (tiisg == 0) data0[row] = {out_ty}(t);
  }}
}}
"""


def _metal_src_q5(N: int, nblk: int, x_half: bool, out_half: bool) -> str:
  """llama.cpp kernel_mul_mv_q5_K_f32_impl for contiguous decode GEMV."""
  x_ty = "half" if x_half else "float"
  out_ty = "half" if out_half else "float"
  def y(e: str) -> str: return _ycast(e, x_half)
  return f"""
#include <metal_stdlib>
using namespace metal;
{_BLOCK[_Q5_K]}
kernel void kquant_gemv(
  device {out_ty}* data0, const device {x_ty}* data1, const device block_q5_K* data2,
  uint3 tgpig [[threadgroup_position_in_grid]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]]
) {{
  constexpr uint N = {N}u, nblk = {nblk}u, NSG = {_NSG}u, NR0 = {_nr0(_Q5_K)}u;
  constexpr uint16_t kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
  const short tid = tiisg / 4;
  const short ix = tiisg % 4;
  const short iq = tid / 4;
  const short ir = tid % 4;
  const short l0 = 8 * ir;
  const short q_offset = 32 * iq + l0;
  const short y_offset = 64 * iq + l0;
  const uint8_t hm1 = 1u << (2 * iq);
  const uint8_t hm2 = hm1 << 1;
  const uint8_t hm3 = hm1 << 4;
  const uint8_t hm4 = hm2 << 4;
  const uint first_row = (tgpig.x * NSG + sgitg) * NR0;
  if (first_row >= N) return;
  float yl[16], yh[16];
  float sumf[4] = {{0.f, 0.f, 0.f, 0.f}};
  uint16_t sc16[4];
  thread const uint8_t* sc8 = (thread const uint8_t*)sc16;
  for (uint ib = ix; ib < nblk; ib += 4u) {{
    float4 sumy = {{0.f, 0.f, 0.f, 0.f}};
    const uint y1 = ib * 256u + y_offset;
    for (short l = 0; l < 8; ++l) {{
      yl[l+0] = {y('data1[y1 + l + 0]')}; sumy[0] += yl[l+0];
      yl[l+8] = {y('data1[y1 + l + 32]')}; sumy[1] += yl[l+8];
      yh[l+0] = {y('data1[y1 + l + 128]')}; sumy[2] += yh[l+0];
      yh[l+8] = {y('data1[y1 + l + 160]')}; sumy[3] += yh[l+8];
    }}
    for (uint r = 0; r < NR0; r++) {{
      const uint row = first_row + r;
      if (row >= N) continue;
      device const block_q5_K& blk = data2[row * nblk + ib];
      device const uint8_t* q1 = blk.qs + q_offset;
      device const uint8_t* q2 = q1 + 64;
      device const uint8_t* qh = blk.qh + l0;
      device const uint16_t* a = (device const uint16_t*)blk.scales + iq;
      const float d = float(blk.d), dmin = float(blk.dmin);
      sc16[0] = a[0] & kmask1;
      sc16[1] = a[2] & kmask1;
      sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
      sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
      float4 acc1 = {{0.f, 0.f, 0.f, 0.f}};
      float4 acc2 = {{0.f, 0.f, 0.f, 0.f}};
      for (short l = 0; l < 8; ++l) {{
        const uint8_t h = qh[l];
        acc1[0] += yl[l+0] * float(q1[l] & 0x0F);
        acc1[1] += yl[l+8] * float(q1[l] & 0xF0);
        acc1[2] += yh[l+0] * float(q2[l] & 0x0F);
        acc1[3] += yh[l+8] * float(q2[l] & 0xF0);
        acc2[0] += (h & hm1) ? yl[l+0] : 0.f;
        acc2[1] += (h & hm2) ? yl[l+8] : 0.f;
        acc2[2] += (h & hm3) ? yh[l+0] : 0.f;
        acc2[3] += (h & hm4) ? yh[l+8] : 0.f;
      }}
      sumf[r] += d * (float(sc8[0]) * (acc1[0] + 16.f * acc2[0]) +
                      float(sc8[1]) * (acc1[1] * (1.f/16.f) + 16.f * acc2[1]) +
                      float(sc8[4]) * (acc1[2] + 16.f * acc2[2]) +
                      float(sc8[5]) * (acc1[3] * (1.f/16.f) + 16.f * acc2[3]))
               - dmin * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                         sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
    }}
  }}
  for (uint r = 0; r < NR0; r++) {{
    const uint row = first_row + r;
    if (row >= N) break;
    float t = simd_sum(sumf[r]);
    if (tiisg == 0) data0[row] = {out_ty}(t);
  }}
}}
"""

def _metal_src_q6(N: int, nblk: int, x_half: bool, out_half: bool) -> str:
  """llama.cpp kernel_mul_mv_q6_K_f32_impl for contiguous decode GEMV."""
  x_ty = "half" if x_half else "float"
  out_ty = "half" if out_half else "float"
  def y(e: str) -> str: return _ycast(e, x_half)
  return f"""
#include <metal_stdlib>
using namespace metal;
{_BLOCK[_Q6_K]}
kernel void kquant_gemv(
  device {out_ty}* data0, const device {x_ty}* data1, const device block_q6_K* data2,
  uint3 tgpig [[threadgroup_position_in_grid]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]]
) {{
  constexpr uint N = {N}u, nblk = {nblk}u, NSG = {_NSG}u, NR0 = {_nr0(_Q6_K)}u;
  constexpr uint8_t kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
  const uint first_row = (tgpig.x * NSG + sgitg) * NR0;
  if (first_row >= N) return;
  const short tid = tiisg / 2;
  const short ix = tiisg % 2;
  const short ip = tid / 8;
  const short il = tid % 8;
  const short l0 = 4 * il;
  const short is = 8 * ip + l0 / 16;
  const short y_offset = 128 * ip + l0;
  const short q_offset_l = 64 * ip + l0;
  const short q_offset_h = 32 * ip + l0;
  float sumf[4] = {{0.f, 0.f, 0.f, 0.f}};
  float yl[16];
  for (uint ib = ix; ib < nblk; ib += 2u) {{
    for (short l = 0; l < 4; ++l) {{
      const uint yb = ib * 256u + y_offset;
      yl[4*l + 0] = {y('data1[yb + l + 0]')};
      yl[4*l + 1] = {y('data1[yb + l + 32]')};
      yl[4*l + 2] = {y('data1[yb + l + 64]')};
      yl[4*l + 3] = {y('data1[yb + l + 96]')};
    }}
    for (uint r = 0; r < NR0; r++) {{
      const uint row = first_row + r;
      if (row >= N) continue;
      device const block_q6_K& blk = data2[row * nblk + ib];
      device const uint8_t* q1 = blk.ql + q_offset_l;
      device const uint8_t* q2 = q1 + 32;
      device const uint8_t* qh = blk.qh + q_offset_h;
      device const int8_t* sc = blk.scales + is;
      const float d = float(blk.d);
      float4 sums = {{0.f, 0.f, 0.f, 0.f}};
      for (short l = 0; l < 4; ++l) {{
        sums[0] += yl[4*l + 0] * float((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
        sums[1] += yl[4*l + 1] * float((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
        sums[2] += yl[4*l + 2] * float((int8_t)((q1[l] >> 4) | ((qh[l] & kmask3) << 0)) - 32);
        sums[3] += yl[4*l + 3] * float((int8_t)((q2[l] >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
      }}
      sumf[r] += d * (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
    }}
  }}
  for (uint r = 0; r < NR0; r++) {{
    const uint row = first_row + r;
    if (row >= N) break;
    float t = simd_sum(sumf[r]);
    if (tiisg == 0) data0[row] = {out_ty}(t);
  }}
}}
"""

def _x_load(x_half: bool) -> str:
  return "\n".join(f"    float xv{g} = {_ycast(f'data1[ib*256 + {g}*32 + it]', x_half)};" for g in range(8))

def _body_q5() -> str:
  return f"""
      device const block_q5_K& b = data2[row * nblk + ib];
      const float d = float(b.d), dmin = float(b.dmin);
      {_SCALE_MIN}
      const uint8_t qh = b.qh[it];
      float acc = 0.f;
      {{
        const float xv[8] = {{xv0,xv1,xv2,xv3,xv4,xv5,xv6,xv7}};
        for (int g = 0; g < 8; g++) {{
          const uint8_t qbyte = b.qs[(g >> 1) * 32 + it];
          float q = float((g & 1) ? (qbyte >> 4) : (qbyte & 0xF));
          if (qh & (1u << g)) q += 16.f;
          acc += (d * float(sc[g]) * q - dmin * float(m[g])) * xv[g];
        }}
      }}
"""

def _body_q6() -> str:
  return r"""
      device const block_q6_K& b = data2[row * nblk + ib];
      const float d = float(b.d);
      float acc = 0.f;
      {
        const int is = it / 16;
        const uint8_t qh0 = b.qh[it];
        const uint8_t qh1 = b.qh[it + 32];
        const int8_t q1 = (int8_t)((b.ql[it] & 0xF) | (((qh0 >> 0) & 3) << 4)) - 32;
        const int8_t q2 = (int8_t)((b.ql[it + 32] & 0xF) | (((qh0 >> 2) & 3) << 4)) - 32;
        const int8_t q3 = (int8_t)((b.ql[it] >> 4) | (((qh0 >> 4) & 3) << 4)) - 32;
        const int8_t q4 = (int8_t)((b.ql[it + 32] >> 4) | (((qh0 >> 6) & 3) << 4)) - 32;
        const int8_t q5 = (int8_t)((b.ql[it + 64] & 0xF) | (((qh1 >> 0) & 3) << 4)) - 32;
        const int8_t q6 = (int8_t)((b.ql[it + 96] & 0xF) | (((qh1 >> 2) & 3) << 4)) - 32;
        const int8_t q7 = (int8_t)((b.ql[it + 64] >> 4) | (((qh1 >> 4) & 3) << 4)) - 32;
        const int8_t q8 = (int8_t)((b.ql[it + 96] >> 4) | (((qh1 >> 6) & 3) << 4)) - 32;
        acc += d * float(b.scales[is + 0]) * float(q1) * xv0;
        acc += d * float(b.scales[is + 2]) * float(q2) * xv1;
        acc += d * float(b.scales[is + 4]) * float(q3) * xv2;
        acc += d * float(b.scales[is + 6]) * float(q4) * xv3;
        acc += d * float(b.scales[is + 8]) * float(q5) * xv4;
        acc += d * float(b.scales[is + 10]) * float(q6) * xv5;
        acc += d * float(b.scales[is + 12]) * float(q7) * xv6;
        acc += d * float(b.scales[is + 14]) * float(q8) * xv7;
      }
"""

_BODY = {_Q5_K: _body_q5, _Q6_K: _body_q6}

def _metal_src_generic(ggml_type: int, N: int, nblk: int, x_half: bool, out_half: bool) -> str:
  x_ty = "half" if x_half else "float"
  out_ty = "half" if out_half else "float"
  return f"""
#include <metal_stdlib>
using namespace metal;
{_BLOCK[ggml_type]}
kernel void kquant_gemv(
  device {out_ty}* data0, const device {x_ty}* data1, const device {_BLOCK_T[ggml_type]}* data2,
  uint3 tgpig [[threadgroup_position_in_grid]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]]
) {{
  constexpr uint N = {N}u, nblk = {nblk}u, NSG = {_NSG}u, NR0 = {_nr0(ggml_type)}u;
  const uint first_row = (tgpig.x * NSG + sgitg) * NR0;
  if (first_row >= N) return;
  const short it = tiisg;
  float sumf[4] = {{0.f, 0.f, 0.f, 0.f}};
  for (uint ib = 0; ib < nblk; ib++) {{
{_x_load(x_half)}
    for (uint r = 0; r < NR0; r++) {{
      const uint row = first_row + r;
      if (row >= N) continue;
{_BODY[ggml_type]()}
      sumf[r] += acc;
    }}
  }}
  for (uint r = 0; r < NR0; r++) {{
    const uint row = first_row + r;
    if (row >= N) break;
    float t = simd_sum(sumf[r]);
    if (tiisg == 0) data0[row] = {out_ty}(t);
  }}
}}
"""

def _metal_src(ggml_type: int, N: int, nblk: int, x_half: bool, out_half: bool) -> str:
  if ggml_type == _Q4_K: return _metal_src_q4(N, nblk, x_half, out_half)
  if ggml_type == _Q5_K: return _metal_src_q5(N, nblk, x_half, out_half)
  if ggml_type == _Q6_K: return _metal_src_q6(N, nblk, x_half, out_half)
  return _metal_src_generic(ggml_type, N, nblk, x_half, out_half)

@functools.cache
def _compiled_lib(ggml_type: int, N: int, nblk: int, nsg: int, nr0: int, x_half: bool, out_half: bool) -> bytes:
  from tinygrad.runtime.ops_metal import MetalCompiler
  return MetalCompiler().compile(_metal_src(ggml_type, N, nblk, x_half, out_half))

def _program(out: UOp, x: UOp, qweight: UOp, *, ggml_type: int, N: int, K: int, x_half: bool, out_half: bool) -> UOp:
  nblk, rpt = K // _QK_K, _rows_per_tg(ggml_type)
  out_f, x_f, qw_f = out.flatten(), x.flatten(), qweight.flatten()
  gidx0 = UOp.special((N + rpt - 1) // rpt, "gidx0")
  lidx0 = UOp.special(32 * _NSG, "lidx0")
  i = (gidx0 * (32 * _NSG) + lidx0) % N
  store = out_f[i].store(x_f[i % K].load().cast(dtypes.float32) * 0.0 + qw_f[0].load().cast(dtypes.float32) * 0.0)
  sink = UOp.sink(store, gidx0, lidx0, arg=KernelInfo(name="kquant_gemv", opts_to_apply=()))
  pi = ProgramInfo(name="kquant_gemv",
                   global_size=((N + rpt - 1) // rpt, 1, 1),
                   local_size=(32 * _NSG, 1, 1),
                   globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target("METAL"))
  src = _metal_src(ggml_type, N, nblk, x_half, out_half)
  lib = _compiled_lib(ggml_type, N, nblk, _NSG, _nr0(ggml_type), x_half, out_half)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=()), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)), arg=pi)

program_uop = _program

_BLOCK_BYTES = {_Q4_K: 144, _Q5_K: 176, _Q6_K: 210}

def _match_packed_gemv(ast: UOp) -> tuple[int, int, int, int, int, int, bool, bool] | None:
  """Isolated k-quant decode GEMV: 3 bufs, 1 REDUCE, concrete N/K, nbytes match."""
  if ast.op is not Ops.SINK: return None
  nodes = list(ast.toposort())
  reduces = [u for u in nodes if u.op is Ops.REDUCE and u.arg[0] is Ops.ADD]
  if len(reduces) != 1: return None
  red_ranges = [s for s in reduces[0].src[1:] if s.op is Ops.RANGE]
  if len(red_ranges) != 1: return None
  try: K = int(cast(int, red_ranges[0].src[0].ssimplify()))
  except Exception: return None
  if K % _QK_K: return None
  nblk = K // _QK_K

  stores = [u for u in nodes if u.op is Ops.STORE]
  if len(stores) != 1: return None
  idx = stores[0].src[0]
  while idx.op is Ops.CAST: idx = idx.src[0]
  if idx.op is not Ops.INDEX or idx.src[0].op is not Ops.PARAM: return None
  out_p = idx.src[0]
  weak = [s for s in idx.src[1:] if s.op is Ops.RANGE and s.arg[1] == AxisType.WEAK]
  if not weak:
    weak = [u for u in stores[0].toposort()
            if u.op is Ops.RANGE and u.arg[1] == AxisType.WEAK and u is not red_ranges[0]]
  if not weak: return None
  try: N = int(prod(int(cast(int, r.src[0].ssimplify())) for r in weak))
  except Exception: return None

  params = [u for u in nodes if u.op is Ops.PARAM and isinstance(u.arg, ParamArg) and u.dtype.itemsize > 0]
  if len({p.arg.slot for p in params}) != 3: return None
  uchar = [p for p in params if p.dtype == dtypes.uchar]
  if len(uchar) != 1: return None
  qw_slot, qw_bytes = uchar[0].arg.slot, int(cast(int, uchar[0].src[0].ssimplify()))
  ggml_type = next((gt for gt, bb in _BLOCK_BYTES.items() if qw_bytes == N * nblk * bb), None)
  if ggml_type is None: return None

  # allow oversized reused buffers; ranges define N/K
  x_cands = [p for p in params if p.arg.slot not in (out_p.arg.slot, qw_slot) and p.dtype.itemsize in (2, 4)]
  if len(x_cands) != 1: return None
  if not any(u.op is Ops.BITCAST for u in nodes): return None
  if not any(u.op in (Ops.AND, Ops.SHR) for u in nodes): return None
  x_p = x_cands[0]
  return (ggml_type, N, K, out_p.arg.slot, x_p.arg.slot, qw_slot,
          x_p.dtype.itemsize == 2 and bool(getenv("KQUANT_X_HALF", 1)), out_p.dtype.itemsize == 2)

def try_packed_kquant_gemv(ast: UOp, renderer: Renderer) -> UOp | None:
  if not getenv("QUANT_GEMV_LOWER", 1) or not isinstance(renderer, MetalRenderer): return None
  if (m := _match_packed_gemv(ast)) is None: return None
  ggml_type, N, K, out_slot, x_slot, qw_slot, x_half, out_half = m
  nblk, rpt = K // _QK_K, _rows_per_tg(ggml_type)
  pi = ProgramInfo(name="kquant_gemv",
                   global_size=((N + rpt - 1) // rpt, 1, 1), local_size=(32 * _NSG, 1, 1),
                   globals=(out_slot, x_slot, qw_slot), outs=(out_slot,), ins=(x_slot, qw_slot),
                   target=renderer.target)
  src = _metal_src(ggml_type, N, nblk, x_half, out_half)
  lib = _compiled_lib(ggml_type, N, nblk, _NSG, _nr0(ggml_type), x_half, out_half)
  return UOp(Ops.PROGRAM, src=(ast, UOp(Ops.LINEAR, src=()), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)), arg=pi)

