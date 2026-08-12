"""Packed ggml k-quant GEMV/GEMM for Metal (llama.cpp mul_mv).

Primary entry: try_packed_kquant_gemv from do_to_program (QUANT_GEMV_LOWER).
T=1/T>1 default: Y=T mul_mv (re-reads weights ×T). QUANT_GEMV_WS=1 TILE=k: WS.
QUANT_GEMV_EXT=1: experimental mul_mv_ext. Goal: mul_mm for T>8.

Optional: program_uop via Tensor.custom_kernel when FUSED_KQUANT_GEMV=1
(deprecated escape hatch; default off; T=1 only).

Launch geometry knobs (KQUANT_NSG / KQUANT_NR0_*) are tuning-only.
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
# Tuning-only launch geometry (defaults = zero-ritual PR path).
# NSG = Metal simdgroups / threadgroup (default 1; NSG=2 is ~3% slower on Muse).
# NR0 = output rows per subgroup (Q4=2, Q5=1, Q6=2).
_NSG = getenv("KQUANT_NSG", 1)
_NR0 = {_Q4_K: getenv("KQUANT_NR0_Q4", 2), _Q5_K: getenv("KQUANT_NR0_Q5", 1), _Q6_K: getenv("KQUANT_NR0_Q6", 2)}
# Small-T packed path (DFlash verify); large prefill stays dequant+GEMM.
_MAX_T = getenv("QUANT_GEMV_MAX_T", 16)


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

def _metal_src_q4(N: int, nblk: int, T: int, x_half: bool, out_half: bool) -> str:
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
  constexpr uint N = {N}u, nblk = {nblk}u, T = {T}u, NSG = {_NSG}u, NR0 = {_nr0(_Q4_K)}u;
  constexpr uint16_t kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
  const short ix = tiisg / 8;
  const short it = tiisg % 8;
  const short iq = it / 4;
  const short ir = it % 4;
  const uint first_row = (tgpig.x * NSG + sgitg) * NR0;
  if (first_row >= N) return;
  const uint col = tgpig.y;
  if (col >= T) return;
  const uint xoff = col * nblk * 256u;
  const uint ooff = col * N;
  float yl[16], yh[16];
  float sumf[4] = {{0.f, 0.f, 0.f, 0.f}};
  uint16_t sc16[4];
  thread const uint8_t* sc8 = (thread const uint8_t*)sc16;
  for (uint ib = ix; ib < nblk; ib += 4u) {{
    float4 sumy = {{0.f, 0.f, 0.f, 0.f}};
    const uint base = ib * 256u + 64u * iq + 8u * ir;
    for (short i = 0; i < 8; ++i) {{
      yl[i+0] = {y('data1[xoff + base + i + 0]')}; sumy[0] += yl[i+0];
      yl[i+8] = {y('data1[xoff + base + i + 32]')}; sumy[1] += yl[i+8];
      yh[i+0] = {y('data1[xoff + base + i + 128]')}; sumy[2] += yh[i+0];
      yh[i+8] = {y('data1[xoff + base + i + 160]')}; sumy[3] += yh[i+8];
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
    if (tiisg == 0) data0[ooff + row] = {out_ty}(t);
  }}
}}
"""


def _metal_src_q5(N: int, nblk: int, T: int, x_half: bool, out_half: bool) -> str:
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
  constexpr uint N = {N}u, nblk = {nblk}u, T = {T}u, NSG = {_NSG}u, NR0 = {_nr0(_Q5_K)}u;
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
  const uint col = tgpig.y;
  if (col >= T) return;
  const uint xoff = col * nblk * 256u;
  const uint ooff = col * N;
  float yl[16], yh[16];
  float sumf[4] = {{0.f, 0.f, 0.f, 0.f}};
  uint16_t sc16[4];
  thread const uint8_t* sc8 = (thread const uint8_t*)sc16;
  for (uint ib = ix; ib < nblk; ib += 4u) {{
    float4 sumy = {{0.f, 0.f, 0.f, 0.f}};
    const uint y1 = ib * 256u + y_offset;
    for (short l = 0; l < 8; ++l) {{
      yl[l+0] = {y('data1[xoff + y1 + l + 0]')}; sumy[0] += yl[l+0];
      yl[l+8] = {y('data1[xoff + y1 + l + 32]')}; sumy[1] += yl[l+8];
      yh[l+0] = {y('data1[xoff + y1 + l + 128]')}; sumy[2] += yh[l+0];
      yh[l+8] = {y('data1[xoff + y1 + l + 160]')}; sumy[3] += yh[l+8];
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
    if (tiisg == 0) data0[ooff + row] = {out_ty}(t);
  }}
}}
"""

def _metal_src_q6(N: int, nblk: int, T: int, x_half: bool, out_half: bool) -> str:
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
  constexpr uint N = {N}u, nblk = {nblk}u, T = {T}u, NSG = {_NSG}u, NR0 = {_nr0(_Q6_K)}u;
  constexpr uint8_t kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
  const uint first_row = (tgpig.x * NSG + sgitg) * NR0;
  if (first_row >= N) return;
  const uint col = tgpig.y;
  if (col >= T) return;
  const uint xoff = col * nblk * 256u;
  const uint ooff = col * N;
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
      yl[4*l + 0] = {y('data1[xoff + yb + l + 0]')};
      yl[4*l + 1] = {y('data1[xoff + yb + l + 32]')};
      yl[4*l + 2] = {y('data1[xoff + yb + l + 64]')};
      yl[4*l + 3] = {y('data1[xoff + yb + l + 96]')};
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
    if (tiisg == 0) data0[ooff + row] = {out_ty}(t);
  }}
}}
"""

def _x_load(x_half: bool) -> str:
  return "\n".join(f"    float xv{g} = {_ycast(f'data1[xoff + ib*256 + {g}*32 + it]', x_half)};" for g in range(8))

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

def _metal_src_generic(ggml_type: int, N: int, nblk: int, T: int, x_half: bool, out_half: bool) -> str:
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
  constexpr uint N = {N}u, nblk = {nblk}u, T = {T}u, NSG = {_NSG}u, NR0 = {_nr0(ggml_type)}u;
  const uint first_row = (tgpig.x * NSG + sgitg) * NR0;
  if (first_row >= N) return;
  const uint col = tgpig.y;
  if (col >= T) return;
  const uint xoff = col * nblk * 256u;
  const uint ooff = col * N;
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
    if (tiisg == 0) data0[ooff + row] = {out_ty}(t);
  }}
}}
"""


def _metal_src_q4_ws(N: int, nblk: int, T: int, x_half: bool, out_half: bool, tile: int) -> str:
  """Weight-stationary Q4_K: NR0/NSG occupancy (=T=1), accumulate `tile` cols/TG."""
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
  constexpr uint N = {N}u, nblk = {nblk}u, T = {T}u, TILE = {tile}u, NSG = {_NSG}u, NR0 = {_nr0(_Q4_K)}u;
  constexpr uint16_t kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
  const short ix = tiisg / 8;
  const short it = tiisg % 8;
  const short iq = it / 4;
  const short ir = it % 4;
  const uint first_row = (tgpig.x * NSG + sgitg) * NR0;
  if (first_row >= N) return;
  const uint col0 = tgpig.y * TILE;
  if (col0 >= T) return;
  const uint ncol = (col0 + TILE <= T) ? TILE : (T - col0);
  float yl[16], yh[16];
  float sumf[4 * TILE];
  for (uint i = 0; i < 4u * TILE; i++) sumf[i] = 0.f;
  uint16_t sc16[4];
  thread const uint8_t* sc8 = (thread const uint8_t*)sc16;
  for (uint ib = ix; ib < nblk; ib += 4u) {{
    uint16_t q1c[4][4], q2c[4][4];
    float d_r[4], dmin_r[4];
    uint16_t sc_r[4][4];
    for (uint r = 0; r < NR0; r++) {{
      const uint row = first_row + r;
      if (row >= N) continue;
      device const block_q4_K& blk = data2[row * nblk + ib];
      device const uint16_t* sc = (device const uint16_t*)blk.scales + iq;
      device const uint16_t* q1p = (device const uint16_t*)blk.qs + 16 * iq + 4 * ir;
      device const uint16_t* q2p = q1p + 32;
      d_r[r] = float(blk.d); dmin_r[r] = float(blk.dmin);
      sc_r[r][0] = sc[0] & kmask1; sc_r[r][1] = sc[2] & kmask1;
      sc_r[r][2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
      sc_r[r][3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
      for (short i = 0; i < 4; ++i) {{ q1c[r][i] = q1p[i]; q2c[r][i] = q2p[i]; }}
    }}
    for (uint tc = 0; tc < ncol; tc++) {{
      const uint col = col0 + tc;
      const uint xoff = col * nblk * 256u;
      float4 sumy = {{0.f, 0.f, 0.f, 0.f}};
      const uint base = ib * 256u + 64u * iq + 8u * ir;
      for (short i = 0; i < 8; ++i) {{
        yl[i+0] = {y('data1[xoff + base + i + 0]')}; sumy[0] += yl[i+0];
        yl[i+8] = {y('data1[xoff + base + i + 32]')}; sumy[1] += yl[i+8];
        yh[i+0] = {y('data1[xoff + base + i + 128]')}; sumy[2] += yh[i+0];
        yh[i+8] = {y('data1[xoff + base + i + 160]')}; sumy[3] += yh[i+8];
      }}
      for (uint r = 0; r < NR0; r++) {{
        const uint row = first_row + r;
        if (row >= N) continue;
        sc16[0] = sc_r[r][0]; sc16[1] = sc_r[r][1]; sc16[2] = sc_r[r][2]; sc16[3] = sc_r[r][3];
        float4 acc1 = {{0.f, 0.f, 0.f, 0.f}};
        float4 acc2 = {{0.f, 0.f, 0.f, 0.f}};
        for (short i = 0; i < 4; ++i) {{
          acc1[0] += yl[2*i + 0] * float(q1c[r][i] & 0x000F);
          acc1[1] += yl[2*i + 1] * float(q1c[r][i] & 0x0F00);
          acc1[2] += yl[2*i + 8] * float(q1c[r][i] & 0x00F0);
          acc1[3] += yl[2*i + 9] * float(q1c[r][i] & 0xF000);
          acc2[0] += yh[2*i + 0] * float(q2c[r][i] & 0x000F);
          acc2[1] += yh[2*i + 1] * float(q2c[r][i] & 0x0F00);
          acc2[2] += yh[2*i + 8] * float(q2c[r][i] & 0x00F0);
          acc2[3] += yh[2*i + 9] * float(q2c[r][i] & 0xF000);
        }}
        sumf[r * TILE + tc] += d_r[r] * ((acc1[0] + (1.f/256.f) * acc1[1]) * float(sc8[0]) +
                        (acc1[2] + (1.f/256.f) * acc1[3]) * float(sc8[1]) * (1.f/16.f) +
                        (acc2[0] + (1.f/256.f) * acc2[1]) * float(sc8[4]) +
                        (acc2[2] + (1.f/256.f) * acc2[3]) * float(sc8[5]) * (1.f/16.f))
                 - dmin_r[r] * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                           sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
      }}
    }}
  }}
  for (uint r = 0; r < NR0; r++) {{
    const uint row = first_row + r;
    if (row >= N) break;
    for (uint tc = 0; tc < ncol; tc++) {{
      float t = simd_sum(sumf[r * TILE + tc]);
      if (tiisg == 0) data0[(col0 + tc) * N + row] = {out_ty}(t);
    }}
  }}
}}
"""


def _metal_src_q5_ws(N: int, nblk: int, T: int, x_half: bool, out_half: bool, tile: int) -> str:
  """Weight-stationary Q5_K: NR0/NSG occupancy, accumulate `tile` cols/TG."""
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
  constexpr uint N = {N}u, nblk = {nblk}u, T = {T}u, TILE = {tile}u, NSG = {_NSG}u, NR0 = {_nr0(_Q5_K)}u;
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
  const uint col0 = tgpig.y * TILE;
  if (col0 >= T) return;
  const uint ncol = (col0 + TILE <= T) ? TILE : (T - col0);
  float yl[16], yh[16];
  float sumf[4 * TILE];
  for (uint i = 0; i < 4u * TILE; i++) sumf[i] = 0.f;
  uint16_t sc16[4];
  thread const uint8_t* sc8 = (thread const uint8_t*)sc16;
  for (uint ib = ix; ib < nblk; ib += 4u) {{
    uint8_t q1c[4][8], q2c[4][8], qhc[4][8];
    float d_r[4], dmin_r[4];
    uint16_t sc_r[4][4];
    for (uint r = 0; r < NR0; r++) {{
      const uint row = first_row + r;
      if (row >= N) continue;
      device const block_q5_K& blk = data2[row * nblk + ib];
      device const uint8_t* q1p = blk.qs + q_offset;
      device const uint8_t* q2p = q1p + 64;
      device const uint8_t* qhp = blk.qh + l0;
      device const uint16_t* a = (device const uint16_t*)blk.scales + iq;
      d_r[r] = float(blk.d); dmin_r[r] = float(blk.dmin);
      sc_r[r][0] = a[0] & kmask1; sc_r[r][1] = a[2] & kmask1;
      sc_r[r][2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
      sc_r[r][3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
      for (short l = 0; l < 8; ++l) {{ q1c[r][l] = q1p[l]; q2c[r][l] = q2p[l]; qhc[r][l] = qhp[l]; }}
    }}
    for (uint tc = 0; tc < ncol; tc++) {{
      const uint col = col0 + tc;
      const uint xoff = col * nblk * 256u;
      float4 sumy = {{0.f, 0.f, 0.f, 0.f}};
      const uint y1 = ib * 256u + y_offset;
      for (short l = 0; l < 8; ++l) {{
        yl[l+0] = {y('data1[xoff + y1 + l + 0]')}; sumy[0] += yl[l+0];
        yl[l+8] = {y('data1[xoff + y1 + l + 32]')}; sumy[1] += yl[l+8];
        yh[l+0] = {y('data1[xoff + y1 + l + 128]')}; sumy[2] += yh[l+0];
        yh[l+8] = {y('data1[xoff + y1 + l + 160]')}; sumy[3] += yh[l+8];
      }}
      for (uint r = 0; r < NR0; r++) {{
        const uint row = first_row + r;
        if (row >= N) continue;
        sc16[0] = sc_r[r][0]; sc16[1] = sc_r[r][1]; sc16[2] = sc_r[r][2]; sc16[3] = sc_r[r][3];
        float4 acc1 = {{0.f, 0.f, 0.f, 0.f}};
        float4 acc2 = {{0.f, 0.f, 0.f, 0.f}};
        for (short l = 0; l < 8; ++l) {{
          const uint8_t h = qhc[r][l];
          acc1[0] += yl[l+0] * float(q1c[r][l] & 0x0F);
          acc1[1] += yl[l+8] * float(q1c[r][l] & 0xF0);
          acc1[2] += yh[l+0] * float(q2c[r][l] & 0x0F);
          acc1[3] += yh[l+8] * float(q2c[r][l] & 0xF0);
          acc2[0] += (h & hm1) ? yl[l+0] : 0.f;
          acc2[1] += (h & hm2) ? yl[l+8] : 0.f;
          acc2[2] += (h & hm3) ? yh[l+0] : 0.f;
          acc2[3] += (h & hm4) ? yh[l+8] : 0.f;
        }}
        sumf[r * TILE + tc] += d_r[r] * (float(sc8[0]) * (acc1[0] + 16.f * acc2[0]) +
                        float(sc8[1]) * (acc1[1] * (1.f/16.f) + 16.f * acc2[1]) +
                        float(sc8[4]) * (acc1[2] + 16.f * acc2[2]) +
                        float(sc8[5]) * (acc1[3] * (1.f/16.f) + 16.f * acc2[3]))
                 - dmin_r[r] * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                           sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
      }}
    }}
  }}
  for (uint r = 0; r < NR0; r++) {{
    const uint row = first_row + r;
    if (row >= N) break;
    for (uint tc = 0; tc < ncol; tc++) {{
      float t = simd_sum(sumf[r * TILE + tc]);
      if (tiisg == 0) data0[(col0 + tc) * N + row] = {out_ty}(t);
    }}
  }}
}}
"""


def _metal_src_q6_ws(N: int, nblk: int, T: int, x_half: bool, out_half: bool, tile: int) -> str:
  """Weight-stationary Q6_K: NR0/NSG occupancy, accumulate `tile` cols/TG."""
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
  constexpr uint N = {N}u, nblk = {nblk}u, T = {T}u, TILE = {tile}u, NSG = {_NSG}u, NR0 = {_nr0(_Q6_K)}u;
  constexpr uint8_t kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
  const uint first_row = (tgpig.x * NSG + sgitg) * NR0;
  if (first_row >= N) return;
  const uint col0 = tgpig.y * TILE;
  if (col0 >= T) return;
  const uint ncol = (col0 + TILE <= T) ? TILE : (T - col0);
  const short tid = tiisg / 2;
  const short ix = tiisg % 2;
  const short ip = tid / 8;
  const short il = tid % 8;
  const short l0 = 4 * il;
  const short is = 8 * ip + l0 / 16;
  const short y_offset = 128 * ip + l0;
  const short q_offset_l = 64 * ip + l0;
  const short q_offset_h = 32 * ip + l0;
  float sumf[4 * TILE];
  for (uint i = 0; i < 4u * TILE; i++) sumf[i] = 0.f;
  float yl[16];
  for (uint ib = ix; ib < nblk; ib += 2u) {{
    uint8_t q1c[4][4], q2c[4][4], qhc[4][4];
    float d_r[4];
    int8_t sc0_r[4], sc2_r[4], sc4_r[4], sc6_r[4];
    for (uint r = 0; r < NR0; r++) {{
      const uint row = first_row + r;
      if (row >= N) continue;
      device const block_q6_K& blk = data2[row * nblk + ib];
      device const uint8_t* q1p = blk.ql + q_offset_l;
      device const uint8_t* q2p = q1p + 32;
      device const uint8_t* qhp = blk.qh + q_offset_h;
      device const int8_t* scp = blk.scales + is;
      d_r[r] = float(blk.d);
      sc0_r[r] = scp[0]; sc2_r[r] = scp[2]; sc4_r[r] = scp[4]; sc6_r[r] = scp[6];
      for (short l = 0; l < 4; ++l) {{ q1c[r][l] = q1p[l]; q2c[r][l] = q2p[l]; qhc[r][l] = qhp[l]; }}
    }}
    for (uint tc = 0; tc < ncol; tc++) {{
      const uint col = col0 + tc;
      const uint xoff = col * nblk * 256u;
      for (short l = 0; l < 4; ++l) {{
        const uint yb = ib * 256u + y_offset;
        yl[4*l + 0] = {y('data1[xoff + yb + l + 0]')};
        yl[4*l + 1] = {y('data1[xoff + yb + l + 32]')};
        yl[4*l + 2] = {y('data1[xoff + yb + l + 64]')};
        yl[4*l + 3] = {y('data1[xoff + yb + l + 96]')};
      }}
      for (uint r = 0; r < NR0; r++) {{
        const uint row = first_row + r;
        if (row >= N) continue;
        float4 sums = {{0.f, 0.f, 0.f, 0.f}};
        for (short l = 0; l < 4; ++l) {{
          sums[0] += yl[4*l + 0] * float((int8_t)((q1c[r][l] & 0xF) | ((qhc[r][l] & kmask1) << 4)) - 32);
          sums[1] += yl[4*l + 1] * float((int8_t)((q2c[r][l] & 0xF) | ((qhc[r][l] & kmask2) << 2)) - 32);
          sums[2] += yl[4*l + 2] * float((int8_t)((q1c[r][l] >> 4) | ((qhc[r][l] & kmask3) << 0)) - 32);
          sums[3] += yl[4*l + 3] * float((int8_t)((q2c[r][l] >> 4) | ((qhc[r][l] & kmask4) >> 2)) - 32);
        }}
        sumf[r * TILE + tc] += d_r[r] * (sums[0] * float(sc0_r[r]) + sums[1] * float(sc2_r[r]) +
                                        sums[2] * float(sc4_r[r]) + sums[3] * float(sc6_r[r]));
      }}
    }}
  }}
  for (uint r = 0; r < NR0; r++) {{
    const uint row = first_row + r;
    if (row >= N) break;
    for (uint tc = 0; tc < ncol; tc++) {{
      float t = simd_sum(sumf[r * TILE + tc]);
      if (tiisg == 0) data0[(col0 + tc) * N + row] = {out_ty}(t);
    }}
  }}
}}
"""


def _metal_src_q4_ws_row(N: int, nblk: int, T: int, x_half: bool, out_half: bool) -> str:
  """Weight-stationary Q4_K: one row/TG, cache quant payload, accumulate all T."""
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
  constexpr uint N = {N}u, nblk = {nblk}u, T = {T}u;
  constexpr uint16_t kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
  const short ix = tiisg / 8;
  const short it = tiisg % 8;
  const short iq = it / 4;
  const short ir = it % 4;
  const uint row = tgpig.x;
  if (row >= N) return;
  float yl[16], yh[16];
  float sumf[T];
  for (uint c = 0; c < T; c++) sumf[c] = 0.f;
  uint16_t sc16[4];
  thread const uint8_t* sc8 = (thread const uint8_t*)sc16;
  for (uint ib = ix; ib < nblk; ib += 4u) {{
    device const block_q4_K& blk = data2[row * nblk + ib];
    device const uint16_t* sc = (device const uint16_t*)blk.scales + iq;
    device const uint16_t* q1p = (device const uint16_t*)blk.qs + 16 * iq + 4 * ir;
    device const uint16_t* q2p = q1p + 32;
    const float d = float(blk.d), dmin = float(blk.dmin);
    sc16[0] = sc[0] & kmask1;
    sc16[1] = sc[2] & kmask1;
    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
    uint16_t q1c[4], q2c[4];
    for (short i = 0; i < 4; ++i) {{ q1c[i] = q1p[i]; q2c[i] = q2p[i]; }}
    for (uint col = 0; col < T; col++) {{
      const uint xoff = col * nblk * 256u;
      float4 sumy = {{0.f, 0.f, 0.f, 0.f}};
      const uint base = ib * 256u + 64u * iq + 8u * ir;
      for (short i = 0; i < 8; ++i) {{
        yl[i+0] = {y('data1[xoff + base + i + 0]')}; sumy[0] += yl[i+0];
        yl[i+8] = {y('data1[xoff + base + i + 32]')}; sumy[1] += yl[i+8];
        yh[i+0] = {y('data1[xoff + base + i + 128]')}; sumy[2] += yh[i+0];
        yh[i+8] = {y('data1[xoff + base + i + 160]')}; sumy[3] += yh[i+8];
      }}
      float4 acc1 = {{0.f, 0.f, 0.f, 0.f}};
      float4 acc2 = {{0.f, 0.f, 0.f, 0.f}};
      for (short i = 0; i < 4; ++i) {{
        acc1[0] += yl[2*i + 0] * float(q1c[i] & 0x000F);
        acc1[1] += yl[2*i + 1] * float(q1c[i] & 0x0F00);
        acc1[2] += yl[2*i + 8] * float(q1c[i] & 0x00F0);
        acc1[3] += yl[2*i + 9] * float(q1c[i] & 0xF000);
        acc2[0] += yh[2*i + 0] * float(q2c[i] & 0x000F);
        acc2[1] += yh[2*i + 1] * float(q2c[i] & 0x0F00);
        acc2[2] += yh[2*i + 8] * float(q2c[i] & 0x00F0);
        acc2[3] += yh[2*i + 9] * float(q2c[i] & 0xF000);
      }}
      sumf[col] += d * ((acc1[0] + (1.f/256.f) * acc1[1]) * float(sc8[0]) +
                      (acc1[2] + (1.f/256.f) * acc1[3]) * float(sc8[1]) * (1.f/16.f) +
                      (acc2[0] + (1.f/256.f) * acc2[1]) * float(sc8[4]) +
                      (acc2[2] + (1.f/256.f) * acc2[3]) * float(sc8[5]) * (1.f/16.f))
               - dmin * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                         sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
    }}
  }}
  for (uint col = 0; col < T; col++) {{
    float t = simd_sum(sumf[col]);
    if (tiisg == 0) data0[col * N + row] = {out_ty}(t);
  }}
}}
"""


def _metal_src_q5_ws_row(N: int, nblk: int, T: int, x_half: bool, out_half: bool) -> str:
  """Weight-stationary Q5_K: one row/TG."""
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
  constexpr uint N = {N}u, nblk = {nblk}u, T = {T}u;
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
  const uint row = tgpig.x;
  if (row >= N) return;
  float yl[16], yh[16];
  float sumf[T];
  for (uint c = 0; c < T; c++) sumf[c] = 0.f;
  uint16_t sc16[4];
  thread const uint8_t* sc8 = (thread const uint8_t*)sc16;
  for (uint ib = ix; ib < nblk; ib += 4u) {{
    device const block_q5_K& blk = data2[row * nblk + ib];
    device const uint8_t* q1p = blk.qs + q_offset;
    device const uint8_t* q2p = q1p + 64;
    device const uint8_t* qhp = blk.qh + l0;
    device const uint16_t* a = (device const uint16_t*)blk.scales + iq;
    const float d = float(blk.d), dmin = float(blk.dmin);
    sc16[0] = a[0] & kmask1;
    sc16[1] = a[2] & kmask1;
    sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
    sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
    uint8_t q1c[8], q2c[8], qhc[8];
    for (short l = 0; l < 8; ++l) {{ q1c[l] = q1p[l]; q2c[l] = q2p[l]; qhc[l] = qhp[l]; }}
    for (uint col = 0; col < T; col++) {{
      const uint xoff = col * nblk * 256u;
      float4 sumy = {{0.f, 0.f, 0.f, 0.f}};
      const uint y1 = ib * 256u + y_offset;
      for (short l = 0; l < 8; ++l) {{
        yl[l+0] = {y('data1[xoff + y1 + l + 0]')}; sumy[0] += yl[l+0];
        yl[l+8] = {y('data1[xoff + y1 + l + 32]')}; sumy[1] += yl[l+8];
        yh[l+0] = {y('data1[xoff + y1 + l + 128]')}; sumy[2] += yh[l+0];
        yh[l+8] = {y('data1[xoff + y1 + l + 160]')}; sumy[3] += yh[l+8];
      }}
      float4 acc1 = {{0.f, 0.f, 0.f, 0.f}};
      float4 acc2 = {{0.f, 0.f, 0.f, 0.f}};
      for (short l = 0; l < 8; ++l) {{
        const uint8_t h = qhc[l];
        acc1[0] += yl[l+0] * float(q1c[l] & 0x0F);
        acc1[1] += yl[l+8] * float(q1c[l] & 0xF0);
        acc1[2] += yh[l+0] * float(q2c[l] & 0x0F);
        acc1[3] += yh[l+8] * float(q2c[l] & 0xF0);
        acc2[0] += (h & hm1) ? yl[l+0] : 0.f;
        acc2[1] += (h & hm2) ? yl[l+8] : 0.f;
        acc2[2] += (h & hm3) ? yh[l+0] : 0.f;
        acc2[3] += (h & hm4) ? yh[l+8] : 0.f;
      }}
      sumf[col] += d * (float(sc8[0]) * (acc1[0] + 16.f * acc2[0]) +
                      float(sc8[1]) * (acc1[1] * (1.f/16.f) + 16.f * acc2[1]) +
                      float(sc8[4]) * (acc1[2] + 16.f * acc2[2]) +
                      float(sc8[5]) * (acc1[3] * (1.f/16.f) + 16.f * acc2[3]))
               - dmin * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                         sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
    }}
  }}
  for (uint col = 0; col < T; col++) {{
    float t = simd_sum(sumf[col]);
    if (tiisg == 0) data0[col * N + row] = {out_ty}(t);
  }}
}}
"""


def _metal_src_q6_ws_row(N: int, nblk: int, T: int, x_half: bool, out_half: bool) -> str:
  """Weight-stationary Q6_K: one row/TG."""
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
  constexpr uint N = {N}u, nblk = {nblk}u, T = {T}u;
  constexpr uint8_t kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
  const uint row = tgpig.x;
  if (row >= N) return;
  const short tid = tiisg / 2;
  const short ix = tiisg % 2;
  const short ip = tid / 8;
  const short il = tid % 8;
  const short l0 = 4 * il;
  const short is = 8 * ip + l0 / 16;
  const short y_offset = 128 * ip + l0;
  const short q_offset_l = 64 * ip + l0;
  const short q_offset_h = 32 * ip + l0;
  float sumf[T];
  for (uint c = 0; c < T; c++) sumf[c] = 0.f;
  float yl[16];
  for (uint ib = ix; ib < nblk; ib += 2u) {{
    device const block_q6_K& blk = data2[row * nblk + ib];
    device const uint8_t* q1p = blk.ql + q_offset_l;
    device const uint8_t* q2p = q1p + 32;
    device const uint8_t* qhp = blk.qh + q_offset_h;
    device const int8_t* scp = blk.scales + is;
    const float d = float(blk.d);
    const int8_t sc0 = scp[0], sc2 = scp[2], sc4 = scp[4], sc6 = scp[6];
    uint8_t q1c[4], q2c[4], qhc[4];
    for (short l = 0; l < 4; ++l) {{ q1c[l] = q1p[l]; q2c[l] = q2p[l]; qhc[l] = qhp[l]; }}
    for (uint col = 0; col < T; col++) {{
      const uint xoff = col * nblk * 256u;
      for (short l = 0; l < 4; ++l) {{
        const uint yb = ib * 256u + y_offset;
        yl[4*l + 0] = {y('data1[xoff + yb + l + 0]')};
        yl[4*l + 1] = {y('data1[xoff + yb + l + 32]')};
        yl[4*l + 2] = {y('data1[xoff + yb + l + 64]')};
        yl[4*l + 3] = {y('data1[xoff + yb + l + 96]')};
      }}
      float4 sums = {{0.f, 0.f, 0.f, 0.f}};
      for (short l = 0; l < 4; ++l) {{
        sums[0] += yl[4*l + 0] * float((int8_t)((q1c[l] & 0xF) | ((qhc[l] & kmask1) << 4)) - 32);
        sums[1] += yl[4*l + 1] * float((int8_t)((q2c[l] & 0xF) | ((qhc[l] & kmask2) << 2)) - 32);
        sums[2] += yl[4*l + 2] * float((int8_t)((q1c[l] >> 4) | ((qhc[l] & kmask3) << 0)) - 32);
        sums[3] += yl[4*l + 3] * float((int8_t)((q2c[l] >> 4) | ((qhc[l] & kmask4) >> 2)) - 32);
      }}
      sumf[col] += d * (sums[0] * float(sc0) + sums[1] * float(sc2) + sums[2] * float(sc4) + sums[3] * float(sc6));
    }}
  }}
  for (uint col = 0; col < T; col++) {{
    float t = simd_sum(sumf[col]);
    if (tiisg == 0) data0[col * N + row] = {out_ty}(t);
  }}
}}
"""


# --- T>1: llama.cpp mul_mv_ext style (dequant once, dot r1ptg Y cols) ---
# For T>8 llama uses mul_mm; EXT with r1ptg=4 still cuts weight traffic ~4x vs Y=T.
_EXT_R1 = getenv("QUANT_GEMV_EXT_R1", 4)  # cols accumulated per TG (2..5)
_EXT_NSG = getenv("QUANT_GEMV_EXT_NSG", 2)
_EXT_NXPSG = getenv("QUANT_GEMV_EXT_NXPSG", 8)  # 4/8/16


def _dequant_helpers(ggml_type: int) -> str:
  scale = r"""
static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
  return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
               : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                        uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}
"""
  if ggml_type == _Q4_K:
    return scale + r"""
void dequantize_q4_K(device const block_q4_K * xb, short il, thread float4x4 & reg) {
  device const uchar * q = xb->qs;
  short is = (il/4) * 2;
  q = q + (il/4) * 32 + 16 * (il&1);
  il = il & 3;
  const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
  const float d = il < 2 ? float(xb->d) : float(xb->d) / 16.f;
  const float minv = float(xb->dmin);
  const float dl = d * float(sc[0]);
  const float ml = minv * float(sc[1]);
  const ushort mask = il < 2 ? 0x0F : 0xF0;
  for (int i = 0; i < 16; ++i) reg[i/4][i%4] = dl * float(q[i] & mask) - ml;
}
"""
  if ggml_type == _Q5_K:
    return scale + r"""
void dequantize_q5_K(device const block_q5_K *xb, short il, thread float4x4 & reg) {
  device const uint8_t * q = xb->qs;
  device const uint8_t * qh = xb->qh;
  short is = (il/4) * 2;
  q = q + 32 * (il/4) + 16 * (il&1);
  qh = qh + 16 * (il&1);
  uint8_t ul = 1 << (il/2);
  il = il & 3;
  const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
  const float d = il < 2 ? float(xb->d) : float(xb->d) / 16.f;
  const float minv = float(xb->dmin);
  const float dl = d * float(sc[0]);
  const float ml = minv * float(sc[1]);
  const ushort mask = il<2 ? 0x0F : 0xF0;
  const float qh_val = il<2 ? 16.f : 256.f;
  for (int i = 0; i < 16; ++i)
    reg[i/4][i%4] = dl * (float(q[i] & mask) + (qh[i] & ul ? qh_val : 0.f)) - ml;
}
"""
  if ggml_type == _Q6_K:
    return scale + r"""
void dequantize_q6_K(device const block_q6_K *xb, short il, thread float4x4 & reg) {
  const float d_all = float(xb->d);
  device const uint16_t * ql = (device const uint16_t *)xb->ql;
  device const uint16_t * qh = (device const uint16_t *)xb->qh;
  device const int8_t * scales = (device const int8_t *)xb->scales;
  ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
  qh = qh + 16*(il/8) + 8*(il&1);
  float sc = float(scales[(il%2) + 2 * ((il/2))]);
  il = (il/2) & 3;
  const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0u : 0x30303030u) : (il>0 ? 0x0C0C0C0Cu : 0x03030303u);
  const uint32_t kmask2 = il>1 ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
  const float ml = d_all * sc * 32.f;
  const float dl0 = d_all * sc;
  const float dl1 = dl0 / 256.f;
  const float dl2 = dl0 / (256.f * 256.f);
  const float dl3 = dl0 / (256.f * 256.f * 256.f);
  const uint8_t shr_h = il>2 ? 2 : 0;
  const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
  const uint8_t shr_l = il>1 ? 4 : 0;
  for (int i = 0; i < 4; ++i) {
    const uint32_t low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
    const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
    const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
    reg[i][0] = dl0 * float(q & 0xFFu) - ml;
    reg[i][1] = dl1 * float(q & 0xFF00u) - ml;
    reg[i][2] = dl2 * float(q & 0xFF0000u) - ml;
    reg[i][3] = dl3 * float(q & 0xFF000000u) - ml;
  }
}
"""
  raise ValueError(ggml_type)

def _ext_r1(T: int) -> int:
  r = int(_EXT_R1)
  if r < 2: r = 2
  if r > 5: r = 5
  return max(1, min(r, T))

def _metal_src_ext(ggml_type: int, N: int, nblk: int, T: int, x_half: bool, out_half: bool) -> str:
  """mul_mv_ext-style: 1 weight dequant, r1ptg Y columns. Grid Y = ceil(T/r1ptg)."""
  x_ty = "half" if x_half else "float"
  out_ty = "half" if out_half else "float"
  r1 = _ext_r1(T)
  nsg = int(_EXT_NSG)
  nxpsg = int(_EXT_NXPSG)
  if nxpsg not in (4, 8, 16): nxpsg = 8
  nypsg = 32 // nxpsg
  block = _BLOCK[ggml_type]
  bt = _BLOCK_T[ggml_type]
  deq = {_Q4_K: "dequantize_q4_K", _Q5_K: "dequantize_q5_K", _Q6_K: "dequantize_q6_K"}[ggml_type]
  # K = nblk*256; each thread walks 16-wide chunks via float4x4
  return f"""
#include <metal_stdlib>
using namespace metal;
{block}
{_dequant_helpers(ggml_type)}
kernel void kquant_gemv(
  device {out_ty}* data0, const device {x_ty}* data1, const device {bt}* data2,
  uint3 tgpig [[threadgroup_position_in_grid]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]]
) {{
  constexpr uint N = {N}u, nblk = {nblk}u, T = {T}u, K = {nblk * 256}u;
  constexpr short NSG = {nsg}, nxpsg = {nxpsg}, nypsg = {nypsg}, r1ptg = {r1};
  constexpr short chpb = 16; // 256/16 float4x4 chunks per QK_K block
  const short tx = tiisg % nxpsg;
  const short ty = tiisg / nxpsg;
  const uint row = tgpig.x * (nypsg * NSG) + nypsg * sgitg + ty;
  const uint col0 = tgpig.y * r1ptg;
  if (col0 >= T) return;
  const uint ncol = (col0 + r1ptg <= T) ? r1ptg : (T - col0);

  device const {bt}* xq = (row < N) ? (data2 + row * nblk) : data2;
  // y pointers: one float4x4 stream per column (K is multiple of 16)
  const device float4x4* y4[{r1}];
  for (short ir = 0; ir < r1ptg; ++ir) {{
    if (ir < (short)ncol) {{
      const device {x_ty}* yp = data1 + (col0 + ir) * K;
      y4[ir] = (const device float4x4*)yp + tx;
    }} else y4[ir] = (const device float4x4*)data1;
  }}
  float sumf[{r1}];
  for (short ir = 0; ir < r1ptg; ++ir) sumf[ir] = 0.f;

  short cch = tx % chpb;
  device const {bt}* xqp = xq + (tx / chpb);
  // ich indexes 16-wide chunks along K
  for (uint ich = tx; 16u * ich < K; ich += nxpsg) {{
    float4x4 lx;
    if (row < N) {deq}(xqp, cch, lx);
    else lx = float4x4(0.f);
    cch += nxpsg;
    if (cch >= chpb) {{ xqp += cch / chpb; cch %= chpb; }}
    for (short ir = 0; ir < r1ptg; ++ir) {{
      if (ir >= (short)ncol) continue;
      // cast half4x4-as-float4x4 load: rebuild from half if needed
      float4x4 yy;
      {{
        const device {x_ty}* yp = data1 + (col0 + ir) * K + 16u * ich;
        for (int i = 0; i < 16; i++) {{
          yy[i/4][i%4] = float(yp[i]);
        }}
      }}
      sumf[ir] += dot(lx[0], yy[0]) + dot(lx[1], yy[1]) + dot(lx[2], yy[2]) + dot(lx[3], yy[3]);
    }}
  }}
  // reduce across nxpsg threads in the row (tx dimension)
  for (short ir = 0; ir < r1ptg; ++ir) {{
    if (nxpsg >= 32) sumf[ir] += simd_shuffle_down(sumf[ir], 16);
    if (nxpsg >= 16) sumf[ir] += simd_shuffle_down(sumf[ir], 8);
    if (nxpsg >= 8) sumf[ir] += simd_shuffle_down(sumf[ir], 4);
    if (nxpsg >= 4) sumf[ir] += simd_shuffle_down(sumf[ir], 2);
    if (nxpsg >= 2) sumf[ir] += simd_shuffle_down(sumf[ir], 1);
  }}
  if (tx == 0 && row < N) {{
    for (short ir = 0; ir < (short)ncol; ++ir)
      data0[(col0 + ir) * N + row] = {out_ty}(sumf[ir]);
  }}
}}
"""


# --- T>8: llama.cpp-style mul_mm (simdgroup MMA, weight tile in threadgroup) ---
# Fixed 64x32 / 4 SG (llama legacy). For T<=16, sgitg 2..3 are padded cols — skip their MMA.
_MM_NSG = 4
_MM_NR0 = 64
_MM_NR1 = 32

def _metal_src_mm(ggml_type: int, N: int, nblk: int, T: int, x_half: bool, out_half: bool) -> str:
  """Quantized GEMM via dequant-to-threadgroup + simdgroup_multiply_accumulate."""
  x_ty = "half" if x_half else "float"
  out_ty = "half" if out_half else "float"
  block = _BLOCK[ggml_type]
  bt = _BLOCK_T[ggml_type]
  deq = {_Q4_K: "dequantize_q4_K", _Q5_K: "dequantize_q5_K", _Q6_K: "dequantize_q6_K"}[ggml_type]
  return f"""
#include <metal_stdlib>
using namespace metal;
{block}
{_dequant_helpers(ggml_type)}
kernel void kquant_gemv(
  device {out_ty}* data0, const device {x_ty}* data1, const device {bt}* data2,
  uint3 tgpig [[threadgroup_position_in_grid]],
  ushort tiitg [[thread_index_in_threadgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]]
) {{
  constexpr uint N = {N}u, nblk = {nblk}u, T = {T}u, K = {nblk*256}u;
  constexpr short NR0 = {_MM_NR0}, NR1 = {_MM_NR1}, NK = 32, NL0 = 2, NL1 = 4, nl = 16;
  threadgroup half sa[2048]; // NR0 * NK
  threadgroup half sb[1024]; // NR1 * NK
  const int r0 = tgpig.y * NR0;
  const int r1 = tgpig.x * NR1;
  const short nr0 = (N - r0 < NR0) ? short(N - r0) : NR0;
  const short nr1 = (T - r1 < NR1) ? short(T - r1) : NR1;
  const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : short(nr0 - 1);
  const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : short(nr1 - 1);
  const short il0 = (tiitg % NL0);
  short il = il0;
  const short offset1 = il0 / nl;
  device const {bt}* x = data2 + (uint)(r0 + lr0) * nblk + offset1;
  const short iy = 8 * (tiitg % NL1);
  const device {x_ty}* y = data1 + (uint)(r1 + lr1) * K + iy;
  // padded col half (sgitg 2..3 when T<=16) still helps load sa, but skips MMA/store
  const bool do_mma = (r1 + 16*(sgitg/2)) < (int)T;
  simdgroup_half8x8 ma[4];
  simdgroup_half8x8 mb[2];
  simdgroup_float8x8 mc[8];
  for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
  for (uint loop_k = 0; loop_k < K; loop_k += NK) {{
    float4x4 temp_a;
    {deq}(x, il, temp_a);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short i = 0; i < 16; i++) {{
      const short sx = 2*il0 + i/8;
      const short sy = (tiitg/NL0)/8;
      const short lx = (tiitg/NL0)%8;
      const short ly = i%8;
      const short ib = 8*sx + sy;
      *(sa + 64*ib + 8*ly + lx) = half(temp_a[i/4][i%4]);
    }}
    {{
      const short sx = (tiitg%NL1);
      const short sy = (tiitg/NL1)/8;
      const short ly = (tiitg/NL1)%8;
      const short ib = 4*sx + sy;
      if (loop_k + iy + 8u <= K) {{
        *(threadgroup half2x4*)(sb + 64*ib + 8*ly) = half2x4(*(const device {x_ty}2x4*)(y));
      }} else {{
        for (short i = 0; i < 8; ++i) {{
          const uint kpos = loop_k + iy + i;
          *(sb + 64*ib + 8*ly + i) = (kpos < K) ? half(y[i]) : half(0);
        }}
      }}
    }}
    il = (il + 2 < nl) ? il + 2 : il % 2;
    x = (il < 2) ? x + (2 + nl - 1)/nl : x;
    y += NK;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (do_mma) {{
      threadgroup const half* lsma = (sa + 4*64*(sgitg%2));
      threadgroup const half* lsmb = (sb + 2*64*(sgitg/2));
      for (short ik = 0; ik < NK/8; ik++) {{
        simdgroup_barrier(mem_flags::mem_none);
        for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8);
        simdgroup_barrier(mem_flags::mem_none);
        for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8);
        simdgroup_barrier(mem_flags::mem_none);
        for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
        lsma += 8*64;
        lsmb += 4*64;
      }}
    }}
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float* temp_str = ((threadgroup float*)sa) + 32*(sgitg&1) + (16*(sgitg>>1))*NR0;
  if (do_mma) {{
    for (short i = 0; i < 8; i++)
      simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0);
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgitg == 0) {{
    for (int j = tiitg; j < nr1; j += 32) {{
      for (int i = 0; i < nr0; i++) {{
        data0[(r1 + j) * N + (r0 + i)] = {out_ty}(temp_str[j*NR0 + i]);
      }}
    }}
  }}
}}
"""


def _ws_tile(T: int) -> int:
  """Cols accumulated in-kernel under QUANT_GEMV_WS. Default=4; 0→T."""
  tile = int(getenv("QUANT_GEMV_WS_TILE", 4))
  if tile <= 0: tile = T
  return max(1, min(tile, T, 16))


def _metal_src(ggml_type: int, N: int, nblk: int, T: int, x_half: bool, out_half: bool) -> str:
  # T>8 default: mul_mm (simdgroup MMA). QUANT_GEMV_MM=0 to disable.
  if T > 8 and getenv("QUANT_GEMV_MM", 1) and ggml_type in (_Q4_K, _Q5_K, _Q6_K):
    return _metal_src_mm(ggml_type, N, nblk, T, x_half, out_half)
  if T > 1 and getenv("QUANT_GEMV_EXT", 0) and ggml_type in (_Q4_K, _Q5_K, _Q6_K):
    return _metal_src_ext(ggml_type, N, nblk, T, x_half, out_half)
  if T > 1 and getenv("QUANT_GEMV_WS", 0):
    tile = _ws_tile(T)
    if ggml_type == _Q4_K: return _metal_src_q4_ws(N, nblk, T, x_half, out_half, tile)
    if ggml_type == _Q5_K: return _metal_src_q5_ws(N, nblk, T, x_half, out_half, tile)
    if ggml_type == _Q6_K: return _metal_src_q6_ws(N, nblk, T, x_half, out_half, tile)
  if ggml_type == _Q4_K: return _metal_src_q4(N, nblk, T, x_half, out_half)
  if ggml_type == _Q5_K: return _metal_src_q5(N, nblk, T, x_half, out_half)
  if ggml_type == _Q6_K: return _metal_src_q6(N, nblk, T, x_half, out_half)
  return _metal_src_generic(ggml_type, N, nblk, T, x_half, out_half)

@functools.cache
def _compiled_lib(ggml_type: int, N: int, nblk: int, T: int, nsg: int, nr0: int, x_half: bool, out_half: bool, ws: int = 0, tile: int = 0) -> bytes:
  from tinygrad.runtime.ops_metal import MetalCompiler
  return MetalCompiler().compile(_metal_src(ggml_type, N, nblk, T, x_half, out_half))

def _program(out: UOp, x: UOp, qweight: UOp, *, ggml_type: int, N: int, K: int, x_half: bool, out_half: bool) -> UOp:
  """FUSED_KQUANT_GEMV escape hatch — T=1 only."""
  nblk, rpt, T = K // _QK_K, _rows_per_tg(ggml_type), 1
  out_f, x_f, qw_f = out.flatten(), x.flatten(), qweight.flatten()
  gidx0 = UOp.special((N + rpt - 1) // rpt, "gidx0")
  lidx0 = UOp.special(32 * _NSG, "lidx0")
  i = (gidx0 * (32 * _NSG) + lidx0) % N
  store = out_f[i].store(x_f[i % K].load().cast(dtypes.float32) * 0.0 + qw_f[0].load().cast(dtypes.float32) * 0.0)
  sink = UOp.sink(store, gidx0, lidx0, arg=KernelInfo(name="kquant_gemv", opts_to_apply=()))
  pi = ProgramInfo(name="kquant_gemv",
                   global_size=((N + rpt - 1) // rpt, T, 1),
                   local_size=(32 * _NSG, 1, 1),
                   globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target("METAL"))
  src = _metal_src(ggml_type, N, nblk, T, x_half, out_half)
  lib = _compiled_lib(ggml_type, N, nblk, T, _NSG, _nr0(ggml_type), x_half, out_half)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=()), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)), arg=pi)

program_uop = _program

_BLOCK_BYTES = {_Q4_K: 144, _Q5_K: 176, _Q6_K: 210}

def _match_packed_gemv(ast: UOp) -> tuple[int, int, int, int, int, int, int, bool, bool] | None:
  """Isolated k-quant GEMV/small-GEMM: 3 bufs, 1 REDUCE, concrete N/K/T, nbytes match.

  T=1 decode: one WEAK range (N). T>1 verify: WEAK ranges whose product is T*N;
  N is recovered from packed uchar nbytes so T = out_elems // N.
  """
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
  try: out_elems = int(prod(int(cast(int, r.src[0].ssimplify())) for r in weak))
  except Exception: return None

  params = [u for u in nodes if u.op is Ops.PARAM and isinstance(u.arg, ParamArg) and u.dtype.itemsize > 0]
  if len({p.arg.slot for p in params}) != 3: return None
  uchar = [p for p in params if p.dtype == dtypes.uchar]
  if len(uchar) != 1: return None
  qw_slot, qw_bytes = uchar[0].arg.slot, int(cast(int, uchar[0].src[0].ssimplify()))
  # Factor out_elems = T * N with N from packed nbytes (not from WEAK prod alone).
  match = None
  for gt, bb in _BLOCK_BYTES.items():
    unit = nblk * bb
    if unit <= 0 or qw_bytes % unit: continue
    N = qw_bytes // unit
    if N < 1 or out_elems % N: continue
    T = out_elems // N
    if T < 1 or T > getenv("QUANT_GEMV_MAX_T", 16): continue
    match = (gt, N, T)
    break
  if match is None: return None
  ggml_type, N, T = match

  # allow oversized reused buffers; ranges define N/K/T
  x_cands = [p for p in params if p.arg.slot not in (out_p.arg.slot, qw_slot) and p.dtype.itemsize in (2, 4)]
  if len(x_cands) != 1: return None
  if not any(u.op is Ops.BITCAST for u in nodes): return None
  if not any(u.op in (Ops.AND, Ops.SHR) for u in nodes): return None
  x_p = x_cands[0]
  try:
    if int(cast(int, x_p.src[0].ssimplify())) < T * K: return None
  except Exception: return None
  return (ggml_type, N, K, T, out_p.arg.slot, x_p.arg.slot, qw_slot,
          x_p.dtype.itemsize == 2 and bool(getenv("KQUANT_X_HALF", 1)), out_p.dtype.itemsize == 2)

def try_packed_kquant_gemv(ast: UOp, renderer: Renderer) -> UOp | None:
  if not getenv("QUANT_GEMV_LOWER", 1) or not isinstance(renderer, MetalRenderer): return None
  if (m := _match_packed_gemv(ast)) is None: return None
  ggml_type, N, K, T, out_slot, x_slot, qw_slot, x_half, out_half = m
  nblk = K // _QK_K
  ws = int(getenv("QUANT_GEMV_WS", 0)) if T > 1 else 0
  use_mm = bool(T > 8 and getenv("QUANT_GEMV_MM", 1) and ggml_type in (_Q4_K, _Q5_K, _Q6_K))
  use_ext = bool((not use_mm) and T > 1 and getenv("QUANT_GEMV_EXT", 0) and ggml_type in (_Q4_K, _Q5_K, _Q6_K))
  rpt = _rows_per_tg(ggml_type)
  if use_mm:
    gsz, lsz = ((T + _MM_NR1 - 1) // _MM_NR1, (N + _MM_NR0 - 1) // _MM_NR0, 1), (32 * _MM_NSG, 1, 1)
    tile, ws = 0, 0
  elif use_ext:
    nsg = int(_EXT_NSG)
    nxpsg = int(_EXT_NXPSG)
    if nxpsg not in (4, 8, 16): nxpsg = 8
    nypsg = 32 // nxpsg
    r0ptg = nypsg * nsg
    r1 = _ext_r1(T)
    gsz, lsz = ((N + r0ptg - 1) // r0ptg, (T + r1 - 1) // r1, 1), (32 * nsg, 1, 1)
    tile, ws = r1, 0
  elif ws:
    tile = _ws_tile(T)
    ny = (T + tile - 1) // tile
    gsz, lsz = ((N + rpt - 1) // rpt, ny, 1), (32 * _NSG, 1, 1)
  else:
    tile = 0
    gsz, lsz = ((N + rpt - 1) // rpt, T, 1), (32 * _NSG, 1, 1)
  pi = ProgramInfo(name="kquant_gemv",
                   global_size=gsz, local_size=lsz,
                   globals=(out_slot, x_slot, qw_slot), outs=(out_slot,), ins=(x_slot, qw_slot),
                   target=renderer.target)
  src = _metal_src(ggml_type, N, nblk, T, x_half, out_half)
  # encode mm nr1 into tile so _compiled_lib cache distinguishes T<=16 vs T>16 kernels
  lib = _compiled_lib(ggml_type, N, nblk, T, _NSG, _nr0(ggml_type), x_half, out_half, ws, tile)
  return UOp(Ops.PROGRAM, src=(ast, UOp(Ops.LINEAR, src=()), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)), arg=pi)

