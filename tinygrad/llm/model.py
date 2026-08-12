from __future__ import annotations
import functools, itertools, pathlib
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes
from tinygrad.nn import Linear
from tinygrad.helpers import prod
from tinygrad.llm.gguf import gguf_load, gguf_load_packed, ggml_data_to_tensor, _GGML_NATIVE
from tinygrad.llm.quant import is_ggml_quant, replace_linear_with_quant
from tinygrad.uop.ops import resolve

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device)

class ExpertWeights:
  """Like Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def attention_mask(T:int|UOp, start_pos:int|UOp, dtype, sliding_window:int=0) -> Tensor|None:
  """Causal mask, optionally with llama.cpp LLAMA_SWA_TYPE_STANDARD windowing.

  Masks key positions where key_pos > query_pos (causal) or query_pos - key_pos >= sliding_window.
  Returns None when no masking is required (decode step with full context inside the window).
  """
  kv_len = start_pos + T
  need_causal = resolve(T != 1)
  need_swa = sliding_window > 0 and resolve(kv_len > sliding_window)
  if not need_causal and not need_swa: return None
  mask = Tensor.zeros((1, 1, T, kv_len), dtype=dtype, buffer=False)
  if need_causal:
    mask = mask + Tensor.full((1, 1, T, kv_len), float("-inf"), dtype=dtype, buffer=False).triu(start_pos+1)
  if need_swa:
    # -inf where col <= row + (start_pos - sliding_window)  <=>  query_pos - key_pos >= window
    mask = mask + Tensor.full((1, 1, T, kv_len), float("-inf"), dtype=dtype, buffer=False).tril(start_pos - sliding_window)
  return mask

def iswa_cache_len(max_context:int, sliding_window:int) -> int:
  """llama.cpp-style ISWA cell count: pad(min(n_ctx, n_swa), 256)."""
  if sliding_window <= 0: return max_context
  padded = ((sliding_window + 255) // 256) * 256
  return min(max_context, padded)

def iswa_attention_mask(T:int|UOp, start_pos:int|UOp, kv_cache_len:int, dtype, sliding_window:int) -> Tensor:
  """Mask for right-aligned shift-register SWA KV (length kv_cache_len).

  Cache slots hold the most recent keys right-aligned; leading slots may be padding.
  abs_key(c) = start_pos + T - kv_cache_len + c.
  """
  mask = Tensor.zeros((1, 1, T, kv_cache_len), dtype=dtype, buffer=False)
  # padding: columns [0, pad_end) invalid for every query row (broadcast a 1-row tril)
  pad_end = kv_cache_len - (start_pos + T)
  mask = mask + Tensor.full((1, 1, 1, kv_cache_len), float("-inf"), dtype=dtype, buffer=False).tril(pad_end - 1)
  # causal: abs_key > abs_query => c > kv_cache_len - T + row
  mask = mask + Tensor.full((1, 1, T, kv_cache_len), float("-inf"), dtype=dtype, buffer=False).triu(kv_cache_len - T + 1)
  if sliding_window > 0 and kv_cache_len > sliding_window:
    # abs_q - abs_k >= window => c <= kv_cache_len - T + row - window
    mask = mask + Tensor.full((1, 1, T, kv_cache_len), float("-inf"), dtype=dtype, buffer=False).tril(kv_cache_len - T - sliding_window)
  return mask

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int
  kda: bool = False

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  v_head_dim: int
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  ssm_layers: tuple[bool, ...] = ()
  attn_output_gate: bool = False  # qwen-style: gate folded into doubled q_proj
  attn_gate: bool = False         # muse/afmoe-style: separate blk.N.attn_gate before o_proj
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False
  # Muse-Glimmer / hybrid attention
  sliding_window: int = 0
  sliding_window_pattern: tuple[bool, ...] = ()  # True = local/SWA layer
  use_rope: bool = True  # False = NoPE (Muse global layers)
  final_logit_softcapping: float = 0.0
  logit_scale: float = 1.0
  post_norm: bool = False  # post_attention_norm + post_ffw_norm (Muse)
  post_norm_eps: float = 1e-8
  embd_norm: bool = False  # weightless RMSNorm after token embedding (Muse)
  rope_interleaved_qk: bool = False  # KEEP_QK_QUANT: permute Q/K acts not weights

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)
    # Muse-Glimmer dual post-norms (GGUF: post_attention_norm / post_ffw_norm)
    if config.post_norm:
      self.post_attention_norm = nn.RMSNorm(config.dim, config.post_norm_eps)
      self.post_ffw_norm = nn.RMSNorm(config.dim, config.post_norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=False)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      logits = self.ffn_gate_inp(x)
      if hasattr(self, 'exp_probs_b'):
        probs = logits.sigmoid()
        _, sel = pairwise_topk(probs + self.exp_probs_b["bias"], self.config.num_experts_per_tok)
        probs = probs.gather(-1, sel)
        if self.config.norm_topk_prob: probs = probs / probs.sum(axis=-1, keepdim=True)
      else:
        vals, sel = pairwise_topk(logits, self.config.num_experts_per_tok)
        probs = vals.softmax(-1) if self.config.norm_topk_prob else logits.softmax(-1).gather(-1, sel)
      probs = probs * self.config.routed_scaling_factor
      x_down = self.ffn_down_exps(sel, (self.ffn_gate_exps(sel, h).silu() * self.ffn_up_exps(sel, h)).contiguous())  # (B, T, k, D)
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(self.ffn_gate_shexp(x).silu().contiguous() * self.ffn_up_shexp(x))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    # TODO: remove the need for this contiguous
    return self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  # return writes that reset this block's state after a cache mismatch
  def _state_reset_ops(self) -> list[Tensor]: return []
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp):
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp):
      attn = self._attention(self.attn_norm(x), start_pos)
      if hasattr(self, "post_attention_norm"): attn = self.post_attention_norm(attn)
      h = x + attn
      ffn = self._feed_forward(self.ffn_norm(h))
      if hasattr(self, "post_ffw_norm"): ffn = self.post_ffw_norm(ffn)
      return (h + ffn).contiguous()
    return _run(x, start_pos)

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = Linear(config.head_dim * config.n_heads, config.dim, bias=False)
    # Muse/afmoe: separate attention output gate (sigmoid) applied before o_proj
    if config.attn_gate:
      self.attn_gate = Linear(config.dim, config.head_dim * config.n_heads, bias=False)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    # Muse/afmoe separate gate from pre-attn hidden; qwen folds gate into doubled q_proj
    muse_gate = self.attn_gate(x).sigmoid() if hasattr(self, "attn_gate") and self.config.attn_gate else None
    if self.config.attn_output_gate:
      qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    # Muse: RoPE on local/SWA layers only; global layers are NoPE
    if self.config.use_rope:
      # ggml NORM stores Q/K interleaved; apply_rope wants half-split. When Q/K stay
      # quantized (KEEP_QK_QUANT), convert on the activation (cheap) instead of dequant+permute weights.
      if getattr(self.config, "rope_interleaved_qk", False):
        def _interleaved_to_half(t: Tensor) -> Tensor:
          rope, rest = t[..., :self.config.rope_dim], t[..., self.config.rope_dim:]
          rope = rope.rearrange("b h t (half two) -> b h t (two half)", two=2)
          return rope.cat(rest, dim=-1) if resolve(rest.shape[-1] != 0) else rope
        q, k = _interleaved_to_half(q), _interleaved_to_half(k)
      q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
      k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    new_kv = Tensor.stack(k, v)
    mask: Tensor|None
    if self.iswa:
      # ISWA shift-register: fixed-shape drop-oldest|append (compact SWA cache like llama.cpp).
      assigned_kv = Tensor(self.cache_kv.uop.after(
        self.cache_kv.uop.store(self.cache_kv[:, :, :, T:, :].cat(new_kv, dim=3).uop)))
      k, v = assigned_kv[0], assigned_kv[1]
      mask = iswa_attention_mask(T, start_pos, self.kv_cache_len, x.dtype, self.config.sliding_window)
    else:
      assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(new_kv.uop)))
      k = assigned_kv[0, :, :, 0:start_pos+T, :]
      v = assigned_kv[1, :, :, 0:start_pos+T, :]
      # NOTE: causal_lower_right (+ optional SWA). not the causal_upper_left from is_causal=True
      mask = attention_mask(T, start_pos, x.dtype, self.config.sliding_window)
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    if muse_gate is not None: attn = attn * muse_gate
    elif self.config.attn_output_gate: attn = attn * gate.sigmoid()
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # TODO: how is the dtype of this determined?
      # ISWA: local/SWA layers allocate only the sliding window (padded to 256), not full max_context.
      # Use shift-register only when the compact cache is smaller than max_context; otherwise keep
      # absolute indexing (same size, cheaper than cat-shift every step).
      sw = self.config.sliding_window
      self.kv_cache_len = iswa_cache_len(self.config.max_context, sw) if sw > 0 else self.config.max_context
      self.iswa = self.config.sliding_window > 0 and self.kv_cache_len < self.config.max_context
      self.cache_kv = Tensor.empty(2, x.shape[0], self.config.n_kv_heads, self.kv_cache_len, self.config.head_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    if not self.config.ssm or not self.config.ssm.kda: q_rope = apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T])
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(q_rope, dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2)
    if not self.config.ssm or not self.config.ssm.kda: k_rope = apply_rope(k_rope, self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.uop)))[:, :, 0:start_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv = Linear(config.dim, self.conv_channels, bias=False)
    if ssm.kda:
      self.ssm_g_a, self.ssm_g_b = Linear(config.dim, self.head_v_dim, bias=False), Linear(self.head_v_dim, ssm.inner_size, bias=False)
      self.ssm_f_a, self.ssm_f_b = Linear(config.dim, self.head_k_dim, bias=False), Linear(self.head_k_dim, ssm.inner_size, bias=False)
    else:
      self.attn_gate = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_alpha = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(ssm.inner_size if ssm.kda else self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads, 1) if ssm.kda else Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    assert T == 1, "GatedDeltaNetBlock currently only supports T=1"

    # input processing
    x = x.half()
    out_gate = self.ssm_g_b(self.ssm_g_a(x)) if hasattr(self, "ssm_g_a") else self.attn_gate(x)
    out_gate = out_gate.reshape(B, 1, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, self.num_v_heads, 1, 1)
    alpha = self.ssm_f_b(self.ssm_f_a(x)) if hasattr(self, "ssm_f_a") else self.ssm_alpha(x)
    alpha = ((alpha.float() + self.ssm_dt["bias"]).softplus().reshape(B, self.num_v_heads, -1) *
             self.ssm_a.reshape(1, self.num_v_heads, -1)).exp().unsqueeze(-2)

    # qkv conv
    conv_window = self.conv_state.cat(self.attn_qkv(x), dim=1)
    conv_out = (conv_window * self.ssm_conv1d["weight"].T.unsqueeze(0)).sum(1).silu()
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    q = q.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1)
    k = k.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1)
    v = v.reshape(B, self.num_v_heads, self.head_v_dim)
    q, k, v = q.mul(self.head_k_dim**-0.5).unsqueeze(-1), k.unsqueeze(-1), v.unsqueeze(-1)

    # recurrent
    recurrent_state = self.recurrent_state * alpha
    recurrent_state = recurrent_state + ((v - recurrent_state@k) * beta)@k.transpose(-1, -2)

    # store the updated state
    conv_state_store = self.conv_state.uop.store(conv_window[:, 1:, :].cast(self.conv_state.dtype).uop)
    recurrent_state_store = self.recurrent_state.uop.store(recurrent_state.cast(self.recurrent_state.dtype).uop)
    recurrent_state = Tensor(self.recurrent_state.uop.after(recurrent_state_store, conv_state_store))

    # output
    core_attn_out = self.ssm_norm((recurrent_state@q).squeeze(-1).reshape(B, 1, self.num_v_heads, self.head_v_dim))
    out_gate = out_gate.sigmoid() if hasattr(self, "ssm_g_a") else out_gate.silu()
    return self.ssm_out((core_attn_out * out_gate).reshape(B, 1, -1).cast(x.dtype))

  # recurrent state can't be partially reused after divergence, force a full rebuild
  def _state_reset_ops(self):
    return [self.conv_state.assign(self.conv_state.const_like(0)),
            self.recurrent_state.assign(self.recurrent_state.const_like(0))] if hasattr(self, "conv_state") else []
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return 0 if prefix_len != cached_len else prefix_len

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim, device=x.device).clone()

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = []
    for i in range(config.num_blocks):
      base = dense_config if i < config.leading_dense_blocks else config
      # Muse hybrid attention: local layers get SWA+RoPE, global layers get full attn+NoPE
      if config.sliding_window_pattern:
        is_local = bool(config.sliding_window_pattern[i])
        base = replace(base, use_rope=is_local, sliding_window=config.sliding_window if is_local else 0)
      if config.ssm and config.ssm_layers[i]:
        self.blk.append(GatedDeltaNetBlock(base, config.ssm))
      else:
        self.blk.append(block_cls(base))
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.embd_norm = nn.RMSNorm(config.dim, config.norm_eps, elementwise_affine=False) if config.embd_norm else None
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.logit_scale = config.logit_scale
    self.final_logit_softcapping = config.final_logit_softcapping
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    x = self.token_embd(tokens).float()                   # (B, T, D)
    if self.embd_norm is not None: x = self.embd_norm(x)
    for block in self.blk: x = block(x, start_pos)
    logits = self.output(self.output_norm(x))[:, -1, :]
    # Muse: output multiplier then optional tanh softcap (llama.cpp muse-glimmer.cpp)
    if self.logit_scale != 1.0: logits = logits * self.logit_scale
    if self.final_logit_softcapping:
      logits = (logits / self.final_logit_softcapping).tanh() * self.final_logit_softcapping
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    """Load a GGUF checkpoint.

    PACKED_QUANT=1 (default for path loads): each quant tensor is copied as a contiguous
    uint8 buffer on device; matmul layers become QuantLinear. Metal decode GEMVs lower
    via QUANT_GEMV_LOWER (default on); FUSED_KQUANT_GEMV=1 is an optional custom_kernel escape hatch.
    RoPE-permuted Q/K stay dense f16 after one-time dequant+permute. This beats a single
    giant lazy GGUF view for decode and avoids REALIZE=1's f16 blowup on Muse-class models.

    PACKED_QUANT=0: legacy lazy-dequant-over-full-GGUF-buffer path.
    REALIZE=1: materialize every parameter in f16 (refused when too large).
    """
    packed_quant = bool(getenv("PACKED_QUANT", 1)) and not isinstance(gguf, Tensor)
    packed: dict[str, tuple[Tensor, int, tuple[int, ...]]] = {}
    if packed_quant:
      assert not isinstance(gguf, Tensor)
      kv, packed = gguf_load_packed(gguf)
      state_dict = {name: (raw if typ in _GGML_NATIVE else ggml_data_to_tensor(raw, prod(shape), typ).reshape(shape))
                   for name, (raw, typ, shape) in packed.items()}
    else:
      # TODO: remove the need for copy to default device
      kv, state_dict = gguf_load(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)

    # Prefer keeping quantized buffers + dequant views. HALF casts the view dtype.
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    ssm_layers: tuple[bool, ...] = ()
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
      ssm_layers = tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(kv[f'{arch}.block_count']))
    elif arch == 'kimi-linear':
      ssm_layers = tuple(x == 0 for x in n_kv_heads)
      n_kv_heads = max(n_kv_heads)
      ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.kda.head_dim'], n_heads, n_heads, n_heads*kv[f'{arch}.kda.head_dim'], kda=True)
      for i, is_ssm in enumerate(ssm_layers):
        if not is_ssm: continue
        state_dict[f"blk.{i}.attn_qkv.weight"] = state_dict.pop(f"blk.{i}.attn_q.weight").cat(
          state_dict.pop(f"blk.{i}.attn_k.weight"), state_dict.pop(f"blk.{i}.attn_v.weight"), dim=0).contiguous()
        state_dict[f"blk.{i}.ssm_conv1d.weight"] = state_dict.pop(f"blk.{i}.ssm_conv1d_q.weight").cat(
          state_dict.pop(f"blk.{i}.ssm_conv1d_k.weight"), state_dict.pop(f"blk.{i}.ssm_conv1d_v.weight"), dim=0).squeeze(1).contiguous()
        state_dict[f"blk.{i}.ssm_out.weight"] = state_dict.pop(f"blk.{i}.attn_output.weight")
        if packed_quant:
          for suffix in ("attn_q", "attn_k", "attn_v", "ssm_conv1d_q", "ssm_conv1d_k", "ssm_conv1d_v", "attn_output"):
            packed.pop(f"blk.{i}.{suffix}.weight", None)
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    # Permute RoPE weights from interleaved (ggml NORM) to half-split layout used by apply_rope.
    # Muse-Glimmer GGUFs store interleaved Q/K (conversion unpermutes HF rotate_half).
    # KEEP_QK_QUANT=1: leave Q/K packed (QuantLinear); convert interleaved→half-split on activations.
    rope_arches = ('llama', 'muse-glimmer')
    keep_qk_quant = bool(getenv("KEEP_QK_QUANT", 1)) and packed_quant
    dense_f16: set[str] = set()
    if not keep_qk_quant:
      for name in list(state_dict):
        if arch == 'kimi-linear': continue
        if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch in rope_arches or kv_lora_rank):
          w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
          prefix = head_dim-rope_dim
          state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
          dense_f16.add(name)
        elif arch in rope_arches and 'attn_k.weight' in name:
          w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
          state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
          dense_f16.add(name)
        elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
          state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
          dense_f16.add(name)
      # One-time materialize of rope-permuted Q/K so QuantLinear isn't fighting rearranges.
      # NOTE: on Muse this is NOT small — attn_q alone is ~2.8GB f16 and re-read every decode token.
      for name in dense_f16:
        w = state_dict[name]
        if getenv("HALF", 1): w = w.cast(dtypes.float16)
        # Must realize: otherwise the dequant+permute graph is re-fused into every decode step.
        state_dict[name] = w.contiguous().realize()

    config = TransformerConfig(
      num_blocks=kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0), dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv[f'{arch}.rope.freq_base'],
      rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe', 'kimi-linear')),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict,
      attn_gate='blk.0.attn_gate.weight' in state_dict,
      sliding_window=int(kv.get(f'{arch}.attention.sliding_window', 0) or 0),
      sliding_window_pattern=tuple(bool(x) for x in kv.get(f'{arch}.attention.sliding_window_pattern', ())),
      final_logit_softcapping=float(kv.get(f'{arch}.final_logit_softcapping', 0.0) or 0.0),
      logit_scale=float(kv.get(f'{arch}.logit_scale', 1.0) or 1.0),
      post_norm='blk.0.post_attention_norm.weight' in state_dict,
      embd_norm=arch == 'muse-glimmer',
      rope_interleaved_qk=keep_qk_quant)
    model = Transformer(config)

    # Upgrade quant matmul weights to QuantLinear (packed qweight Parameter).
    _QUANT_LEAVES = {
      "ffn_gate", "ffn_up", "ffn_down", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "ffn_gate_inp",
      "attn_v", "attn_output", "attn_gate", "output", "ssm_out", "attn_qkv",
      "ssm_beta", "ssm_alpha", "ssm_g_a", "ssm_g_b", "ssm_f_a", "ssm_f_b",
    }
    if keep_qk_quant:
      _QUANT_LEAVES = _QUANT_LEAVES | {"attn_q", "attn_k", "attn_q_b"}
    if packed_quant:
      for name, (raw, typ, shape) in packed.items():
        if not name.endswith(".weight") or not is_ggml_quant(typ): continue
        if name in dense_f16 or name.endswith("token_embd.weight"): continue
        mod_name = name[:-len(".weight")]
        if mod_name.split(".")[-1] not in _QUANT_LEAVES: continue
        replace_linear_with_quant(model, mod_name, raw, typ, shape)
        state_dict[mod_name + ".qweight"] = raw.flatten()
        state_dict.pop(name, None)

    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # Prefer packed/lazy dequant. REALIZE=1 materializes full f16 params (avoid for large models).
    if realize:
      params = nn.state.get_parameters(model)
      nparams = sum(s.numel() for s in params)
      # ~2B+ f16 params is already multi-GB; Muse 28B would be ~56GB realized.
      if nparams > 2_000_000_000:
        raise RuntimeError(
          f"REFUSING REALIZE=1 for {nparams:,} params (~{nparams*2/1e9:.0f}GB f16). "
          "Keep packed/lazy dequant (REALIZE=0 / PACKED_QUANT=1, default).")
      for s in params: s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def warmup(self):
    for _ in range(2): list(zip(range(2), self.generate([0])))

  def get_start_pos(self, tokens:list[int]) -> int:
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0):
    if self.has_recurrent_block: chunk_size = 1
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor([temperature])
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    if start_pos < len(self._cached_tokens) and (resets := [r for b in self.blk for r in b._state_reset_ops()]): Tensor.realize(*resets)
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      n_toks = min(chunk_size, len(tokens) - start_pos)
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()
      start_pos += n_toks
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
