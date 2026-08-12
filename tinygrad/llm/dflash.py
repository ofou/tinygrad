"""DFlash block-diffusion draft (llama.cpp draft_dflash) for Muse-Glimmer."""
from __future__ import annotations
import pathlib, time
from dataclasses import dataclass

from tinygrad import Tensor, TinyJit, UOp, nn, dtypes, getenv
from tinygrad.helpers import prod
from tinygrad.nn import Linear
from tinygrad.llm.gguf import gguf_load_packed, ggml_data_to_tensor, _GGML_NATIVE
from tinygrad.llm.quant import is_ggml_quant, replace_linear_with_quant
from tinygrad.llm.model import TransformerConfig, TransformerBlock, apply_rope, parse_sliding_window_pattern
from tinygrad.uop.ops import resolve

# ---- pure helpers (unit-tested) -------------------------------------------

def parse_target_layers(kv: dict) -> tuple[int, ...]:
  """Return GGUF `dflash.target_layers` as Muse layer-input indices (llama.cpp as-is).

  HF trains on layer-output ids {1,13,25,37,49}; convert_hf_to_gguf stores +1 so
  GGUF has [2,14,26,38,50]. llama_set_embeddings_layer_inp uses those values
  directly → capture blk inputs at {2,14,26,38,50} (= outputs of {1,13,25,37,49}).
  """
  raw = kv.get("dflash.target_layers") or kv.get("target_layers")
  if raw is None: raise ValueError("DFlash GGUF missing dflash.target_layers")
  layers = tuple(int(x) for x in raw)
  # DFLASH_LAYER_OFFSET: subtract to map GGUF layer-inp ids → alternate capture (debug).
  off = int(getenv("DFLASH_LAYER_OFFSET", "0"))
  return tuple(i - off for i in layers)

def target_layers_0based(layers: list[int] | tuple[int, ...]) -> tuple[int, ...]:
  """Deprecated alias: values are already 0-based Muse indices in GGUF."""
  return tuple(int(x) for x in layers)

def noise_block_tokens(id_last: int, n_draft: int, mask_token_id: int) -> list[int]:
  """Noise block [id_last, MASK…] length n_draft+1 (n_draft <= block_size-1)."""
  if n_draft < 0: raise ValueError("n_draft must be >= 0")
  return [id_last] + [mask_token_id] * n_draft

def accept_prefix(draft_ids: list[int], target_preds: list[int]) -> int:
  """Longest matching prefix between draft tokens and target predictions."""
  n = 0
  for d, t in zip(draft_ids, target_preds):
    if d != t: break
    n += 1
  return n

def enable_dflash_capture(model, target_layers: tuple[int, ...] | list[int]) -> None:
  model._dflash_capture = frozenset(int(i) for i in target_layers)
  model._dflash_features = {}

def disable_dflash_capture(model) -> None:
  model._dflash_capture = None
  model._dflash_features = {}

def take_dflash_features(model) -> dict[int, Tensor]:
  feats = getattr(model, "_dflash_features", None) or {}
  model._dflash_features = {}
  return feats

# ---- config / blocks ------------------------------------------------------

@dataclass(frozen=True)
class DFlashConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  norm_eps: float
  max_context: int
  block_size: int
  mask_token_id: int
  target_layers: tuple[int, ...]  # 0-based Muse indices
  sliding_window: int = 0
  sliding_window_pattern: tuple[bool, ...] = ()
  rope_interleaved_qk: bool = False

  @property
  def n_extract(self) -> int: return len(self.target_layers)

  @property
  def n_draft_max(self) -> int: return max(0, self.block_size - 1)

  @staticmethod
  def from_gguf_kv(kv: dict, max_context: int | None = None) -> DFlashConfig:
    arch = kv["general.architecture"]
    if arch != "dflash": raise ValueError(f"expected arch dflash, got {arch}")
    n_heads = kv[f"{arch}.attention.head_count"]
    head_dim = kv.get(f"{arch}.attention.key_length", kv[f"{arch}.embedding_length"] // n_heads)
    ctx = kv[f"{arch}.context_length"]
    max_context = min(max_context, ctx) if max_context is not None else ctx
    layers = tuple(kv[f"{arch}.target_layers"])
    sw = int(kv.get(f"{arch}.attention.sliding_window", 0) or 0)
    n_blocks = kv[f"{arch}.block_count"]
    pattern = parse_sliding_window_pattern(kv.get(f"{arch}.attention.sliding_window_pattern", ()), n_blocks)
    return DFlashConfig(
      num_blocks=n_blocks, dim=kv[f"{arch}.embedding_length"],
      hidden_dim=kv[f"{arch}.feed_forward_length"], n_heads=n_heads,
      n_kv_heads=kv[f"{arch}.attention.head_count_kv"], head_dim=head_dim,
      rope_theta=float(kv[f"{arch}.rope.freq_base"]), rope_dim=head_dim,
      norm_eps=float(kv[f"{arch}.attention.layer_norm_rms_epsilon"]),
      max_context=max_context, block_size=int(kv[f"{arch}.block_size"]),
      mask_token_id=int(kv.get("tokenizer.ggml.mask_token_id", 0)),
      target_layers=parse_target_layers({"dflash.target_layers": layers}),
      sliding_window=sw,
      sliding_window_pattern=pattern or tuple(True for _ in range(n_blocks)),
      # llama.cpp: legacy DFlash backbones use NEOX RoPE (half-split). Muse target is NORM
      # (interleaved); never inherit KEEP_QK_QUANT's interleaved activation permute here.
      rope_interleaved_qk=False)

class DFlashBlock(TransformerBlock):
  """Draft layer: KV inject from fused target features + non-causal noise attn."""

  def inject(self, fused: Tensor, start_pos: int | UOp, *, realize: bool = True) -> Tensor:
    """Write committed context K/V. Returns assign tensor; realize=False defers sync."""
    self._init_state(fused)
    B, T, _ = fused.shape
    k, v = self.attn_k(fused), self.attn_v(fused)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)
    if self.config.qk_norm: k = self.attn_k_norm(k)
    if self.config.use_rope:
      if getattr(self.config, "rope_interleaved_qk", False):
        rope, rest = k[..., :self.config.rope_dim], k[..., self.config.rope_dim:]
        k = rope.rearrange("b h t (half two) -> b h t (two half)", two=2)
        if resolve(rest.shape[-1] != 0): k = k.cat(rest, dim=-1)
      k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)
    new_kv = Tensor.stack(k, v)
    if self.iswa:
      assigned = Tensor(self.cache_kv.uop.after(
        self.cache_kv.uop.store(self.cache_kv[:, :, :, T:, :].cat(new_kv, dim=3).uop)))
    else:
      assigned = Tensor(self.cache_kv.uop.after(
        self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(new_kv.uop)))
    if realize: Tensor.realize(assigned)
    return assigned

  def _attention(self, x: Tensor, start_pos: int | UOp) -> Tensor:
    """Draft noise attention over injected context + noise block.

    Noise K/V is ephemeral (llama.cpp overwrites those slots on the next inject):
    concat committed cache with noise K/V in-graph and do **not** store into
    cache_kv. That removes snapshot/restore and keeps TinyJit buffer identities
    stable across draft steps.

    Default matches llama.cpp (`llama_set_causal_attn(false)`): non-causal within
    the noise block. Muse HF layer_types are all sliding_attention; vLLM treats
    those as causal — set DFLASH_CAUSAL=1 to A/B that heuristic.
    """
    from tinygrad.llm.model import attention_mask, iswa_attention_mask
    causal = bool(getenv("DFLASH_CAUSAL", 0))
    q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    B, T, _ = x.shape
    q = q.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)
    if self.config.qk_norm: q, k = self.attn_q_norm(q), self.attn_k_norm(k)
    if self.config.use_rope:
      if getattr(self.config, "rope_interleaved_qk", False):
        def _half(t: Tensor) -> Tensor:
          rope, rest = t[..., :self.config.rope_dim], t[..., self.config.rope_dim:]
          rope = rope.rearrange("b h t (half two) -> b h t (two half)", two=2)
          return rope.cat(rest, dim=-1) if resolve(rest.shape[-1] != 0) else rope
        q, k = _half(q), _half(k)
      q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
      k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)
    # Ephemeral: view-concat committed KV + noise; cache_kv stays inject-only.
    mask: Tensor | None
    if self.iswa:
      k = self.cache_kv[0, :, :, T:, :].cat(k, dim=2)
      v = self.cache_kv[1, :, :, T:, :].cat(v, dim=2)
      if causal:
        mask = iswa_attention_mask(T, start_pos, self.kv_cache_len, x.dtype, self.config.sliding_window)
      else:
        mask = Tensor.zeros((1, 1, T, self.kv_cache_len), dtype=x.dtype, buffer=False)
        pad_end = self.kv_cache_len - (start_pos + T)
        mask = mask + Tensor.full((1, 1, 1, self.kv_cache_len), float("-inf"), dtype=x.dtype, buffer=False).tril(pad_end - 1)
        if self.config.sliding_window > 0 and self.kv_cache_len > self.config.sliding_window:
          mask = mask + Tensor.full((1, 1, T, self.kv_cache_len), float("-inf"), dtype=x.dtype, buffer=False).tril(
            self.kv_cache_len - T - self.config.sliding_window)
    else:
      k = self.cache_kv[0, :, :, 0:start_pos, :].cat(k, dim=2)
      v = self.cache_kv[1, :, :, 0:start_pos, :].cat(v, dim=2)
      if causal:
        mask = attention_mask(T, start_pos, x.dtype, self.config.sliding_window)
      else:
        sw = self.config.sliding_window
        if sw > 0 and resolve(start_pos + T > sw):
          mask = Tensor.zeros((1, 1, T, start_pos+T), dtype=x.dtype, buffer=False)
          mask = mask + Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).tril(start_pos - sw)
        else:
          mask = None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
    return self.attn_output(attn.transpose(1, 2).reshape(B, T, -1))


# ---- target embd/head only (draft-only smoke; no Muse blk.*) --------------

_HEAD_KEYS = frozenset({
  "token_embd.weight", "output.weight", "output_norm.weight", "embd_norm.weight",
})

class TargetHead:
  """Muse token_embd + lm_head (+ norms) without loading blk.*. For draft-only iteration."""

  def __init__(self, dim: int, vocab_size: int, norm_eps: float = 1e-5, embd_norm: bool = False):
    self.dim, self.vocab_size, self.norm_eps = dim, vocab_size, norm_eps
    self.token_embd = nn.Embedding(vocab_size, dim)
    # Muse uses weightless embd_norm; affine only if GGUF has embd_norm.weight.
    self.embd_norm = nn.RMSNorm(dim, norm_eps, elementwise_affine=False) if embd_norm else None
    self.output_norm = nn.RMSNorm(dim, norm_eps)
    self.output = Linear(dim, vocab_size, bias=False)

  @staticmethod
  def load_from_gguf(gguf: str | pathlib.Path) -> tuple[TargetHead, dict]:
    """Load only embd/head tensors from a Muse (or similar) GGUF."""
    kv, packed = gguf_load_packed(gguf, names=_HEAD_KEYS)
    arch = kv["general.architecture"]
    dim = int(kv[f"{arch}.embedding_length"])
    vocab_size = len(kv["tokenizer.ggml.tokens"])
    norm_eps = float(kv[f"{arch}.attention.layer_norm_rms_epsilon"])
    embd_norm = arch == "muse-glimmer" or "embd_norm.weight" in packed

    state_dict: dict[str, Tensor] = {}
    for name, (raw, typ, shape) in packed.items():
      if typ in _GGML_NATIVE:
        state_dict[name] = raw.cast(dtypes.float16) if getenv("HALF", 1) else raw
      else:
        w = ggml_data_to_tensor(raw, prod(shape), typ).reshape(shape)
        state_dict[name] = w.cast(dtypes.float16) if getenv("HALF", 1) else w

    if "token_embd.weight" not in state_dict:
      raise ValueError(f"GGUF missing token_embd.weight: {gguf}")
    # tied embd if no separate output (same as Transformer.from_gguf); keep Linear weight, not QuantLinear.
    if "output.weight" not in packed:
      state_dict["output.weight"] = state_dict["token_embd.weight"]

    if "embd_norm.weight" in state_dict: embd_norm = True
    model = TargetHead(dim, vocab_size, norm_eps, embd_norm=embd_norm)
    if "embd_norm.weight" in state_dict:
      model.embd_norm = nn.RMSNorm(dim, norm_eps, elementwise_affine=True)

    # QuantLinear for output like Muse; token_embd stays Embedding + dequant view.
    for name, (raw, typ, shape) in packed.items():
      if not name.endswith(".weight") or not is_ggml_quant(typ): continue
      if name.endswith("token_embd.weight"): continue
      mod_name = name[:-len(".weight")]
      if mod_name.split(".")[-1] != "output": continue
      replace_linear_with_quant(model, mod_name, raw, typ, shape)
      state_dict[mod_name + ".qweight"] = raw.flatten()
      state_dict.pop(name, None)

    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False, strict=False)
    return model, kv

  # alias
  from_gguf = load_from_gguf

class _TargetRef:
  """Hold target without nn.state walking it (no __dict__ / not a list)."""
  __slots__ = ("m",)
  def __init__(self, m): self.m = m

class DFlashDraft:
  """Loaded DFlash draft; shares target token_embd + lm_head."""

  def __init__(self, config: DFlashConfig, target):
    self.config = config
    # Keep Muse out of nn.state so load_state_dict doesn't walk 30B target.*
    self._target_ref = _TargetRef(target)
    self.fc = Linear(config.dim * config.n_extract, config.dim, bias=False)
    self.enc_output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.blk: list[DFlashBlock] = []
    for i in range(config.num_blocks):
      is_local = bool(config.sliding_window_pattern[i]) if config.sliding_window_pattern else True
      bcfg = TransformerConfig(
        num_blocks=1, dim=config.dim, hidden_dim=config.hidden_dim,
        n_heads=config.n_heads, n_kv_heads=config.n_kv_heads, norm_eps=config.norm_eps,
        vocab_size=1, head_dim=config.head_dim, rope_theta=config.rope_theta,
        rope_dim=config.rope_dim, v_head_dim=config.head_dim, max_context=config.max_context,
        qk_norm=config.head_dim, use_rope=is_local,
        sliding_window=config.sliding_window if is_local else 0,
        rope_interleaved_qk=config.rope_interleaved_qk)
      self.blk.append(DFlashBlock(bcfg))

  @property
  def target(self):
    return self._target_ref.m

  def encode(self, features: dict[int, Tensor] | list[Tensor]) -> Tensor:
    """Concat extract-layer inputs (target_layers order) -> fc -> enc.output_norm."""
    if isinstance(features, dict):
      layer_inputs = [features[i] for i in self.config.target_layers]
    else:
      layer_inputs = list(features)
    if len(layer_inputs) != self.config.n_extract:
      raise ValueError(f"expected {self.config.n_extract} features, got {len(layer_inputs)}")
    x = layer_inputs[0].cat(*layer_inputs[1:], dim=-1).float()
    return self.enc_output_norm(self.fc(x))

  def _inject_apply(self, fused: Tensor, start_pos: int | UOp):
    """Jitable: write fused K/V into every draft layer; return assign tensors."""
    outs = []
    for block in self.blk:
      block._init_state(fused)
      outs.append(block.inject(fused, start_pos, realize=False))
    return tuple(outs)

  def ensure_inject_jit(self, T: int):
    """TinyJit for fixed inject length T (full-accept batch). Prefill T>block_size stays eager."""
    if getattr(self, "_inject_jit_T", None) != T or not hasattr(self, "inject_jit"):
      self._inject_jit_T = T
      self._inject_start_pos = UOp.variable("dflash_inject_pos", 0, self.config.max_context - 1)
      self.inject_jit = TinyJit(self._inject_apply)
    return self.inject_jit

  def process(self, features: dict[int, Tensor] | list[Tensor], start_pos: int) -> Tensor:
    """Encode Muse layer inputs and inject K/V into every draft layer.

    One realize for fused + all layer assigns (no per-layer host sync).
    Steady-state full-accept (T<=block_size) uses TinyJit; large prefill chunks stay eager.
    """
    fused = self.encode(features).contiguous()
    T = int(fused.shape[1]) if isinstance(fused.shape[1], int) else int(resolve(fused.shape[1]))
    # Only JIT the steady full-accept length (same T as draft noise). Prefill chunks and
    # post-reject remainders must not steal/rebuild the capture.
    use_jit = bool(getenv("DFLASH_INJECT_JIT", 1)) and getattr(self, "_draft_jit_n", None) == T
    if use_jit:
      # Init caches before first capture so buffer identities stay stable.
      for block in self.blk: block._init_state(fused)
      jit = self.ensure_inject_jit(T)
      jit(fused, self._inject_start_pos.bind(start_pos))
    else:
      for block in self.blk: block._init_state(fused)
      assigns = [block.inject(fused, start_pos, realize=False) for block in self.blk]
      Tensor.realize(fused, *assigns)
    return fused

  def _draft_forward(self, tokens: Tensor, start_pos: int | UOp, temperature: Tensor) -> Tensor:
    """Jitable noise forward → argmax ids for positions 1..T-1 (packed T<=QUANT_GEMV_MAX_T)."""
    x = self._target_ref.m.token_embd(tokens).float()
    # Muse target applies weightless embd_norm before layers. llama.cpp draft noise uses
    # raw rows (no embd_norm). HF Muse may differ — DFLASH_NOISE_EMBD_NORM=1 to match target.
    if int(__import__("os").environ.get("DFLASH_NOISE_EMBD_NORM", "0") or "0"):
      en = getattr(self._target_ref.m, "embd_norm", None)
      if en is not None: x = en(x)
    for block in self.blk: x = block(x, start_pos)
    logits = self._target_ref.m.output(self.output_norm(x))
    logits = logits / temperature.maximum(1e-12)
    return logits[:, 1:, :].argmax(-1)

  def ensure_draft_jit(self, n_tokens: int):
    """TinyJit for fixed noise length (n_draft+1). Rebuilds if length changes."""
    if getattr(self, "_draft_jit_n", None) != n_tokens or not hasattr(self, "draft_jit"):
      # Init caches so JIT capture sees stable buffer identities.
      if self.blk and not hasattr(self.blk[0], "cache_kv"):
        raise RuntimeError("draft KV not initialized; call process() before draft_block")
      self._draft_jit_n = n_tokens
      self._draft_start_pos = UOp.variable("dflash_draft_pos", 0, self.config.max_context - 1)
      self.draft_jit = TinyJit(self._draft_forward)
    return self.draft_jit

  def draft_block(self, id_last: int, n_past: int, n_draft: int | None = None,
                  temperature: float = 0.0) -> list[int]:
    """Noise block [id_last, MASK…]; non-causal forward; sample positions 1..

    Noise K/V is ephemeral (see DFlashBlock._attention): no cache store, so no
    snapshot/restore. Steady-state path is TinyJit + packed QUANT_GEMV_LOWER.
    """
    n_draft = self.config.n_draft_max if n_draft is None else min(int(n_draft), self.config.n_draft_max)
    if n_draft <= 0: return []
    toks = noise_block_tokens(id_last, n_draft, self.config.mask_token_id)
    # Caches must exist (process/prefill) before JIT capture.
    for block in self.blk:
      if not hasattr(block, "cache_kv"):
        raise RuntimeError("draft_block before process(): draft KV cache not initialized")
    tokens = Tensor([toks], dtype=dtypes.int32).contiguous()
    temp = Tensor([temperature if temperature and temperature > 1e-12 else 0.0])
    # ensure_draft_jit compares _draft_jit_n — must not pre-assign len(toks) or
    # n_draft changes keep the old TinyJit and raise args mismatch.
    use_jit = bool(getenv("DFLASH_DRAFT_JIT", 1))
    if use_jit:
      jit = self.ensure_draft_jit(len(toks))
      sp = self._draft_start_pos.bind(n_past)
      ids = jit(tokens, sp, temp)
    else:
      self._draft_jit_n = len(toks)  # inject JIT gating when draft jit off
      ids = self._draft_forward(tokens, n_past, temp).realize()
    return [int(x) for x in ids.numpy().reshape(-1).tolist()]

  def reset_cache(self) -> None:
    """Zero draft KV in-place so TinyJit buffer identities stay valid across rejects."""
    for b in self.blk:
      if hasattr(b, "cache_kv"):
        b.cache_kv.assign(Tensor.zeros(*b.cache_kv.shape, dtype=b.cache_kv.dtype, device=b.cache_kv.device)).realize()

  @staticmethod
  def from_gguf(gguf: str | pathlib.Path, target, max_context: int | None = None) -> DFlashDraft:
    kv, packed = gguf_load_packed(gguf)
    config = DFlashConfig.from_gguf_kv(kv, max_context=max_context)
    keep_qk = bool(getenv("KEEP_QK_QUANT", 1))
    state_dict: dict[str, Tensor] = {}
    dense_f16: set[str] = set()
    for name, (raw, typ, shape) in packed.items():
      if typ in _GGML_NATIVE:
        state_dict[name] = raw.cast(dtypes.float16) if getenv("HALF", 1) else raw
      else:
        w = ggml_data_to_tensor(raw, prod(shape), typ).reshape(shape)
        state_dict[name] = w.cast(dtypes.float16) if getenv("HALF", 1) else w

    # DFlash Q/K are NeoX/half-split in GGUF — do not apply Muse interleaved permute.
    if not keep_qk:
      # Optional: materialize Q/K as dense f16 without layout change (debug / A-B only).
      for name in list(state_dict):
        if "attn_q.weight" in name or "attn_k.weight" in name:
          state_dict[name] = state_dict[name].contiguous().realize()
          dense_f16.add(name)

    if "enc.output_norm.weight" in state_dict:
      state_dict["enc_output_norm.weight"] = state_dict.pop("enc.output_norm.weight")

    model = DFlashDraft(config, target)
    _QUANT_LEAVES = {"ffn_gate", "ffn_up", "ffn_down", "attn_v", "attn_output", "fc"}
    if keep_qk: _QUANT_LEAVES = _QUANT_LEAVES | {"attn_q", "attn_k"}
    for name, (raw, typ, shape) in packed.items():
      if not name.endswith(".weight") or not is_ggml_quant(typ): continue
      if name in dense_f16: continue
      mod_name = name[:-len(".weight")].replace("enc.output_norm", "enc_output_norm")
      if mod_name.split(".")[-1] not in _QUANT_LEAVES: continue
      replace_linear_with_quant(model, mod_name, raw, typ, shape)
      state_dict[mod_name + ".qweight"] = raw.flatten()
      state_dict.pop(name, None)

    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False, strict=False)
    return model


def smoke_dflash_only(draft: DFlashDraft, T: int = 16, n_draft: int | None = None,
                      id_last: int = 1, verbose: bool = True) -> list[int]:
  """Synthetic features -> process(encode+inject) -> draft_block. Prints phase timings."""
  dim = draft.config.dim
  feats = {i: Tensor.randn(1, T, dim) for i in draft.config.target_layers}
  t0 = time.perf_counter()
  draft.process(feats, 0)
  t_ei = time.perf_counter()
  ids = draft.draft_block(id_last, T, n_draft=n_draft, temperature=0.0)
  t_dr = time.perf_counter()
  if verbose:
    cfg = draft.config
    print(f"[dflash-only] config blocks={cfg.num_blocks} dim={cfg.dim} block_size={cfg.block_size} "
          f"target_layers={cfg.target_layers} n_draft={n_draft if n_draft is not None else cfg.n_draft_max}")
    print(f"[dflash-only] encode+inject {(t_ei-t0)*1e3:.1f} ms  draft {(t_dr-t_ei)*1e3:.1f} ms  "
          f"total {(t_dr-t0)*1e3:.1f} ms")
    print(f"[dflash-only] draft_ids={ids}")
  return ids

if __name__ == "__main__":
  import argparse
  ap = argparse.ArgumentParser(description="DFlash draft-only Metal smoke")
  ap.add_argument("--model", required=True, help="Muse GGUF (embd/head source)")
  ap.add_argument("--dflash", required=True, help="DFlash draft GGUF")
  ap.add_argument("--max_context", type=int, default=4096)
  ap.add_argument("--T", type=int, default=16)
  ap.add_argument("--n_draft", type=int, default=None)
  args = ap.parse_args()
  head, kv = TargetHead.from_gguf(args.model)
  print(f"target head from {args.model}: dim={head.dim} vocab={head.vocab_size} "
        f"arch={kv.get('general.architecture')}")
  draft = DFlashDraft.from_gguf(args.dflash, head, max_context=args.max_context)
  print(f"dflash: blocks={draft.config.num_blocks} block_size={draft.config.block_size} "
        f"target_layers={draft.config.target_layers} mask={draft.config.mask_token_id}")
  smoke_dflash_only(draft, T=args.T, n_draft=args.n_draft)
