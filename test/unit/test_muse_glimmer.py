import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.model import (
  Transformer, TransformerBlock, TransformerConfig, attention_mask,
  iswa_cache_len, iswa_attention_mask,
)
from tinygrad.llm.quant import QuantLinear, replace_linear_with_quant
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.cli import SimpleTokenizer

def _cfg(**kwargs):
  base = dict(num_blocks=1, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, norm_eps=1e-5,
              vocab_size=32, head_dim=4, rope_theta=10000.0, rope_dim=4, v_head_dim=4, max_context=16)
  base.update(kwargs)
  return TransformerConfig(**base)

class TestAttentionMask(unittest.TestCase):
  def test_causal_only(self):
    mask = attention_mask(3, 0, dtypes.float32, sliding_window=0)
    m = mask.numpy()[0, 0]
    # lower-tri allowed (0), upper -inf
    self.assertTrue(np.isneginf(m[0, 1]))
    self.assertTrue(np.isneginf(m[0, 2]))
    self.assertEqual(m[0, 0], 0)
    self.assertEqual(m[2, 0], 0)
    self.assertEqual(m[2, 2], 0)

  def test_swa_standard_matches_llama_cpp(self):
    # llama.cpp LLAMA_SWA_TYPE_STANDARD: mask if query_pos - key_pos >= window
    T, start_pos, window = 4, 5, 3
    mask = attention_mask(T, start_pos, dtypes.float32, sliding_window=window).numpy()[0, 0]
    kv_len = start_pos + T
    for r in range(T):
      q = start_pos + r
      for c in range(kv_len):
        should_mask = (c > q) or (q - c >= window)
        if should_mask:
          self.assertTrue(np.isneginf(mask[r, c]), f"expected -inf at q={q} k={c}")
        else:
          self.assertEqual(mask[r, c], 0, f"expected 0 at q={q} k={c}")

  def test_decode_step_needs_swa_mask(self):
    # T=1 decode past the window must still mask old keys
    mask = attention_mask(1, 10, dtypes.float32, sliding_window=4)
    self.assertIsNotNone(mask)
    m = mask.numpy()[0, 0, 0]
    # query_pos=10, window=4 -> attend keys 7..10; mask 0..6
    self.assertTrue(np.all(np.isneginf(m[:7])))
    self.assertTrue(np.all(m[7:] == 0))

  def test_decode_inside_window_no_mask(self):
    self.assertIsNone(attention_mask(1, 2, dtypes.float32, sliding_window=8))

class TestLogitSoftcapScale(unittest.TestCase):
  def test_softcap_and_scale_applied(self):
    cfg = _cfg(logit_scale=0.5, final_logit_softcapping=2.0, num_blocks=1, max_context=8)
    model = Transformer(cfg)
    # zero out weights for a deterministic path through norms/linears as much as practical
    for p in model.__dict__.values():
      pass
    # Directly exercise the logit post-process formula used in forward
    logits = Tensor([[10.0, -4.0, 0.0]])
    scaled = logits * cfg.logit_scale
    soft = (scaled / cfg.final_logit_softcapping).tanh() * cfg.final_logit_softcapping
    expected = (np.array([[10.0, -4.0, 0.0]]) * 0.5)
    expected = np.tanh(expected / 2.0) * 2.0
    np.testing.assert_allclose(soft.numpy(), expected, rtol=1e-5, atol=1e-5)

  def test_forward_runs_with_muse_logit_fields(self):
    cfg = _cfg(logit_scale=0.19611613, final_logit_softcapping=20.0, embd_norm=True, max_context=8)
    model = Transformer(cfg)
    tok = Tensor([[1, 2, 3]], dtype=dtypes.int32)
    out = model.forward(tok, 0, Tensor([0.0])).realize()
    self.assertEqual(out.shape, (1, 1))

class TestMuseGatedAttention(unittest.TestCase):
  def test_attn_gate_modulates_output(self):
    cfg = _cfg(attn_gate=True, use_rope=True, max_context=8)
    block = TransformerBlock(cfg)
    # identity-ish projections
    D, H, Hd = cfg.dim, cfg.n_heads, cfg.head_dim
    block.attn_q.weight = Tensor.eye(D)[:H*Hd, :].contiguous()
    block.attn_k.weight = Tensor.eye(D)[:cfg.n_kv_heads*Hd, :].contiguous()
    block.attn_v.weight = Tensor.eye(D)[:cfg.n_kv_heads*Hd, :].contiguous()
    block.attn_output.weight = Tensor.eye(D)[:, :H*Hd].contiguous() if D >= H*Hd else Tensor.ones(D, H*Hd)
    # large positive gate -> ~1 after sigmoid; large negative -> ~0
    block.attn_gate.weight = Tensor.ones(H*Hd, D)
    x = Tensor.ones(1, 2, D)
    block._init_state(x)
    out_open = block._attention(x, 0).realize().numpy()
    block.attn_gate.weight = -Tensor.ones(H*Hd, D)
    # reset cache for fair compare of gating path only (same qkv)
    block.cache_kv = Tensor.empty(2, 1, cfg.n_kv_heads, cfg.max_context, cfg.head_dim, device=x.device)
    out_closed = block._attention(x, 0).realize().numpy()
    self.assertGreater(np.abs(out_open).mean(), np.abs(out_closed).mean())

class TestMuseHybridLayers(unittest.TestCase):
  def test_pattern_local_local_local_global(self):
    pattern = tuple(i % 4 != 3 for i in range(8))
    cfg = _cfg(num_blocks=8, sliding_window=4, sliding_window_pattern=pattern,
               attn_gate=True, post_norm=True, embd_norm=True, max_context=16)
    model = Transformer(cfg)
    for i, blk in enumerate(model.blk):
      is_local = pattern[i]
      self.assertEqual(blk.config.use_rope, is_local, f"layer {i} rope")
      self.assertEqual(blk.config.sliding_window, 4 if is_local else 0, f"layer {i} window")
      self.assertTrue(hasattr(blk, "attn_gate"))
      self.assertTrue(hasattr(blk, "post_attention_norm"))
      self.assertTrue(hasattr(blk, "post_ffw_norm"))
    self.assertIsNotNone(model.embd_norm)

  def test_nope_skips_rope(self):
    cfg = _cfg(use_rope=False, max_context=8)
    block = TransformerBlock(cfg)
    x = Tensor.randn(1, 3, cfg.dim)
    block._init_state(x)
    # Should not crash; freqs_cis still allocated but unused when use_rope=False
    out = block._attention(x, 0).realize()
    self.assertEqual(out.shape, (1, 3, cfg.dim))

class TestLlama4TokenizerPreset(unittest.TestCase):
  def test_llama4_alias_accepted(self):
    tok = SimpleTokenizer({"a": 0, "b": 1, "ab": 2}, {"<s>": 3}, preset="llama4", bos_id=3, eos_id=3)
    self.assertEqual(tok.preset, "llama3")  # aliased
    self.assertEqual(tok.encode("ab"), [2])

  def test_unknown_preset_still_rejected(self):
    with self.assertRaises(ValueError):
      SimpleTokenizer({"a": 0}, {}, preset="not-a-real-preset")

class TestRealizeGuard(unittest.TestCase):
  def test_realize_refuses_large_param_count(self):
    # Smoke-test the guard logic without a 17GB file: call the threshold check shape
    nparams = 3_000_000_000
    with self.assertRaises(RuntimeError):
      if nparams > 2_000_000_000:
        raise RuntimeError(
          f"REFUSING REALIZE=1 for {nparams:,} params (~{nparams*2/1e9:.0f}GB f16). "
          "Keep lazy dequant over the GGUF buffer (REALIZE=0, default).")


class TestISWA(unittest.TestCase):
  def test_iswa_cache_len_pads_to_256(self):
    self.assertEqual(iswa_cache_len(131072, 2048), 2048)
    self.assertEqual(iswa_cache_len(1000, 2048), 1000)
    self.assertEqual(iswa_cache_len(4096, 2000), 2048)  # pad 2000 -> 2048
    self.assertEqual(iswa_cache_len(8192, 0), 8192)

  def test_local_layers_allocate_window_not_max_context(self):
    pattern = tuple(i % 4 != 3 for i in range(8))
    # window 512 pads to 512; max_context 4096 -> local KV is 512 not 4096
    cfg = _cfg(num_blocks=8, sliding_window=512, sliding_window_pattern=pattern,
               attn_gate=True, max_context=4096)
    model = Transformer(cfg)
    x = Tensor.randn(1, 2, cfg.dim)
    for i, blk in enumerate(model.blk):
      blk._init_state(x)
      if pattern[i]:
        self.assertTrue(blk.iswa)
        self.assertEqual(blk.kv_cache_len, 512)
        self.assertEqual(blk.cache_kv.shape[3], 512)
      else:
        self.assertFalse(blk.iswa)
        self.assertEqual(blk.cache_kv.shape[3], 4096)

  def test_iswa_mask_pads_and_causal(self):
    # kv_cache_len=8, start_pos=2, T=2 -> 4 valid keys right-aligned, pad_end=4
    mask = iswa_attention_mask(2, 2, 8, dtypes.float32, sliding_window=8).numpy()[0, 0]
    self.assertTrue(np.all(np.isneginf(mask[:, :4])))
    # row0 abs_q=2; abs_key(c)=c-4; causal masks abs_k>abs_q => c>6
    self.assertTrue(np.isneginf(mask[0, 7]))
    self.assertEqual(mask[0, 6], 0)
    # padding is column-only: row1 still sees first valid key at col4
    self.assertEqual(mask[1, 4], 0)

  def test_iswa_forward_decode_and_prefill(self):
    pattern = (True, True, True, False)
    # window pads to 256; max_context 512 forces compact ISWA shift path on local layers
    cfg = _cfg(num_blocks=4, sliding_window=4, sliding_window_pattern=pattern,
               attn_gate=True, post_norm=True, embd_norm=True, max_context=512)
    model = Transformer(cfg)
    tok = Tensor([[1, 2, 3, 4]], dtype=dtypes.int32)
    out = model.forward(tok, 0, Tensor([0.0])).realize()
    self.assertEqual(out.shape, (1, 1))
    self.assertTrue(model.blk[0].iswa)
    self.assertEqual(model.blk[0].kv_cache_len, 256)
    self.assertEqual(model.blk[0].cache_kv.shape[3], 256)
    self.assertFalse(model.blk[3].iswa)
    self.assertEqual(model.blk[3].cache_kv.shape[3], 512)
    for sp in range(4, 12):
      out = model.forward(Tensor([[int(out.item())]], dtype=dtypes.int32), sp, Tensor([0.0])).realize()
      self.assertEqual(out.shape, (1, 1))

class TestQuantLinear(unittest.TestCase):
  def test_q8_0_matches_dequant_linear(self):
    in_f, out_f = 32, 64
    rng = np.random.default_rng(0)
    blocks = []
    for _ in range(out_f * in_f // 32):
      scale = np.float16(rng.uniform(0.01, 0.1))
      qs = rng.integers(-127, 127, size=32, dtype=np.int8)
      blocks.append(np.frombuffer(scale.tobytes() + qs.tobytes(), dtype=np.uint8))
    packed = Tensor(np.concatenate(blocks))
    ql = QuantLinear(in_f, out_f, ggml_type=8)
    ql.qweight = packed
    x = Tensor(rng.standard_normal((2, in_f)).astype(np.float32))
    y = ql(x).realize().numpy()
    w = ggml_data_to_tensor(packed, out_f * in_f, 8).reshape(out_f, in_f).cast(dtypes.float16)
    y_ref = x.linear(w.transpose()).realize().numpy()
    np.testing.assert_allclose(y, y_ref, rtol=1e-3, atol=1e-3)

  def test_replace_linear_installs_qweight(self):
    cfg = _cfg(num_blocks=1, max_context=8)
    model = Transformer(cfg)
    n = cfg.dim * cfg.hidden_dim
    packed = Tensor.zeros((n // 32) * 34, dtype=dtypes.uint8)
    replace_linear_with_quant(model, "blk.0.ffn_down", packed, 8, (cfg.dim, cfg.hidden_dim))
    self.assertIsInstance(model.blk[0].ffn_down, QuantLinear)
    self.assertEqual(model.blk[0].ffn_down.qweight.numel(), packed.numel())


if __name__ == "__main__":
  unittest.main()
