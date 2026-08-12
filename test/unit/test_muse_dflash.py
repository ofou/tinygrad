import os, unittest
import numpy as np
from tinygrad import Tensor, dtypes, nn
from tinygrad.llm.dflash import (
  target_layers_0based, noise_block_tokens, accept_prefix,
  enable_dflash_capture, disable_dflash_capture, take_dflash_features,
  DFlashConfig, DFlashDraft, TargetHead, _HEAD_KEYS,
)
from tinygrad.llm.model import Transformer, TransformerConfig
from tinygrad.llm.gguf import _gguf_parse_header_file

DFLASH_GGUF = os.path.expanduser(
  "~/.lmstudio/models/meta-models/Muse-Glimmer-30B-GGUF/dflash-kquant.gguf")
MUSE_GGUF = os.path.expanduser(
  "~/.lmstudio/models/lmstudio-community/Muse-Glimmer-30B-GGUF/muse-glimmer-30B-kquant-17gb.gguf")

class TestDFlashHelpers(unittest.TestCase):
  def test_target_layers_as_gguf(self):
    # GGUF stores layer-input indices (llama.cpp uses as-is). HF layer-output ids are these-1.
    self.assertEqual(target_layers_0based([2, 14, 26, 38, 50]), (2, 14, 26, 38, 50))
    from tinygrad.llm.dflash import parse_target_layers
    self.assertEqual(parse_target_layers({"dflash.target_layers": [2, 14, 26, 38, 50]}), (2, 14, 26, 38, 50))

  def test_noise_block_layout(self):
    toks = noise_block_tokens(id_last=7, n_draft=4, mask_token_id=201818)
    self.assertEqual(toks, [7, 201818, 201818, 201818, 201818])
    self.assertEqual(len(toks), 5)  # n_draft+1

  def test_noise_block_zero_draft(self):
    self.assertEqual(noise_block_tokens(3, 0, 9), [3])

  def test_accept_prefix_full(self):
    self.assertEqual(accept_prefix([1, 2, 3], [1, 2, 3, 9]), 3)

  def test_accept_prefix_partial(self):
    self.assertEqual(accept_prefix([1, 2, 9], [1, 2, 3]), 2)

  def test_accept_prefix_none(self):
    self.assertEqual(accept_prefix([5, 6], [1, 2]), 0)

  def test_accept_prefix_empty(self):
    self.assertEqual(accept_prefix([], [1]), 0)

class TestDFlashMetadata(unittest.TestCase):
  @unittest.skipUnless(os.path.isfile(DFLASH_GGUF), "draft GGUF not present")
  def test_gguf_header_metadata(self):
    kv, infos, _ = _gguf_parse_header_file(__import__("pathlib").Path(DFLASH_GGUF))
    self.assertEqual(kv["general.architecture"], "dflash")
    self.assertEqual(kv["dflash.block_count"], 5)
    self.assertEqual(kv["dflash.block_size"], 16)
    self.assertEqual(kv["dflash.embedding_length"], 6656)
    self.assertEqual(kv["dflash.feed_forward_length"], 19968)
    self.assertEqual(kv["dflash.attention.head_count"], 32)
    self.assertEqual(kv["dflash.attention.head_count_kv"], 8)
    self.assertEqual(kv["dflash.attention.sliding_window"], 2048)
    self.assertEqual(list(kv["dflash.target_layers"]), [2, 14, 26, 38, 50])
    self.assertEqual(kv["tokenizer.ggml.mask_token_id"], 201818)
    names = {n for n, *_ in infos}
    self.assertIn("fc.weight", names)
    self.assertIn("enc.output_norm.weight", names)
    self.assertIn("output_norm.weight", names)
    self.assertIn("blk.0.attn_q.weight", names)
    self.assertIn("blk.4.ffn_down.weight", names)
    self.assertNotIn("token_embd.weight", names)
    self.assertNotIn("output.weight", names)
    self.assertTrue(all(not n.endswith("attn_gate.weight") for n in names))

  @unittest.skipUnless(os.path.isfile(DFLASH_GGUF), "draft GGUF not present")
  def test_config_from_kv(self):
    kv, _, _ = _gguf_parse_header_file(__import__("pathlib").Path(DFLASH_GGUF))
    cfg = DFlashConfig.from_gguf_kv(kv, max_context=4096)
    self.assertEqual(cfg.num_blocks, 5)
    self.assertEqual(cfg.dim, 6656)
    self.assertEqual(cfg.block_size, 16)
    self.assertEqual(cfg.n_draft_max, 15)
    self.assertEqual(cfg.mask_token_id, 201818)
    self.assertEqual(cfg.target_layers, (2, 14, 26, 38, 50))
    self.assertEqual(cfg.max_context, 4096)
    self.assertEqual(cfg.dim * cfg.n_extract, 6656 * 5)
    # llama.cpp LLM_ARCH_DFLASH → NEOX (not Muse NORM / interleaved)
    self.assertFalse(cfg.rope_interleaved_qk)

class TestDFlashRopeLayout(unittest.TestCase):
  @unittest.skipUnless(os.path.isfile(DFLASH_GGUF), "draft GGUF not present")
  def test_dflash_is_neox_not_muse_interleaved(self):
    """KEEP_QK_QUANT must not flip DFlash to Muse interleaved activation permute."""
    import os as _os
    prev = _os.environ.get("KEEP_QK_QUANT")
    try:
      _os.environ["KEEP_QK_QUANT"] = "1"
      kv, _, _ = _gguf_parse_header_file(__import__("pathlib").Path(DFLASH_GGUF))
      cfg = DFlashConfig.from_gguf_kv(kv, max_context=256)
      self.assertFalse(cfg.rope_interleaved_qk)
    finally:
      if prev is None: _os.environ.pop("KEEP_QK_QUANT", None)
      else: _os.environ["KEEP_QK_QUANT"] = prev

class TestDFlashCaptureHooks(unittest.TestCase):
  def test_capture_zero_cost_when_off(self):
    cfg = TransformerConfig(
      num_blocks=2, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, norm_eps=1e-5,
      vocab_size=32, head_dim=4, rope_theta=10000.0, rope_dim=4, v_head_dim=4, max_context=16)
    model = Transformer(cfg)
    self.assertIsNone(getattr(model, "_dflash_capture", None))
    out = model.forward(Tensor([[1, 2]], dtype=dtypes.int32), 0, Tensor([0.0])).realize()
    self.assertEqual(out.shape, (1, 1))
    self.assertEqual(take_dflash_features(model), {})

  def test_capture_layer_inputs(self):
    cfg = TransformerConfig(
      num_blocks=4, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, norm_eps=1e-5,
      vocab_size=32, head_dim=4, rope_theta=10000.0, rope_dim=4, v_head_dim=4, max_context=16)
    model = Transformer(cfg)
    enable_dflash_capture(model, (1, 3))
    _ = model.forward(Tensor([[1, 2, 3]], dtype=dtypes.int32), 0, Tensor([0.0])).realize()
    feats = take_dflash_features(model)
    self.assertEqual(sorted(feats), [1, 3])
    for v in feats.values():
      self.assertEqual(tuple(v.shape), (1, 3, 8))
    disable_dflash_capture(model)
    self.assertIsNone(model._dflash_capture)

class TestDFlashVerifyOutputs(unittest.TestCase):
  def test_forward_verify_no_side_channel(self):
    """Verify returns preds+feats as outputs; does not require _dflash_capture."""
    cfg = TransformerConfig(
      num_blocks=4, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, norm_eps=1e-5,
      vocab_size=32, head_dim=4, rope_theta=10000.0, rope_dim=4, v_head_dim=4, max_context=16)
    model = Transformer(cfg)
    model._dflash_verify_layers = (1, 3)
    self.assertIsNone(getattr(model, "_dflash_capture", None))
    ret = model.forward_verify(Tensor([[1, 2, 3]], dtype=dtypes.int32), 0, Tensor([0.0]))
    preds, f1, f3 = ret
    Tensor.realize(preds, f1, f3)
    self.assertEqual(tuple(preds.shape), (1, 3))
    self.assertEqual(tuple(f1.shape), (1, 3, 8))
    self.assertEqual(tuple(f3.shape), (1, 3, 8))
    self.assertEqual(take_dflash_features(model), {})

  def test_forward_verify_matches_capture_inputs(self):
    """Layer taps from forward_verify match capture side-channel (pre-block inputs)."""
    cfg = TransformerConfig(
      num_blocks=4, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, norm_eps=1e-5,
      vocab_size=32, head_dim=4, rope_theta=10000.0, rope_dim=4, v_head_dim=4, max_context=16)
    model = Transformer(cfg)
    sd = nn.state.get_state_dict(model)
    nn.state.load_state_dict(model, {k: Tensor.randn(*v.shape) * 0.02 for k, v in sd.items()}, verbose=False)
    toks = Tensor([[4, 5, 6, 7]], dtype=dtypes.int32)
    # capture path
    enable_dflash_capture(model, (1, 3))
    _ = model.forward_next_ids(toks, 0, Tensor([0.0])).realize()
    cap = take_dflash_features(model)
    Tensor.realize(*cap.values())
    disable_dflash_capture(model)
    # output path (fresh caches)
    for b in model.blk:
      for attr in ("cache_kv", "freqs_cis", "iswa", "kv_cache_len"):
        if hasattr(b, attr): delattr(b, attr)
    model._dflash_verify_layers = (1, 3)
    preds, f1, f3 = model.forward_verify(toks, 0, Tensor([0.0]))
    Tensor.realize(preds, f1, f3)
    self.assertLess(float(np.abs(f1.numpy() - cap[1].numpy()).max()), 1e-5)
    self.assertLess(float(np.abs(f3.numpy() - cap[3].numpy()).max()), 1e-5)

  def test_sequential_verify_feature_cat(self):
    """T=1 verify steps cat to (1,T,D) like a parallel tap."""
    cfg = TransformerConfig(
      num_blocks=4, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, norm_eps=1e-5,
      vocab_size=32, head_dim=4, rope_theta=10000.0, rope_dim=4, v_head_dim=4, max_context=32)
    model = Transformer(cfg)
    model.ensure_verify_jit((1, 3))
    from tinygrad import UOp
    vsp = UOp.variable("sp", 0, 31)
    batch = [4, 5, 6]
    feat_lists = {1: [], 3: []}
    preds = []
    for i, tok in enumerate(batch):
      ret = model.verify_jit(Tensor([[tok]], dtype=dtypes.int32).contiguous(), vsp.bind(i), Tensor([0.0]))
      preds.append(int(ret[0].item()))
      feat_lists[1].append(ret[1])
      feat_lists[3].append(ret[2])
    f1 = feat_lists[1][0].cat(*feat_lists[1][1:], dim=1).realize()
    f3 = feat_lists[3][0].cat(*feat_lists[3][1:], dim=1).realize()
    self.assertEqual(len(preds), 3)
    self.assertEqual(tuple(f1.shape), (1, 3, 8))
    self.assertEqual(tuple(f3.shape), (1, 3, 8))

  def test_ensure_verify_jit_shapes(self):
    cfg = TransformerConfig(
      num_blocks=3, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, norm_eps=1e-5,
      vocab_size=32, head_dim=4, rope_theta=10000.0, rope_dim=4, v_head_dim=4, max_context=32)
    model = Transformer(cfg)
    jit = model.ensure_verify_jit((0, 2))
    self.assertEqual(model._dflash_verify_layers, (0, 2))
    from tinygrad import UOp
    vsp = UOp.variable("sp", 0, 31)
    for sp in (0, 3):
      ret = jit(Tensor([[1, 2]], dtype=dtypes.int32).contiguous(), vsp.bind(sp), Tensor([0.0]))
      preds, fa, fb = ret
      self.assertEqual(tuple(preds.shape), (1, 2))
      self.assertEqual(tuple(fa.shape), (1, 2, 8))
      self.assertEqual(tuple(fb.shape), (1, 2, 8))

class TestDFlashTinyShapes(unittest.TestCase):

  def test_encode_inject_shapes(self):
    # tiny stand-in target + draft config
    tcfg = TransformerConfig(
      num_blocks=4, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, norm_eps=1e-5,
      vocab_size=32, head_dim=4, rope_theta=10000.0, rope_dim=4, v_head_dim=4, max_context=32)
    target = Transformer(tcfg)
    dcfg = DFlashConfig(
      num_blocks=2, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, head_dim=4,
      rope_theta=10000.0, rope_dim=4, norm_eps=1e-5, max_context=32, block_size=8,
      mask_token_id=31, target_layers=(1, 3), sliding_window=0,
      sliding_window_pattern=(False, False), rope_interleaved_qk=False)
    draft = DFlashDraft(dcfg, target)
    self.assertEqual(draft.fc.weight.shape, (8, 16))  # out, in = dim, dim*n_extract
    feats = {1: Tensor.randn(1, 4, 8), 3: Tensor.randn(1, 4, 8)}
    fused = draft.encode(feats).realize()
    self.assertEqual(tuple(fused.shape), (1, 4, 8))
    draft.process(feats, 0)
    # noise layout + draft forward (random weights — just shape/smoke)
    toks = noise_block_tokens(5, 3, 31)
    self.assertEqual(len(toks), 4)
    out = draft.draft_block(5, 4, n_draft=3, temperature=0.0)
    self.assertEqual(len(out), 3)
    self.assertTrue(all(isinstance(x, int) for x in out))


class TestDFlashTargetIsolation(unittest.TestCase):
  def test_target_not_in_state_dict(self):
    from tinygrad.llm.dflash import DFlashDraft, DFlashConfig, _TargetRef
    tcfg = TransformerConfig(
      num_blocks=2, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, norm_eps=1e-5,
      vocab_size=32, head_dim=4, rope_theta=10000.0, rope_dim=4, v_head_dim=4, max_context=16)
    target = Transformer(tcfg)
    dcfg = DFlashConfig(
      num_blocks=1, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, head_dim=4,
      rope_theta=10000.0, rope_dim=4, norm_eps=1e-5, max_context=16, block_size=4,
      mask_token_id=31, target_layers=(1,), sliding_window=0,
      sliding_window_pattern=(False,), rope_interleaved_qk=False)
    draft = DFlashDraft(dcfg, target)
    keys = list(nn.state.get_state_dict(draft))
    self.assertTrue(all(not k.startswith("_target") for k in keys))
    self.assertIs(draft.target, target)
    self.assertIsInstance(draft._target_ref, _TargetRef)

class TestDFlashInjectCache(unittest.TestCase):
  def test_inject_writes_absolute_kv(self):
    """Injected K must land at start_pos (absolute) when ISWA is off."""
    import numpy as np
    from tinygrad.llm.model import apply_rope
    tcfg = TransformerConfig(
      num_blocks=2, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, norm_eps=1e-5,
      vocab_size=32, head_dim=4, rope_theta=10000.0, rope_dim=4, v_head_dim=4, max_context=32)
    target = Transformer(tcfg)
    dcfg = DFlashConfig(
      num_blocks=1, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, head_dim=4,
      rope_theta=10000.0, rope_dim=4, norm_eps=1e-5, max_context=32, block_size=8,
      mask_token_id=31, target_layers=(1,), sliding_window=0,
      sliding_window_pattern=(True,), rope_interleaved_qk=False)
    draft = DFlashDraft(dcfg, target)
    feats = {1: Tensor.randn(1, 3, 8)}
    fused = draft.encode(feats).realize()
    draft.process(feats, 0)
    b = draft.blk[0]
    self.assertFalse(b.iswa)
    cache = b.cache_kv.realize().numpy()
    k = b.attn_k(fused).reshape(1, 3, 1, 4).transpose(1, 2)
    if b.config.qk_norm: k = b.attn_k_norm(k)
    if b.config.use_rope:
      k = apply_rope(k[..., :4], b.freqs_cis[0:3])
    k = k.realize().numpy()
    self.assertLess(float(np.abs(k[0] - cache[0, 0, :, 0:3, :]).max()), 1e-3)


class TestSlidingWindowPattern(unittest.TestCase):
  def test_sliding_window_pattern_period(self):
    from tinygrad.llm.model import parse_sliding_window_pattern
    self.assertEqual(parse_sliding_window_pattern(4, 8),
                     (True, True, True, False, True, True, True, False))
    self.assertEqual(parse_sliding_window_pattern([1, 1, 0], 3), (True, True, False))


class TestDFlashInjectEqualsJoint(unittest.TestCase):
  def test_inject_matches_joint_attention(self):
    """llama.cpp inject+noise must match HF joint ctx||noise attention (absolute KV)."""
    import numpy as np
    from tinygrad import dtypes
    from tinygrad.llm.model import apply_rope
    Tensor.manual_seed(0)
    tcfg = TransformerConfig(
      num_blocks=2, dim=16, hidden_dim=32, n_heads=4, n_kv_heads=2, norm_eps=1e-5,
      vocab_size=64, head_dim=4, rope_theta=10000.0, rope_dim=4, v_head_dim=4, max_context=64, qk_norm=4)
    target = Transformer(tcfg)
    sd = nn.state.get_state_dict(target)
    nn.state.load_state_dict(target, {k: Tensor.randn(*v.shape) * 0.02 for k, v in sd.items()}, verbose=False)
    dcfg = DFlashConfig(
      num_blocks=2, dim=16, hidden_dim=32, n_heads=4, n_kv_heads=2, head_dim=4,
      rope_theta=10000.0, rope_dim=4, norm_eps=1e-5, max_context=64, block_size=4,
      mask_token_id=63, target_layers=(1,), sliding_window=0,
      sliding_window_pattern=(True, True), rope_interleaved_qk=False)
    draft = DFlashDraft(dcfg, target)
    dsd = nn.state.get_state_dict(draft)
    nn.state.load_state_dict(draft, {k: Tensor.randn(*v.shape) * 0.02 for k, v in dsd.items() if getattr(v, "shape", None)},
                           verbose=False, strict=False)
    ctx_len, n_draft = 5, 3
    feats = {1: Tensor.randn(1, ctx_len, 16)}
    fused = draft.encode(feats).realize()
    toks = [7] + [63] * n_draft
    noise = target.token_embd(Tensor([toks], dtype=dtypes.int32)).float().realize()
    draft.reset_cache()
    draft.process(feats, 0)
    x = noise
    for b in draft.blk: x = b(x, ctx_len)
    logits_inj = target.output(draft.output_norm(x)).realize().numpy()
    # HF joint
    x = noise
    for block in draft.blk:
      block._init_state(x)
      xn = block.attn_norm(x)
      B, Tn, _ = xn.shape
      Ct = fused.shape[1]
      q = block.attn_q(xn).reshape(B, Tn, block.config.n_heads, 4).transpose(1, 2)
      k_ctx = block.attn_k(fused).reshape(B, Ct, block.config.n_kv_heads, 4).transpose(1, 2)
      k_n = block.attn_k(xn).reshape(B, Tn, block.config.n_kv_heads, 4).transpose(1, 2)
      v_ctx = block.attn_v(fused).reshape(B, Ct, block.config.n_kv_heads, 4).transpose(1, 2)
      v_n = block.attn_v(xn).reshape(B, Tn, block.config.n_kv_heads, 4).transpose(1, 2)
      if block.config.qk_norm:
        q, k_ctx, k_n = block.attn_q_norm(q), block.attn_k_norm(k_ctx), block.attn_k_norm(k_n)
      if block.config.use_rope:
        k_ctx = apply_rope(k_ctx, block.freqs_cis[0:Ct])
        q = apply_rope(q, block.freqs_cis[ctx_len:ctx_len+Tn])
        k_n = apply_rope(k_n, block.freqs_cis[ctx_len:ctx_len+Tn])
      attn = q.scaled_dot_product_attention(k_ctx.cat(k_n, dim=2), v_ctx.cat(v_n, dim=2), enable_gqa=True)
      attn = block.attn_output(attn.transpose(1, 2).reshape(B, Tn, -1))
      h = x + attn
      x = h + block._feed_forward(block.ffn_norm(h))
    logits_joint = target.output(draft.output_norm(x)).realize().numpy()
    self.assertLess(float(np.abs(logits_inj - logits_joint).max()), 1e-4)


class TestTargetHead(unittest.TestCase):
  def test_head_state_has_no_blocks(self):
    head = TargetHead(dim=8, vocab_size=32, norm_eps=1e-5, embd_norm=True)
    keys = list(nn.state.get_state_dict(head))
    self.assertTrue(all(not k.startswith("blk.") for k in keys))
    self.assertIn("token_embd.weight", keys)
    self.assertIn("output.weight", keys)
    self.assertIn("output_norm.weight", keys)
    self.assertIsNotNone(head.embd_norm)

  def test_draft_tiny_against_target_head(self):
    head = TargetHead(dim=8, vocab_size=32, norm_eps=1e-5, embd_norm=True)
    dcfg = DFlashConfig(
      num_blocks=2, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, head_dim=4,
      rope_theta=10000.0, rope_dim=4, norm_eps=1e-5, max_context=32, block_size=8,
      mask_token_id=31, target_layers=(1, 3), sliding_window=0,
      sliding_window_pattern=(False, False), rope_interleaved_qk=False)
    draft = DFlashDraft(dcfg, head)
    self.assertIs(draft.target, head)
    feats = {1: Tensor.randn(1, 4, 8), 3: Tensor.randn(1, 4, 8)}
    draft.process(feats, 0)
    out = draft.draft_block(5, 4, n_draft=3, temperature=0.0)
    self.assertEqual(len(out), 3)

  def test_head_keys_constant(self):
    self.assertTrue({"token_embd.weight", "output.weight", "output_norm.weight"} <= _HEAD_KEYS)
    self.assertTrue(all(not k.startswith("blk.") for k in _HEAD_KEYS))

  @unittest.skipUnless(os.path.isfile(MUSE_GGUF), "Muse GGUF not present")
  def test_load_from_gguf_isolates_head(self):
    head, kv = TargetHead.load_from_gguf(MUSE_GGUF)
    self.assertEqual(kv["general.architecture"], "muse-glimmer")
    self.assertEqual(head.dim, 6656)
    self.assertEqual(head.vocab_size, 202048)
    keys = list(nn.state.get_state_dict(head))
    self.assertTrue(all(not k.startswith("blk.") for k in keys), keys[:20])
    self.assertIn("token_embd.weight", keys)
    self.assertTrue("output.qweight" in keys or "output.weight" in keys)
    self.assertIn("output_norm.weight", keys)

  @unittest.skipUnless(os.path.isfile(MUSE_GGUF) and os.path.isfile(DFLASH_GGUF), "GGUFs not present")
  def test_draft_binds_target_head(self):
    head, _ = TargetHead.load_from_gguf(MUSE_GGUF)
    draft = DFlashDraft.from_gguf(DFLASH_GGUF, head, max_context=256)
    self.assertEqual(draft.config.mask_token_id, 201818)
    self.assertIs(draft.target, head)
    dkeys = list(nn.state.get_state_dict(draft))
    self.assertTrue(all(not k.startswith("_target") for k in dkeys))


if __name__ == "__main__":
  unittest.main()
