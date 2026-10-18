"""Vendored nano-vllm-fusion backbone on CPU: import, build, prefill/decode consistency."""

import sys

import pytest
import torch

from nanovllm_drama3.backbone import (
    allocate_caches,
    attention_layers,
    build_backbone,
    fusion_root,
    fusion_state,
    gdn_layers,
    import_fusion,
    reset_state_slot,
)
from nanovllm_drama3.backbone.testing import tiny_text_config

CPU_ONLY = not torch.cuda.is_available()


@pytest.fixture(scope="module")
def fusion():
    return import_fusion()


def test_vendored_checkout_present():
    root = fusion_root()
    assert (root / "LICENSE").exists()
    assert (root / "nanovllm" / "models" / "qwen3_5_moe.py").exists()
    assert (root / "nanovllm" / "layers" / "gated_delta_net.py").exists()


def test_import_fusion_cpu(fusion):
    from nanovllm.models import qwen3_5_moe
    from nanovllm.layers import attention, gated_delta_net  # noqa: F401

    assert hasattr(qwen3_5_moe, "Qwen3_5MoeForCausalLM")
    assert qwen3_5_moe.Qwen3_5MoeBlock.__name__ == "Drama3MoeBlock"  # patch applied
    assert qwen3_5_moe._UpstreamQwen3_5MoeBlock.__name__ == "Qwen3_5MoeBlock"
    st = fusion_state()
    if CPU_ONLY:
        assert st["cpu_fallback"] is True
        assert st["stubs"] == {"triton": True, "flash_attn": True}
        # stubs must not leak into sys.modules (they break torch._inductor imports)
        assert not getattr(sys.modules.get("triton"), "__nanovllm_drama3_stub__", False)
        assert "flash_attn" not in sys.modules or not getattr(sys.modules["flash_attn"], "__nanovllm_drama3_stub__", False)
        from nanovllm_drama3.backbone import cpu_fallback

        assert attention.store_kvcache is cpu_fallback.store_kvcache
    # lazy engine exports on the light package
    import nanovllm

    assert nanovllm.SamplingParams(temperature=0.5).temperature == 0.5
    # re-import is idempotent
    assert import_fusion() is fusion


def _seeded_tiny_model(seed: int = 0):
    cfg = tiny_text_config()
    torch.manual_seed(seed)
    model = build_backbone(cfg, "cpu", torch.float32)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
        for layer in gdn_layers(model):
            layer.A_log.copy_(torch.rand(layer.A_log.shape, generator=g).log())  # A_log < 0
            layer.dt_bias.zero_()
    model.eval()
    return cfg, model


def test_build_tiny_backbone(fusion):
    cfg, model = _seeded_tiny_model()
    names = dict(model.named_parameters())
    assert len(gdn_layers(model)) == 3 and len(attention_layers(model)) == 1
    assert "model.layers.0.linear_attn.in_proj_qkv.weight" in names
    assert "model.layers.3.self_attn.q_proj.weight" in names
    assert names["model.layers.3.self_attn.q_proj.weight"].shape == (cfg.num_attention_heads * cfg.head_dim * 2, cfg.hidden_size)
    assert names["model.layers.0.mlp.experts.0.gate_up_proj.weight"].shape == (2 * cfg.moe_intermediate_size, cfg.hidden_size)
    assert names["model.layers.0.mlp.shared_expert_gate.weight"].shape == (1, cfg.hidden_size)
    assert names["model.layers.0.linear_attn.A_log"].dtype == torch.float32
    assert names["lm_head.weight"].shape == (cfg.vocab_size, cfg.hidden_size)


def _prefill(model, cfg, seqs, slots, block_size):
    from nanovllm.utils.context import reset_context, set_context

    ids = torch.cat(seqs)
    positions = torch.cat([torch.arange(len(s)) for s in seqs])
    cu = torch.tensor([0] + [len(s) for s in seqs], dtype=torch.int32).cumsum(0).to(torch.int32)
    # one block per sequence: block id == slot id
    slot_mapping = torch.cat([torch.arange(len(s)) + slot * block_size for s, slot in zip(seqs, slots)]).to(torch.int32)
    set_context(True, cu, cu, max(map(len, seqs)), max(map(len, seqs)), slot_mapping, None, None,
                state_indices=torch.tensor(slots, dtype=torch.int32))
    with torch.inference_mode():
        hidden = model(ids, positions)
        logits = model.compute_logits(hidden)
    reset_context()
    return hidden, logits


def _decode(model, cfg, seqs, slots, block_size):
    from nanovllm.utils.context import reset_context, set_context

    ids = torch.tensor([s[-1] for s in seqs])
    positions = torch.tensor([len(s) - 1 for s in seqs])
    context_lens = torch.tensor([len(s) for s in seqs], dtype=torch.int32)
    slot_mapping = torch.tensor([slot * block_size + len(s) - 1 for s, slot in zip(seqs, slots)], dtype=torch.int32)
    block_tables = torch.tensor([[slot] for slot in slots], dtype=torch.int32)
    set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables,
                state_indices=torch.tensor(slots, dtype=torch.int32))
    with torch.inference_mode():
        hidden = model(ids, positions)
        logits = model.compute_logits(hidden)
    reset_context()
    return hidden, logits


def test_prefill_then_decode_matches_full_prefill(fusion):
    """GDN recurrent decode + paged-KV decode must agree with chunked prefill over
    the same tokens (validates the state-slot path end to end on CPU)."""
    cfg, model = _seeded_tiny_model()
    block_size = 16
    allocate_caches(model, cfg, num_blocks=4, block_size=block_size, num_state_slots=4, device="cpu", dtype=torch.float32)
    g = torch.Generator().manual_seed(1)
    seq_a = torch.randint(0, cfg.vocab_size, (9,), generator=g)
    seq_b = torch.randint(0, cfg.vocab_size, (6,), generator=g)
    next_a, next_b = seq_a[-1:], seq_b[-1:]

    # Step 1: prefill the prefixes into slots 0/1, then decode one token each.
    _, logits_prefill = _prefill(model, cfg, [seq_a[:-1], seq_b[:-1]], [0, 1], block_size)
    assert logits_prefill.shape == (2, cfg.vocab_size)
    _, logits_decode = _decode(model, cfg, [seq_a, seq_b], [0, 1], block_size)
    assert logits_decode.shape == (2, cfg.vocab_size)
    assert torch.isfinite(logits_decode).all()

    # Step 2: prefill the full sequences fresh into slots 2/3.
    _, logits_full = _prefill(model, cfg, [seq_a, seq_b], [2, 3], block_size)
    torch.testing.assert_close(logits_decode, logits_full, rtol=1e-4, atol=1e-4)

    # Step 3: slot reuse without reset would leak conv/recurrent state.
    reset_state_slot(model, 0)
    for layer in gdn_layers(model):
        assert layer.recurrent_states[0].abs().sum() == 0 and layer.conv_states[0].abs().sum() == 0


def test_shared_expert_gate_is_per_token(fusion):
    """The patched gate is sigmoid(x @ W^T) with W (1, hidden): different tokens get
    different gates, and the block equals a hand computation."""
    cfg, model = _seeded_tiny_model()
    block = model.model.layers[0].mlp
    x = torch.randn(5, cfg.hidden_size)
    gate = block.shared_expert_gate(x)
    assert gate.shape == (5, 1) and gate.std() > 0
    with torch.inference_mode():
        out = block(x)
        # reference: routed part + sigmoid(gate) * shared
        import torch.nn.functional as F

        logits = block.gate(x)
        probs = F.softmax(logits, dim=1, dtype=torch.float)
        w, idx = torch.topk(probs, block.top_k, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        ref = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for j in range(block.top_k):
                ref[t] += w[t, j] * block.experts[idx[t, j]](x[t : t + 1])[0]
        ref += torch.sigmoid(F.linear(x, block.shared_expert_gate.weight)) * block.shared_expert(x)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_embedding_injection_hook(fusion):
    """`image_embeds`/`image_token_mask` lets a caller override input embeddings per
    row - the hook package 4 can use to inject audio-frame (codebook-sum) embeddings."""
    cfg, model = _seeded_tiny_model()
    from nanovllm.utils.context import reset_context, set_context

    ids = torch.randint(0, cfg.vocab_size, (4,))
    pos = torch.arange(4)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    set_context(True, cu, cu, 4, 4, torch.arange(4, dtype=torch.int32), None, None, state_indices=None)
    with torch.inference_mode():
        base = model(ids, pos)
        mask = torch.tensor([False, True, False, True])
        emb = model.model.embed_tokens.weight[ids[mask]] * 0 + 1.0
        injected = model.model(ids, pos, image_embeds=emb, image_token_mask=mask)
    reset_context()
    assert base.shape == injected.shape == (4, cfg.hidden_size)
    assert not torch.allclose(base, injected)
