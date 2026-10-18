"""End-to-end engine wiring on CPU with tiny random backbone + tower (no codec).

Checks the package-4 plumbing that does not need real weights: prompt frames ->
mixed input embeddings -> backbone prefill (KV + GDN slot) -> tower prefill ->
sampler -> tower step -> backbone decode step, and that the incremental decode
path agrees with one fresh teacher-forced prefill over prompt + generated rows.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm_drama3.backbone import build_backbone, gdn_layers  # noqa: E402
from nanovllm_drama3.backbone.testing import TINY_TOP, tiny_text_config  # noqa: E402
from nanovllm_drama3.engine import Drama3Engine, import_fusion_for  # noqa: E402
from nanovllm_drama3.tower import CodebookTower, TowerConfig  # noqa: E402

SEM_BEGIN = TINY_TOP["semantic_start_token_id"]  # 100
SEM_END = TINY_TOP["semantic_end_token_id"]      # 107 -> 8 codes
EOS = TINY_TOP["eos_token_id"]                   # 90
Q = 3


class StubTokenizer:
    """Offline tokenizer: control tokens get ids 80..89, every other character one
    id in 1..70 (all below the toy semantic band 100..107 and the EOS id 90)."""

    _CONTROL = re.compile(r"<\|[^|]+\|>")
    _IDS = {"<|im_start|>": 80, "<|im_end|>": EOS, "<|voice|>": 81, **{f"<|speaker:{i}|>": 82 + i for i in range(8)}}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        cursor = 0
        for m in self._CONTROL.finditer(text):
            ids += [ord(c) % 70 + 1 for c in text[cursor : m.start()]]
            ids.append(self._IDS[m.group(0)])
            cursor = m.end()
        ids += [ord(c) % 70 + 1 for c in text[cursor:]]
        return ids


def _tiny_engine(seed: int = 0) -> Drama3Engine:
    import_fusion_for(torch.device("cpu"))
    cfg = tiny_text_config()
    torch.manual_seed(seed)
    model = build_backbone(cfg, "cpu", torch.float32)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
        for layer in gdn_layers(model):
            layer.A_log.copy_(torch.rand(layer.A_log.shape, generator=g).log())
            layer.dt_bias.zero_()
    model.eval()
    tower_cfg = TowerConfig.tiny(vocab_size=SEM_END - SEM_BEGIN + 1, dim=cfg.hidden_size, num_codebooks=Q,
                                 head_dim=8, n_head=4, n_local_heads=2, intermediate_size=64)
    torch.manual_seed(seed + 1)
    tower = CodebookTower(tower_cfg, semantic_begin_id=SEM_BEGIN).eval()
    return Drama3Engine(
        model=model, text_cfg=cfg, tower=tower, codec=None, tokenizer=StubTokenizer(),
        eos_token_id=EOS, semantic_begin_id=SEM_BEGIN, semantic_end_id=SEM_END, num_codebooks=Q,
        device="cpu", dtype=torch.float32, max_seq_len=512, block_size=16,
    )


@pytest.fixture(scope="module")
def engine() -> Drama3Engine:
    return _tiny_engine()


def _ref_codes(frames: int = 5, seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, SEM_END - SEM_BEGIN + 1, (frames, Q), generator=g)


def test_synthesize_frames_and_timings(engine: Drama3Engine):
    wav, frames, t = engine.synthesize("hi there", ref_text="ref words", ref_codes=_ref_codes(),
                                       max_new_tokens=6, seed=11, top_k=4, top_p=0.9)
    assert wav is None  # no codec in the tiny engine
    assert frames.dtype == torch.long and frames.ndim == 2 and frames.shape[1] == Q
    assert 0 <= frames.shape[0] <= 6
    assert t["frames"] == frames.shape[0]
    assert t["stop_reason"] in ("eos", "max_new_tokens")
    if frames.numel():
        assert int(frames.min()) >= 0 and int(frames.max()) < SEM_END - SEM_BEGIN + 1
        assert int(frames[:, 0].min()) >= 1  # semantic code 0 excluded from the main-token candidates
    prompt_rows = engine.build_frames("hi there", "ref words", _ref_codes()).shape[0]
    assert t["prompt_rows"] == prompt_rows and prompt_rows > 5
    assert t["prefill_ms"] > 0 and t["time_to_first_frame_ms"] >= t["prefill_ms"] or frames.shape[0] == 0
    n = frames.shape[0]
    assert len(t["per_step"]["tower_ms"]) == n
    # one backbone decode step per generated frame except the last one (its output is never sampled)
    expected_bb = n if t["stop_reason"] == "eos" else max(n - 1, 0)
    assert len(t["per_step"]["backbone_ms"]) == expected_bb


def test_same_seed_is_deterministic_across_requests(engine: Drama3Engine):
    """Also exercises the per-request GDN slot reset and KV reuse: request 2 must
    not see request 1's state."""
    kw = dict(ref_text="ref words", ref_codes=_ref_codes(), max_new_tokens=5, seed=5, top_k=4)
    _, a, _ = engine.synthesize("one two", **kw)
    engine.synthesize("something else entirely", ref_text="x", ref_codes=_ref_codes(3, 9), max_new_tokens=3, seed=1)
    _, b, _ = engine.synthesize("one two", **kw)
    assert torch.equal(a, b)


def test_decode_steps_match_fresh_teacher_forced_prefill(engine: Drama3Engine):
    """Candidate logits sampled at decode step k (row T-1+k, computed through the
    recurrent GDN state + paged KV decode path with mixed frame embeddings) equal a
    fresh full prefill over prompt + generated frames."""
    text, ref_text, codes = "abc def", "ref words", _ref_codes(6, 4)
    _, frames, t = engine.synthesize(text, ref_text=ref_text, ref_codes=codes, max_new_tokens=5, seed=2,
                                     top_k=4, collect_logits=True)
    step_logits = t["candidate_logits"]
    assert len(step_logits) >= 1
    prompt = engine.build_frames(text, ref_text, codes)
    T = prompt.shape[0]
    gen_rows = torch.cat((frames[:, :1] + SEM_BEGIN, frames), dim=1) if frames.numel() else torch.empty(0, 1 + Q, dtype=torch.long)
    all_frames = torch.cat((prompt, gen_rows))
    full = engine.teacher_forced_logits(all_frames)
    for k, lg in enumerate(step_logits):
        row = T - 1 + k
        assert row < all_frames.shape[0]
        torch.testing.assert_close(lg, full[row].cpu(), rtol=1e-4, atol=1e-4)


def test_ref_codes_in_codebook_major_layout_are_transposed(engine: Drama3Engine):
    codes = _ref_codes(7)
    a = engine.build_frames("x", "r", codes)
    b = engine.build_frames("x", "r", codes.t().contiguous())  # [Q, F] as stored in the manifest
    assert torch.equal(a, b)
    assert (a[:, 0] >= SEM_BEGIN).sum() == 7


def test_prompt_longer_than_budget_is_rejected(engine: Drama3Engine):
    with pytest.raises(ValueError):
        engine.synthesize("x", ref_text="r", ref_codes=_ref_codes(engine.max_seq_len), max_new_tokens=1)


def test_cli_helpers(tmp_path: Path):
    from nanovllm_drama3 import cli

    wav = torch.sin(torch.arange(4410) / 10.0) * 0.5
    cli.write_wav(tmp_path / "a.wav", wav, 44100)
    data, sr = cli.read_wav(tmp_path / "a.wav")
    assert sr == 44100 and data.shape == (4410,) and np.abs(data - wav.numpy()).max() < 1e-3
    row = {"utt": "u", "codes": torch.randint(0, 10, (10, 9)).tolist()}
    assert cli.manifest_codes(row).shape == (9, 10)
    assert cli.normalize_text("Hello, World! It's me.") == "hello world its me"
    assert cli.word_errors("a b c", "a x c d") == (2, 3)
    assert cli.word_errors("a b", "a b") == (0, 2)


def test_gpu_sdpa_replacements_match_reference_fallbacks():
    """The CUDA flash-attn stand-ins in engine.py are device-agnostic torch; check
    them here against the float32 reference implementations."""
    from nanovllm_drama3.backbone import cpu_fallback as ref
    from nanovllm_drama3.engine import sdpa_decode_with_kvcache, sdpa_varlen_prefill

    torch.manual_seed(0)
    H, Hkv, D, bs = 4, 2, 16, 8
    L1, L2 = 11, 6
    q = torch.randn(L1 + L2, H, D)
    k = torch.randn(L1 + L2, Hkv, D)
    v = torch.randn(L1 + L2, Hkv, D)
    cu = torch.tensor([0, L1, L1 + L2], dtype=torch.int32)
    a = ref.flash_attn_varlen_func(q, k, v, cu, cu, L1, L1, softmax_scale=D**-0.5, causal=True)
    b = sdpa_varlen_prefill(q, k, v, cu, cu, L1, L1, softmax_scale=D**-0.5, causal=True)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)

    nblocks = 3
    k_cache = torch.randn(nblocks, bs, Hkv, D)
    v_cache = torch.randn(nblocks, bs, Hkv, D)
    qd = torch.randn(2, 1, H, D)
    lens = torch.tensor([13, 20], dtype=torch.int32)
    table = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int32)
    a = ref.flash_attn_with_kvcache(qd, k_cache, v_cache, cache_seqlens=lens, block_table=table, softmax_scale=D**-0.5)
    b = sdpa_decode_with_kvcache(qd, k_cache, v_cache, cache_seqlens=lens, block_table=table, softmax_scale=D**-0.5)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
