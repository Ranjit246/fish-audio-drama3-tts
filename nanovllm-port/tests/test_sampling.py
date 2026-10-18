# SPDX-License-Identifier: Apache-2.0
"""Main-token sampler (candidates, RAS re-draw, seeds) and codebook-head draws."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm_drama3.sampling import (  # noqa: E402
    RAS_WINDOW,
    MainTokenSampler,
    SamplingParams,
    main_candidate_ids,
    sample_codebook,
    sample_rows,
    sample_top_k_top_p,
)

EOS, BEGIN, END = 5, 100, 115  # 16 semantic codes 100..115 in a 200-token vocabulary
VOCAB = 200


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# ---------------------------------------------------------------------------
# sample_rows / top-k / top-p
# ---------------------------------------------------------------------------


def test_sample_rows_is_seed_deterministic():
    logits = torch.randn(4, 50, generator=gen(0))
    a = sample_top_k_top_p(logits, 1.0, 30, 0.9, gen(1))
    b = sample_top_k_top_p(logits, 1.0, 30, 0.9, gen(1))
    assert torch.equal(a, b)
    draws = {tuple(sample_top_k_top_p(logits, 1.0, 30, 0.9, gen(s)).tolist()) for s in range(20)}
    assert len(draws) > 1


def test_greedy_takes_argmax():
    logits = torch.randn(3, 40, generator=gen(0))
    out = sample_top_k_top_p(logits, 0.0, 30, 0.9, gen(1))
    assert torch.equal(out, logits.argmax(-1))
    # vLLM marks greedy rows with temperature -1
    out = sample_rows(logits, torch.full((3,), -1.0), torch.full((3,), 40), torch.ones(3), gen(2))
    assert torch.equal(out, logits.argmax(-1))


def test_top_k_restricts_candidates():
    logits = torch.randn(1, 64, generator=gen(0)).expand(200, 64).contiguous()
    top3 = set(logits[0].topk(3).indices.tolist())
    draws = set(sample_top_k_top_p(logits, 1.0, 3, 1.0, gen(3)).tolist())
    assert draws <= top3 and len(draws) > 1
    assert set(sample_top_k_top_p(logits, 1.0, 1, 1.0, gen(3)).tolist()) == {int(logits[0].argmax())}


def test_top_p_keeps_only_the_nucleus():
    logits = torch.zeros(300, 10)
    logits[:, 2] = 6.0  # p(2) ~ 0.98
    draws = set(sample_top_k_top_p(logits, 1.0, 0, 0.5, gen(4)).tolist())
    assert draws == {2}
    draws = set(sample_top_k_top_p(logits, 1.0, 0, 1.0, gen(4)).tolist())
    assert len(draws) > 1


def test_top_k_zero_disables_filter_and_width_is_clamped():
    logits = torch.randn(2, 8, generator=gen(0))
    assert sample_top_k_top_p(logits, 1.0, 0, 1.0, gen(1)).shape == (2,)
    assert sample_top_k_top_p(logits, 1.0, 999, 1.0, gen(1)).shape == (2,)


def test_sample_codebook_shape_and_range():
    logits = torch.randn(3, 16, generator=gen(0))
    codes = sample_codebook(logits, SamplingParams(1.0, 30, 0.9), gen(1))
    assert codes.shape == (3,) and codes.dtype == torch.long
    assert bool(((codes >= 0) & (codes < 16)).all())


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


def test_candidates_exclude_semantic_zero_by_default():
    assert main_candidate_ids(EOS, BEGIN, END) == [EOS] + list(range(BEGIN + 1, END + 1))
    assert main_candidate_ids(EOS, BEGIN, END, exclude_semantic_zero=False) == [EOS] + list(range(BEGIN, END + 1))
    assert main_candidate_ids(eos_token_id=5, semantic_begin_id=100, semantic_end_id=103) == [5, 101, 102, 103]


def test_released_checkpoint_candidates():
    ids = main_candidate_ids(248046, 248099, 252194)
    assert 248046 in ids and 248099 not in ids and ids[-1] == 252194
    assert len(ids) == 252194 - 248099 + 1  # eos replaces code 0


def test_candidate_restriction_never_samples_text_or_code_zero():
    sampler = MainTokenSampler(eos_token_id=EOS, semantic_begin_id=BEGIN, semantic_end_id=END)
    logits = torch.zeros(VOCAB)
    logits[42] = 50.0  # a text token with an overwhelming logit
    logits[BEGIN] = 40.0  # semantic code 0
    for seed in range(50):
        draw = sampler.sample(logits, [], SamplingParams(1.0, 30, 0.9), gen(seed))
        assert draw.token != 42 and draw.token != BEGIN
        assert draw.token == EOS or BEGIN < draw.token <= END
    masked = sampler.masked_logits(logits)
    assert masked[42] == float("-inf") and masked[BEGIN] == float("-inf")
    assert masked[EOS] == 0.0 and masked[END] == 0.0
    assert sampler.candidate_logits(logits.unsqueeze(0)).shape == (1, 1 + (END - BEGIN))


def test_code_zero_can_be_re_enabled():
    sampler = MainTokenSampler(eos_token_id=EOS, semantic_begin_id=BEGIN, semantic_end_id=END, exclude_semantic_zero=False)
    logits = torch.zeros(VOCAB)
    logits[BEGIN] = 50.0
    assert sampler.sample(logits, [], SamplingParams(0.0, 1, 1.0)).token == BEGIN


# ---------------------------------------------------------------------------
# RAS
# ---------------------------------------------------------------------------


def ras_sampler() -> MainTokenSampler:
    return MainTokenSampler(eos_token_id=EOS, semantic_begin_id=BEGIN, semantic_end_id=END)


def test_ras_triggers_on_forced_repetition():
    sampler = ras_sampler()
    x = BEGIN + 7
    logits = torch.zeros(VOCAB)
    logits[x] = 30.0  # first draw (top_k 1) is always x
    first_params = SamplingParams(1.0, 1, 1.0)
    fresh = sampler.sample(logits, [], first_params, gen(0))
    assert fresh.first == x and not fresh.repeated and fresh.token == x
    repeated = sampler.sample(logits, [BEGIN + 1, x, BEGIN + 2], first_params, gen(0))
    assert repeated.first == x and repeated.repeated and repeated.token == repeated.redraw
    # The re-draw (temperature 1 / top_k 30 / top_p 0.95) is a real sample over the
    # candidates: with flat alternatives it picks something other than x at least once.
    # (Three ties, not two: with two the second's inclusive cumulative mass is 1.0 > 0.95
    # and the reference top-p rule drops it.)
    logits[BEGIN + 8] = 30.0
    logits[BEGIN + 9] = 30.0
    tokens = {sampler.sample(logits, [x], first_params, gen(s)).token for s in range(100)}
    assert tokens <= {x, BEGIN + 8, BEGIN + 9} and len(tokens) >= 2


def test_ras_ignores_end_of_speech_repeats():
    sampler = ras_sampler()
    logits = torch.zeros(VOCAB)
    logits[EOS] = 30.0
    draw = sampler.sample(logits, [EOS] * 5, SamplingParams(1.0, 1, 1.0), gen(0))
    assert draw.first == EOS and not draw.repeated and draw.token == EOS


def test_ras_window_is_sixteen_frames():
    sampler = ras_sampler()
    x = BEGIN + 3
    logits = torch.zeros(VOCAB)
    logits[x] = 30.0
    params = SamplingParams(1.0, 1, 1.0)
    filler = [BEGIN + 9] * (RAS_WINDOW - 1)
    assert sampler.sample(logits, [x] + filler, params, gen(0)).repeated  # x is 16 tokens back
    assert not sampler.sample(logits, [x] + filler + [BEGIN + 9], params, gen(0)).repeated  # 17 back
    assert RAS_WINDOW == 16


def test_main_token_sampler_is_seed_deterministic():
    sampler = ras_sampler()
    logits = torch.randn(VOCAB, generator=gen(0))
    a = [sampler.sample(logits, [], generator=gen(11)).token for _ in range(3)]
    b = [sampler.sample(logits, [], generator=gen(11)).token for _ in range(3)]
    assert a == b
    many = {sampler.sample(logits, [], generator=gen(s)).token for s in range(30)}
    assert len(many) > 1


def test_greedy_first_draw_still_samples_redraw():
    sampler = ras_sampler()
    logits = torch.randn(VOCAB, generator=gen(0))
    tied = {BEGIN + 1, BEGIN + 2, BEGIN + 3}
    for t in tied:
        logits[t] = 20.0
    draw = sampler.sample(logits, [], SamplingParams(0.0, 30, 0.9), gen(0))
    best = int(sampler.candidate_logits(logits.unsqueeze(0)).argmax())
    assert draw.first == int(sampler.candidates[best])
    redraws = {sampler.sample(logits, [], SamplingParams(0.0, 30, 0.9), gen(s)).redraw for s in range(50)}
    assert redraws <= tied and len(redraws) >= 2


def test_sample_rejects_batches():
    with pytest.raises(ValueError):
        ras_sampler().sample(torch.zeros(2, VOCAB), [])
