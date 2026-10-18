# SPDX-License-Identifier: Apache-2.0
"""Codebook tower: cache consistency, parameter names, embedding mix, weight loading."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm_drama3.sampling import SamplingParams  # noqa: E402
from nanovllm_drama3.tower import TOWER_PARAM_PREFIXES, CodebookTower, TowerConfig  # noqa: E402

SEMANTIC_BEGIN = 100  # tiny vocabulary: semantic ids 100..115, text ids below 100
PARAM_NAMES_FILE = Path(__file__).with_name("test_tower_param_names.txt")


@pytest.fixture
def config() -> TowerConfig:
    return TowerConfig.tiny()


@pytest.fixture
def tower(config: TowerConfig) -> CodebookTower:
    torch.manual_seed(0)
    model = CodebookTower(config, semantic_begin_id=SEMANTIC_BEGIN)
    # Random gains so the zero-centered norms are not identity-scaled.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("enorm.weight") or name.endswith("hnorm.weight"):
                param.normal_(0.0, 0.3)
    return model.eval()


def make_frames(config: TowerConfig, text_rows: int, audio_rows: int, seed: int = 1) -> torch.Tensor:
    """[text_rows + audio_rows, 1 + Q] prompt-like frames: text then reference audio."""
    g = torch.Generator().manual_seed(seed)
    total = text_rows + audio_rows
    frames = torch.zeros(total, 1 + config.num_codebooks, dtype=torch.long)
    frames[:text_rows, 0] = torch.randint(0, SEMANTIC_BEGIN, (text_rows,), generator=g)
    codes = torch.randint(0, config.vocab_size, (audio_rows, config.num_codebooks), generator=g)
    frames[text_rows:, 1:] = codes
    frames[text_rows:, 0] = SEMANTIC_BEGIN + codes[:, 0]
    return frames


# ---------------------------------------------------------------------------
# (b) parameter names
# ---------------------------------------------------------------------------


def test_parameter_names_match_checkpoint_index(tower: CodebookTower):
    expected = set(PARAM_NAMES_FILE.read_text().split())
    assert len(expected) == 192
    actual = {name for name, _ in tower.named_parameters()}
    assert actual == expected
    assert all(name.startswith(TOWER_PARAM_PREFIXES) for name in actual)


def test_parameter_shapes_for_released_config():
    """Shapes of the real tower (meta device, no memory) follow the checkpoint layout."""
    config = TowerConfig.drama3()
    with torch.device("meta"):
        tower = CodebookTower(config, semantic_begin_id=248099)
    shapes = {name: tuple(p.shape) for name, p in tower.named_parameters()}
    assert shapes["audio_decoder.codebook_embeddings.weight"] == (10 * 4096, 2048)
    assert shapes["audio_decoder_mtp_shared.layers.0.semantic_embedding.weight"] == (4096, 2048)
    assert shapes["audio_decoder_mtp.layers.8.codebook_embedding.weight"] == (4096, 2048)
    assert shapes["audio_decoder_mtp.heads.8.weight"] == (4096, 2048)
    # [query | gate | key | value] rows: 2 * 16*256 + 2 * 2*256
    assert shapes["audio_decoder_mtp.layers.0.transformer_layer.attention.wqkv.weight"] == (2 * 4096 + 2 * 512, 2048)
    assert shapes["audio_decoder_mtp.layers.0.eh_proj.weight"] == (2048, 4096)
    assert tower.num_layers == 13


# ---------------------------------------------------------------------------
# (a) cache consistency
# ---------------------------------------------------------------------------


def test_incremental_teacher_forcing_matches_full_forward(tower: CodebookTower, config: TowerConfig):
    """Running the prompt row by row through the KV caches gives the same hidden
    states and head logits as one teacher-forced pass over all rows."""
    frames = make_frames(config, text_rows=4, audio_rows=6)
    T = frames.shape[0]
    torch.manual_seed(2)
    hidden = torch.randn(T, config.dim)

    full_cache = tower.new_cache()
    full = tower.prefill(hidden, frames, full_cache, return_head_logits=True)
    assert full is not None and full.hidden.shape == (T - 1, config.dim)
    assert len(full.head_logits) == config.num_codebooks - 1
    assert len(full_cache) == T - 1 and full_cache.next_position == T - 1

    inc_cache = tower.new_cache()
    # First three rows in one chunk, then one row at a time (chunked extend).
    hiddens, logits = [], [[] for _ in range(config.num_codebooks - 1)]
    chunks = [(0, 3)] + [(p, p + 1) for p in range(3, T - 1)]
    for start, end in chunks:
        inputs, targets = frames[start:end], frames[start + 1 : end + 1]
        out = tower.run(
            hidden[start:end],
            input_codes=inputs[:, 1:],
            input_is_semantic=tower.is_semantic(inputs[:, 0]),
            next_main_tokens=targets[:, 0],
            next_codes=targets[:, 1:],
            cache=inc_cache,
            return_head_logits=True,
        )
        hiddens.append(out.hidden)
        for r, l in enumerate(out.head_logits):
            logits[r].append(l)
    inc_hidden = torch.cat(hiddens)
    assert inc_hidden.shape == full.hidden.shape
    torch.testing.assert_close(inc_hidden, full.hidden, rtol=1e-5, atol=1e-5)
    for r in range(config.num_codebooks - 1):
        torch.testing.assert_close(torch.cat(logits[r]), full.head_logits[r], rtol=1e-5, atol=1e-5)
    for layer_full, layer_inc in zip(full_cache.layers, inc_cache.layers):
        torch.testing.assert_close(layer_inc.keys, layer_full.keys, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(layer_inc.values, layer_full.values, rtol=1e-5, atol=1e-5)


def test_greedy_steps_match_full_teacher_forced_forward(tower: CodebookTower, config: TowerConfig):
    """Greedy ``step`` after ``prefill`` produces frames whose codes equal the argmax
    of a fresh full teacher-forced pass over prompt + generated frames (the step
    path and the batched path see identical caches)."""
    prompt = make_frames(config, text_rows=3, audio_rows=4)
    T = prompt.shape[0]
    steps = 5
    torch.manual_seed(3)
    hidden = torch.randn(T + steps, config.dim)
    greedy = SamplingParams(temperature=0.0, top_k=0, top_p=1.0)

    cache = tower.new_cache()
    tower.prefill(hidden[:T], prompt, cache)
    generated = []
    main_tokens = SEMANTIC_BEGIN + torch.randint(0, config.vocab_size, (steps,), generator=torch.Generator().manual_seed(4))
    for k in range(steps):
        # Step k consumes the hidden state of row T-1+k (the row whose logits gave main token k).
        frame = tower.step(hidden[T - 1 + k], int(main_tokens[k]), cache, sampling=greedy)
        assert frame.shape == (1 + config.num_codebooks,)
        assert int(frame[0]) == int(main_tokens[k])
        assert int(frame[1]) == int(main_tokens[k]) - SEMANTIC_BEGIN
        generated.append(frame)
    assert cache.next_position == T - 1 + steps
    assert len(cache.generated) == steps

    all_frames = torch.cat((prompt, torch.stack(generated)))
    fresh = tower.new_cache()
    full = tower.prefill(hidden[: T + steps - 1 + 1][: all_frames.shape[0]], all_frames, fresh, return_head_logits=True)
    # Row p predicts frame p+1; generated frame k sits at row T+k and was predicted at row T-1+k.
    for k in range(steps):
        row = T - 1 + k
        for r in range(config.num_codebooks - 1):
            predicted = int(full.head_logits[r][row].argmax())
            assert predicted == int(generated[k][2 + r]), (k, r)


def test_step_is_seed_deterministic(tower: CodebookTower, config: TowerConfig):
    prompt = make_frames(config, text_rows=2, audio_rows=3)
    hidden = torch.randn(prompt.shape[0] + 3, config.dim, generator=torch.Generator().manual_seed(5))

    def run(seed: int) -> list[list[int]]:
        cache = tower.new_cache()
        tower.prefill(hidden[: prompt.shape[0]], prompt, cache)
        g = torch.Generator().manual_seed(seed)
        return [tower.step(hidden[prompt.shape[0] - 1 + k], SEMANTIC_BEGIN + 3 + k, cache, generator=g).tolist() for k in range(3)]

    assert run(7) == run(7)
    # The default sampling is stochastic; a different seed should differ somewhere over 27 codes.
    assert run(7) != run(8)


def test_step_for_end_of_speech_has_zero_codes(tower: CodebookTower, config: TowerConfig):
    prompt = make_frames(config, text_rows=2, audio_rows=2)
    cache = tower.new_cache()
    tower.prefill(torch.randn(prompt.shape[0], config.dim), prompt, cache)
    eos = 7  # any non-semantic id
    frame = tower.step(torch.randn(config.dim), eos, cache)
    assert frame.tolist() == [eos] + [0] * config.num_codebooks


def test_step_requires_prefill(tower: CodebookTower, config: TowerConfig):
    with pytest.raises(RuntimeError):
        tower.step(torch.randn(config.dim), SEMANTIC_BEGIN, tower.new_cache())


def test_prefill_single_row_only_stores_input_frame(tower: CodebookTower, config: TowerConfig):
    frames = make_frames(config, text_rows=1, audio_rows=0)
    cache = tower.new_cache()
    assert tower.prefill(torch.randn(1, config.dim), frames, cache) is None
    assert len(cache) == 0 and cache.next_position == 0
    assert torch.equal(cache.input_frame, frames[0])


# ---------------------------------------------------------------------------
# backbone embedding mix and weight loading
# ---------------------------------------------------------------------------


def test_backbone_embedding_mixes_codebook_one_for_semantic_rows(tower: CodebookTower, config: TowerConfig):
    frames = make_frames(config, text_rows=2, audio_rows=3)
    token_embeds = torch.randn(frames.shape[0], config.dim)
    mixed = tower.backbone_input_embedding(token_embeds, frames)
    torch.testing.assert_close(mixed[:2], token_embeds[:2])  # text rows pass through
    table = tower.audio_decoder.codebook_embeddings.weight
    for i in range(2, 5):
        code1 = int(frames[i, 2])  # code 1 lives in column 2; codebook 1 rows are offset by vocab_size
        expected = (token_embeds[i] + table[config.vocab_size + code1]) / (2.0**0.5)
        torch.testing.assert_close(mixed[i], expected)


def test_load_state_roundtrip(tower: CodebookTower, config: TowerConfig):
    weights = {k: v.clone() for k, v in tower.state_dict().items()}
    assert all(k.startswith(TOWER_PARAM_PREFIXES) for k in weights)
    torch.manual_seed(99)
    other = CodebookTower(config, semantic_begin_id=SEMANTIC_BEGIN).eval()
    loaded = other.load_state({**weights, "model.layers.0.ignored": torch.zeros(1)})
    assert loaded == set(weights)
    frames = make_frames(config, text_rows=2, audio_rows=3)
    hidden = torch.randn(frames.shape[0], config.dim)
    a = tower.prefill(hidden, frames, tower.new_cache(), return_head_logits=True)
    b = other.prefill(hidden, frames, other.new_cache(), return_head_logits=True)
    torch.testing.assert_close(a.hidden, b.hidden)
    with pytest.raises(ValueError):
        other.load_state({"audio_decoder.codebook_embeddings.weight": torch.zeros(1, 1)})
    with pytest.raises(ValueError):
        other.load_state({"audio_decoder.not_a_param": torch.zeros(1)})
