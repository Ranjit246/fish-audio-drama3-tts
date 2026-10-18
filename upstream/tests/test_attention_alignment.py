# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Attention-alignment logits -> word timestamps (``vllm_omni.utils.attention_alignment``).

Synthetic matrices with a planted reading path, the literal reference DP, the
incremental decoder, and the Phase 0 offline decode of one Drama 3 utterance
(reproduced with the real tokenizer when it is cached).
"""

import numpy as np
import pytest

from vllm_omni.utils import attention_alignment as aa

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

FRAME = aa.FRAME_MS


def _dtw_reference(a: np.ndarray) -> np.ndarray:
    """Literal port of the monotonic path search the kernels implement."""
    n_steps, n_text = a.shape
    dp = np.full((n_steps, n_text), aa.NEG)
    back = np.zeros((n_steps, n_text), dtype=np.int32)
    dp[0, : min(n_text, 3)] = a[0, : min(n_text, 3)]
    for i in range(1, n_steps):
        run, arg = aa.NEG, 0
        for j in range(n_text):
            if dp[i - 1, j] > run:
                run, arg = dp[i - 1, j], j
            if run > aa.NEG:
                dp[i, j] = run + a[i, j]
                back[i, j] = arg
    path = np.zeros(n_steps, dtype=np.int32)
    path[-1] = int(dp[-1].argmax())
    for i in range(n_steps - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def _planted(n_steps: int, n_text: int, heads: int = 3, seed: int = 0, strength: float = 6.0):
    """Raw logits whose heads read the text in a staircase; returns (logits, planted path)."""
    rng = np.random.default_rng(seed)
    path = np.minimum((np.arange(n_steps) * n_text) // n_steps, n_text - 1)
    logits = rng.normal(0, 1, (n_steps, heads, n_text))
    logits[np.arange(n_steps), :, path] += strength
    return logits.astype(np.float32), path


def _one_token_per_word(text: str) -> list[tuple[int, int]]:
    return [(start, end) for start, end, _ in aa.split_words(text)]


def test_frame_clock():
    assert FRAME == pytest.approx(46.44, abs=0.005)


@pytest.mark.parametrize("seed", range(4))
def test_dtw_backends_match_the_reference(seed):
    rng = np.random.default_rng(seed)
    # Quantised values make ties, where the back-pointer convention matters.
    a = np.round(rng.normal(size=(30, 7)) * 2) / 2
    reference = _dtw_reference(a)
    assert np.array_equal(aa.dtw_path(a, backend="numpy"), reference)
    if aa.HAVE_NUMBA:
        assert np.array_equal(aa.dtw_path(a, backend="numba"), reference)
    assert np.array_equal(aa.dtw_path(a), reference)


def test_reduce_then_dtw_recovers_a_planted_path_and_monotonic_spans():
    text = "one two three four five six seven eight"
    logits, planted = _planted(80, 8)
    fused = aa.reduce_matrix(logits)
    assert fused.shape == (80, 8) and fused.min() == 0
    path = aa.dtw_path(fused)
    assert np.array_equal(path, planted)
    # The offline (Phase 0) variant with Whisper's time normalisation agrees on a whole utterance.
    assert np.array_equal(aa.dtw_path(aa.reduce_matrix(logits, time_norm=True)), planted)
    # A prefix's matrix is the prefix of the matrix, up to the shift to a zero minimum
    # (not so with time normalisation, whose prefix path wanders off the planted one).
    prefix = aa.reduce_matrix(logits[:7])
    assert np.allclose(prefix - prefix.min(), fused[:7] - fused[:7].min())
    assert np.array_equal(aa.dtw_path(prefix), planted[:7])
    assert not np.array_equal(aa.dtw_path(aa.reduce_matrix(logits[:7], time_norm=True)), planted[:7])
    words = aa.words_from_path(path, text, _one_token_per_word(text))
    assert [w["word"] for w in words] == text.split()
    starts = [w["start_ms"] for w in words]
    assert starts == sorted(starts) and starts[0] == 0
    for current, following in zip(words, words[1:]):
        assert current["end_ms"] == following["start_ms"]  # contiguous, disjoint
    assert words[-1]["end_ms"] == pytest.approx(80 * FRAME)
    assert not any(w.get("interpolated") for w in words)


def test_words_from_path_interpolates_skipped_words_and_caps_overlaps():
    text = "a b c"
    # Token 1 ("b") is never visited; the path jumps from a to c.
    path = np.array([0, 0, 2, 2, 2])
    words = aa.words_from_path(path, text, _one_token_per_word(text))
    assert words[0]["start_ms"] == 0 and words[0]["end_ms"] == pytest.approx(2 * FRAME)
    assert words[1]["interpolated"] and words[1]["start_ms"] == words[0]["end_ms"] == words[1]["end_ms"]
    assert words[2]["start_ms"] == pytest.approx(2 * FRAME) and words[2]["end_ms"] == pytest.approx(5 * FRAME)
    # Two words sharing one token keep the same span; a word's end is capped
    # only at a later start.
    text = "cd ef gh"
    offsets = [(0, 5), (6, 8)]
    words = aa.words_from_path(np.array([0, 0, 0, 0, 1]), text, offsets)
    assert [(w["start_ms"], w["end_ms"]) for w in words[:2]] == [(0, pytest.approx(4 * FRAME))] * 2
    assert words[2]["start_ms"] == pytest.approx(4 * FRAME)
    text = "a b"
    words = aa.words_from_path(np.array([0, 1, 0, 1]), text, [(0, 1), (2, 3)])  # the path cannot go back, but a word may
    assert words[0]["end_ms"] == words[1]["start_ms"] == pytest.approx(FRAME)


def test_tail_silence_is_estimated_from_the_other_words():
    words = [
        {"word": "Hello", "start_ms": 0, "end_ms": 300},
        {"word": "there", "start_ms": 300, "end_ms": 600},
        {"word": "friend.", "start_ms": 600, "end_ms": 1800},
    ]
    # 60 ms per character spoken; "friend." is 7 characters -> 420 ms of speech, the rest silence.
    assert aa.estimate_tail_silence_ms(words) == pytest.approx(1800 - 600 - 420)
    assert aa.estimate_tail_silence_ms(words[:2]) == 0.0


def test_decoder_streams_final_words_monotonically_and_matches_the_one_shot_decode():
    text = "one two three four five six seven eight"
    logits, _ = _planted(80, 8, seed=3)
    offsets = _one_token_per_word(text)

    one_shot = aa.WordAlignmentDecoder(text, offsets)
    one_shot.push(logits)
    expected = one_shot.finish()

    decoder = aa.WordAlignmentDecoder(text, offsets)
    finals: dict[str, tuple[int, int]] = {}
    events = []
    for start in range(0, 80, 7):
        decoder.push(logits[start : start + 7])
        words = decoder.update()
        events.append(words)
        # Final words first, then the provisional ones; a final word is emitted once.
        flags = [w["final"] for w in words]
        assert flags == sorted(flags, reverse=True)
        for w in words:
            if w["final"]:
                assert w["word"] not in finals
                finals[w["word"]] = (w["start_ms"], w["end_ms"])
    assert len(finals) >= 5  # most words settled before the end
    last = decoder.finish()
    assert [w["word"] for w in last] == text.split() and all(w["final"] for w in last)
    assert "tail_silence_ms" in last[-1] and "tail_silence_ms" not in last[0]
    # Frozen words keep their streamed spans; the whole list equals the one-shot decode.
    for w in last:
        if w["word"] in finals:
            assert (w["start_ms"], w["end_ms"]) == finals[w["word"]]
    assert [(w["word"], w["start_ms"], w["end_ms"]) for w in last] == [
        (w["word"], w["start_ms"], w["end_ms"]) for w in expected
    ]
    assert decoder.frames == 80
    event = aa.WordAlignmentDecoder.event(last)
    assert event["type"] == "speech.alignment.delta" and event["resolution_ms"] == 46.44


def test_decoder_ignores_empty_rows_and_fails_softly_on_a_text_mismatch():
    import torch

    decoder = aa.WordAlignmentDecoder("a b", [(0, 1), (2, 3)])
    decoder.push(None)
    decoder.push(torch.zeros((0, 1, 1), dtype=torch.float16))  # another request's placeholder
    decoder.push([torch.zeros((2, 3, 2), dtype=torch.float16)])
    assert decoder.frames == 2 and decoder.update()
    bad = aa.WordAlignmentDecoder("a b", [(0, 1), (2, 3)])
    bad.push(np.zeros((4, 3, 5)))
    assert bad.failed and bad.update() == [] and bad.finish() == []


def test_header_timestamps_are_integers():
    words = [{"word": "a", "start_ms": 0, "end_ms": 46, "final": True, "tail_silence_ms": 10}]
    assert aa.header_timestamps(words) == [{"word": "a", "start_ms": 0, "end_ms": 46, "tail_silence_ms": 10}]


# ---------------------------------------------------------------------------
# Phase 0 offline decode of one Drama 3 utterance (timestamps/results/example_req_10.words.json)
# ---------------------------------------------------------------------------

PHASE0_TEXT = "It is situated across the northeastern rim of the older and larger crater Danjon."
# Text column per decode step from the Phase 0 DTW (126 steps, 16 target columns).
PHASE0_PATH = (
    [0] + [1] * 3 + [2] * 7 + [3] * 11 + [4] * 8 + [5] * 10 + [6] * 12 + [7] * 8 + [8] * 7
    + [9] * 7 + [10] * 7 + [11] * 8 + [12] * 9 + [13] * 4 + [14] * 7 + [15] * 17
)
PHASE0_WORDS = [
    ("It", 0.0, 46.4), ("is", 46.4, 185.8), ("situated", 185.8, 510.8), ("across", 510.8, 1021.7),
    ("the", 1021.7, 1393.2), ("northeastern", 1393.2, 1857.6), ("rim", 1857.6, 2414.9), ("of", 2414.9, 2786.4),
    ("the", 2786.4, 3111.5), ("older", 3111.5, 3436.6), ("and", 3436.6, 3761.6), ("larger", 3761.6, 4133.2),
    ("crater", 4133.2, 4551.1), ("Danjon.", 4551.1, 5851.4),
]  # fmt: skip


def _phase0_path() -> np.ndarray:
    path = np.asarray(PHASE0_PATH)
    assert len(path) == 126 and path.max() == 15
    return path


def test_phase0_example_words_from_the_published_path_with_one_token_per_column():
    """The Phase 0 spans follow from its path alone, given the words' columns."""
    # Column -> word as Phase 0 reported it: one column per word, three for "Danjon.".
    columns = [[j] for j in range(13)] + [[13, 14, 15]]
    words = aa.split_words(PHASE0_TEXT)
    offsets = [None] * 16
    for (start, end, _), cols in zip(words, columns):
        for col in cols:
            offsets[col] = (start, end)
    result = aa.words_from_path(_phase0_path(), PHASE0_TEXT, offsets)
    assert [(w["word"], round(w["start_ms"], 1), round(w["end_ms"], 1)) for w in result] == PHASE0_WORDS



def _load_drama3_tokenizer():
    """The cached Drama 3 tokenizer (its snapshot directory, so no config.json or network is needed)."""
    import glob
    import os

    from transformers import AutoTokenizer

    root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    snapshots = sorted(glob.glob(os.path.join(root, "models--fishaudio--drama-3", "snapshots", "*", "tokenizer.json")))
    if snapshots:
        return AutoTokenizer.from_pretrained(os.path.dirname(snapshots[-1]))
    return AutoTokenizer.from_pretrained("fishaudio/drama-3", local_files_only=True)


@pytest.fixture(scope="module")
def drama3_tokenizer():
    try:
        return _load_drama3_tokenizer()
    except Exception as exc:  # pragma: no cover - offline without the cached tokenizer
        pytest.skip(f"Drama 3 tokenizer not cached: {exc}")


def test_phase0_example_is_reproduced_with_the_real_tokenizer(drama3_tokenizer):
    from vllm_omni.model_executor.models.fish_omni.prompt import target_text_token_offsets

    offsets = target_text_token_offsets(drama3_tokenizer, PHASE0_TEXT)
    assert len(offsets) == 16  # the 16 target columns of the Phase 0 request
    result = aa.words_from_path(_phase0_path(), PHASE0_TEXT, offsets)
    assert [(w["word"], round(w["start_ms"], 1), round(w["end_ms"], 1)) for w in result] == PHASE0_WORDS
    decoder = aa.WordAlignmentDecoder(PHASE0_TEXT, offsets)
    # Planted logits that reproduce the Phase 0 path exactly.
    logits = np.zeros((126, 6, 16), dtype=np.float32)
    logits[np.arange(126), :, _phase0_path()] = 8.0
    decoder.push(logits)
    final = decoder.finish()
    # Integer milliseconds against Phase 0's one-decimal values: within 1 ms.
    assert [w["word"] for w in final] == [word for word, _, _ in PHASE0_WORDS]
    for got, (_, start, end) in zip(final, PHASE0_WORDS):
        assert abs(got["start_ms"] - start) <= 1 and abs(got["end_ms"] - end) <= 1
    # "Danjon." spans 1.3 s of a 5.85 s clip (the audio was 5.80 s): most of it is trailing silence.
    assert final[-1]["tail_silence_ms"] > 500


@pytest.mark.parametrize("window", [1, 3, 7])
def test_windowed_pushes_give_the_one_shot_spans_and_never_move_a_final_word(window):
    """The frame base advances by exactly the rows pushed, whatever the window size."""
    text = "one two three four five six seven eight"
    logits, _ = _planted(80, 8, seed=5)
    offsets = _one_token_per_word(text)
    one_shot = aa.WordAlignmentDecoder(text, offsets)
    one_shot.push(logits)
    expected = [(w["word"], w["start_ms"], w["end_ms"]) for w in one_shot.finish()]

    decoder = aa.WordAlignmentDecoder(text, offsets)
    frozen: dict[str, tuple[int, int]] = {}
    for start in range(0, 80, window):
        decoder.push(logits[start : start + window])
        assert decoder.frames == min(start + window, 80)
        for w in decoder.update():
            if w["final"]:
                frozen[w["word"]] = (w["start_ms"], w["end_ms"])
            elif w["word"] in frozen:
                raise AssertionError(f"{w['word']} moved after being final")
    final = decoder.finish()
    assert [(w["word"], w["start_ms"], w["end_ms"]) for w in final] == expected
    assert all((w["start_ms"], w["end_ms"]) == frozen[w["word"]] for w in final if w["word"] in frozen)
    assert final[-1]["end_ms"] == round(80 * FRAME)
