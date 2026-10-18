"""Token/column bookkeeping and word spans with the real Drama 3 tokenizer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from d3align import FRAME_MS
from d3align.decode import decode_words, reduce_matrix, split_words, target_token_offsets, words_from_path
from d3align.dp import dtw_path
from d3align.prompt_layout import Reference, build_prompt_layout, prompt_length, target_text_columns

ROOT = Path(__file__).resolve().parents[1]

TEXTS = [
    "Get the trust fund to the bank early.",
    "We asked over twenty different people, and they all said it was his.",
    "Hello world",
    "  leading spaces and 2 numbers: 3,500 dollars!",
    "<|speaker:0|> Already tagged by the caller.",
]


def test_frame_clock():
    assert FRAME_MS == pytest.approx(46.44, abs=0.005)


@pytest.mark.parametrize("text", TEXTS)
def test_target_offsets_match_target_columns(drama3_tokenizer, text):
    tok = drama3_tokenizer
    ref = Reference("a short reference transcript", torch.zeros((37, 10), dtype=torch.long))
    for refs in ((), (ref,)):
        cols = target_text_columns(tok, text, refs)
        ids, offs = target_token_offsets(tok, text)
        assert len(ids) == len(cols) == len(offs)
        layout = build_prompt_layout(tok, text, refs)
        seg_ids = [int(v) for v in layout.frames[cols.start : cols.stop, 0]]
        assert seg_ids == ids  # column j <-> token ids[j], exactly
        assert prompt_length(tok, text, [r.transcript for r in refs], [int(r.codes.shape[0]) for r in refs]) == len(layout)
    # offsets are relative to the user's text (which may itself carry the tag), clamped at 0,
    # monotone and covering every word
    tagged = text.startswith("<|speaker:0|>")
    assert offs[0][0] == (len("<|speaker:0|>") if tagged else 0)
    assert all(a <= b for a, b in offs) and all(offs[i][1] <= offs[i + 1][1] for i in range(len(offs) - 1))
    for ws, we, w in split_words(text):
        if w.startswith("<|speaker:"):
            continue
        assert any(a < we and b > ws for a, b in offs), (w, offs)


def test_words_from_path_staircase(drama3_tokenizer):
    tok = drama3_tokenizer
    text = "Get the trust fund to the bank early."
    ids, offs = target_token_offsets(tok, text)
    n_text = len(ids)  # 9 tokens: ' Get',' the',' trust',' fund',' to',' the',' bank',' early','.'
    assert n_text == 9
    steps = 45
    path = np.array([min(n_text - 1, (s * n_text) // steps) for s in range(steps)])  # 5 steps per token
    words = words_from_path(path, text, offs, frame_ms=FRAME_MS)
    assert [w["word"] for w in words] == text.split()
    assert words[0]["start_ms"] == 0.0 and words[0]["end_ms"] == pytest.approx(5 * FRAME_MS, abs=0.1)
    assert words[1]["start_ms"] == pytest.approx(5 * FRAME_MS, abs=0.1)
    # "early." spans two tokens (' early' and '.') -> 10 frames, ends at the last frame
    assert words[-1]["columns"] == [7, 8]
    assert words[-1]["start_ms"] == pytest.approx(35 * FRAME_MS, abs=0.1)
    assert words[-1]["end_ms"] == pytest.approx(steps * FRAME_MS, abs=0.1)
    for a, b in zip(words, words[1:]):
        assert a["end_ms"] <= b["start_ms"] + 1e-6
        assert "interpolated" not in a


def test_words_from_path_interpolates_skipped_word(drama3_tokenizer):
    tok = drama3_tokenizer
    text = "one two three"
    _, offs = target_token_offsets(tok, text)
    path = np.array([0, 0, 2, 2, 2])  # token 1 ("two") never visited
    words = words_from_path(path, text, offs, frame_ms=100.0)
    assert words[1]["interpolated"] is True
    assert words[1]["start_ms"] == words[0]["end_ms"] == 200.0
    assert words[1]["end_ms"] == words[2]["start_ms"] == 200.0


def test_reduce_matrix_then_dtw_recovers_planted_alignment(drama3_tokenizer):
    tok = drama3_tokenizer
    text = "Get the trust fund to the bank early."
    cols = target_text_columns(tok, text)
    T = cols.stop + 6
    n_text, steps, nsel = len(cols), 60, 3
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 0.3, size=(steps, nsel, T))
    truth = np.array([min(n_text - 1, (s * n_text) // steps) for s in range(steps)])
    for s in range(steps):
        raw[s, :, cols.start + truth[s]] += 5.0
    raw[:, :, : cols.start] += 8.0  # heavy mass on prefix columns: must be sliced away BEFORE softmax
    A = reduce_matrix(raw, np.arange(cols.start, cols.stop))
    assert A.shape == (steps, n_text) and A.min() == 0.0
    path = dtw_path(A)
    np.testing.assert_array_equal(path, truth)
    words, path2, _ = decode_words(raw, np.arange(cols.start, cols.stop), text, tok)
    np.testing.assert_array_equal(path2, truth)
    assert [w["word"] for w in words] == text.split()
    assert all(w["start_ms"] is not None for w in words)


def test_align_decode_cli(drama3_tokenizer, tmp_path):
    """End-to-end CLI on a synthetic capture whose planted head follows the text."""
    tok = drama3_tokenizer
    text = "Get the trust fund to the bank early."
    cols = target_text_columns(tok, text)
    layout = build_prompt_layout(tok, text)
    T, steps = len(layout), 54
    n_text = len(cols)
    rng = np.random.default_rng(1)
    arrays = {"prompt_len": np.int64(T), "layers": np.array([3, 7]), "steps": np.int64(steps), "counter": np.int64(0)}
    for layer in (3, 7):
        M = rng.normal(0, 0.3, size=(steps, 16, T))
        for s in range(steps):
            M[s, 5, cols.start + min(n_text - 1, (s * n_text) // steps)] += 6.0
        arrays[f"logits_L{layer}"] = M.astype(np.float16)
    cap_dir = tmp_path / "cap"
    cap_dir.mkdir()
    np.savez(cap_dir / "req_0.npz", **arrays)
    req = {"order": 0, "condition": "zeroshot", "utt": "u", "text": text, "ref_text": None, "prompt_len": T,
           "target_cols": [cols.start, cols.stop], "n_text": n_text, "audio_s": steps * 2048 / 44100, "expected_steps": steps}
    (tmp_path / "requests.jsonl").write_text(json.dumps(req) + "\n")
    (tmp_path / "heads.json").write_text(json.dumps({"heads": [{"layer": 7, "head": 5}, {"layer": 3, "head": 5}]}))
    out = tmp_path / "words.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "decode" / "align_decode.py"), "--npz", str(cap_dir / "req_0.npz"),
         "--requests", str(tmp_path / "requests.jsonl"), "--heads", str(tmp_path / "heads.json"), "--out", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    d = json.loads(out.read_text())
    assert [w["word"] for w in d["words"]] == text.split()
    assert d["words"][0]["start_ms"] == 0.0
    assert d["words"][-1]["end_ms"] == pytest.approx(steps * FRAME_MS, abs=0.1)
    assert "Get" in proc.stdout and "early." in proc.stdout
