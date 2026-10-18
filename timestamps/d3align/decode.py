"""Alignment logits -> word timestamps. CPU, cheap, no model.

Maths identical to VoxCPM2's decode_runtime.py (reduce_matrix / words_from_matrix),
with Drama 3's frame clock (one decode step = one 46.44 ms codec frame) and
Drama 3's prompt layout (the target text is its own segment, tokenized alone,
prefixed by one ``<|speaker:0|>`` row that ``target_text_columns`` excludes).

Column <-> token bookkeeping (be exact here, this is where the VoxCPM2 port
silently mapped every word to None once):

    segment literal  = with_speaker_tag(text) = "<|speaker:0|> " + text
    segment ids      = tokenizer.encode(literal)           -> [speaker_tag, t1, ..., tn]
    target columns   = segment rows minus the speaker row  -> n columns  <-> t1..tn
    column j         <-> token id ids[j+1], char offsets offsets[j+1] - len("<|speaker:0|> ")

So the first column's token carries the leading space (" Hello"); its offsets
are clamped at 0 so it still overlaps the first word.

Frame clock: decode step s has frame s (the frame sampled in the previous
step; frame 0 comes out of the prefill's last row) as its INPUT row, and its
attention row is what the model reads from the text while emitting frame
s + 1. We time step s as frame s: ``[s * FRAME_MS, (s + 1) * FRAME_MS)``. That
is an off-by-at-most-one-frame convention (46 ms, the design's expected
boundary error); ``frame_offset`` shifts it if listening says so.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import FRAME_MS
from .dp import dtw_path
from .prompt_layout import DRAMA3_TOKENS, speaker_tag_match, with_speaker_tag


def reduce_matrix(raw_logits: np.ndarray, target_cols: Sequence[int] | np.ndarray | None) -> np.ndarray:
    """``[steps, n_sel, T]`` raw logits -> ``[steps, n_text]`` fused, DTW-ready matrix.

    Per selected head: slice to the target columns, softmax, z-score over time
    (Whisper's normalisation, so one confident head cannot dominate), then mean
    over heads and shift so the minimum is 0.
    """
    M = np.asarray(raw_logits, dtype=np.float64)
    if M.ndim != 3 or M.shape[0] == 0 or M.shape[1] == 0:
        return np.zeros((0, 0))
    if target_cols is not None:
        cols = np.asarray(list(target_cols), dtype=np.int64)
        M = M[:, :, cols]
    if M.shape[2] == 0:
        return np.zeros((0, 0))
    out = []
    for h in range(M.shape[1]):
        A = M[:, h, :]
        A = A - A.max(axis=1, keepdims=True)
        A = np.exp(A)
        A = A / np.clip(A.sum(axis=1, keepdims=True), 1e-9, None)
        A = (A - A.mean(0, keepdims=True)) / (A.std(0, keepdims=True) + 1e-8)
        out.append(A)
    F = np.mean(out, axis=0)
    return F - F.min()


def split_words(text: str) -> list[tuple[int, int, str]]:
    """Whitespace words as ``(char_start, char_end, word)``."""
    out, cur, start = [], "", None
    for i, ch in enumerate(text):
        if ch.isspace():
            if cur:
                out.append((start, i, cur))
                cur, start = "", None
        else:
            if not cur:
                start = i
            cur += ch
    if cur:
        out.append((start, len(text), cur))
    return out


def target_token_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids and char offsets (relative to ``text``) of the target-text COLUMNS.

    Mirrors ``target_text_columns``: the segment literal is tokenized on its
    own and the leading speaker-tag token is dropped. The returned lists have
    exactly ``len(target_text_columns(tokenizer, text, refs))`` entries (the
    reference segments never change the target segment's tokens).
    """
    literal = with_speaker_tag(text)
    m = speaker_tag_match(literal)
    # chars of the literal before the user's own text (tag + one space when we added it)
    shift = len(literal) - len(text) if literal != text else 0
    enc = tokenizer(literal, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offs = [tuple(o) for o in enc["offset_mapping"]]
    plain = list(tokenizer.encode(literal, add_special_tokens=False))
    if plain != ids:
        raise RuntimeError("tokenizer(...) and tokenizer.encode(...) disagree on the target segment")
    if ids and ids[0] in DRAMA3_TOKENS.speaker_token_ids:
        ids, offs = ids[1:], offs[1:]
    elif m is not None:
        # a user-supplied tag that tokenized into several pieces: drop all tokens inside it
        keep = [i for i, (a, b) in enumerate(offs) if b > m.end()]
        ids = [ids[i] for i in keep]
        offs = [offs[i] for i in keep]
    rel = [(max(0, a - shift), max(0, b - shift)) for a, b in offs]
    return ids, rel


def words_from_path(
    path: np.ndarray,
    text: str,
    tokenizer_offsets: Sequence[tuple[int, int]],
    frame_ms: float = FRAME_MS,
    *,
    frame_offset: int = 0,
) -> list[dict[str, Any]]:
    """DTW path (text column per decode step) -> ``[{word, start_ms, end_ms}]``.

    ``tokenizer_offsets[j]`` are the char offsets, relative to ``text``, of the
    token in target column ``j`` (from ``target_token_offsets``). Words the
    path never visits are interpolated between their neighbours and flagged
    ``interpolated: True``.
    """
    path = np.asarray(path, dtype=np.int64)
    n_cols = len(tokenizer_offsets)
    words: list[dict[str, Any]] = []
    for ws, we, w in split_words(text):
        cols = {j for j, (a, b) in enumerate(tokenizer_offsets) if j < n_cols and a < we and b > ws}
        steps = np.nonzero(np.isin(path, list(cols)))[0] if cols else np.zeros(0, dtype=np.int64)
        if steps.size == 0:
            words.append({"word": w, "start_ms": None, "end_ms": None, "columns": sorted(cols)})
            continue
        s0, s1 = int(steps.min()) + frame_offset, int(steps.max()) + 1 + frame_offset
        words.append(
            {
                "word": w,
                "start_ms": round(max(0, s0) * frame_ms, 1),
                "end_ms": round(max(0, s1) * frame_ms, 1),
                "columns": sorted(cols),
            }
        )
    return _fill_gaps(words, len(path), frame_ms)


def _fill_gaps(words: list[dict[str, Any]], n_steps: int, frame_ms: float) -> list[dict[str, Any]]:
    timed = [i for i, w in enumerate(words) if w["start_ms"] is not None]
    if not timed:
        return words
    total = n_steps * frame_ms
    for i, w in enumerate(words):
        if w["start_ms"] is not None:
            continue
        prev = max((j for j in timed if j < i), default=None)
        nxt = min((j for j in timed if j > i), default=None)
        lo = words[prev]["end_ms"] if prev is not None else 0.0
        hi = words[nxt]["start_ms"] if nxt is not None else total
        if hi < lo:
            hi = lo
        w["start_ms"], w["end_ms"] = round(lo, 1), round(hi, 1)
        w["interpolated"] = True
    return words


def decode_words(
    raw_logits: np.ndarray,
    target_cols: Sequence[int] | np.ndarray,
    text: str,
    tokenizer: Any,
    *,
    frame_ms: float = FRAME_MS,
    frame_offset: int = 0,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """One-call decode: ``[steps, n_sel, T]`` raw logits -> (words, path, fused matrix)."""
    A = reduce_matrix(raw_logits, target_cols)
    if A.size == 0:
        return [], np.zeros(0, dtype=np.int32), A
    _, offs = target_token_offsets(tokenizer, text)
    if len(offs) != A.shape[1]:
        raise ValueError(f"{len(offs)} target tokens but {A.shape[1]} target columns; text/layout mismatch")
    path = dtw_path(A)
    return words_from_path(path, text, offs, frame_ms, frame_offset=frame_offset), path, A


__all__ = [
    "decode_words",
    "reduce_matrix",
    "split_words",
    "target_token_offsets",
    "words_from_path",
]
