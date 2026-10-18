# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Word timestamps from attention-alignment logits (API server side).

Stage 0 of a model such as Fish Audio Drama 3 (``fish_omni``) captures, per
generated codec frame, the raw attention logits of a few alignment heads over
the request's text tokens (``[frames, heads, text]``; see
``fish_omni/alignment.py``). This module turns them into word spans:

1. ``reduce_matrix``: per head, softmax over the text columns, z-score over
   time (so one confident head cannot dominate), mean over heads, minus min.
2. ``dtw_path``: the monotonic path of maximal attention through the fused
   ``[frames, text]`` matrix (it may stay on a token or move forward, and
   may start on any of the first three tokens, for leading silence).
3. ``words_from_path``: token per frame -> span per whitespace word, through
   the tokenizer's character offsets of the text tokens.

``WordAlignmentDecoder`` runs this incrementally as frames arrive: a word is
``final`` once the path has moved past its last token; its span is then
frozen. The last word absorbs the trailing silence (the model keeps reading
it until end of speech), so the final list adds an estimate of that
silence, ``tail_silence_ms``, from the utterance's own speaking rate.

Frame clock: frame ``k`` of the codec covers ``[k, k + 1) * frame_ms``; the
row scored at the step that sampled frame ``k`` is frame ``k``'s row. CPU
only, numpy; the O(frames x text) path search uses numba when importable.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np

# Drama 3 / DAC: 44.1 kHz, 2048 samples per frame, one frame per decode step.
FRAME_MS = 1000.0 * 2048 / 44100
NEG = -1e9
EVENT_TYPE = "speech.alignment.delta"

try:  # numba is optional; the numpy path gives the same result
    if os.environ.get("VLLM_OMNI_ALIGNMENT_NO_NUMBA") == "1":
        raise ImportError("disabled")
    import numba as _numba

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised only without numba
    _numba = None
    HAVE_NUMBA = False


# ---------------------------------------------------------------------------
# Matrix reduction and the monotonic path
# ---------------------------------------------------------------------------


def reduce_matrix(raw_logits: np.ndarray, *, time_norm: bool = False) -> np.ndarray:
    """``[frames, heads, text]`` raw logits -> ``[frames, text]`` fused, path-ready matrix.

    Per head: softmax over the text columns, then the mean over heads. A
    frame's row depends on that frame alone, so the matrix of a prefix is the
    prefix of the matrix: what an incremental decoder needs.

    ``time_norm`` adds Whisper's z-score of each column over time before the
    mean (the offline Phase 0 decode). It equalises heads over a whole
    utterance but degenerates on a prefix (a column attended in every frame
    so far scores 0 while unreached columns' noise is amplified), so the
    serving path leaves it off.
    """
    matrix = np.asarray(raw_logits, dtype=np.float64)
    if matrix.ndim != 3 or 0 in matrix.shape:
        return np.zeros((0, 0))
    fused = np.zeros((matrix.shape[0], matrix.shape[2]))
    for head in range(matrix.shape[1]):
        a = matrix[:, head, :]
        a = np.exp(a - a.max(axis=1, keepdims=True))
        a = a / np.clip(a.sum(axis=1, keepdims=True), 1e-9, None)
        if time_norm:
            a = (a - a.mean(0, keepdims=True)) / (a.std(0, keepdims=True) + 1e-8)
        fused += a
    fused /= matrix.shape[1]
    return fused - fused.min()


def _dtw_path_numpy(a: np.ndarray) -> np.ndarray:
    n_steps, n_text = a.shape
    dp = np.full((n_steps, n_text), NEG)
    back = np.zeros((n_steps, n_text), dtype=np.int32)
    k = min(n_text, 3)
    dp[0, :k] = a[0, :k]
    index = np.arange(n_text)
    for i in range(1, n_steps):
        prev = dp[i - 1]
        run = np.maximum.accumulate(prev)
        # The back-pointer is the first column attaining the running maximum.
        prev_run = np.empty_like(run)
        prev_run[0] = NEG
        prev_run[1:] = run[:-1]
        arg = np.maximum.accumulate(np.where(prev > prev_run, index, 0))
        ok = run > NEG
        dp[i] = np.where(ok, run + a[i], NEG)
        back[i] = np.where(ok, arg, 0)
    path = np.zeros(n_steps, dtype=np.int32)
    path[-1] = int(dp[-1].argmax())
    for i in range(n_steps - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


if HAVE_NUMBA:

    @_numba.njit(cache=True, nogil=True)
    def _dtw_path_numba(a):  # pragma: no cover - compiled
        n_steps, n_text = a.shape
        dp = np.full((n_steps, n_text), NEG)
        back = np.zeros((n_steps, n_text), dtype=np.int32)
        k = min(n_text, 3)
        for j in range(k):
            dp[0, j] = a[0, j]
        for i in range(1, n_steps):
            run = NEG
            arg = 0
            for j in range(n_text):
                if dp[i - 1, j] > run:
                    run = dp[i - 1, j]
                    arg = j
                if run > NEG:
                    dp[i, j] = run + a[i, j]
                    back[i, j] = arg
        path = np.zeros(n_steps, dtype=np.int32)
        best = 0
        for j in range(1, n_text):
            if dp[n_steps - 1, j] > dp[n_steps - 1, best]:
                best = j
        path[n_steps - 1] = best
        for i in range(n_steps - 1, 0, -1):
            path[i - 1] = back[i, path[i]]
        return path


def dtw_path(matrix: np.ndarray, *, backend: str | None = None) -> np.ndarray:
    """Text column per frame (int32) along the best monotonic path through ``matrix[frames, text]``."""
    a = np.ascontiguousarray(matrix, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected a [frames, text] matrix, got shape {a.shape}")
    if 0 in a.shape:
        return np.zeros(a.shape[0], dtype=np.int32)
    if backend is None:
        backend = "numba" if HAVE_NUMBA else "numpy"
    if backend == "numba":
        if not HAVE_NUMBA:
            raise RuntimeError("numba is not available")
        return _dtw_path_numba(a)
    if backend == "numpy":
        return _dtw_path_numpy(a)
    raise ValueError(backend)


# ---------------------------------------------------------------------------
# Words
# ---------------------------------------------------------------------------


def split_words(text: str) -> list[tuple[int, int, str]]:
    """Whitespace-delimited words as ``(char_start, char_end, word)``."""
    out: list[tuple[int, int, str]] = []
    start = None
    for i, ch in enumerate(text):
        if ch.isspace():
            if start is not None:
                out.append((start, i, text[start:i]))
                start = None
        elif start is None:
            start = i
    if start is not None:
        out.append((start, len(text), text[start:]))
    return out


def word_columns(text: str, token_offsets: Sequence[tuple[int, int]]) -> list[tuple[str, list[int]]]:
    """Each word with the text columns (tokens) overlapping it."""
    words = []
    for ws, we, word in split_words(text):
        cols = [j for j, (a, b) in enumerate(token_offsets) if a < we and b > ws]
        words.append((word, cols))
    return words


def words_from_path(
    path: np.ndarray,
    text: str,
    token_offsets: Sequence[tuple[int, int]],
    frame_ms: float = FRAME_MS,
) -> list[dict[str, Any]]:
    """Path (text column per frame) -> ``[{word, start_ms, end_ms}]`` in text order.

    A word's span runs from the first to the last frame whose column belongs
    to it; a word the path skips is interpolated between its neighbours
    (``interpolated: True``). Spans are kept disjoint: a word's end is capped
    at the next word's start.
    """
    path = np.asarray(path, dtype=np.int64)
    words: list[dict[str, Any]] = []
    for word, cols in word_columns(text, token_offsets):
        steps = np.nonzero(np.isin(path, cols))[0] if cols else np.zeros(0, dtype=np.int64)
        if steps.size == 0:
            words.append({"word": word, "start_ms": None, "end_ms": None})
            continue
        words.append(
            {
                "word": word,
                "start_ms": float(steps.min()) * frame_ms,
                "end_ms": float(steps.max() + 1) * frame_ms,
            }
        )
    _fill_gaps(words, len(path) * frame_ms)
    for current, following in zip(words, words[1:]):
        if following["start_ms"] > current["start_ms"]:
            current["end_ms"] = min(current["end_ms"], following["start_ms"])
    return words


def _fill_gaps(words: list[dict[str, Any]], total_ms: float) -> None:
    timed = [i for i, w in enumerate(words) if w["start_ms"] is not None]
    if not timed:
        for w in words:
            w["start_ms"], w["end_ms"], w["interpolated"] = 0.0, total_ms, True
        return
    for i, w in enumerate(words):
        if w["start_ms"] is not None:
            continue
        prev = max((j for j in timed if j < i), default=None)
        nxt = min((j for j in timed if j > i), default=None)
        lo = words[prev]["end_ms"] if prev is not None else 0.0
        hi = words[nxt]["start_ms"] if nxt is not None else total_ms
        w["start_ms"], w["end_ms"], w["interpolated"] = lo, max(lo, hi), True


def estimate_tail_silence_ms(words: Sequence[dict[str, Any]]) -> float:
    """Trailing silence absorbed by the last word, from the other words' speaking rate.

    The attention keeps reading the last token until end of speech, so the
    last span includes the pause before it. With at least two other timed
    words, the median duration per character of those words times the last
    word's length is taken as its spoken duration; the rest of its span is
    the estimate. 0 when nothing can be estimated.
    """
    if len(words) < 3:
        return 0.0
    rates = [
        (w["end_ms"] - w["start_ms"]) / max(1, len(w["word"]))
        for w in words[:-1]
        if not w.get("interpolated") and w["end_ms"] > w["start_ms"]
    ]
    if len(rates) < 2:
        return 0.0
    last = words[-1]
    spoken = float(np.median(rates)) * max(1, len(last["word"]))
    return max(0.0, (last["end_ms"] - last["start_ms"]) - spoken)


# ---------------------------------------------------------------------------
# Incremental decoding for a stream
# ---------------------------------------------------------------------------


class WordAlignmentDecoder:
    """Accumulates alignment rows of one request and emits word spans as they settle.

    ``update()`` after new rows returns the words whose state changed: words
    that became ``final`` (the path has moved past their last token and on
    for ``final_lag`` more frames; their spans are frozen from then on) and,
    after them, the words the path has reached whose spans still move
    (``final: false``). ``finish()`` returns every word, all final, the last
    with ``tail_silence_ms``. Milliseconds are integers; the resolution is
    one frame (``frame_ms``).
    """

    #: Frames the path must have moved on past a word before it is final:
    #: the newest frames' assignment can still shift by a frame or two.
    final_lag = 4

    def __init__(self, text: str, token_offsets: Sequence[tuple[int, int]], *, frame_ms: float = FRAME_MS) -> None:
        self.text = text
        self.offsets = [tuple(pair) for pair in token_offsets]
        self.frame_ms = float(frame_ms)
        self.n_text = len(self.offsets)
        self.words = word_columns(text, self.offsets)
        self._rows: list[np.ndarray] = []
        self._frames = 0
        self._frozen: dict[int, tuple[int, int]] = {}
        self._provisional: dict[int, tuple[int, int]] = {}
        self._mismatch = False

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def failed(self) -> bool:
        """The rows did not match the text's token count; no timestamps for this request."""
        return self._mismatch

    def push(self, rows: Any) -> None:
        """Add ``[frames, heads, text]`` rows (numpy, torch or a list of either); empty ones are ignored."""
        if rows is None:
            return
        if isinstance(rows, (list, tuple)):
            for item in rows:
                self.push(item)
            return
        if hasattr(rows, "detach"):
            rows = rows.detach().to("cpu").float().numpy()
        array = np.asarray(rows)
        if array.ndim != 3 or array.shape[0] == 0:
            return
        if array.shape[2] != self.n_text:
            self._mismatch = True
            return
        self._rows.append(np.ascontiguousarray(array, dtype=np.float32))
        self._frames += array.shape[0]

    def _spans(self) -> tuple[list[dict[str, Any]], int] | None:
        if not self._rows or self._mismatch or self.n_text == 0:
            return None
        rows = self._rows[0] if len(self._rows) == 1 else np.concatenate(self._rows)
        if len(self._rows) > 1:
            self._rows = [rows]
        path = dtw_path(reduce_matrix(rows))
        return words_from_path(path, self.text, self.offsets, self.frame_ms), int(path[-1])

    def _entry(self, index: int, span: tuple[int, int], final: bool) -> dict[str, Any]:
        return {"word": self.words[index][0], "start_ms": span[0], "end_ms": span[1], "final": final}

    def update(self) -> list[dict[str, Any]]:
        """Words whose state changed since the last call (see the class docstring)."""
        result = self._spans()
        if result is None:
            return []
        spans, column = result
        # The word the path is on: the last word whose tokens it has reached.
        # Words before it are final once the path is ``final_lag`` frames past
        # them; up to it, spans are provisional while they still move.
        reached = [index for index, (_, cols) in enumerate(self.words) if cols and min(cols) <= column]
        current = max(reached) if reached else 0
        settled_ms = (self._frames - self.final_lag) * self.frame_ms
        final: list[dict[str, Any]] = []
        provisional: list[dict[str, Any]] = []
        for index in range(min(current + 1, len(self.words))):
            if index in self._frozen:
                continue
            span = self._clamped(index, spans)
            if index < current and span[1] <= settled_ms:
                self._frozen[index] = span
                self._provisional.pop(index, None)
                final.append(self._entry(index, span, True))
            elif self._provisional.get(index) != span:
                self._provisional[index] = span
                provisional.append(self._entry(index, span, False))
        return final + provisional

    def _clamped(self, index: int, spans: list[dict[str, Any]]) -> tuple[int, int]:
        start, end = spans[index]["start_ms"], spans[index]["end_ms"]
        if index > 0 and (index - 1) in self._frozen:
            start = max(start, self._frozen[index - 1][1])
        end = max(end, start)
        return int(round(start)), int(round(end))

    def finish(self) -> list[dict[str, Any]]:
        """Every word, final; the last word carries ``tail_silence_ms``."""
        result = self._spans()
        if result is None:
            return []
        spans, _ = result
        out: list[dict[str, Any]] = []
        for index in range(len(self.words)):
            span = self._frozen.get(index) or self._clamped(index, spans)
            self._frozen[index] = span
            entry = self._entry(index, span, True)
            if spans[index].get("interpolated"):
                entry["interpolated"] = True
            out.append(entry)
        if out:
            timed = [
                {"word": w["word"], "start_ms": w["start_ms"], "end_ms": w["end_ms"], "interpolated": w.get("interpolated")}
                for w in out
            ]
            out[-1]["tail_silence_ms"] = int(round(estimate_tail_silence_ms(timed)))
        return out

    @staticmethod
    def event(words: Sequence[dict[str, Any]], frame_ms: float = FRAME_MS) -> dict[str, Any]:
        """The ``speech.alignment.delta`` SSE payload for ``words``."""
        return {"type": EVENT_TYPE, "words": list(words), "resolution_ms": round(frame_ms, 2)}


def header_timestamps(words: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ``X-Word-Timestamps`` form: ``{word, start_ms, end_ms}`` (+ ``tail_silence_ms`` on the last word)."""
    out = []
    for w in words:
        entry = {"word": w["word"], "start_ms": int(w["start_ms"]), "end_ms": int(w["end_ms"])}
        if "tail_silence_ms" in w:
            entry["tail_silence_ms"] = int(w["tail_silence_ms"])
        out.append(entry)
    return out


__all__ = [
    "EVENT_TYPE",
    "FRAME_MS",
    "HAVE_NUMBA",
    "WordAlignmentDecoder",
    "dtw_path",
    "estimate_tail_silence_ms",
    "header_timestamps",
    "reduce_matrix",
    "split_words",
    "word_columns",
    "words_from_path",
]
