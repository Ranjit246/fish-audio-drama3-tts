"""Per-head sweep metrics, identical to VoxCPM2's align_head/align_sweep.py.

Per head, over its ``[audio_step x text_token]`` matrix (rows normalised):

- path_mass    mean attention along the best monotonic path; uniform = 1/n_text
- monotonicity fraction of consecutive steps whose argmax does not move back
- coverage     distinct tokens visited / min(n_steps, n_text)

``score = path_mass * monotonicity * min(2 * coverage, 1)``

Never rank on path_mass alone: a head parked on one token traces a "perfect"
monotonic path. Coverage is what separates alignment from degenerate attention.
"""

from __future__ import annotations

import numpy as np

from .dp import monotonic_path_mass


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis in float64."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= np.clip(x.sum(axis=-1, keepdims=True), 1e-300, None)
    return x


def coverage(A: np.ndarray) -> float:
    if A.size == 0:
        return 0.0
    n_steps, n_text = A.shape
    return len(np.unique(A.argmax(axis=1))) / max(1, min(n_steps, n_text))


def monotonicity(A: np.ndarray) -> float:
    if A.shape[0] < 2:
        return 0.0
    am = A.argmax(axis=1)
    return float((np.diff(am) >= 0).mean())


def composite(path_mass: float, mono: float, cov: float) -> float:
    return float(path_mass * mono * min(cov * 2.0, 1.0))


def head_metrics(A: np.ndarray) -> dict[str, float]:
    """Metrics of one normalised ``[n_steps, n_text]`` attention matrix."""
    A = np.asarray(A, dtype=np.float64)
    A = A / np.clip(A.sum(axis=1, keepdims=True), 1e-9, None)
    pm = monotonic_path_mass(A)
    mo = monotonicity(A)
    co = coverage(A)
    return {"path_mass": pm, "monotonicity": mo, "coverage": co, "score": composite(pm, mo, co)}


def layer_metrics(raw_logits: np.ndarray, target_cols) -> tuple[list[dict[str, float]], np.ndarray]:
    """Metrics of every head of one layer from RAW logits over all prompt columns.

    ``raw_logits``: ``[n_steps, n_heads, T]`` (fp16/fp32), ``target_cols``: the
    target-text columns within ``[0, T)``. Slices to the target columns BEFORE
    the softmax (equivalent to softmax-then-renormalise within the slice).

    Also returns per-head ``target_mass``: the share of the FULL-row softmax
    that lands on the target columns, mean over steps. This is the number the
    VoxCPM2 README says to check before trusting a head: a head whose mass sits
    on the reference transcript / system prompt is not aligning the target text
    even if its renormalised slice looks diagonal.
    """
    M = np.asarray(raw_logits, dtype=np.float64)
    if M.ndim != 3:
        raise ValueError(f"expected [steps, heads, T], got {M.shape}")
    cols = np.asarray(list(target_cols), dtype=np.int64)
    full = softmax_rows(M)  # [steps, heads, T]
    target_mass = full[:, :, cols].sum(axis=-1).mean(axis=0)  # [heads]
    sliced = softmax_rows(M[:, :, cols])  # [steps, heads, n_text]
    out = []
    for h in range(sliced.shape[1]):
        m = head_metrics(sliced[:, h, :])
        m["target_mass"] = float(target_mass[h])
        out.append(m)
    return out, target_mass


__all__ = [
    "composite",
    "coverage",
    "head_metrics",
    "layer_metrics",
    "monotonicity",
    "softmax_rows",
]
