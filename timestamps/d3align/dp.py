"""Monotonic alignment DP kernels.

Ported 1:1 from the VoxCPM2 aligner
(TTS-duration-predictor/VOXCPM2/nano-vllm/aligner/decode_runtime.py ``dtw_path``
and align_head/align_sweep.py ``monotonic_path_mass``). Same semantics, same
constants: NEG = -1e9, the path may start on any of the first three text
columns (leading silence), and it may stay or jump forward, never back.

Both kernels are O(n_steps x n_text) and were pure-Python loops in the
reference, which under the GIL is a per-request stall inside an API server.
They are compiled with ``numba.njit`` when numba is importable and fall back
to vectorised numpy otherwise. ``tests/test_dp.py`` checks that the numba,
numpy and literal-reference implementations agree bit for bit.
"""

from __future__ import annotations

import os

import numpy as np

NEG = -1e9

try:  # numba is optional
    if os.environ.get("D3ALIGN_NO_NUMBA") == "1":
        raise ImportError("disabled by D3ALIGN_NO_NUMBA")
    import numba as _nb

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised only without numba
    _nb = None
    HAVE_NUMBA = False


# --------------------------------------------------------------------------
# Reference (literal port, pure Python). Kept for tests and as documentation.
# --------------------------------------------------------------------------

def dtw_path_reference(A: np.ndarray) -> np.ndarray:
    """Monotonic max-attention path; returns the text column per audio step."""
    n_steps, n_text = A.shape
    dp = np.full((n_steps, n_text), NEG)
    bk = np.zeros((n_steps, n_text), dtype=np.int32)
    dp[0, : min(n_text, 3)] = A[0, : min(n_text, 3)]
    for i in range(1, n_steps):
        run, arg = NEG, 0
        for j in range(n_text):
            if dp[i - 1, j] > run:
                run, arg = dp[i - 1, j], j
            if run > NEG:
                dp[i, j] = run + A[i, j]
                bk[i, j] = arg
    path = np.zeros(n_steps, dtype=np.int32)
    path[-1] = int(dp[-1].argmax())
    for i in range(n_steps - 1, 0, -1):
        path[i - 1] = bk[i, path[i]]
    return path


def monotonic_path_mass_reference(A: np.ndarray) -> float:
    """Mean attention along the best monotonic path (the papers' OAS)."""
    n_steps, n_text = A.shape
    if n_steps == 0 or n_text == 0:
        return 0.0
    dp = np.full((n_steps, n_text), NEG, dtype=np.float64)
    dp[0, 0] = A[0, 0]
    for j in range(1, min(n_text, 3)):
        dp[0, j] = A[0, j]
    for i in range(1, n_steps):
        run = NEG
        for j in range(n_text):
            if dp[i - 1, j] > run:
                run = dp[i - 1, j]
            if run > NEG:
                dp[i, j] = run + A[i, j]
    return float(dp[n_steps - 1].max() / n_steps)


# --------------------------------------------------------------------------
# numpy fallback (vectorised over text columns, loop over steps)
# --------------------------------------------------------------------------

def _dtw_path_numpy(A: np.ndarray) -> np.ndarray:
    n_steps, n_text = A.shape
    dp = np.full((n_steps, n_text), NEG)
    bk = np.zeros((n_steps, n_text), dtype=np.int32)
    k = min(n_text, 3)
    dp[0, :k] = A[0, :k]
    idx = np.arange(n_text)
    for i in range(1, n_steps):
        prev = dp[i - 1]
        run = np.maximum.accumulate(prev)
        # the reference updates ``arg`` only on a STRICT new maximum, so the
        # back-pointer is the first index attaining the running max
        prev_run = np.empty_like(run)
        prev_run[0] = NEG
        prev_run[1:] = run[:-1]
        is_new = prev > prev_run
        arg = np.maximum.accumulate(np.where(is_new, idx, 0))
        ok = run > NEG
        dp[i] = np.where(ok, run + A[i], NEG)
        bk[i] = np.where(ok, arg, 0)
    path = np.zeros(n_steps, dtype=np.int32)
    path[-1] = int(dp[-1].argmax())
    for i in range(n_steps - 1, 0, -1):
        path[i - 1] = bk[i, path[i]]
    return path


def _monotonic_path_mass_numpy(A: np.ndarray) -> float:
    n_steps, n_text = A.shape
    if n_steps == 0 or n_text == 0:
        return 0.0
    dp = np.full(n_text, NEG, dtype=np.float64)
    k = min(n_text, 3)
    dp[:k] = A[0, :k]
    for i in range(1, n_steps):
        run = np.maximum.accumulate(dp)
        dp = np.where(run > NEG, run + A[i], NEG)
    return float(dp.max() / n_steps)


# --------------------------------------------------------------------------
# numba kernels (same loops as the reference, compiled)
# --------------------------------------------------------------------------

if HAVE_NUMBA:

    @_nb.njit(cache=True, nogil=True)
    def _dtw_path_numba(A):  # pragma: no cover - compiled
        n_steps, n_text = A.shape
        dp = np.full((n_steps, n_text), NEG)
        bk = np.zeros((n_steps, n_text), dtype=np.int32)
        k = min(n_text, 3)
        for j in range(k):
            dp[0, j] = A[0, j]
        for i in range(1, n_steps):
            run = NEG
            arg = 0
            for j in range(n_text):
                if dp[i - 1, j] > run:
                    run = dp[i - 1, j]
                    arg = j
                if run > NEG:
                    dp[i, j] = run + A[i, j]
                    bk[i, j] = arg
        path = np.zeros(n_steps, dtype=np.int32)
        best = 0
        for j in range(1, n_text):
            if dp[n_steps - 1, j] > dp[n_steps - 1, best]:
                best = j
        path[n_steps - 1] = best
        for i in range(n_steps - 1, 0, -1):
            path[i - 1] = bk[i, path[i]]
        return path

    @_nb.njit(cache=True, nogil=True)
    def _monotonic_path_mass_numba(A):  # pragma: no cover - compiled
        n_steps, n_text = A.shape
        if n_steps == 0 or n_text == 0:
            return 0.0
        dp = np.full((n_steps, n_text), NEG)
        k = min(n_text, 3)
        for j in range(k):
            dp[0, j] = A[0, j]
        for i in range(1, n_steps):
            run = NEG
            for j in range(n_text):
                if dp[i - 1, j] > run:
                    run = dp[i - 1, j]
                if run > NEG:
                    dp[i, j] = run + A[i, j]
        best = dp[n_steps - 1, 0]
        for j in range(1, n_text):
            if dp[n_steps - 1, j] > best:
                best = dp[n_steps - 1, j]
        return best / n_steps


# --------------------------------------------------------------------------
# public dispatch
# --------------------------------------------------------------------------

def _prep(A) -> np.ndarray:
    A = np.ascontiguousarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"expected a 2-D [n_steps, n_text] matrix, got shape {A.shape}")
    return A


def dtw_path(A: np.ndarray, *, backend: str | None = None) -> np.ndarray:
    """Monotonic max-attention path through ``A[n_steps, n_text]``.

    Returns the text column index per audio step (int32, length n_steps).
    ``backend`` forces ``"numba"``, ``"numpy"`` or ``"reference"``; default is
    numba when available, else numpy.
    """
    A = _prep(A)
    if A.shape[0] == 0 or A.shape[1] == 0:
        return np.zeros(A.shape[0], dtype=np.int32)
    if backend is None:
        backend = "numba" if HAVE_NUMBA else "numpy"
    if backend == "numba":
        if not HAVE_NUMBA:
            raise RuntimeError("numba is not available")
        return _dtw_path_numba(A)
    if backend == "numpy":
        return _dtw_path_numpy(A)
    if backend == "reference":
        return dtw_path_reference(A)
    raise ValueError(backend)


def monotonic_path_mass(A: np.ndarray, *, backend: str | None = None) -> float:
    """Mean attention per step along the best monotonic path through ``A``."""
    A = _prep(A)
    if A.shape[0] == 0 or A.shape[1] == 0:
        return 0.0
    if backend is None:
        backend = "numba" if HAVE_NUMBA else "numpy"
    if backend == "numba":
        if not HAVE_NUMBA:
            raise RuntimeError("numba is not available")
        return float(_monotonic_path_mass_numba(A))
    if backend == "numpy":
        return _monotonic_path_mass_numpy(A)
    if backend == "reference":
        return monotonic_path_mass_reference(A)
    raise ValueError(backend)


__all__ = [
    "HAVE_NUMBA",
    "NEG",
    "dtw_path",
    "dtw_path_reference",
    "monotonic_path_mass",
    "monotonic_path_mass_reference",
]
