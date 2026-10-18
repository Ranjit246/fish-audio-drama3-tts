"""numba vs numpy vs literal-reference DP kernels must agree exactly."""

from __future__ import annotations

import numpy as np
import pytest

from d3align import dp

BACKENDS = ["numpy", "reference"] + (["numba"] if dp.HAVE_NUMBA else [])


def _matrices():
    rng = np.random.default_rng(1234)
    shapes = [(1, 1), (1, 5), (2, 1), (3, 2), (7, 3), (40, 9), (120, 33), (9, 60), (64, 64)]
    for n, m in shapes:
        yield rng.random((n, m))
        # heavy ties: quantised values exercise the strict-'>' back-pointer rule
        yield np.round(rng.random((n, m)) * 3) / 3
        # softmax-like rows with a staircase
        A = rng.random((n, m)) * 0.05
        for i in range(n):
            A[i, min(m - 1, (i * m) // max(n, 1))] += 1.0
        yield A / A.sum(1, keepdims=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dtw_path_matches_reference(backend):
    for A in _matrices():
        ref = dp.dtw_path_reference(A)
        got = dp.dtw_path(A, backend=backend)
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, ref, err_msg=f"shape {A.shape} backend {backend}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_path_mass_matches_reference(backend):
    for A in _matrices():
        ref = dp.monotonic_path_mass_reference(A)
        got = dp.monotonic_path_mass(A, backend=backend)
        assert got == pytest.approx(ref, rel=0, abs=1e-12), f"shape {A.shape} backend {backend}"


def test_default_backend_is_numba_when_available():
    A = np.random.default_rng(0).random((20, 7))
    np.testing.assert_array_equal(dp.dtw_path(A), dp.dtw_path_reference(A))
    if dp.HAVE_NUMBA:
        assert dp.dtw_path.__module__ == "d3align.dp"
        # the compiled kernels exist and are what the dispatcher calls
        assert hasattr(dp, "_dtw_path_numba") and hasattr(dp, "_monotonic_path_mass_numba")


def test_path_is_monotonic_and_follows_a_clean_diagonal():
    n, m = 50, 10
    A = np.full((n, m), 0.01)
    for i in range(n):
        A[i, (i * m) // n] = 1.0
    path = dp.dtw_path(A)
    assert (np.diff(path) >= 0).all()
    assert path[0] == 0 and path[-1] == m - 1
    assert dp.monotonic_path_mass(A / A.sum(1, keepdims=True)) > 0.9


def test_empty_inputs():
    assert dp.dtw_path(np.zeros((0, 5))).shape == (0,)
    assert dp.monotonic_path_mass(np.zeros((0, 5))) == 0.0
    assert dp.monotonic_path_mass(np.zeros((5, 0))) == 0.0
