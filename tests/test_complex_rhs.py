"""Real A with complex B keeps B's imaginary part (trnsolver#66).

``solve``, ``solve_spd`` and ``eigh_generalized`` cast B to A's dtype. For a
real A and a complex B that dropped the imaginary part (a UserWarning only),
where ``torch.linalg.solve`` raises on the mismatch.
"""

import warnings

import pytest
import torch

import trnsolver

N = 6


def _spd(n, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, n, generator=g, dtype=torch.float64)
    return (X @ X.T / n + 2 * torch.eye(n, dtype=torch.float64)).to(dtype)


def _crand(*shape, dtype, seed=3):
    g = torch.Generator().manual_seed(seed)
    re = torch.randn(*shape, generator=g, dtype=torch.float64)
    im = torch.randn(*shape, generator=g, dtype=torch.float64)
    return torch.complex(re, im).to(dtype)


@pytest.fixture(autouse=True)
def _no_discard_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Casting complex values to real")
        yield


@pytest.mark.parametrize("a_dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("fn", ["solve", "solve_spd"])
@pytest.mark.parametrize("k", [None, 2])
def test_real_matrix_complex_rhs(a_dtype, fn, k):
    A = _spd(N, a_dtype)
    B = _crand(N, dtype=torch.complex64) if k is None else _crand(N, k, dtype=torch.complex64)
    X = getattr(trnsolver, fn)(A, B)
    assert X.is_complex()
    ref = torch.linalg.solve(A.to(torch.complex128), B.to(torch.complex128))
    tol = 5e-2 if a_dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(X.to(torch.complex128), ref, atol=tol, rtol=tol)


def test_eigh_generalized_real_a_complex_b():
    A = _spd(N, torch.float64, seed=1) - 2.5 * torch.eye(N, dtype=torch.float64)
    X = _crand(N, N, dtype=torch.complex128, seed=4)
    B = X @ X.mH / N + 2 * torch.eye(N, dtype=torch.complex128)
    w, V = trnsolver.eigh_generalized(A, B)
    Ac = A.to(torch.complex128)
    torch.testing.assert_close(Ac @ V, B @ V * w.to(V.dtype))
    Li = torch.linalg.inv(torch.linalg.cholesky(B))
    torch.testing.assert_close(w, torch.linalg.eigvalsh(Li @ Ac @ Li.mH))


def test_real_rhs_unchanged():
    A = _spd(N, torch.float32)
    B = torch.randn(N, 2, dtype=torch.float64)
    X = trnsolver.solve(A, B)
    assert X.dtype == torch.float32  # B is still cast to A's real dtype
