"""Tests for trnsolver.orthogonalizer and eigh_generalized(..., X=X) (#72)."""

import pytest
import torch

import trnsolver
from trnsolver import orthogonalize as orth_mod

scipy_linalg = pytest.importorskip("scipy.linalg")

METHODS = ("canonical", "symmetric", "cholesky")


def _spd(n, cond=1e3, seed=0, dtype=torch.float64):
    """Q diag(s) Q^H with eigenvalues log-spaced in [1, cond]."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g, dtype=dtype)
    Q, _ = torch.linalg.qr(A)
    s = torch.logspace(0, torch.log10(torch.tensor(cond)).item(), n, dtype=torch.float64)
    S = (Q * s.to(dtype)) @ Q.mH
    return 0.5 * (S + S.mH)


def _overlap(n, seed=0, dtype=torch.float64):
    """Unit-diagonal Gram matrix of random vectors (an AO-overlap stand-in)."""
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(2 * n, n, generator=g, dtype=dtype)
    V = V / torch.linalg.vector_norm(V, dim=0)
    S = V.mH @ V
    return 0.5 * (S + S.mH)


def _sym(n, seed=1, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g, dtype=dtype)
    return 0.5 * (A + A.mH)


def _duplicate(M, idx):
    """Append copies of basis functions ``idx`` (rows and columns)."""
    M = torch.cat([M, M[:, idx]], dim=1)
    return torch.cat([M, M[idx, :]], dim=0)


def _eye(m, dtype):
    return torch.eye(m, dtype=dtype)


# ---------------------------------------------------------------------------
# Well-conditioned S against scipy / torch references
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_orthonormal_full_rank(method):
    S = _overlap(40)
    r = trnsolver.orthogonalizer(S, method)
    assert r.rank == 40
    assert r.X.shape == (40, 40)
    assert r.X.dtype == torch.float64
    torch.testing.assert_close(r.X.T @ S @ r.X, _eye(40, S.dtype), atol=1e-12, rtol=0)
    # X X^T = S^{-1} for any full-rank orthogonalizer (gauge invariant).
    Sinv = torch.from_numpy(scipy_linalg.inv(S.numpy()))
    torch.testing.assert_close(r.X @ r.X.T, Sinv, atol=1e-10, rtol=1e-10)


def test_symmetric_is_loewdin():
    S = _overlap(30)
    X = trnsolver.orthogonalizer(S).X
    ref = torch.from_numpy(scipy_linalg.fractional_matrix_power(S.numpy(), -0.5).real)
    torch.testing.assert_close(X, ref, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(X, trnsolver.inv_sqrt_spd(S), atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(X, X.T, atol=0, rtol=0)


def test_canonical_columns_and_eigenvalues():
    S = _overlap(25)
    r = trnsolver.orthogonalizer(S, "canonical")
    w, U = torch.linalg.eigh(S)
    torch.testing.assert_close(r.eigenvalues, w)
    # Column j is the S eigenvector j scaled by s_j^{-1/2} (up to sign).
    scale = torch.linalg.vector_norm(r.X, dim=0) ** -2
    torch.testing.assert_close(scale, w, atol=1e-12, rtol=1e-10)
    assert r.pivots is None


def test_cholesky_pivots_match_lapack_and_torch(monkeypatch):
    S = _duplicate(_overlap(30), [3, 7])
    r = trnsolver.orthogonalizer(S, "cholesky")
    _, piv, rank, _ = scipy_linalg.lapack.dpstrf(S.numpy(), lower=1, tol=1e-7)
    assert r.rank == rank == 30
    assert r.pivots.tolist() == (piv[:rank] - 1).tolist()
    assert r.eigenvalues is None
    # The torch fallback (no SciPy) picks the same pivots.
    monkeypatch.setattr(orth_mod, "_pivots_lapack", lambda S, tol: None)
    r2 = trnsolver.orthogonalizer(S, "cholesky")
    assert r2.pivots.tolist() == r.pivots.tolist()
    torch.testing.assert_close(r2.X, r.X)
    # X is zero outside the kept functions.
    dropped = sorted(set(range(32)) - set(r.pivots.tolist()))
    assert len(dropped) == 2
    assert torch.all(r.X[dropped] == 0)


# ---------------------------------------------------------------------------
# Near-singular S: dependencies are dropped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_duplicate_functions_dropped(method):
    S0 = _overlap(30)
    S = _duplicate(S0, [0, 5, 17])
    r = trnsolver.orthogonalizer(S, method)
    assert r.rank == 30
    assert r.X.shape == (33, 30)
    torch.testing.assert_close(r.X.T @ S @ r.X, _eye(30, S.dtype), atol=1e-11, rtol=0)


def test_planted_eigenvalues_drop_exactly_below_threshold():
    n = 40
    g = torch.Generator().manual_seed(3)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    s = torch.logspace(-3, 1, n, dtype=torch.float64)
    s[:4] = torch.tensor([1e-12, 1e-9, 5e-7, 2e-6], dtype=torch.float64) * s.max()
    S = (Q * s) @ Q.T
    S = 0.5 * (S + S.T)
    # X^T dS X for a rounding dS grows like 1 / (smallest kept eigenvalue).
    for lindep, m, atol in ((1e-7, n - 2, 1e-9), (1e-6, n - 3, 1e-9), (1e-10, n - 1, 1e-6)):
        for method in ("canonical", "symmetric"):
            r = trnsolver.orthogonalizer(S, method, lindep)
            assert r.rank == m, (lindep, method)
            kept = r.eigenvalues > lindep * r.eigenvalues.max()
            assert int(kept.sum()) == m
            torch.testing.assert_close(r.X.T @ S @ r.X, _eye(m, S.dtype), atol=atol, rtol=0)


def test_cholesky_handles_near_singular():
    S = _duplicate(_overlap(30), [2])
    S[30, 30] += 1e-12  # duplicate plus noise: Schur pivot 1e-12 < 1e-7
    r = trnsolver.orthogonalizer(S, "cholesky")
    assert r.rank == 30
    torch.testing.assert_close(r.X.T @ S @ r.X, _eye(30, S.dtype), atol=1e-10, rtol=0)


def test_lindep_zero_keeps_every_positive_eigenvalue():
    S = _spd(20, cond=1e10)
    for method in METHODS:
        assert trnsolver.orthogonalizer(S, method, 0.0).rank == 20


# ---------------------------------------------------------------------------
# eigh_generalized with X
# ---------------------------------------------------------------------------


def test_eigh_generalized_with_x_matches_scipy():
    n = 32
    S, F = _overlap(n), _sym(n)
    wref, Vref = scipy_linalg.eigh(F.numpy(), S.numpy())
    for method in METHODS:
        X = trnsolver.orthogonalizer(S, method).X
        w, V = trnsolver.eigh_generalized(F, None, X=X)
        torch.testing.assert_close(w, torch.from_numpy(wref), atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(F @ V, S @ V * w, atol=1e-9, rtol=0)
        torch.testing.assert_close(V.T @ S @ V, _eye(n, F.dtype), atol=1e-10, rtol=0)
    # The Cholesky path agrees.
    w2, _ = trnsolver.eigh_generalized(F, S)
    torch.testing.assert_close(w2, torch.from_numpy(wref), atol=1e-10, rtol=1e-10)


def test_eigh_generalized_duplicates_where_cholesky_fails():
    n = 24
    S0, F0 = _overlap(n), _sym(n)
    dup = [1, 9]
    S, F = _duplicate(S0, dup), _duplicate(F0, dup)
    with pytest.raises(torch.linalg.LinAlgError):
        trnsolver.eigh_generalized(F, S)
    wref = torch.from_numpy(scipy_linalg.eigh(F0.numpy(), S0.numpy(), eigvals_only=True))
    for method in METHODS:
        X = trnsolver.orthogonalizer(S, method).X
        w, V = trnsolver.eigh_generalized(F, S, X=X)
        assert w.shape == (n,) and V.shape == (n + 2, n)
        torch.testing.assert_close(w, wref, atol=1e-9, rtol=1e-9)


def test_eigh_generalized_fp32_application():
    """X built in fp64 on the host, applied in fp32 (the device path)."""
    n = 48
    S, F = _overlap(n), _sym(n)
    wref = torch.from_numpy(scipy_linalg.eigh(F.numpy(), S.numpy(), eigvals_only=True))
    X = trnsolver.orthogonalizer(S, dtype=torch.float32).X
    assert X.dtype == torch.float32
    w, V = trnsolver.eigh_generalized(F.float(), None, X=X)
    assert w.dtype == V.dtype == torch.float32
    torch.testing.assert_close(w.double(), wref, atol=1e-4, rtol=1e-4)
    # A float64 X is cast to F's dtype.
    w2, _ = trnsolver.eigh_generalized(F.float(), None, X=trnsolver.orthogonalizer(S).X)
    torch.testing.assert_close(w2, w)


def test_eigh_generalized_needs_b_or_x():
    with pytest.raises(ValueError, match="needs B"):
        trnsolver.eigh_generalized(_sym(4), None)
    with pytest.raises(ValueError, match="rows"):
        trnsolver.eigh_generalized(_sym(4), None, X=torch.eye(5, dtype=torch.float64))


# ---------------------------------------------------------------------------
# Complex Hermitian, batched, dtype handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_complex_hermitian(method):
    n = 20
    S = _duplicate(_overlap(n, dtype=torch.complex128), [4])
    F = _duplicate(_sym(n, dtype=torch.complex128), [4])
    r = trnsolver.orthogonalizer(S, method)
    assert r.X.dtype == torch.complex128
    assert r.rank == n
    torch.testing.assert_close(r.X.mH @ S @ r.X, _eye(n, S.dtype), atol=1e-11, rtol=0)
    w, V = trnsolver.eigh_generalized(F, None, X=r.X)
    wref = scipy_linalg.eigh(F[:n, :n].numpy(), S[:n, :n].numpy(), eigvals_only=True)
    torch.testing.assert_close(w, torch.from_numpy(wref), atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_batched(method):
    S1 = _duplicate(_overlap(16, seed=1), [0])
    S2 = _overlap(17, seed=2)
    S3 = _duplicate(_overlap(15, seed=3), [1, 2])
    S = torch.stack([S1, S2, S3]).reshape(3, 1, 17, 17)
    r = trnsolver.orthogonalizer(S, method)
    assert r.rank.shape == (3, 1)
    assert r.rank.flatten().tolist() == [16, 17, 15]
    assert r.X.shape == (3, 1, 17, 17)
    for i, Si in enumerate((S1, S2, S3)):
        one = trnsolver.orthogonalizer(Si, method)
        Xi = r.X[i, 0]
        torch.testing.assert_close(Xi[:, : one.rank], one.X)
        assert torch.all(Xi[:, one.rank :] == 0)
        if method == "cholesky":
            assert r.pivots[i, 0, : one.rank].tolist() == one.pivots.tolist()
            assert torch.all(r.pivots[i, 0, one.rank :] == -1)
        else:
            torch.testing.assert_close(r.eigenvalues[i, 0], one.eigenvalues)


def test_low_precision_input_computed_in_fp64():
    S = _spd(24, cond=1e6)
    r64 = trnsolver.orthogonalizer(S)
    r32 = trnsolver.orthogonalizer(S.float())
    assert r32.X.dtype == torch.float64
    # Only the input rounding differs: fp64 work, not an fp32 eigensolve.
    err = (r32.X.T @ S @ r32.X - _eye(24, S.dtype)).abs().max()
    assert err < 1e-1
    torch.testing.assert_close(r64.X.T @ S @ r64.X, _eye(24, S.dtype), atol=1e-9, rtol=0)
    rbf = trnsolver.orthogonalizer(S.bfloat16(), dtype=torch.bfloat16)
    assert rbf.X.dtype == torch.bfloat16


def test_bad_arguments():
    S = _overlap(4)
    with pytest.raises(ValueError, match="method"):
        trnsolver.orthogonalizer(S, "qr")
    with pytest.raises(ValueError, match="lindep"):
        trnsolver.orthogonalizer(S, lindep=-1.0)
    with pytest.raises(ValueError, match="shape"):
        trnsolver.orthogonalizer(S[:, :3])
    with pytest.raises(ValueError, match="positive"):
        trnsolver.orthogonalizer(-S)
    with pytest.raises(ValueError, match="positive"):
        trnsolver.orthogonalizer(-S, "cholesky")


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------


def _sym_input(n, seed=0):
    S = _overlap(n, seed=seed)
    return S.clone().requires_grad_(True)


def test_gradcheck_symmetric():
    S = _sym_input(6)
    assert torch.autograd.gradcheck(
        lambda A: trnsolver.orthogonalizer(A, "symmetric").X, (S,), atol=1e-6
    )


def test_gradcheck_cholesky():
    # Distinct diagonal entries: a unit diagonal ties the first pivot, and the
    # finite differences would flip it.
    S = _spd(6, cond=30, seed=1).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda A: trnsolver.orthogonalizer(A, "cholesky").X, (S,), atol=1e-6
    )


def test_gradcheck_canonical_gauge_invariant():
    # Canonical X is defined up to column signs; X X^T (= S^{-1} on the kept
    # subspace) is not, also with a dropped direction.
    S = _duplicate(_overlap(5, seed=2), [0]) + 1e-9 * torch.eye(6, dtype=torch.float64)
    S = S.clone().requires_grad_(True)

    def f(A):
        X = trnsolver.orthogonalizer(A, "canonical", 1e-6).X
        assert X.shape[-1] == 5
        return X @ X.T

    assert torch.autograd.gradcheck(f, (S,), atol=1e-5)
