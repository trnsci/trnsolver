"""Complex Hermitian and batched inputs against torch.linalg (trnsolver#64).

``.T`` was used where the math needs the conjugate transpose of the last
two dims (``.mH``). For complex Hermitian input that silently dropped the
conjugation; for a batch it reversed every dim, which for a batch of size n
kept the shape and returned wrong values with no error.
"""

import pytest
import torch

import trnsolver

N = 6
CDT = (torch.complex64, torch.complex128)


def _hpd(n, dtype, batch=(), seed=0):
    g = torch.Generator().manual_seed(seed)
    re = torch.randn(*batch, n, n, generator=g, dtype=torch.float64)
    im = torch.randn(*batch, n, n, generator=g, dtype=torch.float64) if dtype.is_complex else 0 * re
    X = torch.complex(re, im) if dtype.is_complex else re
    A = X @ X.mH / n + 2 * torch.eye(n, dtype=X.dtype)
    return A.to(dtype)


def _rhs(*shape, dtype, seed=5):
    g = torch.Generator().manual_seed(seed)
    if dtype.is_complex:
        return torch.complex(
            torch.randn(*shape, generator=g, dtype=torch.float64),
            torch.randn(*shape, generator=g, dtype=torch.float64),
        ).to(dtype)
    return torch.randn(*shape, generator=g, dtype=torch.float64).to(dtype)


def _tol(dtype):
    return 1e-4 if dtype in (torch.float32, torch.complex64) else 1e-10


def _close(x, ref, dtype):
    assert x.shape == ref.shape
    torch.testing.assert_close(x, ref.to(x.dtype), atol=_tol(dtype), rtol=_tol(dtype))


@pytest.mark.parametrize("dtype", CDT)
class TestComplexHermitian:
    def test_cholesky_upper(self, dtype):
        A = _hpd(N, dtype)
        U = trnsolver.cholesky(A, upper=True)
        _close(U, torch.linalg.cholesky(A, upper=True), dtype)
        _close(U.mH @ U, A, dtype)

    @pytest.mark.parametrize("k", [None, 3])
    @pytest.mark.parametrize("ir", [False, True])
    def test_solve_spd(self, dtype, k, ir):
        A = _hpd(N, dtype)
        B = _rhs(N, dtype=dtype) if k is None else _rhs(N, k, dtype=dtype)
        X = trnsolver.solve_spd(A, B, iterative_refinement=ir)
        _close(X, torch.linalg.solve(A, B), dtype)

    def test_inv_spd(self, dtype):
        A = _hpd(N, dtype)
        _close(trnsolver.inv_spd(A), torch.linalg.inv(A), dtype)

    def test_inv_sqrt_spd(self, dtype):
        A = _hpd(N, dtype)
        X = trnsolver.inv_sqrt_spd(A)
        _close(X @ A @ X, torch.eye(N, dtype=dtype), dtype)
        _close(X, X.mH, dtype)

    def test_eigh_generalized(self, dtype):
        A = _hpd(N, dtype, seed=1) - 2.5 * torch.eye(N, dtype=dtype)
        B = _hpd(N, dtype, seed=2)
        w, V = trnsolver.eigh_generalized(A, B)
        L = torch.linalg.cholesky(B.to(torch.complex128))
        Li = torch.linalg.inv(L)
        ref = torch.linalg.eigvalsh(Li @ A.to(torch.complex128) @ Li.mH)
        _close(w, ref, dtype)
        _close(A @ V, B @ V * w.to(dtype), dtype)

    def test_ssor_preconditioner(self, dtype):
        A = _hpd(N, dtype).to(torch.complex128)
        r = _rhs(N, dtype=torch.complex128)
        w = 1.3
        D = torch.diag(torch.diagonal(A))
        L = torch.tril(A, -1)
        M = (D + w * L) @ torch.linalg.inv(D) @ (D + w * L.mH) / (w * (2 - w))
        z = trnsolver.ssor_preconditioner(A, w)(r)
        _close(z, torch.linalg.solve(M, r), torch.complex128)


# Batch of size n: ``.T`` reverses (n, n, n) into (n, n, n), so the shape
# matches and only the values are wrong. Batch (2, 3) checks other shapes.
@pytest.mark.parametrize("batch", [(N,), (2, 3)])
class TestBatched:
    def test_cholesky_upper(self, batch):
        A = _hpd(N, torch.float64, batch)
        _close(
            trnsolver.cholesky(A, upper=True), torch.linalg.cholesky(A, upper=True), torch.float64
        )

    @pytest.mark.parametrize("ir", [False, True])
    def test_solve_spd_matrix_rhs(self, batch, ir):
        A = _hpd(N, torch.float64, batch)
        B = _rhs(*batch, N, 2, dtype=torch.float64)
        X = trnsolver.solve_spd(A, B, iterative_refinement=ir)
        _close(X, torch.linalg.solve(A, B), torch.float64)

    def test_solve_spd_shared_vector_rhs(self, batch):
        A = _hpd(N, torch.float64, batch)
        b = _rhs(N, dtype=torch.float64)
        X = trnsolver.solve_spd(A, b)
        _close(
            X, torch.linalg.solve(A, b.expand(*batch, N).unsqueeze(-1)).squeeze(-1), torch.float64
        )

    def test_inv_spd(self, batch):
        A = _hpd(N, torch.float64, batch)
        _close(trnsolver.inv_spd(A), torch.linalg.inv(A), torch.float64)

    def test_inv_sqrt_spd(self, batch):
        A = _hpd(N, torch.float64, batch)
        w, V = torch.linalg.eigh(A)
        ref = V @ torch.diag_embed(w.rsqrt()) @ V.mT
        _close(trnsolver.inv_sqrt_spd(A), ref, torch.float64)

    def test_pinv(self, batch):
        g = torch.Generator().manual_seed(3)
        A = torch.randn(*batch, N + 3, N, generator=g, dtype=torch.float64)
        A[..., 0, :] *= 1e3  # different largest singular value per matrix
        _close(trnsolver.pinv(A), torch.linalg.pinv(A), torch.float64)

    def test_eigh_generalized(self, batch):
        A = _hpd(N, torch.float64, batch, seed=1) - 2.5 * torch.eye(N, dtype=torch.float64)
        B = _hpd(N, torch.float64, batch, seed=2)
        w, V = trnsolver.eigh_generalized(A, B)
        _close(A @ V, B @ V * w.unsqueeze(-2), torch.float64)
        L = torch.linalg.cholesky(B)
        Li = torch.linalg.inv(L)
        _close(w, torch.linalg.eigvalsh(Li @ A @ Li.mT), torch.float64)


def test_pinv_cutoff_is_per_matrix():
    """A batch must not share the first matrix's cutoff (or its batch-dim size)."""
    big = torch.diag(torch.tensor([1e6, 1.0], dtype=torch.float64))
    small = torch.diag(torch.tensor([1.0, 1e-3], dtype=torch.float64))
    A = torch.stack([big, small])
    P = trnsolver.pinv(A, rcond=1e-5)
    torch.testing.assert_close(P[1], torch.linalg.inv(small))
    torch.testing.assert_close(P[0], torch.diag(torch.tensor([1e-6, 0.0], dtype=torch.float64)))


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0), (2, 0, 4)])
def test_pinv_empty(shape):
    A = torch.zeros(shape, dtype=torch.float64)
    assert trnsolver.pinv(A).shape == torch.linalg.pinv(A).shape


@pytest.mark.parametrize("dtype", CDT)
def test_eigh_randomized_complex_hermitian(dtype):
    n, k = 40, 3
    g = torch.Generator().manual_seed(7)
    Q, _ = torch.linalg.qr(
        torch.complex(
            torch.randn(n, n, generator=g, dtype=torch.float64),
            torch.randn(n, n, generator=g, dtype=torch.float64),
        )
    )
    lam = torch.cat(
        [
            torch.tensor([30.0, -20.0, 10.0], dtype=torch.float64),
            1e-3 * torch.rand(n - 3, generator=g, dtype=torch.float64),
        ]
    )
    A = ((Q * lam.to(Q.dtype)) @ Q.mH).to(dtype)
    w, V = trnsolver.eigh_randomized(A, k, seed=0)
    # Sketch error from the 1e-3 tail: compare at 1e-5, not to rounding.
    ref = torch.tensor([-20.0, 10.0, 30.0], dtype=w.dtype)
    torch.testing.assert_close(w, ref, atol=1e-5, rtol=1e-5)
    _close(V.mH @ V, torch.eye(k, dtype=dtype), dtype)


@pytest.mark.parametrize(
    "dtype,batch",
    [(torch.complex64, ()), (torch.complex128, ()), (torch.float64, (N,)), (torch.float32, (2, 3))],
)
def test_eigh_complex_batch_skip_householder_under_require_nki(monkeypatch, dtype, batch):
    # The Householder kernels take one real (n, n) matrix. Under
    # TRNSOLVER_REQUIRE_NKI=1 on a Neuron host, complex or batched eigh (the
    # inner call of eigh_generalized) raised there instead of taking aten.
    from trnsolver import eigen

    def _no_householder(*a, **k):
        raise AssertionError("Householder path called for complex/batched input")

    monkeypatch.setattr(eigen, "_REQUIRE_NKI", True)
    monkeypatch.setattr(eigen, "_use_nki", lambda: True)
    monkeypatch.setattr(eigen, "_householder_qr_eigh", _no_householder)
    A = _hpd(N, dtype, batch=batch)
    w, V = trnsolver.eigh(A)
    w_ref, _ = torch.linalg.eigh(A)
    _close(w, w_ref, dtype)
    _close(V @ torch.diag_embed(w.to(V.dtype)) @ V.mH, A, dtype)


# On a Neuron device matmul ignores the lazy conj bit of a ``.mH`` view
# (aws-neuron/aws-neuron-sdk#1455): ``A.mH @ B`` computes ``A.mT @ B``. CPU
# matmul honours the bit, so the value tests above cannot see it; instead,
# check that no conj view reaches a matmul or solve, or is returned.

_LINALG = {
    torch.matmul,
    torch.Tensor.__matmul__,
    torch.Tensor.__rmatmul__,
    torch.Tensor.matmul,
    torch.mm,
    torch.bmm,
    torch.linalg.solve_triangular,
    torch.linalg.solve,
}


class _NoConjViews(torch.overrides.TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.seen = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _LINALG:
            for a in (*args, *kwargs.values()):
                if isinstance(a, torch.Tensor) and a.is_conj():
                    self.seen.append(getattr(func, "__name__", str(func)))
        return func(*args, **kwargs)


def test_mH_resolves_conj():
    from trnsolver.factor import _mH

    A = _rhs(2, N, N, dtype=torch.complex64)
    assert A.mH.is_conj()  # the lazy view this guards against
    H = _mH(A)
    assert not H.is_conj()
    torch.testing.assert_close(H, A.conj().transpose(-2, -1).contiguous())
    R = _rhs(N, N, dtype=torch.float32)
    assert _mH(R).data_ptr() == R.data_ptr()  # real: no copy


_CONJ_CASES = {
    "cholesky_upper": lambda A, B, S, X: trnsolver.cholesky(A, upper=True),
    "solve_spd": lambda A, B, S, X: trnsolver.solve_spd(A, B),
    "solve_spd_ir": lambda A, B, S, X: trnsolver.solve_spd(A, B, iterative_refinement=True),
    "inv_spd": lambda A, B, S, X: trnsolver.inv_spd(A),
    "inv_sqrt_spd": lambda A, B, S, X: trnsolver.inv_sqrt_spd(A),
    "pinv": lambda A, B, S, X: trnsolver.pinv(A),
    "eigh_generalized": lambda A, B, S, X: trnsolver.eigh_generalized(A, S),
    "eigh_generalized_X": lambda A, B, S, X: trnsolver.eigh_generalized(A, None, X=X),
    "eigh_randomized": lambda A, B, S, X: trnsolver.eigh_randomized(A, k=2, oversampling=2),
    "ssor": lambda A, B, S, X: trnsolver.ssor_preconditioner(A)(B),
}


@pytest.mark.parametrize("dtype", CDT)
@pytest.mark.parametrize("op", sorted(_CONJ_CASES))
def test_no_conj_view_reaches_matmul(op, dtype):
    # Inputs are built outside the mode: only the op's own calls are checked
    # (the orthogonalizer X is built once on the host in float64).
    A = _hpd(N, dtype)
    B = _rhs(N, dtype=dtype)
    S = _hpd(N, dtype, seed=3)
    X = trnsolver.orthogonalizer(S)[0]
    mode = _NoConjViews()
    with mode:
        out = _CONJ_CASES[op](A, B, S, X)
    assert mode.seen == [], f"{op}: conj view passed to {mode.seen} (aws-neuron-sdk#1455)"
    for t in out if isinstance(out, tuple) else (out,):
        assert not t.is_conj(), f"{op} returned a lazy conj view"
