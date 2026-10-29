"""Stage-2 tridiagonal eigensolve of the NKI eigh path (#42).

``TRNSOLVER_EIGH_TRIDIAG_SOLVER`` picks the host solve of the tridiagonal
(d, e): ``lapack`` (default, one vectorized call) or ``qr`` (the scalar
implicit-shift QR iteration, kept for one release). CPU-only: the NKI
reduction and reflector kernels are replaced by pure-torch equivalents so the
rest of ``_householder_qr_eigh`` (Q_right, reflector assembly, Rayleigh-Ritz)
runs as it does on device.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

import trnsolver.eigen as eigen

SIZES = list(range(2, 129))
DTYPES = [torch.float32, torch.float64]
SOLVERS = ["lapack", "qr"]


def _sym(n, dtype, seed=0):
    g = torch.Generator().manual_seed(seed + n)
    A = torch.randn(n, n, generator=g, dtype=torch.float64)
    return (0.5 * (A + A.T)).to(dtype)


def _fix_signs(V):
    """Flip each column so its largest-magnitude entry is positive."""
    idx = V.abs().argmax(dim=0)
    signs = torch.sign(V[idx, torch.arange(V.shape[1])])
    signs[signs == 0] = 1.0
    return V * signs


def _householder_torch(A):
    """Host Householder reduction with the kernel's (diag, subdiag, V_refs) layout."""
    n = A.shape[0]
    T = A.clone()
    V_refs = torch.zeros(n, max(n - 1, 1), dtype=A.dtype)
    for k in range(n - 2):
        x = T[k + 1 :, k]
        nx = torch.linalg.norm(x)
        if nx == 0:
            continue
        v = torch.zeros(n, dtype=A.dtype)
        v[k + 1 :] = x
        v[k + 1] += nx if x[0] >= 0 else -nx
        v /= torch.linalg.norm(v)
        V_refs[:, k] = v
        H = torch.eye(n, dtype=A.dtype) - 2.0 * torch.outer(v, v)
        T = H @ T @ H
    return torch.diagonal(T).clone(), torch.diagonal(T, 1).clone(), V_refs


@pytest.fixture
def cpu_pipeline(monkeypatch):
    """Route _householder_qr_eigh through pure-torch stage-1 and assembly."""
    monkeypatch.setattr(eigen, "_use_device_tridiag", lambda n: False)
    monkeypatch.setattr(eigen, "_householder_tridiag", _householder_torch)
    monkeypatch.setattr(eigen, "_call_apply_reflectors", eigen._apply_reflectors)


class TestSwitch:
    def test_default_lapack(self, monkeypatch):
        monkeypatch.delenv("TRNSOLVER_EIGH_TRIDIAG_SOLVER", raising=False)
        assert eigen._tridiag_solver() == "lapack"

    @pytest.mark.parametrize(
        "value,expected",
        [("lapack", "lapack"), (" LAPACK ", "lapack"), ("qr", "qr"), ("", "lapack")],
    )
    def test_values(self, monkeypatch, value, expected):
        monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG_SOLVER", value)
        assert eigen._tridiag_solver() == expected

    def test_invalid(self, monkeypatch):
        monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG_SOLVER", "jacobi")
        with pytest.raises(ValueError, match="TRNSOLVER_EIGH_TRIDIAG_SOLVER"):
            eigen._tridiag_solver()

    def test_qr_routes_to_qr_iterate(self, monkeypatch):
        calls = []
        real = eigen._qr_iterate

        def spy(d, e, tol):
            calls.append(d.shape[0])
            return real(d, e, tol)

        monkeypatch.setattr(eigen, "_qr_iterate", spy)
        d, e = torch.randn(6, dtype=torch.float64), torch.randn(5, dtype=torch.float64)
        monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG_SOLVER", "lapack")
        eigen._tridiag_eigh_host(d, e, 1e-10)
        assert calls == []
        monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG_SOLVER", "qr")
        eigen._tridiag_eigh_host(d, e, 1e-10)
        assert calls == [6]


class TestLapackTridiag:
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("scipy_present", [True, False])
    def test_decomposes_tridiagonal(self, monkeypatch, dtype, scipy_present):
        if scipy_present:
            pytest.importorskip("scipy")
        else:
            # Force the torch fallback: `import scipy.linalg` raises ImportError.
            import sys

            monkeypatch.setitem(sys.modules, "scipy", None)
        n = 64
        g = torch.Generator().manual_seed(1)
        d = torch.randn(n, generator=g, dtype=torch.float64).to(dtype)
        e = torch.randn(n - 1, generator=g, dtype=torch.float64).to(dtype)
        w, Q = eigen._lapack_tridiag_eigh(d, e)
        assert w.dtype == dtype and Q.dtype == dtype
        assert torch.all(w[1:] >= w[:-1])
        T = torch.diag(d.double()) + torch.diag(e.double(), 1) + torch.diag(e.double(), -1)
        atol = 1e-4 if dtype == torch.float32 else 1e-10
        np.testing.assert_allclose(
            (Q.double() @ torch.diag(w.double()) @ Q.double().T).numpy(), T.numpy(), atol=atol
        )
        np.testing.assert_allclose((Q.double().T @ Q.double()).numpy(), np.eye(n), atol=atol)


class TestEquivalence:
    @pytest.mark.parametrize("solver", SOLVERS)
    @pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "fp64"])
    @pytest.mark.parametrize("n", SIZES)
    def test_vs_torch_eigh(self, cpu_pipeline, monkeypatch, n, dtype, solver):
        monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG_SOLVER", solver)
        A = _sym(n, dtype)
        w, V = eigen._householder_qr_eigh(A)
        assert w.dtype == dtype and V.dtype == dtype
        w_ref, V_ref = torch.linalg.eigh(A.double())
        if dtype == torch.float32:
            w_tol, v_tol, r_tol = 1e-4, 2e-3, 1e-4
        else:
            w_tol, v_tol, r_tol = 1e-10, 1e-7, 1e-10
        scale = max(1.0, w_ref.abs().max().item())
        np.testing.assert_allclose(w.double().numpy(), w_ref.numpy(), atol=w_tol * scale)
        np.testing.assert_allclose(
            _fix_signs(V.double()).numpy(), _fix_signs(V_ref).numpy(), atol=v_tol
        )
        resid = (A.double() @ V.double() - V.double() * w.double()).abs().max().item()
        assert resid < r_tol * scale

    @pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "fp64"])
    def test_solvers_agree(self, cpu_pipeline, monkeypatch, dtype):
        A = _sym(96, dtype)
        out = {}
        for solver in SOLVERS:
            monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG_SOLVER", solver)
            w, V = eigen._householder_qr_eigh(A)
            out[solver] = (w.double(), _fix_signs(V.double()))
        atol = 1e-4 if dtype == torch.float32 else 1e-9
        np.testing.assert_allclose(out["lapack"][0].numpy(), out["qr"][0].numpy(), atol=atol)
        np.testing.assert_allclose(out["lapack"][1].numpy(), out["qr"][1].numpy(), atol=10 * atol)


class TestTiming:
    def test_lapack_50x_faster_than_qr_at_128(self, monkeypatch):
        n = 128
        d, e, _ = _householder_torch(_sym(n, torch.float32))

        def best(solver, reps):
            monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG_SOLVER", solver)
            eigen._tridiag_eigh_host(d, e, 1e-10)  # warm
            times = []
            for _ in range(reps):
                t0 = time.perf_counter()
                eigen._tridiag_eigh_host(d, e, 1e-10)
                times.append(time.perf_counter() - t0)
            return min(times)

        t_lapack = best("lapack", 20)
        t_qr = best("qr", 2)
        assert t_qr >= 50 * t_lapack, f"qr {t_qr * 1e3:.2f} ms vs lapack {t_lapack * 1e3:.3f} ms"
