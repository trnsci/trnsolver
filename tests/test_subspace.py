"""``trnsolver.occupied_subspace``: Chebyshev-filtered subspace iteration.

Layers:

* CPU: the filter coefficients against a direct Chebyshev evaluation, the
  PyTorch reference filter against its definition, the Rayleigh-Ritz, and
  the driver (cold start, warm start, tolerance, guesses, Fock-like spectra
  with deep core levels). These run in ``pytest -m "not neuron"``.
* ``nki_simulator``: the kernel through ``nki.simulate`` against the
  reference, the driver end to end on the kernel, and the NKI 0.7 MLIR
  verifier at lnc = 1 and 2.
* ``neuron``: the kernel on the native ``neuron`` device.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

import trnsolver
from trnsolver import subspace as SS
from trnsolver.nki import subspace as SK
from trnsolver.nki.dispatch import HAS_NKI


def _spectrum_matrix(w, seed=0, dtype=torch.float64):
    """Q diag(w) Q^T with a random orthogonal Q."""
    n = w.numel()
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    A = (Q * w.double()) @ Q.T
    return (0.5 * (A + A.T)).to(dtype)


def _fock_like(n, k, seed=0, dtype=torch.float64):
    """A Fock-like spectrum: a few deep core levels, a valence band, a gap, virtuals."""
    g = torch.Generator().manual_seed(seed)
    ncore = max(1, k // 5)
    core = -11.0 - 0.3 * torch.rand(ncore, generator=g, dtype=torch.float64)
    val = torch.linspace(-1.5, -0.35, k - ncore, dtype=torch.float64)
    virt = 0.15 + torch.sort(torch.rand(n - k, generator=g, dtype=torch.float64) ** 2 * 30).values
    return _spectrum_matrix(torch.cat([core, val, virt]), seed=seed, dtype=dtype)


def _perturb(A, eps, seed=1):
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(A.shape, generator=g, dtype=torch.float64)
    return (A.double() + eps * 0.5 * (D + D.T)).to(A.dtype)


def _proj_err(V, A, k):
    """Spectral-norm distance between span(V) and A's lowest-k eigenspace."""
    w, U = torch.linalg.eigh(A.double())
    V = V.double()
    return float(torch.linalg.matrix_norm(V @ V.T - U[:, :k] @ U[:, :k].T, 2))


def _cheb(m, x):
    x = torch.as_tensor(x, dtype=torch.float64)
    return torch.where(
        x.abs() <= 1,
        torch.cos(m * torch.arccos(torch.clamp(x, -1, 1))),
        torch.cosh(m * torch.arccosh(x.abs())) * torch.sign(x) ** m,
    )


@pytest.fixture
def pytorch_backend():
    old = trnsolver.get_backend()
    trnsolver.set_backend("pytorch")
    yield
    trnsolver.set_backend(old)


# ----------------------------------------------------------------------
# CPU
# ----------------------------------------------------------------------


class TestCoefficients:
    @pytest.mark.parametrize("m", [1, 2, 5, 8, 16])
    def test_scaled_chebyshev(self, m):
        a0, a, b = -3.0, 0.5, 4.0
        coef = SS.chebyshev_coefficients(m, a0, a, b)
        lam = torch.linspace(a0, b, 41, dtype=torch.float64)
        # Run the recurrence on scalars (diagonal F).
        prev = torch.ones_like(lam)
        cur = coef[0, 0] * lam + coef[1, 0]
        for i in range(1, m):
            prev, cur = cur, coef[0, i] * lam * cur + coef[1, i] * cur + coef[2, i] * prev
        x = (lam - 0.5 * (a + b)) / (0.5 * (b - a))
        x0 = (a0 - 0.5 * (a + b)) / (0.5 * (b - a))
        torch.testing.assert_close(cur, _cheb(m, x) / _cheb(m, x0), rtol=1e-10, atol=1e-12)
        assert float(cur[0]) == pytest.approx(1.0)

    def test_rejects_bad_interval(self):
        with pytest.raises(ValueError):
            SS.chebyshev_coefficients(4, 1.0, 0.5, 2.0)

    def test_choose_degree(self):
        theta = torch.tensor([-11.0, -1.0, -0.5, 0.2], dtype=torch.float64)
        assert SS.choose_degree(theta, 3, 30.0) == SS._MAX_DEGREE
        assert SS.choose_degree(theta, 3, 30.0, max_degree=64, amp_max=1e3) < 64
        m = SS.choose_degree(theta, 3, 30.0, max_degree=64, amp_max=1e3)
        a = 0.2
        ratio = math.cosh(m * SS._acosh_abs_x(-11.0, a, 30.0)) / math.cosh(
            m * SS._acosh_abs_x(-0.5, a, 30.0)
        )
        assert ratio <= 1e3
        assert SS.choose_degree(theta, 3, 30.0, max_degree=1, amp_max=1.0) == SS.DEGREES[0]

    def test_table_layout(self):
        coef = SS.chebyshev_coefficients(3, -2.0, 0.0, 1.0)
        theta = torch.tensor([-2.0, -1.0], dtype=torch.float64)
        t = SS.filter_table(coef, theta)
        assert t.dtype == torch.float32 and t.numel() == 3 * 3 + 2
        torch.testing.assert_close(t[-2:], torch.tensor([2.0, 1.0]))


class TestReference:
    def test_matches_definition(self):
        n, kb, m = 30, 6, 5
        A = _spectrum_matrix(torch.linspace(-2, 3, n), seed=2)
        X, _ = torch.linalg.qr(torch.randn(n, kb, dtype=torch.float64))
        theta = torch.linspace(-2, -1, kb, dtype=torch.float64)
        coef = SS.chebyshev_coefficients(m, -2.0, -0.8, 3.1)
        out = SS.filter_reference(A, X, SS.filter_table(coef, theta).double())
        assert out.shape == (n + 6 * kb + 1, kb)
        w, U = torch.linalg.eigh(A)
        x = (w - 0.5 * (-0.8 + 3.1)) / (0.5 * 3.9)
        x0 = (-2.0 - 0.5 * (-0.8 + 3.1)) / (0.5 * 3.9)
        Y = U @ torch.diag(_cheb(m, x) / _cheb(m, x0)) @ U.T @ X
        E = Y - X @ (X.T @ Y)
        torch.testing.assert_close(out[:n], E, rtol=1e-5, atol=1e-6)
        G, H, r = SS.unpack_grams(out[n:], kb)
        Z = torch.cat([X, E], 1)
        torch.testing.assert_close(G, Z.T @ Z, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(H, Z.T @ A @ Z, rtol=1e-5, atol=1e-6)
        R = A @ X - X * theta
        torch.testing.assert_close(r, (R * R).sum(0), rtol=1e-6, atol=1e-9)

    def test_rayleigh_ritz_exact_in_span(self):
        n, k = 20, 4
        A = _spectrum_matrix(torch.arange(n, dtype=torch.float64), seed=3)
        w, U = torch.linalg.eigh(A)
        g = torch.Generator().manual_seed(0)
        mix = torch.randn(k, k, generator=g, dtype=torch.float64)
        Z = torch.cat([U[:, :k] @ mix, torch.zeros(n, 2, dtype=torch.float64)], 1)
        theta, Q = SS.rayleigh_ritz(Z.T @ Z, Z.T @ A @ Z)
        assert theta.numel() == k  # zero columns dropped
        torch.testing.assert_close(theta, w[:k])
        torch.testing.assert_close((Z @ Q).T @ (Z @ Q), torch.eye(k, dtype=torch.float64))

    def test_rayleigh_ritz_augmented_basis(self):
        n, kb = 50, 6
        A = _spectrum_matrix(torch.linspace(-5, 5, n), seed=4)
        g = torch.Generator().manual_seed(1)
        X, _ = torch.linalg.qr(torch.randn(n, kb, generator=g, dtype=torch.float64))
        E = torch.randn(n, kb, generator=g, dtype=torch.float64) * 1e-3
        E = E - X @ (X.T @ E)
        E[:, -1] = 0.0  # a dependent (zero) correction column is dropped
        Z = torch.cat([X, E], 1)
        G, H = Z.T @ Z, Z.T @ A @ Z
        t1, Q1 = SS.rayleigh_ritz(G, H)
        t2, Q2 = SS.rayleigh_ritz(G, H, kb=kb)
        assert t1.numel() == t2.numel() == 2 * kb - 1
        torch.testing.assert_close(t2, t1, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(Q2.T @ G @ Q2, torch.eye(2 * kb - 1, dtype=torch.float64))


@pytest.mark.usefixtures("pytorch_backend")
class TestDriver:
    def test_cold_start_is_exact(self):
        A = _fock_like(60, 9, seed=4)
        w, V, st = trnsolver.occupied_subspace(A, 9)
        wr, Ur = torch.linalg.eigh(A)
        torch.testing.assert_close(w, wr[:9])
        assert _proj_err(V, A, 9) < 1e-10
        assert st.X.shape == (60, 9 + SS.default_buffer(9)) and st.passes == 0

    @pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-6), (torch.float32, 5e-6)])
    def test_warm_start_converges(self, dtype, tol):
        n, k = 120, 20
        A0 = _fock_like(n, k, seed=5, dtype=dtype)
        A1 = _perturb(A0, 2e-2)
        st = trnsolver.occupied_subspace(A0, k).state
        # About 2.7x per degree-8 pass on this spectrum (core at -11, virtuals
        # to +30, HOMO-LUMO gap 0.5); FP32 floors near 1e-6.
        w, V, st1 = trnsolver.occupied_subspace(A1, k, guess=st, passes=12)
        assert w.dtype == dtype and V.shape == (n, k)
        assert _proj_err(V, A1, k) < tol
        wr = torch.linalg.eigvalsh(A1.double())[:k]
        assert float((w.double() - wr).abs().max()) < 10 * tol * float(wr.abs().max())
        torch.testing.assert_close(
            V.double().T @ V.double(), torch.eye(k, dtype=torch.float64), atol=10 * tol, rtol=0
        )
        assert st1.passes == 12 and st1.degree == SS._MAX_DEGREE

    def test_one_pass_contracts(self):
        n, k = 120, 20
        A0 = _fock_like(n, k, seed=6)
        A1 = _perturb(A0, 2e-2)
        st = trnsolver.occupied_subspace(A0, k).state
        before = _proj_err(st.X[:, :k], A1, k)
        _, V, _ = trnsolver.occupied_subspace(A1, k, guess=st, passes=1)
        assert _proj_err(V, A1, k) < 0.3 * before

    def test_deep_core_high_degree_stays_correct(self):
        # The case that breaks a Rayleigh-Ritz on Y^T Y: deep core levels,
        # high degree, FP32. The correction form keeps span(X).
        n, k = 120, 20
        A0 = _fock_like(n, k, seed=7, dtype=torch.float32)
        A1 = _perturb(A0, 1e-2)
        st = trnsolver.occupied_subspace(A0, k).state
        before = _proj_err(st.X[:, :k], A1, k)
        for _ in range(4):
            _, V, st = trnsolver.occupied_subspace(A1, k, guess=st, degree=32)
        assert _proj_err(V, A1, k) < before

    def test_tol_stops_early(self):
        n, k = 80, 10
        A = _fock_like(n, k, seed=8)
        st = trnsolver.occupied_subspace(A, k).state
        w, V, st2 = trnsolver.occupied_subspace(A, k, guess=st, passes=5, tol=1e-8)
        assert st2.passes == 1  # the exact block passes the check before filtering
        assert float(st2.residual[:k].max()) < 1e-8 * 30
        assert _proj_err(V, A, k) < 1e-10

    def test_tensor_guess(self):
        n, k = 80, 10
        A = _fock_like(n, k, seed=9)
        _, U = torch.linalg.eigh(_perturb(A, 5e-2))
        w, V, st = trnsolver.occupied_subspace(A, k, guess=U[:, :k], passes=8)
        assert _proj_err(V, A, k) < 1e-4  # from about 1e-1
        assert st.X.shape[1] == k + SS.default_buffer(k)

    def test_cached_and_explicit_upper(self):
        n, k = 80, 10
        A = _fock_like(n, k, seed=10)
        A1 = _perturb(A, 1e-3)
        st = trnsolver.occupied_subspace(A, k).state
        top = float(torch.linalg.eigvalsh(A1)[-1])
        for upper in ("cached", top + 1.0):
            _, V, s2 = trnsolver.occupied_subspace(A1, k, guess=st, passes=4, upper=upper)
            assert _proj_err(V, A1, k) < 1e-7
        assert s2.upper == pytest.approx(top + 1.0)
        with pytest.raises(ValueError, match="upper"):
            trnsolver.occupied_subspace(A1, k, guess=st, upper="bogus")

    def test_full_block_and_bad_args(self):
        A = _fock_like(12, 6, seed=11)
        w, V, _ = trnsolver.occupied_subspace(A, 6, guess=torch.eye(12))  # kb >= n: eigh
        torch.testing.assert_close(w, torch.linalg.eigvalsh(A)[:6])
        with pytest.raises(ValueError):
            trnsolver.occupied_subspace(A, 0)
        with pytest.raises(ValueError):
            trnsolver.occupied_subspace(A[:, :5], 2)

    def test_degenerate_block_refreshes(self):
        n, k = 40, 4
        A = _fock_like(n, k, seed=12)
        st = trnsolver.occupied_subspace(A, k).state
        st.theta = torch.zeros_like(st.theta)  # a0 == a: cannot build a filter
        w, V, _ = trnsolver.occupied_subspace(A, k, guess=st)
        torch.testing.assert_close(w, torch.linalg.eigvalsh(A)[:k])

    def test_bf16_input_is_upcast(self):
        A = _fock_like(40, 5, seed=13, dtype=torch.float32).to(torch.bfloat16)
        w, V, _ = trnsolver.occupied_subspace(A, 5)
        assert w.dtype == torch.float32


def test_kernel_fits():
    assert SK.kernel_fits(970, 185)
    assert SK.kernel_fits(1024, 512)
    assert not SK.kernel_fits(1024, 513)
    assert not SK.kernel_fits(4096, 64)
    assert not SK.kernel_fits(10, 11)
    assert SK.layout(970, 185) == (8, 2)


# ----------------------------------------------------------------------
# NKI simulator and MLIR verifier
# ----------------------------------------------------------------------


def _sim_enabled():
    return HAS_NKI and os.environ.get("TRNSOLVER_USE_SIMULATOR", "").lower() in ("1", "true", "yes")


def _case(n, kb, m, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    F = (A + A.T) / (2 * math.sqrt(n))
    X, _ = torch.linalg.qr(torch.randn(n, kb, generator=g))
    w = torch.linalg.eigvalsh(F.double())
    theta = torch.linspace(float(w[0]), float(w[kb - 1]), kb, dtype=torch.float64)
    coef = SS.chebyshev_coefficients(m, float(w[0]), float(w[kb]), float(w[-1]) + 0.1)
    return F, X, SS.filter_table(coef, theta)


@pytest.mark.nki_simulator
class TestSimulator:
    @pytest.fixture(autouse=True)
    def _need_sim(self):
        if not _sim_enabled():
            pytest.skip("TRNSOLVER_USE_SIMULATOR=1 and nki required")

    @pytest.mark.parametrize(
        "n,kb,m", [(64, 16, 1), (100, 24, 2), (130, 120, 3), (256, 160, 4), (200, 8, 8)]
    )
    def test_kernel_matches_reference(self, n, kb, m):
        F, X, table = _case(n, kb, m, seed=n)
        ref = SS.filter_reference(F.double(), X.double(), table.double())
        got = SK.run_filter_kernel(F, X, table).double()
        assert got.shape == ref.shape
        err = float((got - ref).abs().max() / ref.abs().max())
        assert err < 1e-5

    def test_driver_uses_kernel(self, monkeypatch):
        calls = []
        real = SK.run_filter_kernel

        def spy(*a):
            calls.append(a[1].shape)
            return real(*a)

        monkeypatch.setattr(SK, "run_filter_kernel", spy)
        n, k = 150, 12
        A0 = _fock_like(n, k, seed=14, dtype=torch.float32)
        A1 = _perturb(A0, 1e-2)
        st = trnsolver.occupied_subspace(A0, k).state
        w, V, _ = trnsolver.occupied_subspace(A1, k, guess=st, passes=4)
        assert len(calls) == 4 and calls[0] == (n, k + SS.default_buffer(k))
        # Same iteration on the PyTorch reference filter.
        trnsolver.set_backend("pytorch")
        try:
            w_ref, V_ref, _ = trnsolver.occupied_subspace(A1, k, guess=st, passes=4)
        finally:
            trnsolver.set_backend("auto")
        assert len(calls) == 4
        torch.testing.assert_close(w, w_ref, rtol=0, atol=1e-4)
        assert _proj_err(V, A1, k) < 1.5 * _proj_err(V_ref, A1, k) + 1e-5


@pytest.mark.nki_simulator
@pytest.mark.parametrize(
    "n,kb,m", [(64, 16, 1), (130, 56, 2), (250, 56, 8), (512, 96, 8), (970, 185, 8)]
)
@pytest.mark.parametrize("lnc", [1, 2])
def test_mlir_verifies(n, kb, m, lnc):
    if not HAS_NKI:
        pytest.skip("nki not importable on this host")
    try:
        from nki.compiler.frontend import FrontendKind, source_to_mlir
    except ImportError:
        pytest.skip("nki.compiler.frontend.source_to_mlir not available in this nki")
    source_to_mlir(
        SK.subspace_filter_kernel,
        np.zeros((n, n), np.float32),
        np.zeros((n, kb), np.float32),
        np.zeros((128, 3 * m + kb), np.float32),
        np.zeros((m, 1), np.float32),
        target="trn2",
        lnc=lnc,
        frontend=FrontendKind.TRACER,
    ).verify()


# ----------------------------------------------------------------------
# Hardware (native TorchNeuron path)
# ----------------------------------------------------------------------


def _native():
    if not HAS_NKI or _sim_enabled():
        return False
    from trnsolver.nki.dispatch import _get_dispatch_path

    try:
        return _get_dispatch_path() == "native"
    except Exception:
        return False


@pytest.mark.neuron
class TestNeuron:
    @pytest.fixture(autouse=True)
    def _need_native(self, monkeypatch):
        if not _native():
            pytest.skip("requires TorchNeuron (native) dispatch path")
        monkeypatch.setattr(SS, "_REQUIRE_NKI", True)

    @pytest.mark.parametrize("n,kb,m", [(64, 16, 1), (250, 56, 8), (490, 101, 8), (970, 185, 8)])
    def test_kernel_matches_reference(self, n, kb, m):
        F, X, table = _case(n, kb, m, seed=n)
        ref = SS.filter_reference(F.double(), X.double(), table.double())
        got = SK.run_filter_kernel(F.to("neuron"), X.to("neuron"), table).cpu().double()
        err = float((got - ref).abs().max() / ref.abs().max())
        assert err < 1e-4

    @pytest.mark.parametrize("n,k", [(250, 41), (490, 81)])
    def test_occupied_subspace_on_device(self, n, k):
        # A 1e-2 perturbation is large for this spectrum (start error ~0.1),
        # and a degree-8 pass contracts it about 2-3x, so 4 passes reach
        # ~1e-2 on any backend (Trn2: 6.1e-3 / 1.4e-2 at n = 250 / 490; the
        # host reference filter agrees to < 1%). Check the device run
        # against the same calls on the host, and that it contracts.
        A0 = _fock_like(n, k, seed=n, dtype=torch.float32)
        A1 = _perturb(A0, 1e-2)
        st = trnsolver.occupied_subspace(A0.to("neuron"), k).state
        before = _proj_err(st.X[:, :k].cpu(), A1, k)
        w, V, st = trnsolver.occupied_subspace(A1.to("neuron"), k, guess=st, passes=4)
        assert V.device.type == "neuron"
        st_h = trnsolver.occupied_subspace(A0, k).state
        w_h, V_h, _ = trnsolver.occupied_subspace(A1, k, guess=st_h, passes=4)
        err, err_h = _proj_err(V.cpu(), A1, k), _proj_err(V_h, A1, k)
        assert err < 0.2 * before
        assert abs(err - err_h) <= 0.05 * err_h + 1e-5
        scale = float(torch.linalg.eigvalsh(A1.double()).abs().max())
        assert float((w.cpu().double() - w_h.double()).abs().max()) < 1e-4 * scale
