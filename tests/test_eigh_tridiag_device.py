"""Single-launch on-device Householder tridiagonalization (#42).

Three layers:

* CPU-only: the ``TRNSOLVER_EIGH_TRIDIAG`` switch and the eigh wiring. These
  run in the default ``-m "not neuron"`` job, with or without ``nki``.
* ``nki_simulator``: kernel correctness through ``nki.simulate`` against
  ``torch.linalg.eigh`` at n = 8 .. 128, and the NKI 0.7 MLIR verifier for
  trn1 and trn2 (the simulator does not enforce partition-start or
  partition-count rules).
* ``neuron``: the same contract on hardware.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import trnsolver
import trnsolver.eigen as eigen
from trnsolver.nki import tridiag
from trnsolver.nki.dispatch import HAS_NKI

SIZES = [8, 16, 32, 64, 100, 128]


def _sym(n, seed=42):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    return 0.5 * (A + A.T)


def _fix_signs(V):
    """Flip each column so its largest-magnitude entry is positive."""
    idx = V.abs().argmax(dim=0)
    signs = torch.sign(V[idx, torch.arange(V.shape[1])])
    signs[signs == 0] = 1.0
    return V * signs


def _q_from_reflectors(V_refs):
    """Q1 = H_0 H_1 ... with H_k = I - 2 v_k v_k^T, in float64."""
    n = V_refs.shape[0]
    Q = torch.eye(n, dtype=torch.float64)
    for k in range(V_refs.shape[1]):
        v = V_refs[:, k].double()
        Q = Q - 2.0 * torch.outer(Q @ v, v)
    return Q


def _householder_fp64(A):
    """Reference reduction in FP64 with the host loop's sign convention."""
    A = A.double().clone()
    n = A.shape[0]
    for k in range(n - 2):
        x = A[k + 1 :, k].clone()
        nx = torch.linalg.norm(x)
        if nx == 0:
            continue
        v = x.clone()
        v[0] += nx if x[0] >= 0 else -nx
        v /= torch.linalg.norm(v)
        H = torch.eye(n, dtype=torch.float64)
        H[k + 1 :, k + 1 :] -= 2.0 * torch.outer(v, v)
        A = H @ A @ H
    return torch.diagonal(A), torch.diagonal(A, 1)


def _tridiag_matrix(d, e):
    return torch.diag(d.double()) + torch.diag(e.double(), 1) + torch.diag(e.double(), -1)


def _check_tridiag(A, d, e, V_refs):
    n = A.shape[0]
    assert d.shape == (n,)
    assert e.shape == (n - 1,)
    assert V_refs.shape == (n, max(n - 1, 1))
    Q = _q_from_reflectors(V_refs)
    T = _tridiag_matrix(d, e)
    np.testing.assert_allclose((Q.T @ A.double() @ Q).numpy(), T.numpy(), atol=2e-5)
    np.testing.assert_allclose((Q.T @ Q).numpy(), np.eye(n), atol=2e-5)


def _check_eigh(A, w, V):
    w_ref, V_ref = torch.linalg.eigh(A.double())
    np.testing.assert_allclose(w.double().numpy(), w_ref.numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(_fix_signs(V.double()).numpy(), _fix_signs(V_ref).numpy(), atol=2e-3)
    resid = (A.double() @ V.double() - V.double() * w.double()).abs().max().item()
    assert resid < 1e-4


# ----------------------------------------------------------------------
# CPU-only: switch and wiring
# ----------------------------------------------------------------------


class TestSwitch:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("TRNSOLVER_EIGH_TRIDIAG", raising=False)
        assert not tridiag.use_device_tridiag(16)

    @pytest.mark.parametrize("value", ["device", "DEVICE", " device "])
    def test_on(self, monkeypatch, value):
        monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG", value)
        assert tridiag.use_device_tridiag(16)

    @pytest.mark.parametrize("value", ["", "host", "1", "on"])
    def test_other_values_off(self, monkeypatch, value):
        monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG", value)
        assert not tridiag.use_device_tridiag(16)

    @pytest.mark.parametrize("n,expected", [(1, False), (2, True), (128, True), (129, False)])
    def test_single_tile_only(self, monkeypatch, n, expected):
        monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG", "device")
        assert tridiag.use_device_tridiag(n) is expected

    def test_constants(self):
        c = tridiag.tridiag_constants(5)
        assert c.shape == (5, 10)
        torch.testing.assert_close(c[:, :5], torch.eye(5))
        torch.testing.assert_close(c[3, 5:], torch.arange(5, dtype=torch.float32))


class _Called(Exception):
    pass


class TestWiring:
    """_householder_qr_eigh picks the reduction from the switch."""

    def _spies(self, monkeypatch):
        def host(A):
            raise _Called("host")

        def device(A):
            raise _Called("device")

        monkeypatch.setattr(eigen, "_householder_tridiag", host)
        monkeypatch.setattr(eigen, "_householder_tridiag_device", device)

    def test_device_when_set(self, monkeypatch):
        self._spies(monkeypatch)
        monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG", "device")
        with pytest.raises(_Called, match="device"):
            eigen._householder_qr_eigh(_sym(16))

    def test_host_by_default(self, monkeypatch):
        self._spies(monkeypatch)
        monkeypatch.delenv("TRNSOLVER_EIGH_TRIDIAG", raising=False)
        with pytest.raises(_Called, match="host"):
            eigen._householder_qr_eigh(_sym(16))

    def test_host_above_single_tile(self, monkeypatch):
        self._spies(monkeypatch)
        monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG", "device")
        with pytest.raises(_Called, match="host"):
            eigen._householder_qr_eigh(_sym(130))


# ----------------------------------------------------------------------
# Simulator + MLIR verifier
# ----------------------------------------------------------------------


def _simulator_on():
    return os.environ.get("TRNSOLVER_USE_SIMULATOR", "").lower() in ("1", "true", "yes")


@pytest.mark.nki_simulator
class TestDeviceTridiagSimulator:
    @pytest.fixture(autouse=True)
    def _need_sim(self, monkeypatch):
        if not _simulator_on():
            pytest.skip("TRNSOLVER_USE_SIMULATOR=1 required")
        if not HAS_NKI:
            pytest.skip("nki package not importable on this host")
        monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG", "device")

    @pytest.mark.parametrize("n", [2, 3] + SIZES)
    def test_reduction(self, n):
        """Q1^T A Q1 = T with Q1 from the returned unit-norm reflectors."""
        A = _sym(n)
        d, e, V_refs = tridiag.householder_tridiag_device(A)
        _check_tridiag(A, d, e, V_refs)

    @pytest.mark.parametrize("n", SIZES)
    def test_matches_fp64_reduction(self, n):
        """Same sign convention as the host loop, so (d, e) match an FP64 reduction.

        The tridiagonal's forward error is conditioning-limited: two FP32
        reductions (this kernel, the host loop) both land 1e-4 to 8e-4 from
        the FP64 one at n = 64..128, in no consistent order. The similarity
        check in test_reduction is the tight one.
        """
        A = _sym(n)
        d, e, _ = tridiag.householder_tridiag_device(A)
        d_ref, e_ref = _householder_fp64(A)
        np.testing.assert_allclose(d.double().numpy(), d_ref.numpy(), atol=2e-3)
        np.testing.assert_allclose(e.double().numpy(), e_ref.numpy(), atol=2e-3)

    @pytest.mark.parametrize("n", SIZES)
    def test_eigh_vs_torch(self, monkeypatch, n):
        """Public eigh with the device reduction: eigenvalues and eigenvectors."""
        # Host reduction must not run, and a kernel failure must not fall back.
        monkeypatch.setattr(eigen, "_householder_tridiag", None)
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", True)
        A = _sym(n)
        w, V = trnsolver.eigh(A)
        _check_eigh(A, w, V)

    def test_zero_column(self):
        """An already-reduced column gives a zero reflector, not NaN."""
        A = torch.diag(torch.arange(1.0, 9.0))
        d, e, V_refs = tridiag.householder_tridiag_device(A)
        assert torch.isfinite(V_refs).all()
        torch.testing.assert_close(d, torch.arange(1.0, 9.0))
        torch.testing.assert_close(e, torch.zeros(7))


@pytest.mark.nki_simulator
@pytest.mark.parametrize("target", ["trn1", "trn2"])
@pytest.mark.parametrize("n", [2, 8, 16, 32, 100, 128])
def test_mlir_verifies(n, target):
    if not HAS_NKI:
        pytest.skip("nki not importable on this host")
    try:
        from nki.compiler.frontend import FrontendKind, source_to_mlir
    except ImportError:
        pytest.skip("nki.compiler.frontend.source_to_mlir not available in this nki")
    A = np.zeros((n, n), dtype=np.float32)
    consts = tridiag.tridiag_constants(n).numpy()
    source_to_mlir(
        tridiag.householder_tridiag_kernel,
        A,
        consts,
        target=target,
        frontend=FrontendKind.TRACER,
    ).verify()


# ----------------------------------------------------------------------
# Hardware
# ----------------------------------------------------------------------


@pytest.mark.neuron
class TestDeviceTridiagNeuron:
    @pytest.fixture(autouse=True)
    def _device_tridiag(self, monkeypatch):
        monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG", "device")
        monkeypatch.setenv("TRNSOLVER_EIGH_NKI_MIN_N", "0")

    @pytest.mark.parametrize("n", SIZES)
    def test_reduction(self, n):
        A = _sym(n)
        d, e, V_refs = tridiag.householder_tridiag_device(A)
        _check_tridiag(A, d.cpu(), e.cpu(), V_refs.cpu())

    @pytest.mark.parametrize("n", SIZES)
    def test_eigh_vs_torch(self, monkeypatch, n):
        # Host reduction must not run, and a kernel failure must not fall back.
        monkeypatch.setattr(eigen, "_householder_tridiag", None)
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", True)
        A = _sym(n)
        w, V = trnsolver.eigh(A)
        _check_eigh(A, w.cpu(), V.cpu())
