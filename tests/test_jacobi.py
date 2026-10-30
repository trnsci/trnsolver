"""Batched Jacobi eigh: ``trnsolver.eigh(A, method="jacobi")`` (#48).

Layers:

* CPU: the round-robin schedules and step masks, the compact tile layout,
  the torch Jacobi path against ``torch.linalg.eigh``, and the device host
  driver (`nki.jacobi.jacobi_eigh_device`) with the kernel replaced by a
  torch emulation of one launch. These run in ``pytest -m "not neuron"``.
* ``nki_simulator``: the kernel through ``nki.simulate`` against the same
  emulation, end-to-end eigh in the simulator, and the NKI 0.7 MLIR
  verifier (the simulator does not check partition or ISA-version rules).
* ``neuron``: the kernel path on the native ``neuron`` device.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import trnsolver
import trnsolver.eigen as eigen
from trnsolver import _jacobi as J
from trnsolver.nki import jacobi as JK
from trnsolver.nki.dispatch import HAS_NKI


def _sym_batch(nb, n, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(nb, n, n, generator=g, dtype=torch.float64)
    return (0.5 * (A + A.transpose(1, 2))).to(dtype)


def _separated(nb, n, seed=0, gap=0.5):
    """Random orthogonal Q diag(w) Q^T with eigenvalue gaps >= gap."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(nb, n, n, generator=g, dtype=torch.float64))
    w = gap * torch.arange(n, dtype=torch.float64) - gap * n / 2
    A = Q @ torch.diag_embed(w.expand(nb, n)) @ Q.transpose(1, 2)
    return (0.5 * (A + A.transpose(1, 2))).float()


def _fix_signs(V):
    """Flip each column so its largest-magnitude entry is positive."""
    idx = V.abs().argmax(dim=-2, keepdim=True)
    signs = torch.sign(torch.gather(V, -2, idx))
    signs[signs == 0] = 1.0
    return V * signs


def _check(A, w, V, *, tol=2e-5, vecs=True):
    """Eigenvalues, residual, orthogonality, eigenvectors up to sign (FP64 ref)."""
    A64 = A.double().reshape(-1, A.shape[-1], A.shape[-1])
    n = A64.shape[-1]
    w64 = w.double().reshape(-1, n)
    V64 = V.double().reshape(-1, n, n)
    wr, Vr = torch.linalg.eigh(A64)
    scale = max(float(wr.abs().max()), 1e-30)
    np.testing.assert_allclose(w64.numpy(), wr.numpy(), atol=tol * scale, rtol=0)
    res = (A64 @ V64 - V64 * w64.unsqueeze(-2)).norm(dim=(1, 2)) / A64.norm(dim=(1, 2)).clamp(
        min=1e-30
    )
    assert float(res.max()) < tol * 4
    orth = (V64.transpose(1, 2) @ V64 - torch.eye(n, dtype=torch.float64)).abs().max()
    assert float(orth) < tol * 4
    if vecs:
        np.testing.assert_allclose(
            _fix_signs(V64).numpy(), _fix_signs(Vr).numpy(), atol=max(tol * 50, 1e-3)
        )


# ----------------------------------------------------------------------
# Schedules and layout
# ----------------------------------------------------------------------


class TestSchedules:
    @pytest.mark.parametrize("n", [2, 4, 8, 18, 64])
    def test_round_robin_covers_every_pair_once(self, n):
        partner, sigma = J.round_robin_schedule(n)
        seen = set()
        for r in range(n - 1):
            p = partner[r]
            assert torch.equal(p[p], torch.arange(n))  # involution
            assert bool((p != torch.arange(n)).all())
            for i in range(n):
                if i < int(p[i]):
                    seen.add((i, int(p[i])))
            assert torch.equal(sigma[r], torch.where(torch.arange(n) < p, 1.0, -1.0))
        assert len(seen) == n * (n - 1) // 2

    @pytest.mark.parametrize("n", [2, 3, 8, 32, 33, 128])
    def test_kernel_step_codes(self, n):
        ne, b, P = JK.block_layout(n)
        consts = JK.jacobi_constants(n)
        m = ne - 1
        sig, Q = consts[:, :m], consts[:, m:]
        assert Q.shape == (P, P)
        cols = torch.arange(P)
        seen = set()
        for r, v in enumerate(JK.step_targets(ne)):
            Pi = torch.eq(Q, v).float()
            assert torch.equal(Pi, Pi.T)
            assert torch.equal(Pi.sum(1), torch.ones(P))  # one partner each
            partner = (Pi * cols).sum(1).long()
            assert bool(((partner // ne) == (cols // ne)).all())  # within a block
            assert bool((partner != cols).all())
            assert torch.equal(sig[:, r], -sig[partner, r])
            assert torch.equal(sig[:, r], torch.where(cols < partner, 1.0, -1.0))
            for i in range(ne):
                seen.add((min(i, int(partner[i])), max(i, int(partner[i]))))
        assert len(seen) == ne * (ne - 1) // 2

    @pytest.mark.parametrize(
        "n,expected", [(1, (2, 64, 128)), (2, (2, 64, 128)), (3, (4, 32, 128)), (32, (32, 4, 128))]
    )
    def test_block_layout(self, n, expected):
        assert JK.block_layout(n) == expected

    def test_block_layout_rejects_large_n(self):
        with pytest.raises(NotImplementedError):
            JK.block_layout(129)

    @pytest.mark.parametrize("n,nb", [(5, 3), (32, 9), (64, 2)])
    def test_pack_roundtrip(self, n, nb):
        A = _sym_batch(nb, n)
        tiles = JK.pack_compact(A, n)
        ne, b, P = JK.block_layout(n)
        assert tiles.shape == (-(-nb // b), P, ne)
        back = JK.unpack_compact(tiles, n, nb)
        torch.testing.assert_close(back[:, :n, :n], A)
        assert float(back[:, n:, :].abs().sum()) == 0.0

    def test_group_plan(self, monkeypatch):
        monkeypatch.delenv("TRNSOLVER_JACOBI_GROUP", raising=False)
        assert JK.group_plan(3, 128) == [(0, 3, 4)]
        assert JK.group_plan(40, 128) == [(0, 16, 16), (16, 16, 16), (32, 8, 8)]
        assert JK.group_plan(300, 32) == [(0, 128, 128), (128, 128, 128), (256, 44, 64)]
        monkeypatch.setenv("TRNSOLVER_JACOBI_GROUP", "3")  # rounded down to 2
        assert JK.group_plan(5, 32) == [(0, 2, 2), (2, 2, 2), (4, 1, 1)]


# ----------------------------------------------------------------------
# Torch Jacobi path (CPU)
# ----------------------------------------------------------------------


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv("TRNSOLVER_EIGH_METHOD", raising=False)
    monkeypatch.delenv("TRNSOLVER_JACOBI_GROUP", raising=False)
    monkeypatch.delenv("TRNSOLVER_JACOBI_CHAINS", raising=False)


class TestTorchJacobi:
    @pytest.mark.parametrize("n", [2, 3, 8, 17, 32, 64])
    @pytest.mark.parametrize("nb", [1, 4])
    def test_matches_eigh(self, no_env, n, nb):
        A = _separated(nb, n, seed=n)
        w, V = trnsolver.eigh(A, method="jacobi")
        assert w.shape == (nb, n) and V.shape == (nb, n, n)
        _check(A, w, V)

    def test_random_symmetric(self, no_env):
        A = _sym_batch(6, 48)
        w, V = trnsolver.eigh(A, method="jacobi")
        _check(A, w, V, vecs=False)

    def test_two_dim_input(self, no_env):
        A = _separated(1, 12)[0]
        w, V = trnsolver.eigh(A, method="jacobi")
        assert w.shape == (12,) and V.shape == (12, 12)
        _check(A, w, V)

    def test_extra_batch_dims(self, no_env):
        A = _separated(6, 9).reshape(2, 3, 9, 9)
        w, V = trnsolver.eigh(A, method="jacobi")
        assert w.shape == (2, 3, 9) and V.shape == (2, 3, 9, 9)
        _check(A, w, V)

    def test_float64_is_tight(self, no_env):
        A = _sym_batch(3, 20, dtype=torch.float64)
        w, V = trnsolver.eigh(A, method="jacobi")
        assert w.dtype == torch.float64
        _check(A, w, V, tol=1e-12, vecs=False)

    def test_bfloat16_roundtrip(self, no_env):
        A = _separated(2, 8).to(torch.bfloat16)
        w, V = trnsolver.eigh(A, method="jacobi")
        assert w.dtype == torch.bfloat16 and V.dtype == torch.bfloat16

    def test_lower_triangle_is_used(self, no_env):
        A = _separated(2, 10)
        garbage = torch.triu(torch.randn(2, 10, 10), 1)
        w, V = trnsolver.eigh(torch.tril(A) + garbage, method="jacobi")
        _check(A, w, V)

    @pytest.mark.parametrize("kind", ["identity", "repeated", "clustered"])
    def test_degenerate_spectra(self, no_env, kind):
        n = 16
        Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
        if kind == "identity":
            w0 = torch.ones(n, dtype=torch.float64)
        elif kind == "repeated":
            w0 = torch.tensor([1.0] * 8 + [3.0] * 8, dtype=torch.float64)
        else:
            w0 = torch.tensor([1.0] * 8 + [2.0] * 8, dtype=torch.float64)
            w0 = w0 + 1e-4 * torch.arange(n, dtype=torch.float64)
        A = (Q @ torch.diag(w0) @ Q.T).float()
        w, V = trnsolver.eigh(A, method="jacobi")
        _check(A, w, V, vecs=False)

    def test_one_by_one_and_empty(self, no_env):
        A = torch.tensor([[[3.0]], [[-2.0]]])
        w, V = trnsolver.eigh(A, method="jacobi")
        torch.testing.assert_close(w, torch.tensor([[3.0], [-2.0]]))
        torch.testing.assert_close(V, torch.ones(2, 1, 1))
        w, V = trnsolver.eigh(torch.empty(0, 4, 4), method="jacobi")
        assert w.shape == (0, 4) and V.shape == (0, 4, 4)

    def test_env_selects_jacobi(self, no_env, monkeypatch):
        calls = []
        real = eigen._jacobi_eigh
        monkeypatch.setattr(eigen, "_jacobi_eigh", lambda A: calls.append(1) or real(A))
        monkeypatch.setenv("TRNSOLVER_EIGH_METHOD", "jacobi")
        trnsolver.eigh(_separated(1, 6)[0])
        assert calls == [1]
        monkeypatch.setenv("TRNSOLVER_EIGH_METHOD", "auto")
        trnsolver.eigh(_separated(1, 6)[0])
        assert calls == [1]

    def test_invalid_method(self, no_env):
        with pytest.raises(ValueError):
            trnsolver.eigh(torch.eye(3), method="qr")

    def test_non_square_rejected(self, no_env):
        with pytest.raises(ValueError):
            trnsolver.eigh(torch.zeros(2, 3, 4), method="jacobi")

    def test_default_dispatch_unchanged(self, no_env, monkeypatch):
        monkeypatch.setattr(
            eigen, "_jacobi_eigh", lambda A: pytest.fail("auto must not route to Jacobi")
        )
        A = _separated(1, 6)[0]
        trnsolver.eigh(A)

    def test_sweep_counts(self):
        # Random symmetric FP32 matrices converge in 6-8 sweeps (see #48).
        for n, lo, hi in [(32, 5, 8), (64, 6, 9), (128, 7, 10)]:
            _, _, sweeps = J.jacobi_eigh_torch(_sym_batch(4, n, seed=n))
            assert lo <= sweeps <= hi, (n, sweeps)


# ----------------------------------------------------------------------
# Device host driver with the kernel emulated in torch (CPU)
# ----------------------------------------------------------------------


def _emulate_launch(Z, V, n, chains):
    """Torch model of one `jacobi_sweeps_kernel` launch (one sweep)."""
    ne, b, P = JK.block_layout(n)
    consts = JK.jacobi_constants(n)
    m = ne - 1
    sig, Q = consts[:, :m], consts[:, m:]
    X = torch.cat([Z, V], dim=-1)  # (G, P, 2 ne)
    for r, v in enumerate(JK.step_targets(ne)):
        Pi = torch.eq(Q, v).to(X.dtype)
        M = torch.einsum("kp,gkc->gpc", Pi, X)
        Zc = X[..., :ne]
        gpp = (Zc * Zc).sum(-1)
        gpq = (M[..., :ne] * Zc).sum(-1)
        gqq = gpp @ Pi
        _, s, k = J.rotation_csk(gpp, gqq, gpq, sig[:, r])
        X = X + (k.unsqueeze(-1) * X - s.unsqueeze(-1) * M)
    return torch.stack([X[..., :ne], X[..., ne:]])


@pytest.fixture
def emulated_kernel(monkeypatch, no_env):
    launches = []

    def run(Z, V, n, chains):
        launches.append((tuple(Z.shape), chains))
        return _emulate_launch(Z, V, n, chains)

    monkeypatch.setattr(JK, "_run_kernel", run)
    monkeypatch.setattr(JK, "_use_simulator", lambda: True)
    monkeypatch.setattr(eigen, "_jacobi_device_ok", lambda n, dtype: dtype == torch.float32)
    return launches


class TestDeviceDriverEmulated:
    @pytest.mark.parametrize("n,nb", [(2, 3), (5, 7), (8, 1), (32, 9), (33, 5), (64, 3)])
    def test_matches_eigh(self, emulated_kernel, n, nb):
        A = _separated(nb, n, seed=n + nb)
        w, V = trnsolver.eigh(A, method="jacobi")
        _check(A, w, V)
        assert JK._SWEEPS0[JK.block_layout(n)[0]] <= JK.last_sweeps <= JK._MAX_SWEEPS

    def test_random_n128(self, emulated_kernel):
        A = _sym_batch(2, 128)
        w, V = trnsolver.eigh(A, method="jacobi")
        _check(A, w, V, vecs=False)
        assert 8 <= JK.last_sweeps <= 10

    def test_groups_and_tail_padding(self, emulated_kernel, monkeypatch):
        monkeypatch.setenv("TRNSOLVER_JACOBI_GROUP", "2")
        A = _separated(5 * 32 + 3, 4, seed=3)  # 32 matrices per tile -> 6 tiles
        w, V = trnsolver.eigh(A, method="jacobi")
        _check(A, w, V)
        shapes = {s for s, _ in emulated_kernel}
        assert shapes == {(2, 128, 4)}  # 6 tiles = 3 groups of 2

    def test_scale_invariance(self, emulated_kernel):
        A = _separated(3, 16)
        for s in (1e-8, 1.0, 1e8):
            w, V = trnsolver.eigh(A * s, method="jacobi")
            _check(A * s, w, V)

    def test_indefinite_pm_pairs(self, emulated_kernel):
        # +-lambda pairs are degenerate for A^2; the Gershgorin shift keeps
        # one-sided Jacobi on A's ordering.
        n = 16
        Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
        w0 = torch.cat([torch.arange(1.0, 9.0), -torch.arange(1.0, 9.0)]).double()
        A = (Q @ torch.diag(w0) @ Q.T).float().unsqueeze(0)
        w, V = trnsolver.eigh(A, method="jacobi")
        _check(A, w, V)

    def test_kernel_failure_falls_back(self, emulated_kernel, monkeypatch):
        def boom(*a):
            raise RuntimeError("compile failed")

        monkeypatch.setattr(JK, "_run_kernel", boom)
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", False)
        A = _separated(2, 8)
        w, V = trnsolver.eigh(A, method="jacobi")
        _check(A, w, V)
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", True)
        with pytest.raises(RuntimeError):
            trnsolver.eigh(A, method="jacobi")

    def test_float64_skips_kernel(self, emulated_kernel):
        trnsolver.eigh(_sym_batch(2, 8, dtype=torch.float64), method="jacobi")
        assert emulated_kernel == []


# ----------------------------------------------------------------------
# NKI simulator and MLIR verifier
# ----------------------------------------------------------------------


def _sim_enabled():
    return HAS_NKI and os.environ.get("TRNSOLVER_USE_SIMULATOR", "").lower() in ("1", "true", "yes")


@pytest.mark.nki_simulator
class TestSimulator:
    @pytest.fixture(autouse=True)
    def _need_sim(self, no_env):
        if not _sim_enabled():
            pytest.skip("TRNSOLVER_USE_SIMULATOR=1 and nki required")

    @pytest.mark.parametrize("n,G,chains", [(4, 1, 1), (8, 2, 2), (16, 1, 1), (5, 3, 2)])
    def test_kernel_matches_emulation(self, n, G, chains):
        ne, b, P = JK.block_layout(n)
        A = _sym_batch(G * b, n)
        sigma = A.abs().sum(-1).amax(-1)
        Z = JK.pack_compact(A + sigma[:, None, None] * torch.eye(n), n)
        V = JK.pack_compact(torch.eye(n).expand(G * b, n, n), n)
        got = JK._run_kernel(Z, V, n, chains)
        ref = _emulate_launch(Z, V, n, chains)
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("n,nb", [(3, 2), (8, 20)])
    def test_eigh_end_to_end(self, n, nb):
        A = _separated(nb, n, seed=nb)
        w, V = trnsolver.eigh(A, method="jacobi")
        _check(A, w, V)


@pytest.mark.nki_simulator
@pytest.mark.parametrize(
    "n,G,chains",
    [(2, 1, 1), (8, 2, 2), (32, 4, 2), (33, 2, 1), (64, 2, 2), (128, 1, 1), (128, 8, 2)],
)
def test_mlir_verifies(n, G, chains):
    if not HAS_NKI:
        pytest.skip("nki not importable on this host")
    try:
        from nki.compiler.frontend import FrontendKind, source_to_mlir
    except ImportError:
        pytest.skip("nki.compiler.frontend.source_to_mlir not available in this nki")
    ne, b, P = JK.block_layout(n)
    Z = np.zeros((G, P, ne), dtype=np.float32)
    source_to_mlir(
        JK.jacobi_sweeps_kernel,
        Z,
        Z,
        JK.jacobi_constants(n).numpy(),
        np.zeros((1, ne, chains), dtype=np.float32),
        target="trn2",
        frontend=FrontendKind.TRACER,
    ).verify()


# ----------------------------------------------------------------------
# Hardware (native TorchNeuron path)
# ----------------------------------------------------------------------


def _native():
    if not HAS_NKI or _sim_enabled():
        return False
    from trnsolver.nki.dispatch import _get_dispatch_path

    return _get_dispatch_path() == "native"


@pytest.mark.neuron
class TestJacobiNeuron:
    @pytest.fixture(autouse=True)
    def _need_native(self, no_env, monkeypatch):
        if not _native():
            pytest.skip("requires TorchNeuron (native) dispatch path")
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", True)

    @pytest.mark.parametrize("n,nb", [(8, 1), (8, 100), (32, 16), (33, 40), (64, 9), (128, 3)])
    def test_eigh_vs_torch(self, n, nb):
        A = _separated(nb, n, seed=n)
        w, V = trnsolver.eigh(A, method="jacobi")
        assert w.device.type == "cpu"
        _check(A, w, V, tol=5e-5)

    def test_random_symmetric(self):
        A = _sym_batch(64, 32)
        w, V = trnsolver.eigh(A, method="jacobi")
        _check(A, w, V, tol=5e-5, vecs=False)
        assert 6 <= JK.last_sweeps <= 9

    def test_device_resident_input(self):
        A = _separated(16, 32)
        w, V = trnsolver.eigh(A.to("neuron"), method="jacobi")
        assert w.device.type == "neuron" and V.device.type == "neuron"
        _check(A, w.cpu(), V.cpu(), tol=5e-5)

    def test_multi_group(self, monkeypatch):
        monkeypatch.setenv("TRNSOLVER_JACOBI_GROUP", "2")
        A = _separated(5 * 4 + 1, 32, seed=7)  # 6 tiles -> 3 launches per sweep
        w, V = trnsolver.eigh(A, method="jacobi")
        _check(A, w, V, tol=5e-5)
