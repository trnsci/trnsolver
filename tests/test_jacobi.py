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
        monkeypatch.setenv("TRNSOLVER_JACOBI_BLOCK", "0")
        assert JK.group_plan(3, 128) == [(0, 3, 4)]
        assert JK.group_plan(40, 128) == [(0, 16, 16), (16, 16, 16), (32, 8, 8)]
        monkeypatch.delenv("TRNSOLVER_JACOBI_BLOCK")
        assert JK.group_plan(200, 128) == [(0, 128, 128), (128, 72, 128)]
        assert JK.group_plan(300, 32) == [(0, 128, 128), (128, 128, 128), (256, 44, 64)]
        monkeypatch.setenv("TRNSOLVER_JACOBI_GROUP", "3")  # rounded down to 2
        assert JK.group_plan(5, 32) == [(0, 2, 2), (2, 2, 2), (4, 1, 1)]


class TestBlockConstants:
    @pytest.mark.parametrize("nb", [2, 4, 8, 16])
    def test_circle_perm_meets_every_block_pair(self, nb):
        pos = list(range(nb))  # pos[p] = block at position p
        met = set()
        for _ in range(nb - 1):
            for t in range(nb // 2):
                met.add(frozenset((pos[2 * t], pos[2 * t + 1])))
            perm = JK.circle_block_perm(nb)
            pos = [pos[perm[p]] for p in range(nb)]
        assert len(met) == nb * (nb - 1) // 2
        assert pos == list(range(nb))  # order nb - 1: back in place

    @pytest.mark.parametrize("n", [33, 64, 128])
    def test_layout(self, n):
        ne, b, P = JK.block_layout(n)
        bs = JK._BLOCK[ne]
        S = 2 * bs
        c = JK.block_constants(n, bs)
        assert c.shape == (P, 4 * P + 2 * S)
        m = S - 1
        mask = c[:, m + P : m + 2 * P]
        pi = c[:, m + 2 * P + S : m + 3 * P + S]
        # Cross steps: every pair (i in the first block, j in the second) once.
        qx = c[:, m + 3 * P + S + 1 :]
        assert torch.equal(qx, qx.T)
        assert torch.equal(
            (qx >= 0).float(),
            mask
            * (1 - torch.eye(P))
            * ((torch.arange(P) % S < bs)[:, None] != (torch.arange(P) % S < bs)[None, :]).float(),
        )
        for r in range(bs):
            assert torch.equal((qx == r).float().sum(1), torch.ones(P))
        assert torch.equal(mask, mask.T) and float(mask.sum()) == P * S
        assert torch.equal(pi.sum(0), torch.ones(P)) and torch.equal(pi.sum(1), torch.ones(P))
        # The shuffle stays within each matrix, and has order nb - 1.
        blockdiag = torch.block_diag(*[torch.ones(ne, ne)] * b)
        assert float((pi * (1 - blockdiag)).abs().sum()) == 0
        assert torch.equal(torch.linalg.matrix_power(pi, ne // bs - 1), torch.eye(P))

    def test_bad_block_size(self):
        with pytest.raises(ValueError):
            JK.block_constants(64, 64)


class TestAutoRegion:
    def test_region_env(self, no_env, monkeypatch):
        assert eigen._jacobi_auto_region() == eigen._JACOBI_AUTO_REGION
        monkeypatch.setenv("TRNSOLVER_EIGH_JACOBI_AUTO", "off")
        assert eigen._jacobi_auto_region() == ()
        monkeypatch.setenv("TRNSOLVER_EIGH_JACOBI_AUTO", "1-32:256, 33-64:1024")
        assert eigen._jacobi_auto_region() == ((1, 32, 256), (33, 64, 1024))
        monkeypatch.setenv("TRNSOLVER_EIGH_JACOBI_AUTO", "junk")
        assert eigen._jacobi_auto_region() == eigen._JACOBI_AUTO_REGION

    def test_default_table(self, no_env, monkeypatch):
        monkeypatch.setattr(eigen, "_NATIVE_DEVICE_TYPE", "cpu")
        monkeypatch.setattr(eigen, "_jacobi_device_ok", lambda n, dtype: True)
        for (n, b), want in {
            (32, 512): True,
            (32, 256): False,
            (40, 256): True,
            (40, 64): False,
            (80, 256): False,
            (80, 1024): True,
            (96, 256): True,
            (128, 64): True,
            (127, 64): False,
        }.items():
            assert eigen._jacobi_auto(torch.zeros(b, n, n)) is want, (n, b)

    def test_cpu_input_never_auto(self, no_env):
        assert not eigen._jacobi_auto(torch.zeros(1024, 16, 16))

    def test_decision(self, no_env, monkeypatch):
        # Pretend CPU is the native device so the region logic is exercised.
        monkeypatch.setattr(eigen, "_NATIVE_DEVICE_TYPE", "cpu")
        monkeypatch.setattr(eigen, "_jacobi_device_ok", lambda n, dtype: True)
        monkeypatch.setenv("TRNSOLVER_EIGH_JACOBI_AUTO", "1-32:512,65-128:1024")
        assert eigen._jacobi_auto(torch.zeros(512, 32, 32))
        assert eigen._jacobi_auto(torch.zeros(2, 256, 8, 8))  # batch dims multiply
        assert not eigen._jacobi_auto(torch.zeros(511, 32, 32))
        assert not eigen._jacobi_auto(torch.zeros(1024, 33, 33))
        assert eigen._jacobi_auto(torch.zeros(1024, 65, 65))
        assert not eigen._jacobi_auto(torch.zeros(512, 128, 128))
        assert not eigen._jacobi_auto(torch.zeros(32, 32))  # unbatched
        assert not eigen._jacobi_auto(torch.zeros(1024, 32, 32, dtype=torch.float64))
        monkeypatch.setenv("TRNSOLVER_EIGH_JACOBI_AUTO", "off")
        assert not eigen._jacobi_auto(torch.zeros(1024, 32, 32))

    def test_auto_routes_to_jacobi(self, no_env, monkeypatch):
        monkeypatch.setattr(eigen, "_NATIVE_DEVICE_TYPE", "cpu")
        monkeypatch.setattr(eigen, "_jacobi_device_ok", lambda n, dtype: False)
        monkeypatch.setattr(eigen, "_jacobi_auto", lambda A: True)
        calls = []
        real = eigen._jacobi_eigh
        monkeypatch.setattr(eigen, "_jacobi_eigh", lambda A: calls.append(1) or real(A))
        A = _separated(4, 8)
        w, V = trnsolver.eigh(A)
        assert calls == [1]
        _check(A, w, V)


# ----------------------------------------------------------------------
# Torch Jacobi path (CPU)
# ----------------------------------------------------------------------


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv("TRNSOLVER_EIGH_METHOD", raising=False)
    monkeypatch.delenv("TRNSOLVER_JACOBI_GROUP", raising=False)
    monkeypatch.delenv("TRNSOLVER_JACOBI_CHAINS", raising=False)
    monkeypatch.delenv("TRNSOLVER_JACOBI_BLOCK", raising=False)
    monkeypatch.delenv("TRNSOLVER_EIGH_JACOBI_AUTO", raising=False)
    monkeypatch.delenv("TRNSOLVER_EIGH_CORES", raising=False)
    monkeypatch.delenv("TRNSOLVER_EIGH_POOL", raising=False)
    monkeypatch.delenv("TRNSOLVER_EIGH_POOL_CORES", raising=False)


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


def _emulate_block_launch(Z, V, n, chains, nsteps=None):
    """Torch model of one `jacobi_block_kernel` launch (one block sweep)."""
    ne, b, P = JK.block_layout(n)
    bs = JK._BLOCK[ne]
    S = 2 * bs
    m = S - 1
    c = JK.block_constants(n, bs)
    sig, Q = c[:, :m], c[:, m : m + P]
    mask = c[:, m + P : m + 2 * P]
    R = c[:, m + 2 * P : m + 2 * P + S]
    c0 = m + 2 * P + S
    pblk = c[:, c0 : c0 + P]
    sigx, Qx = c[:, c0 + P], c[:, c0 + P + 1 :]
    full = [(torch.eq(Q, v), sig[:, r]) for r, v in enumerate(JK.step_targets(S))]
    cross = [(torch.eq(Qx, float(r)), sigx) for r in range(bs)]
    X = torch.cat([Z, V], dim=-1)  # (G, P, 2 ne)
    for st in range(ne // bs - 1 if nsteps is None else nsteps):
        zm = X[..., :ne].transpose(1, 2)
        Gc = ((zm.transpose(1, 2) @ zm) * mask).transpose(1, 2) @ R  # (G, P, S)
        Y = torch.cat([Gc, R.expand_as(Gc)], -1)
        for _ in range(JK._INNER_SWEEPS):
            for Pi, sg in full if st == 0 else cross:
                Pi = Pi.to(X.dtype)
                M = torch.einsum("kp,gkc->gpc", Pi, Y)
                Yc = Y[..., :S]
                gpp = (Yc * Yc).sum(-1)
                gpq = (M[..., :S] * Yc).sum(-1)
                c_, s, _ = J.rotation_csk(gpp, gpp @ Pi, gpq, sg)
                Y = c_.unsqueeze(-1) * Y - s.unsqueeze(-1) * M
        E = Y[..., S:].unsqueeze(2).expand(-1, P, P // S, S).reshape(-1, P, P) * mask
        X = (E.transpose(1, 2) @ pblk).transpose(1, 2) @ X
    return torch.stack([X[..., :ne], X[..., ne:]])


@pytest.fixture
def emulated_kernel(monkeypatch, no_env):
    launches = []

    def run(Z, V, n, chains):
        launches.append((tuple(Z.shape), chains))
        return _emulate_launch(Z, V, n, chains)

    def run_block(Z, V, n, chains):
        launches.append((tuple(Z.shape), chains))
        return _emulate_block_launch(Z, V, n, chains)

    monkeypatch.setattr(JK, "_run_kernel", run)
    monkeypatch.setattr(JK, "_run_block_kernel", run_block)
    monkeypatch.setattr(JK, "_use_simulator", lambda: True)
    monkeypatch.setattr(eigen, "_jacobi_device_ok", lambda n, dtype: dtype == torch.float32)
    return launches


class TestDeviceDriverEmulated:
    @pytest.mark.parametrize("n,nb", [(2, 3), (5, 7), (8, 1), (32, 9), (33, 5), (64, 3)])
    def test_matches_eigh(self, emulated_kernel, n, nb):
        A = _separated(nb, n, seed=n + nb)
        w, V = trnsolver.eigh(A, method="jacobi")
        _check(A, w, V)
        ne = JK.block_layout(n)[0]
        sweeps0 = (JK._BLOCK_SWEEPS0 if JK._use_block(ne) else JK._SWEEPS0)[ne]
        assert sweeps0 <= JK.last_sweeps <= JK._MAX_SWEEPS

    def test_random_n128(self, emulated_kernel, monkeypatch):
        monkeypatch.setenv("TRNSOLVER_JACOBI_BLOCK", "0")
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

    @pytest.mark.parametrize("n", [33, 64, 100, 128])
    def test_block_and_scalar_agree(self, emulated_kernel, monkeypatch, n):
        A = _separated(3, n, seed=n)
        w_b, V_b = trnsolver.eigh(A, method="jacobi")
        _check(A, w_b, V_b)
        monkeypatch.setenv("TRNSOLVER_JACOBI_BLOCK", "0")
        w_s, V_s = trnsolver.eigh(A, method="jacobi")
        _check(A, w_s, V_s)
        torch.testing.assert_close(w_b, w_s, rtol=0, atol=1e-4 * float(w_s.abs().max()))

    def test_block_random_n128_sweeps(self, emulated_kernel):
        A = _sym_batch(2, 128, seed=5)
        w, V = trnsolver.eigh(A, method="jacobi")
        _check(A, w, V, vecs=False)
        assert JK._BLOCK_SWEEPS0[128] <= JK.last_sweeps <= 9

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


class TestMulticore:
    @pytest.mark.parametrize(
        "B,n,parts,expected",
        [
            (1024, 128, 4, [(0, 256), (256, 512), (512, 768), (768, 1024)]),
            (10, 128, 4, [(0, 3), (3, 6), (6, 9), (9, 10)]),
            (3, 128, 4, [(0, 1), (1, 2), (2, 3)]),
            (9, 32, 2, [(0, 8), (8, 9)]),  # whole tiles of 4 matrices
            (3, 32, 4, [(0, 3)]),  # one tile: no split
            (100, 64, 1, [(0, 100)]),
        ],
    )
    def test_split_batch(self, B, n, parts, expected):
        assert JK.split_batch(B, n, parts) == expected

    def test_matches_single_core(self, emulated_kernel):
        A = _separated(13, 32, seed=11)
        w1, V1 = JK.jacobi_eigh_device(A)
        cpu = torch.device("cpu")
        w, V = JK.jacobi_eigh_multicore(A, [cpu, cpu, cpu])
        _check(A, w, V)
        torch.testing.assert_close(w, w1)
        assert JK.last_sweeps >= JK._SWEEPS0[32]

    def test_cores_env(self, no_env, monkeypatch):
        assert eigen._eigh_cores() == 1
        for raw, want in [("4", 4), ("auto", "auto"), (" AUTO ", "auto"), ("0", 1), ("x", 1)]:
            monkeypatch.setenv("TRNSOLVER_EIGH_CORES", raw)
            assert eigen._eigh_cores() == want

    def test_default_is_one_core(self, no_env):
        X = _sym_batch(4096, 128)
        assert eigen._jacobi_devices(X) == [X.device]

    def test_device_choice(self, no_env, monkeypatch):
        import sys
        import types

        monkeypatch.setattr(eigen, "_NATIVE_DEVICE_TYPE", "meta")
        monkeypatch.setattr(eigen, "_use_simulator", lambda: False)
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", types.SimpleNamespace(device_count=lambda: 4)
        )
        X = torch.empty(1024, 128, 128, device="meta")
        monkeypatch.setenv("TRNSOLVER_EIGH_CORES", "2")
        assert [d.index for d in eigen._jacobi_devices(X)] == [0, 1]
        monkeypatch.setenv("TRNSOLVER_EIGH_CORES", "8")
        assert len(eigen._jacobi_devices(X)) == 4
        monkeypatch.setenv("TRNSOLVER_EIGH_CORES", "auto")
        per = eigen._MULTICORE_MIN_PER_CORE[128]
        assert len(eigen._jacobi_devices(X[: 4 * per])) == 4
        assert len(eigen._jacobi_devices(X[: 2 * per])) == 2
        assert len(eigen._jacobi_devices(X[: per - 1])) == 1
        monkeypatch.setenv("TRNSOLVER_EIGH_CORES", "2")
        assert eigen._jacobi_devices(torch.empty(8, 32, 32)) == [
            torch.device("meta", 0),
            torch.device("meta", 1),
        ]
        X2 = types.SimpleNamespace(device=torch.device("meta", 2), shape=(64, 32, 32))
        monkeypatch.setenv("TRNSOLVER_EIGH_CORES", "4")
        assert [d.index for d in eigen._jacobi_devices(X2)] == [2, 3, 0, 1]

    def test_multicore_failure_falls_back_to_one_core(self, emulated_kernel, monkeypatch):
        cpu = torch.device("cpu")
        monkeypatch.setattr(eigen, "_jacobi_devices", lambda X: [cpu, cpu])

        def boom(X, d):
            raise RuntimeError("core busy")

        monkeypatch.setattr(JK, "jacobi_eigh_multicore", boom)
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", False)
        A = _separated(9, 32, seed=4)
        w, V = trnsolver.eigh(A, method="jacobi")
        _check(A, w, V)
        assert emulated_kernel  # the single-core kernel ran, not the torch path
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", True)
        with pytest.raises(RuntimeError, match="core busy"):
            trnsolver.eigh(A, method="jacobi")

    def test_eigh_routes_to_multicore(self, emulated_kernel, monkeypatch):
        monkeypatch.setenv("TRNSOLVER_EIGH_CORES", "3")
        calls = []
        cpu = torch.device("cpu")
        monkeypatch.setattr(eigen, "_jacobi_devices", lambda X: [cpu, cpu, cpu])
        real = JK.jacobi_eigh_multicore
        monkeypatch.setattr(
            JK, "jacobi_eigh_multicore", lambda X, d: calls.append(len(d)) or real(X, d)
        )
        A = _separated(9, 32, seed=2)
        w, V = trnsolver.eigh(A, method="jacobi")
        _check(A, w, V)
        assert calls == [3]


class TestProcessPool:
    """`trnsolver/nki/eigh_pool.py` with the torch Jacobi in the workers."""

    @pytest.mark.parametrize(
        "raw,cores",
        [("0-3", [0, 1, 2, 3]), ("2", [2]), ("0,2", [0, 2]), (" 3,0-1 ", [0, 1, 3]), ("", [])],
    )
    def test_parse_cores(self, raw, cores):
        from trnsolver.nki.eigh_pool import parse_cores

        assert parse_cores(raw) == cores

    def test_split_choice(self, no_env, monkeypatch):
        monkeypatch.setattr(eigen, "_use_simulator", lambda: False)
        X = torch.empty(1024, 128, 128)
        assert eigen._jacobi_pool_split(X) == ([], 1)  # not requested: unchanged
        monkeypatch.setenv("TRNSOLVER_EIGH_POOL", "proc")
        monkeypatch.setenv("TRNSOLVER_EIGH_POOL_CORES", "0-3")
        # Requested: every host call uses the pool, so this process never
        # opens a core a worker holds.
        assert eigen._jacobi_pool_split(X) == ([0, 1, 2, 3], 1)
        monkeypatch.setenv("TRNSOLVER_EIGH_CORES", "2")
        assert eigen._jacobi_pool_split(X) == ([0, 1, 2, 3], 2)
        monkeypatch.setenv("TRNSOLVER_EIGH_CORES", "8")
        assert eigen._jacobi_pool_split(X) == ([0, 1, 2, 3], 4)
        monkeypatch.setenv("TRNSOLVER_EIGH_CORES", "auto")
        per = eigen._MULTICORE_MIN_PER_CORE[128]
        assert eigen._jacobi_pool_split(X[: 2 * per]) == ([0, 1, 2, 3], 2)
        meta = torch.empty(1024, 128, 128, device="meta")
        assert eigen._jacobi_pool_split(meta) == ([], 1)  # device input: threads

    def test_pool_matches_eigh(self, no_env):
        from trnsolver.nki import eigh_pool

        try:
            for nb, n, parts in [(9, 8, 2), (40, 8, 2), (5, 7, 1), (12, 16, 2)]:
                A = _separated(nb, n, seed=nb + n)
                w, V = eigh_pool.jacobi_eigh_pool(A, [0, 1], parts, simulate=True)
                assert w.shape == (nb, n) and V.shape == (nb, n, n)
                _check(A, w, V)
                assert JK.last_sweeps > 0
            pool = eigh_pool._POOL
            assert pool is not None and pool.cores == [0, 1]
            # Results are views of a shared output buffer: one the caller
            # still holds is never overwritten; a dropped one is reused.
            A1, A2 = _separated(16, 8, seed=21), _separated(16, 8, seed=22)
            w1, V1 = eigh_pool.jacobi_eigh_pool(A1, [0, 1], simulate=True)
            w1c = w1.clone()
            w2, V2 = eigh_pool.jacobi_eigh_pool(A2, [0, 1], simulate=True)
            torch.testing.assert_close(w1, w1c)
            assert w1.untyped_storage().data_ptr() != w2.untyped_storage().data_ptr()
            _check(A2, w2, V2)
            del w1, V1, w2, V2
            made = pool._next_id
            w3, V3 = eigh_pool.jacobi_eigh_pool(A2, [0, 1], simulate=True)
            assert pool._next_id == made  # reused, nothing allocated
            # In use: the loop's last (w, V) and (w3, V3); plus spares.
            assert sum(not b.free() for b in pool.bufs) == 2
            assert len(pool.bufs) <= 2 + eigh_pool._KEEP_FREE
            _check(A2, w3, V3)
            del w3, V3
            pool.procs[1].kill()
            pool.procs[1].join()
            A = _separated(9, 8, seed=1)
            w, V = eigh_pool.jacobi_eigh_pool(A, [0, 1], simulate=True)  # restarts
            _check(A, w, V)
            assert eigh_pool._POOL is not pool
            # A failure mid-call closes the pool, and later calls skip it.
            eigh_pool._POOL.conns[0].close()
            with pytest.raises(Exception):  # noqa: B017
                eigh_pool.jacobi_eigh_pool(A, [0, 1], simulate=True)
            assert eigh_pool._POOL is None and eigh_pool._FAILED
            with pytest.raises(RuntimeError, match="failed earlier"):
                eigh_pool.jacobi_eigh_pool(A, [0, 1], simulate=True)
        finally:
            eigh_pool.shutdown()
        assert eigh_pool._POOL is None and not eigh_pool._FAILED

    def test_pool_failure_falls_back(self, emulated_kernel, monkeypatch):
        from trnsolver.nki import eigh_pool

        monkeypatch.setenv("TRNSOLVER_EIGH_POOL", "proc")
        monkeypatch.setenv("TRNSOLVER_EIGH_POOL_CORES", "0-1")
        monkeypatch.setenv("TRNSOLVER_EIGH_CORES", "2")
        monkeypatch.setattr(eigen, "_use_simulator", lambda: False)

        def boom(X, cores, parts):
            assert (cores, parts) == ([0, 1], 2)
            raise RuntimeError("worker died")

        monkeypatch.setattr(eigh_pool, "jacobi_eigh_pool", boom)
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", False)
        A = _separated(9, 32, seed=4)
        w, V = trnsolver.eigh(A, method="jacobi")
        _check(A, w, V)
        assert emulated_kernel  # fell back to the single-core kernel
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", True)
        with pytest.raises(RuntimeError, match="worker died"):
            trnsolver.eigh(A, method="jacobi")


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

    @pytest.mark.parametrize("n,G,chains", [(64, 1, 1), (40, 2, 2), (128, 1, 1)])
    def test_block_kernel_one_step(self, n, G, chains):
        # One outer step (a full sweep amplifies rounding differences in
        # near-degenerate inner angles; the end-to-end test covers it).
        ne, b, P = JK.block_layout(n)
        bs = JK._BLOCK[ne]
        A = _sym_batch(G * b, n)
        sigma = A.abs().sum(-1).amax(-1)
        Z = JK.pack_compact(A + sigma[:, None, None] * torch.eye(n), n)
        V = JK.pack_compact(torch.eye(n).expand(G * b, n, n), n)
        import nki

        got = torch.from_numpy(
            nki.simulate(JK.jacobi_block_kernel)(
                Z.numpy(),
                V.numpy(),
                JK.block_constants(n, bs).numpy(),
                np.zeros((JK._INNER_SWEEPS, 2 * bs, chains, 1), dtype=np.float32),
                int(JK._ONEZERO),
            )
        )
        ref = _emulate_block_launch(Z, V, n, chains, nsteps=1)
        scale = float(ref.abs().max())
        torch.testing.assert_close(got, ref, rtol=0, atol=1e-4 * scale)

    @pytest.mark.parametrize("n,nb", [(40, 3), (64, 4), (128, 1)])
    def test_block_eigh_end_to_end(self, n, nb):
        A = _sym_batch(nb, n, seed=n)
        w, V = trnsolver.eigh(A, method="jacobi")
        _check(A, w, V, vecs=False)

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
@pytest.mark.parametrize("lnc", [1, 2])
def test_mlir_verifies(n, G, chains, lnc):
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
        int(JK._ONEZERO),
        target="trn2",
        lnc=lnc,
        frontend=FrontendKind.TRACER,
    ).verify()


@pytest.mark.nki_simulator
@pytest.mark.parametrize(
    "n,G,chains", [(33, 1, 1), (64, 2, 2), (64, 128, 2), (128, 4, 2), (128, 128, 2)]
)
@pytest.mark.parametrize("lnc", [1, 2])
def test_block_mlir_verifies(n, G, chains, lnc):
    if not HAS_NKI:
        pytest.skip("nki not importable on this host")
    try:
        from nki.compiler.frontend import FrontendKind, source_to_mlir
    except ImportError:
        pytest.skip("nki.compiler.frontend.source_to_mlir not available in this nki")
    ne, b, P = JK.block_layout(n)
    bs = JK._BLOCK[ne]
    Z = np.zeros((G, P, ne), dtype=np.float32)
    source_to_mlir(
        JK.jacobi_block_kernel,
        Z,
        Z,
        JK.block_constants(n, bs).numpy(),
        np.zeros((JK._INNER_SWEEPS, 2 * bs, chains), dtype=np.float32),
        int(JK._ONEZERO),
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

    @pytest.mark.parametrize("n", [64, 128])
    def test_block_matches_scalar_kernel(self, monkeypatch, n):
        A = _separated(4, n, seed=3)
        w_b, V_b = trnsolver.eigh(A, method="jacobi")
        _check(A, w_b, V_b, tol=5e-5)
        monkeypatch.setenv("TRNSOLVER_JACOBI_BLOCK", "0")
        w_s, V_s = trnsolver.eigh(A, method="jacobi")
        _check(A, w_s, V_s, tol=5e-5)

    def test_multi_group(self, monkeypatch):
        monkeypatch.setenv("TRNSOLVER_JACOBI_GROUP", "2")
        A = _separated(5 * 4 + 1, 32, seed=7)  # 6 tiles -> 3 launches per sweep
        w, V = trnsolver.eigh(A, method="jacobi")
        _check(A, w, V, tol=5e-5)

    @pytest.mark.parametrize("on_device", [False, True])
    def test_multicore_split(self, monkeypatch, on_device):
        import torch_neuronx

        if torch_neuronx.device_count() < 2:
            pytest.skip("needs >= 2 visible NeuronCores (NEURON_RT_VISIBLE_CORES)")
        monkeypatch.setenv("TRNSOLVER_EIGH_CORES", "auto")
        A = _separated(300, 128, seed=9)  # 300 // 128 -> 2 cores
        X = A.to("neuron") if on_device else A
        assert len(eigen._jacobi_devices(X)) == 2
        for _ in range(2):  # first call runs the chunks in turn, then concurrently
            w, V = trnsolver.eigh(X, method="jacobi")
            assert w.device == X.device and V.device == X.device
            # 300 x 128 columns: the sign normalisation in _check is ambiguous
            # for a few, so check residual and orthogonality, not V itself.
            _check(A, w.cpu(), V.cpu(), tol=5e-5, vecs=False)

    def test_process_pool(self):
        # The pool's workers must own their cores, and this test process
        # holds every core it can see once it has used one. So the pool
        # runs from a fresh interpreter on cores this process does not see,
        # e.g. NEURON_RT_VISIBLE_CORES=0-1 TRNSOLVER_TEST_POOL_CORES=2-3.
        import subprocess
        import sys

        cores = os.environ.get("TRNSOLVER_TEST_POOL_CORES", "").strip()
        if not cores:
            pytest.skip("set TRNSOLVER_TEST_POOL_CORES to >= 2 cores this process does not see")
        code = (
            "import os, torch, trnsolver\n"
            "from trnsolver.nki import jacobi as JK\n"
            "g = torch.Generator().manual_seed(5)\n"
            "X = torch.randn(300, 128, 128, generator=g, dtype=torch.float64)\n"
            "A = ((X + X.transpose(1, 2)) / 2)\n"
            "for _ in range(2):\n"
            "    w, V = trnsolver.eigh(A.float(), method='jacobi')\n"
            "    ref = torch.linalg.eigvalsh(A)\n"
            "    err = float((w.double() - ref).abs().max() / ref.abs().max())\n"
            "    r = A.float() @ V - V * w[:, None, :]\n"
            "    assert err < 5e-5 and float(r.abs().max()) < 1e-3, (err, float(r.abs().max()))\n"
            "print('OK', JK.last_sweeps)\n"
        )
        env = dict(
            os.environ,
            NEURON_RT_VISIBLE_CORES=cores,
            TRNSOLVER_EIGH_POOL_CORES=cores,
            TRNSOLVER_EIGH_POOL="proc",
            TRNSOLVER_EIGH_CORES="2",
            TRNSOLVER_REQUIRE_NKI="1",
        )
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=900
        )
        assert out.returncode == 0 and "OK" in out.stdout, out.stderr[-3000:]
