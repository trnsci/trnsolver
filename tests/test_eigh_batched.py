"""Batched small eigh: many n <= 128 problems per launch (#48).

Four layers:

* CPU-only: chunking, the batched dispatch policy (B = 1 single path, aten
  cutover, fallback) and, with the ``trnsolver::`` ops replaced by torch
  references on an aliased ``neuron`` device, the batched host pipeline and
  its launch count. These run in the default ``-m "not neuron"`` job.
* ``nki_simulator``: the batched kernels through ``nki.simulate`` against
  ``torch.linalg.eigh`` for B in {1, 2, 7, 16}, n in {8, 64, 128}.
* ``nki_simulator``: the NKI 0.7 MLIR verifier on trn1 and trn2 for every
  batched kernel (the simulator does not check partition rules).
* ``neuron``: the same contract on hardware.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

import trnsolver
import trnsolver.eigen as eigen
from trnsolver.nki import batched, dispatch, tridiag
from trnsolver.nki.dispatch import HAS_NKI

BATCHES = [1, 2, 7, 16]
SIZES = [8, 64, 128]


def _sym_batch(nb, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(nb, n, n, generator=g)
    return 0.5 * (A + A.transpose(1, 2))


def _fix_signs(V):
    """Flip each column of each matrix so its largest-magnitude entry is positive."""
    idx = V.abs().argmax(dim=-2, keepdim=True)
    signs = torch.sign(torch.gather(V, -2, idx))
    signs[signs == 0] = 1.0
    return V * signs


def _check_batched_eigh(A, w, V, w_atol=1e-4, resid_tol=1e-4, v_atol=2e-3):
    nb, n = A.shape[0], A.shape[-1]
    assert w.shape == (nb, n) and V.shape == (nb, n, n)
    Ad, wd, Vd = A.double(), w.double(), V.double()
    w_ref, V_ref = torch.linalg.eigh(Ad)
    np.testing.assert_allclose(wd.numpy(), w_ref.numpy(), atol=w_atol, rtol=1e-4)
    resid = (Ad @ Vd - Vd * wd.unsqueeze(-2)).abs().max().item()
    assert resid < resid_tol
    orth = (Vd.transpose(1, 2) @ Vd - torch.eye(n, dtype=torch.float64)).abs().max().item()
    assert orth < 1e-4
    # Eigenvectors up to sign, where the spectrum is well separated (random
    # symmetric matrices at these sizes are).
    np.testing.assert_allclose(_fix_signs(Vd).numpy(), _fix_signs(V_ref).numpy(), atol=v_atol)


# ----------------------------------------------------------------------
# CPU-only: chunking
# ----------------------------------------------------------------------


class TestChunking:
    def test_default_chunk(self, monkeypatch):
        monkeypatch.delenv("TRNSOLVER_EIGH_BATCH_CHUNK", raising=False)
        assert batched.batch_chunk_max() == 16

    @pytest.mark.parametrize("raw,expected", [("8", 8), ("12", 8), ("1", 1), ("64", 64)])
    def test_env_rounds_down_to_power_of_two(self, monkeypatch, raw, expected):
        monkeypatch.setenv("TRNSOLVER_EIGH_BATCH_CHUNK", raw)
        assert batched.batch_chunk_max() == expected

    @pytest.mark.parametrize("raw", ["0", "-4", "lots"])
    def test_env_garbage_uses_default(self, monkeypatch, raw):
        monkeypatch.setenv("TRNSOLVER_EIGH_BATCH_CHUNK", raw)
        assert batched.batch_chunk_max() == 16

    @pytest.mark.parametrize(
        "nb,expected",
        [
            (1, [(0, 1, 1)]),
            (2, [(0, 2, 2)]),
            (7, [(0, 7, 8)]),
            (16, [(0, 16, 16)]),
            (17, [(0, 16, 16), (16, 1, 1)]),
            (40, [(0, 16, 16), (16, 16, 16), (32, 8, 8)]),
            (45, [(0, 16, 16), (16, 16, 16), (32, 13, 16)]),
        ],
    )
    def test_chunks(self, nb, expected):
        assert batched.batch_chunks(nb, 16) == expected

    @pytest.mark.parametrize("nb", [2, 7, 16, 64, 256, 1000, 1024])
    def test_chunks_cover_batch_with_few_compiled_shapes(self, nb):
        chunks = batched.batch_chunks(nb, 16)
        assert sum(c for _, c, _ in chunks) == nb
        assert all(c <= p <= 16 and p & (p - 1) == 0 for _, c, p in chunks)
        assert len({p for _, _, p in chunks}) <= 2
        assert batched.n_launches(nb, 16) == 4 * -(-nb // 16)


# ----------------------------------------------------------------------
# CPU-only: dispatch policy
# ----------------------------------------------------------------------


class TestBatchedPolicy:
    @pytest.mark.parametrize("nb", BATCHES)
    @pytest.mark.parametrize("n", SIZES)
    def test_pytorch_backend_matches_torch(self, monkeypatch, nb, n):
        monkeypatch.setattr(eigen, "_use_nki", lambda: False)
        A = _sym_batch(nb, n)
        w, V = trnsolver.eigh(A)
        _check_batched_eigh(A, w, V)

    def test_extra_batch_dims(self, monkeypatch):
        monkeypatch.setattr(eigen, "_use_nki", lambda: False)
        A = _sym_batch(6, 8).reshape(2, 3, 8, 8)
        w, V = trnsolver.eigh(A)
        assert w.shape == (2, 3, 8) and V.shape == (2, 3, 8, 8)
        _check_batched_eigh(A.reshape(6, 8, 8), w.reshape(6, 8), V.reshape(6, 8, 8))

    def test_bf16_restored(self, monkeypatch):
        monkeypatch.setattr(eigen, "_use_nki", lambda: False)
        w, V = trnsolver.eigh(_sym_batch(3, 8).to(torch.bfloat16))
        assert w.dtype == V.dtype == torch.bfloat16

    @pytest.fixture
    def spies(self, monkeypatch):
        calls = []

        def single(A, tol=1e-10):
            calls.append(("single", tuple(A.shape)))
            return torch.linalg.eigh(A)

        def batch(A, tol=1e-10):
            calls.append(("batched", tuple(A.shape)))
            return torch.linalg.eigh(A)

        monkeypatch.setattr(eigen, "_use_nki", lambda: True)
        monkeypatch.setattr(eigen, "_householder_qr_eigh", single)
        monkeypatch.setattr(eigen, "_householder_qr_eigh_batched", batch)
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", False)
        monkeypatch.delenv("TRNSOLVER_EIGH_NKI_MIN_N", raising=False)
        return calls

    def test_batch_of_one_takes_single_path(self, spies):
        A = _sym_batch(1, 16)
        w, V = trnsolver.eigh(A)
        assert spies == [("single", (16, 16))]
        assert w.shape == (1, 16) and V.shape == (1, 16, 16)

    def test_two_d_input_unchanged(self, spies):
        trnsolver.eigh(_sym_batch(1, 16)[0])
        assert spies == [("single", (16, 16))]

    def test_batch_takes_batched_path(self, spies):
        trnsolver.eigh(_sym_batch(4, 16))
        assert spies == [("batched", (4, 16, 16))]

    def test_cutover_applies_to_batches(self, spies, monkeypatch):
        monkeypatch.setattr(eigen, "_NATIVE_DEVICE_TYPE", "cpu")
        trnsolver.eigh(_sym_batch(4, 16))
        assert spies == []
        monkeypatch.setenv("TRNSOLVER_EIGH_NKI_MIN_N", "16")
        trnsolver.eigh(_sym_batch(4, 16))
        assert spies == [("batched", (4, 16, 16))]

    def test_failure_falls_back_to_torch(self, spies, monkeypatch):
        def boom(A, tol=1e-10):
            raise RuntimeError("kernel failed")

        monkeypatch.setattr(eigen, "_householder_qr_eigh_batched", boom)
        A = _sym_batch(3, 8)
        w, V = trnsolver.eigh(A)
        _check_batched_eigh(A, w, V)
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", True)
        with pytest.raises(RuntimeError, match="kernel failed"):
            trnsolver.eigh(A)

    def test_host_reduction_loops_single_path(self, monkeypatch):
        # Without TRNSOLVER_EIGH_TRIDIAG=device the reduction is host-driven,
        # one matrix at a time.
        calls = []

        def single(A, tol=1e-10):
            calls.append(tuple(A.shape))
            return torch.linalg.eigh(A)

        monkeypatch.delenv("TRNSOLVER_EIGH_TRIDIAG", raising=False)
        monkeypatch.setattr(eigen, "_householder_qr_eigh", single)
        A = _sym_batch(3, 8)
        w, V = eigen._householder_qr_eigh_batched(A)
        assert calls == [(8, 8)] * 3
        _check_batched_eigh(A, w, V)

    def test_n_one(self):
        A = torch.randn(3, 1, 1)
        w, V = eigen._householder_qr_eigh_batched(A)
        torch.testing.assert_close(w, A[:, 0, :])
        torch.testing.assert_close(V, torch.ones(3, 1, 1))


class TestHostTridiagBatched:
    @pytest.mark.parametrize("n", [2, 8, 64])
    def test_matches_single(self, n):
        g = torch.Generator().manual_seed(1)
        d = torch.randn(5, n, generator=g)
        e = torch.randn(5, n - 1, generator=g)
        w, Q = eigen._lapack_tridiag_eigh_batched(d, e)
        assert w.shape == (5, n) and Q.shape == (5, n, n)
        for b in range(5):
            T = torch.diag(d[b]) + torch.diag(e[b], 1) + torch.diag(e[b], -1)
            torch.testing.assert_close(w[b], torch.linalg.eigvalsh(T), rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(Q[b] @ torch.diag(w[b]) @ Q[b].T, T, rtol=1e-5, atol=1e-5)

    def test_without_scipy(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.linalg", None)
        d, e = torch.randn(4, 6), torch.randn(4, 5)
        w, Q = eigen._lapack_tridiag_eigh_batched(d, e)
        T = torch.diag_embed(d) + torch.diag_embed(e, 1) + torch.diag_embed(e, -1)
        torch.testing.assert_close(w, torch.linalg.eigvalsh(T), rtol=1e-5, atol=1e-5)

    def test_qr_solver(self, monkeypatch):
        monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG_SOLVER", "qr")
        d, e = torch.randn(3, 6), torch.randn(3, 5)
        w, Q = eigen._tridiag_eigh_host_batched(d, e, 1e-10)
        T = torch.diag_embed(d) + torch.diag_embed(e, 1) + torch.diag_embed(e, -1)
        torch.testing.assert_close(
            w.sort(dim=-1).values, torch.linalg.eigvalsh(T), rtol=1e-4, atol=1e-4
        )


# ----------------------------------------------------------------------
# CPU-only: the batched pipeline on torch reference ops (aliased device)
# ----------------------------------------------------------------------


def _ref_tridiag_one(A):
    """Packed (n, n_out_cols(n)) output of householder_tridiag, in float64."""
    n = A.shape[0]
    A = A.double().clone()
    out = torch.zeros(n, tridiag.n_out_cols(n), dtype=torch.float64)
    for k in range(n - 2):
        x = A[k + 1 :, k]
        nx = torch.linalg.norm(x)
        v = torch.zeros(n, dtype=torch.float64)
        v[k + 1 :] = x
        v[k + 1] += nx if x[0] >= 0 else -nx
        vn = torch.linalg.norm(v)
        if vn > 0:
            v = v / vn
        H = torch.eye(n, dtype=torch.float64) - 2.0 * torch.outer(v, v)
        A = H @ A @ H
        out[:, 2 + k] = v
    out[:, 0] = torch.diagonal(A)
    out[: n - 1, 1] = torch.diagonal(A, 1)
    return out.float()


def _ref_apply_one(V_refs, Q):
    Q = Q.clone()
    for k in range(V_refs.shape[1] - 1, -1, -1):
        v = V_refs[:, k]
        Q = Q - 2.0 * torch.outer(v, v @ Q)
    return Q.T.contiguous()


def _batched_reference_ops(calls):
    def rec(name, fn):
        def op(*args):
            calls.append((name, args[0].shape[0]))
            return fn(*args)

        return op

    return {
        "householder_tridiag_batched": rec(
            "householder_tridiag_batched",
            lambda A, consts: torch.stack([_ref_tridiag_one(a) for a in A]),
        ),
        "apply_reflectors_batched": rec(
            "apply_reflectors_batched",
            lambda V, Q: torch.stack([_ref_apply_one(v, q) for v, q in zip(V, Q, strict=True)]),
        ),
        "rr_project_batched": rec("rr_project_batched", lambda A, VT: VT @ A @ VT.transpose(1, 2)),
        "rr_rotate_batched": rec("rr_rotate_batched", lambda VT, Q: VT.transpose(1, 2) @ Q),
    }


class TestBatchedPipelineNative:
    """_householder_qr_eigh_batched_device on reference ops: routing, chunking, plumbing."""

    @pytest.fixture
    def native(self, monkeypatch):
        calls = []
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
        monkeypatch.setattr(dispatch, "_native_ops", _batched_reference_ops(calls))
        monkeypatch.setitem(sys.modules, "torch_xla", None)
        for name in batched._KERNELS.values():
            monkeypatch.setattr(batched, name, object(), raising=False)
        monkeypatch.setattr(batched, "HAS_NKI", True)
        monkeypatch.setattr(batched, "_use_simulator", lambda: False)
        monkeypatch.setattr(tridiag, "_CONSTS_CACHE", {})
        monkeypatch.setattr(eigen, "HAS_NKI", True)
        monkeypatch.setattr(eigen, "_use_nki", lambda: True)
        monkeypatch.setattr(eigen, "_use_simulator", lambda: False)
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", True)
        monkeypatch.setattr(eigen, "_householder_qr_eigh", None)  # B >= 2 never loops
        monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG", "device")
        monkeypatch.delenv("TRNSOLVER_EIGH_BATCH_CHUNK", raising=False)
        return calls

    @pytest.mark.parametrize("nb", [2, 7, 16])
    @pytest.mark.parametrize("n", SIZES)
    def test_matches_torch(self, native, nb, n):
        A = _sym_batch(nb, n)
        w, V = trnsolver.eigh(A)
        _check_batched_eigh(A, w, V)

    @pytest.mark.parametrize("nb", [2, 7, 16, 17, 40])
    def test_launch_count(self, native, nb):
        A = _sym_batch(nb, 8)
        trnsolver.eigh(A)
        chunks = batched.batch_chunks(nb, 16)
        assert len(native) == batched.n_launches(nb) == 4 * len(chunks)
        for op in batched._KERNELS:
            sizes = [c for name, c in native if name == op]
            assert sizes == [p for _, _, p in chunks], op
        assert "torch_xla" not in sys.modules or sys.modules["torch_xla"] is None

    def test_chunk_env(self, native, monkeypatch):
        monkeypatch.setenv("TRNSOLVER_EIGH_BATCH_CHUNK", "4")
        A = _sym_batch(10, 8)
        w, V = trnsolver.eigh(A)
        _check_batched_eigh(A, w, V)
        assert [c for name, c in native if name == "rr_rotate_batched"] == [4, 4, 2]

    def test_n_two_skips_apply(self, native):
        A = _sym_batch(3, 2)
        w, V = trnsolver.eigh(A)
        _check_batched_eigh(A, w, V)
        assert [name for name, _ in native] == [
            "householder_tridiag_batched",
            "rr_project_batched",
            "rr_rotate_batched",
        ]

    def test_padding_does_not_leak(self, native):
        # 7 matrices run in a chunk of 8; the identity pad must not appear.
        A = _sym_batch(7, 8) * 10.0
        w, _ = trnsolver.eigh(A)
        assert w.shape == (7, 8)
        torch.testing.assert_close(w, torch.linalg.eigvalsh(A), rtol=1e-4, atol=1e-3)


# ----------------------------------------------------------------------
# Simulator + MLIR verifier
# ----------------------------------------------------------------------


def _simulator_on():
    return os.environ.get("TRNSOLVER_USE_SIMULATOR", "").lower() in ("1", "true", "yes")


@pytest.mark.nki_simulator
class TestBatchedSimulator:
    @pytest.fixture(autouse=True)
    def _need_sim(self, monkeypatch):
        if not _simulator_on():
            pytest.skip("TRNSOLVER_USE_SIMULATOR=1 required")
        if not HAS_NKI:
            pytest.skip("nki package not importable on this host")
        monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG", "device")
        monkeypatch.delenv("TRNSOLVER_EIGH_BATCH_CHUNK", raising=False)

    @pytest.mark.parametrize("nb", BATCHES)
    @pytest.mark.parametrize("n", SIZES)
    def test_eigh_vs_torch(self, monkeypatch, nb, n):
        # No host-driven reduction, and no silent torch fallback.
        monkeypatch.setattr(eigen, "_householder_tridiag", None)
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", True)
        A = _sym_batch(nb, n)
        w, V = trnsolver.eigh(A)
        _check_batched_eigh(A, w, V)

    @pytest.mark.parametrize("nb", [2, 7])
    @pytest.mark.parametrize("n", [3, 8, 64])
    def test_reduction(self, nb, n):
        A = _sym_batch(nb, n)
        d, e, V_refs = batched.householder_tridiag_batched_device(A)
        assert d.shape == (nb, n) and e.shape == (nb, n - 1) and V_refs.shape == (nb, n, n - 2)
        for b in range(nb):
            T = torch.diag(d[b]) + torch.diag(e[b], 1) + torch.diag(e[b], -1)
            np.testing.assert_allclose(
                torch.linalg.eigvalsh(T.double()).numpy(),
                torch.linalg.eigvalsh(A[b].double()).numpy(),
                atol=2e-5 * n,
            )

    def test_matches_single_kernel(self):
        # Each matrix of the batch gets the same reduction as a single launch.
        A = _sym_batch(3, 16)
        d, e, V_refs = batched.householder_tridiag_batched_device(A)
        for b in range(3):
            d1, e1, V1 = tridiag.householder_tridiag_device(A[b])
            torch.testing.assert_close(d[b], d1, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(e[b], e1, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(V_refs[b], V1[:, : 16 - 2], rtol=1e-6, atol=1e-6)

    def test_zero_matrix_in_batch(self):
        A = _sym_batch(3, 8)
        A[1] = torch.diag(torch.arange(1.0, 9.0))
        d, e, V_refs = batched.householder_tridiag_batched_device(A)
        assert torch.isfinite(V_refs).all()
        torch.testing.assert_close(d[1], torch.arange(1.0, 9.0))
        torch.testing.assert_close(e[1], torch.zeros(7))


def _verify_args(kernel_name, nb, n):
    A = np.zeros((nb, n, n), dtype=np.float32)
    if kernel_name == "householder_tridiag_batched_kernel":
        return A, tridiag.tridiag_constants(n).numpy()
    if kernel_name == "apply_reflectors_batched_kernel":
        return np.zeros((nb, n, n - 2), dtype=np.float32), A
    return A, A


@pytest.mark.nki_simulator
@pytest.mark.parametrize("target", ["trn1", "trn2"])
@pytest.mark.parametrize("kernel_name", sorted(batched._KERNELS.values()))
@pytest.mark.parametrize("nb,n", [(1, 8), (2, 3), (2, 32), (8, 100), (16, 128)])
def test_mlir_verifies(kernel_name, nb, n, target):
    if not HAS_NKI:
        pytest.skip("nki not importable on this host")
    try:
        from nki.compiler.frontend import FrontendKind, source_to_mlir
    except ImportError:
        pytest.skip("nki.compiler.frontend.source_to_mlir not available in this nki")
    source_to_mlir(
        getattr(batched, kernel_name),
        *_verify_args(kernel_name, nb, n),
        target=target,
        frontend=FrontendKind.TRACER,
    ).verify()


# ----------------------------------------------------------------------
# Hardware
# ----------------------------------------------------------------------


@pytest.mark.neuron
class TestBatchedNeuron:
    @pytest.fixture(autouse=True)
    def _device_tridiag(self, monkeypatch):
        monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG", "device")
        monkeypatch.setenv("TRNSOLVER_EIGH_NKI_MIN_N", "0")
        monkeypatch.delenv("TRNSOLVER_EIGH_BATCH_CHUNK", raising=False)

    @pytest.mark.parametrize("nb", BATCHES)
    @pytest.mark.parametrize("n", SIZES)
    def test_eigh_vs_torch(self, monkeypatch, nb, n):
        monkeypatch.setattr(eigen, "_householder_tridiag", None)
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", True)
        A = _sym_batch(nb, n)
        w, V = trnsolver.eigh(A)
        _check_batched_eigh(A, w.cpu(), V.cpu(), w_atol=1e-3, resid_tol=1e-3, v_atol=1e-2)

    @pytest.mark.parametrize("nb", [17, 64])
    def test_multi_chunk(self, monkeypatch, nb):
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", True)
        A = _sym_batch(nb, 64)
        w, V = trnsolver.eigh(A)
        _check_batched_eigh(A, w.cpu(), V.cpu(), w_atol=1e-3, resid_tol=1e-3, v_atol=1e-2)

    def test_device_resident_input(self, monkeypatch):
        if dispatch._get_dispatch_path() != "native":
            pytest.skip("native TorchNeuron path only")
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", True)
        A = _sym_batch(16, 128)
        w, V = trnsolver.eigh(A.to("neuron"))
        assert w.device.type == "neuron" and V.device.type == "neuron"
        _check_batched_eigh(A, w.cpu(), V.cpu(), w_atol=1e-3, resid_tol=1e-3, v_atol=1e-2)
