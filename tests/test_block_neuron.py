"""Hardware gate for the device pivoted-Cholesky build and the projected apply (#114).

Run on Trn2 (TorchNeuron):

    pytest tests/test_block_neuron.py -v -m neuron

The CPU contracts are in test_cg_block.py; this file checks what only the
device can show: that the projected apply keeps its fp32 accuracy through
the device's matmuls (an accidental bf16 matmul would show up as a ~1e-2
error here), that preconditioned CG on the device takes the host-fp64
apply's iteration count, and that a build from device rows matches one
from host rows. These comparisons are within tolerance.

LNC invariance (#128): ``cg_block``'s default ``reduction="pairwise"`` and
the projected apply's Qᵀ V must give the same bits at LNC=1 and LNC=2. A
process has one LNC setting, so ``test_cg_block_lnc1_vs_lnc2_bitwise`` runs
scripts/probe_cg_reduction.py once at each and compares SHA-256 hashes. It
runs an LNC=1 process, which on a shared box needs the whole chip, so it only
runs with ``TRNSOLVER_TEST_LNC=1``; ``TRNSOLVER_TEST_LNC1_CORES`` /
``TRNSOLVER_TEST_LNC2_CORES`` set ``NEURON_RT_VISIBLE_CORES`` for the two
runs (physical / logical core ids, default 0).
"""

import json
import math
import os
import pathlib
import subprocess
import sys

import pytest
import torch

import trnsolver
from trnsolver import block

pytestmark = pytest.mark.neuron

DEV = "neuron"


def _gp_kernel(n, noise, seed=0, ell=0.15):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 3, generator=g, dtype=torch.float64)
    K = torch.exp(-0.5 * torch.cdist(X, X).square() / ell**2)
    return X, K, K + noise * torch.eye(n, dtype=torch.float64)


def _sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


@pytest.fixture(scope="module")
def gp():
    n, noise, rank = 8192, 1e-3, 512
    X, K, A = _gp_kernel(n, noise)
    L = trnsolver.pivoted_cholesky(
        lambda: torch.diagonal(K).clone(), lambda i: K[i], rank, block=256, relax=0.5,
        sub_fn=lambda i: K[i][:, i],
    )  # fmt: skip
    return dict(n=n, noise=noise, X=X, K=K, A=A, L=L)


def test_projected_apply_accuracy_on_device(gp):
    L, noise, n = gp["L"], gp["noise"], gp["n"]
    ref_P = trnsolver.PivotedLowRankPreconditioner(L, noise, apply="native")  # fp64, exact
    P = trnsolver.PivotedLowRankPreconditioner(L.float().to(DEV), noise, apply="projected")
    assert P.Q.device.type == DEV
    g = torch.Generator().manual_seed(1)
    # Residual-like: mostly along the captured subspace, where fp32 cancels.
    V = L @ torch.randn(L.shape[1], 33, generator=g, dtype=torch.float64)
    V = V + 1e-2 * torch.randn(n, 33, generator=g, dtype=torch.float64)
    ref = ref_P(V)
    out = P(V.float().to(DEV)).cpu().double()
    err = float(((out - ref).norm(dim=0) / ref.norm(dim=0)).max())
    cpu = trnsolver.PivotedLowRankPreconditioner(L.float(), noise, apply="projected")
    err_cpu = float(((cpu(V.float()).double() - ref).norm(dim=0) / ref.norm(dim=0)).max())
    print(f"projected apply rel err: device {err:.2e}, cpu fp32 {err_cpu:.2e}")
    assert err < 1e-4


def test_projected_cg_on_device_takes_host64_iterations(gp):
    A, L, noise, n = gp["A"], gp["L"], gp["noise"], gp["n"]
    A_dev = A.float().to(DEV)
    B = torch.randn(n, 33, generator=torch.Generator().manual_seed(2))
    B_dev = B.to(DEV)
    host = trnsolver.PivotedLowRankPreconditioner(L.float().to(DEV), noise, apply="host64")
    proj = trnsolver.PivotedLowRankPreconditioner(L.float().to(DEV), noise, apply="projected")
    Xh, ih = trnsolver.cg_block(A_dev, B_dev, host, tol=1e-5, maxiter=300)
    Xp, ip = trnsolver.cg_block(A_dev, B_dev, proj, tol=1e-5, maxiter=300)
    print(f"iterations host64 {ih['iterations']} projected {ip['iterations']}")
    assert bool(ih["converged"].all()) and bool(ip["converged"].all())
    assert abs(ip["iterations"] - ih["iterations"]) <= 2
    assert float(ip["true_residual"].max()) < 3 * float(ih["true_residual"].max()) + 1e-6
    x64 = torch.linalg.solve(A, B.double())
    fe_h = float(((Xh.cpu().double() - x64).norm(dim=0) / x64.norm(dim=0)).max())
    fe_p = float(((Xp.cpu().double() - x64).norm(dim=0) / x64.norm(dim=0)).max())
    print(f"forward error host64 {fe_h:.2e} projected {fe_p:.2e}")
    assert fe_p < 2 * fe_h + 1e-4


def test_block_build_from_device_rows(gp):
    K = gp["K"]
    K_dev = K.float().to(DEV)
    asked = []

    def rows(idx):
        asked.append(idx.numel())
        return K_dev[idx.to(DEV)]

    diag = lambda: torch.diagonal(K).clone()  # noqa: E731
    sub = lambda i: K[i][:, i]  # noqa: E731
    L_dev = trnsolver.pivoted_cholesky(diag, rows, 512, block=256, relax=0.5, sub_fn=sub)
    assert L_dev.device.type == "cpu" and L_dev.dtype == torch.float64
    assert sum(asked) == 512 and len(asked) < 64
    rem_dev = float(torch.trace(K) - (L_dev * L_dev).sum())
    rem_host = float(torch.trace(K) - (gp["L"] * gp["L"]).sum())
    assert rem_dev == pytest.approx(rem_host, rel=0.05)


def test_block_build_from_trnsparse_kernel_matmul():
    trnsparse = pytest.importorskip("trnsparse")
    n, ell, noise = 20000, 0.15, 1e-2
    g = torch.Generator().manual_seed(0)
    X = torch.rand(n, 3, generator=g, dtype=torch.float64)
    trnsparse.set_backend("nki")
    try:
        op = trnsparse.KernelOperator(X, "rbf", ell, 1.0, noise, device=DEV)

        def rows(idx):
            eye = torch.eye(idx.numel(), dtype=torch.float32, device=DEV)
            return trnsparse.kernel_matmul(X.float(), X[idx].float(), eye, "rbf", ell, 1.0).T

        def sub(idx):
            Xi = X[idx]
            eye = torch.eye(idx.numel(), dtype=torch.float64)
            return trnsparse.kernel_matmul_reference(Xi, Xi, eye, "rbf", ell, 1.0)

        L_dev = trnsolver.pivoted_cholesky(op.diag, rows, 1024, block=512, relax=0.5, sub_fn=sub)
        L_host = trnsolver.pivoted_cholesky(op.diag, op.rows, 1024)
    finally:
        trnsparse.set_backend("auto")
    tr = float(op.diag().sum())
    rem_dev = tr - float((L_dev * L_dev).sum())
    rem_host = tr - float((L_host * L_host).sum())
    print(f"remaining trace: device-row block build {rem_dev:.4e}, host sequential {rem_host:.4e}")
    assert L_dev.shape == (n, 1024)
    assert math.isfinite(rem_dev) and rem_dev < 1.15 * rem_host


# ── LNC-invariant reductions (#128) ──────────────────────────────────────────


def test_pairwise_reductions_on_device_vs_cpu():
    """The fold is elementwise adds only; report whether the device gives the CPU's bits."""
    n, R = 100003, 33  # ragged: padded to 49 · 2^11
    g = torch.Generator().manual_seed(3)
    X, Y = torch.randn(n, R, generator=g), torch.randn(n, R, generator=g)
    Xd, Yd = X.to(DEV), Y.to(DEV)
    d_dev = block._cdot(Xd, Yd, "pairwise").cpu()
    n_dev = block._col_norm(Xd, "pairwise").cpu()
    d_cpu, n_cpu = block._cdot(X, Y, "pairwise"), block._col_norm(X, "pairwise")
    ref = (X.double() * Y.double()).sum(0)
    scale = (X.double() * Y.double()).abs().sum(0)
    err = float(((d_dev.double() - ref).abs() / scale).max())
    print(f"pairwise dot rel err {err:.2e}; device == cpu: "
          f"dot {torch.equal(d_dev, d_cpu)} norm {torch.equal(n_dev, n_cpu)}")  # fmt: skip
    assert err < 13 * 2.0**-24
    assert torch.allclose(n_dev, n_cpu, rtol=1e-6)
    Q = torch.randn(n, 256, generator=g) / math.sqrt(n)
    c = block._tmatmul(Q.to(DEV), Yd, "pairwise").cpu()
    c_ref = Q.double().T @ Y.double()
    c_scale = Q.double().abs().T @ Y.double().abs()
    assert float(((c.double() - c_ref).abs() / c_scale).max()) < 1e-5


def _probe(lnc, cores, out):
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "probe_cg_reduction.py"
    env = dict(os.environ, NEURON_LOGICAL_NC_CONFIG=str(lnc), NEURON_RT_VISIBLE_CORES=cores)
    if "TORCH_NEURONX_NEFF_CACHE_DIR" in env:  # keep the two LNC settings' NEFFs apart
        env["TORCH_NEURONX_NEFF_CACHE_DIR"] += f"/lnc{lnc}"
    args = [sys.executable, "-P", str(script), "--n", "1e5", "--R", "33", "--out", str(out)]
    args += ["--no-ref", "--micro-reps", "2", "--modes", "pairwise,host64,device"]
    proc = subprocess.run(args, env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads((out / "res.json").read_text())


@pytest.mark.skipif(
    os.environ.get("TRNSOLVER_TEST_LNC") != "1",
    reason="runs an LNC=1 and an LNC=2 process; needs the whole chip (TRNSOLVER_TEST_LNC=1)",
)
def test_cg_block_lnc1_vs_lnc2_bitwise(tmp_path):
    r1 = _probe(1, os.environ.get("TRNSOLVER_TEST_LNC1_CORES", "0"), tmp_path / "lnc1")
    r2 = _probe(2, os.environ.get("TRNSOLVER_TEST_LNC2_CORES", "0"), tmp_path / "lnc2")
    keys = ("sha_X", "sha_MB")
    micro = ("sha_cdot", "sha_norm", "sha_qtv")
    for mode in ("pairwise", "host64", "device"):
        s1, s2 = r1["solves"][mode], r2["solves"][mode]
        same = {k: s1[k] == s2[k] for k in keys} | {
            k: r1["micro"][mode][k] == r2["micro"][mode][k] for k in micro
        }
        print(f"{mode}: iterations {s1['iterations']} / {s2['iterations']}, bitwise {same}")
        if mode != "device":  # "device" is the old, LNC-dependent path
            assert s1["iterations"] == s2["iterations"]
            assert all(same.values()), (mode, same)
