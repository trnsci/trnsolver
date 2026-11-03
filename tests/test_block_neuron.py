"""Hardware gate for the device pivoted-Cholesky build and the projected apply (#114).

Run on Trn2 (TorchNeuron):

    pytest tests/test_block_neuron.py -v -m neuron

The CPU contracts are in test_cg_block.py; this file checks what only the
device can show: that the projected apply keeps its fp32 accuracy through
the device's matmuls (an accidental bf16 matmul would show up as a ~1e-2
error here), that preconditioned CG on the device takes the host-fp64
apply's iteration count, and that a build from device rows matches one
from host rows. Device reductions depend on the LNC split (trnsolver#107),
so comparisons are within tolerance, not bitwise.
"""

import math

import pytest
import torch

import trnsolver

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
