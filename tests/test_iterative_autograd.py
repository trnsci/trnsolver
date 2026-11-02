"""Autograd through cg / gmres / schur (#76).

cg and gmres differentiate implicitly (adjoint solve); schur raises.
"""

import pytest
import torch

import trnsolver

F64 = torch.float64


def _spd(n, dtype=F64, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, n, generator=g, dtype=dtype)
    return X @ X.mH + n * torch.eye(n, dtype=dtype)


def _grads(f, A, b):
    A = A.clone().requires_grad_(True)
    b = b.clone().requires_grad_(True)
    gout = torch.randn(b.shape, generator=torch.Generator().manual_seed(9), dtype=b.dtype)
    return torch.autograd.grad(f(A, b), (A, b), gout)


@pytest.mark.parametrize("n", [5, 20])
def test_cg_gradient_wrt_A_matches_dense_solve(n):
    # Before #76 the A-gradient had rel. error 0.55 (n=5) and 7.2 (n=20).
    A = _spd(n)
    b = torch.randn(n, generator=torch.Generator().manual_seed(1), dtype=F64)
    gA, gb = _grads(lambda A, b: trnsolver.cg(A, b, tol=1e-13)[0], A, b)
    rA, rb = _grads(torch.linalg.solve, A, b)
    torch.testing.assert_close(gA, rA, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(gb, rb, rtol=1e-7, atol=1e-9)


def test_cg_returns_iters_and_residual_with_grad():
    A = _spd(8).requires_grad_(True)
    b = torch.ones(8, dtype=F64)
    x, iters, res = trnsolver.cg(A, b, tol=1e-10)
    assert x.requires_grad and iters > 0 and res < 1e-10


def test_gmres_backward_matches_dense_solve():
    # Before #76 backward raised "modified by an inplace operation".
    n = 7
    g = torch.Generator().manual_seed(3)
    A = torch.randn(n, n, generator=g, dtype=F64) + n * torch.eye(n, dtype=F64)
    b = torch.randn(n, generator=g, dtype=F64)
    gA, gb = _grads(lambda A, b: trnsolver.gmres(A, b, tol=1e-13, restart=n)[0], A, b)
    rA, rb = _grads(torch.linalg.solve, A, b)
    torch.testing.assert_close(gA, rA, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(gb, rb, rtol=1e-7, atol=1e-9)


def test_gmres_callable_A_with_grad_raises():
    A = _spd(5)
    b = torch.ones(5, dtype=F64, requires_grad=True)
    with pytest.raises(RuntimeError, match="callable A"):
        trnsolver.gmres(lambda x: A @ x, b)
    with torch.no_grad():
        x, _, res = trnsolver.gmres(lambda x: A @ x, b, tol=1e-10)
    assert res < 1e-10


def test_schur_requires_grad_raises_not_detaches():
    A = _spd(4).requires_grad_(True)
    with pytest.raises(RuntimeError, match="not differentiable"):
        trnsolver.schur(A)
    Q, T = trnsolver.schur(A.detach())
    torch.testing.assert_close(Q @ T @ Q.mT, A.detach(), rtol=1e-8, atol=1e-8)
