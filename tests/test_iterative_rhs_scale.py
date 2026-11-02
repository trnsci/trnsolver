"""cg / gmres with zero or small-scale b, and the Jacobi preconditioner on a block (trnsolver#68)."""

import pytest
import torch

import trnsolver

N = 6


def _spd(n=N, scale=1.0):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(n, n, generator=g, dtype=torch.float64)
    return scale * (X @ X.T / n + 2 * torch.eye(n, dtype=torch.float64))


def _run(name, A, b, **kw):
    if name == "cg":
        return trnsolver.cg(A, b, tol=1e-10, **kw)
    return trnsolver.gmres(A, b, tol=1e-10, restart=N, **kw)


@pytest.mark.parametrize("name", ["cg", "gmres"])
def test_zero_rhs_with_x0_returns_zero(name):
    A, b = _spd(), torch.zeros(N, dtype=torch.float64)
    x0 = torch.ones(N, dtype=torch.float64)
    x, iters, res = _run(name, A, b, x0=x0)
    torch.testing.assert_close(x, torch.zeros_like(b))
    assert torch.equal(x0, torch.ones(N, dtype=torch.float64))
    assert x.untyped_storage().data_ptr() != x0.untyped_storage().data_ptr()


@pytest.mark.parametrize("name", ["cg", "gmres"])
@pytest.mark.parametrize("scale", [1e-16, 1e-20])
def test_small_scale_problem_is_solved(name, scale):
    # A and b both scaled: x = A^{-1} b is O(1), the relative residual is
    # scale-free, and the solver must not return zeros with residual 0.0.
    A = _spd(scale=scale)
    b = scale * torch.linspace(1, 2, N, dtype=torch.float64)
    x, iters, res = _run(name, A, b)
    ref = torch.linalg.solve(A, b)
    torch.testing.assert_close(x, ref, rtol=1e-8, atol=0)
    true_res = (torch.linalg.norm(b - A @ x) / torch.linalg.norm(b)).item()
    assert res == pytest.approx(true_res, rel=1e-3, abs=1e-12)


def test_jacobi_preconditioner_block_scales_rows():
    A = _spd()
    R = torch.randn(N, N, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    M = trnsolver.jacobi_preconditioner(A)
    torch.testing.assert_close(M(R), torch.diag(1.0 / torch.diagonal(A)) @ R)
    r = R[:, 0]
    torch.testing.assert_close(M(r), r / torch.diagonal(A))
