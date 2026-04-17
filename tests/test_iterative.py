"""Test iterative solvers."""

import numpy as np
import pytest
import torch

import trnsolver


class TestCG:
    def test_identity(self):
        A = torch.eye(4)
        b = torch.tensor([1.0, 2.0, 3.0, 4.0])
        x, iters, res = trnsolver.cg(A, b)
        np.testing.assert_allclose(x.numpy(), b.numpy(), atol=1e-5)
        assert iters <= 4

    def test_spd(self, spd_matrix):
        n = 32
        A = spd_matrix(n)
        b = torch.randn(n)
        x, iters, res = trnsolver.cg(A, b, tol=1e-6)
        residual = A @ x - b
        np.testing.assert_allclose(residual.numpy(), np.zeros(n), atol=1e-4)
        assert res < 1e-5

    def test_callable_matvec(self, spd_matrix):
        n = 16
        A = spd_matrix(n)
        b = torch.randn(n)

        def matvec(x):
            return torch.mv(A, x)

        x, iters, res = trnsolver.cg(matvec, b)
        residual = A @ x - b
        np.testing.assert_allclose(residual.numpy(), np.zeros(n), atol=1e-4)

    def test_with_initial_guess(self, spd_matrix):
        n = 16
        A = spd_matrix(n)
        x_true = torch.randn(n)
        b = A @ x_true
        # Start close to solution
        x0 = x_true + 0.01 * torch.randn(n)
        x, iters, res = trnsolver.cg(A, b, x0=x0)
        np.testing.assert_allclose(x.numpy(), x_true.numpy(), atol=1e-3)

    def test_jacobi_preconditioner_reduces_iterations(self):
        # Diagonally dominant but with widely-varying diagonal scales — the
        # regime where Jacobi preconditioning pays off.
        torch.manual_seed(0)
        n = 64
        scales = torch.logspace(0, 3, n)
        A = torch.diag(scales) + 0.01 * torch.randn(n, n)
        A = 0.5 * (A + A.T) + n * torch.eye(n)
        b = torch.randn(n)

        _, iters_plain, _ = trnsolver.cg(A, b, tol=1e-8, maxiter=500)
        M = trnsolver.jacobi_preconditioner(A)
        _, iters_precond, _ = trnsolver.cg(A, b, tol=1e-8, maxiter=500, M=M)

        assert iters_precond < iters_plain
        assert iters_precond <= iters_plain // 2

    def test_preconditioner_tensor_form(self, spd_matrix):
        n = 32
        A = spd_matrix(n)
        b = torch.randn(n)
        M_tensor = torch.diag(1.0 / torch.diagonal(A))
        x, _, res = trnsolver.cg(A, b, tol=1e-6, M=M_tensor)
        assert res < 1e-5


class TestBlockJacobiPreconditioner:
    def test_reduces_iterations_vs_diagonal(self):
        # Block-tridiagonal-dominant matrix with 4×4 coupling blocks —
        # scalar Jacobi misses off-diagonal coupling within the block.
        torch.manual_seed(42)
        n, bs = 64, 8
        A = torch.zeros(n, n)
        for i in range(0, n, bs):
            e = min(i + bs, n)
            blk = torch.randn(e - i, e - i)
            blk = blk @ blk.T + bs * torch.eye(e - i)
            A[i:e, i:e] = blk
        b = torch.randn(n)

        M_diag = trnsolver.jacobi_preconditioner(A)
        M_blk = trnsolver.block_jacobi_preconditioner(A, block_size=bs)

        _, iters_diag, _ = trnsolver.cg(A, b, tol=1e-8, maxiter=500, M=M_diag)
        _, iters_blk, _ = trnsolver.cg(A, b, tol=1e-8, maxiter=500, M=M_blk)

        assert iters_blk < iters_diag

    def test_block_size_n_is_full_cholesky(self, spd_matrix):
        # block_size=n → single block = full Cholesky → exact solve in 1 CG step
        n = 16
        A = spd_matrix(n)
        b = torch.randn(n)
        M = trnsolver.block_jacobi_preconditioner(A, block_size=n)
        _, iters, res = trnsolver.cg(A, b, tol=1e-8, maxiter=50, M=M)
        # Exact preconditioning → 1 step in theory; FP32 rounding allows 2.
        assert iters <= 2
        assert res < 1e-7

    def test_block_size_1_matches_jacobi(self, spd_matrix):
        # Single-element blocks == scalar Jacobi preconditioning
        n = 16
        A = spd_matrix(n)
        r = torch.randn(n)
        M_scalar = trnsolver.jacobi_preconditioner(A)
        M_block1 = trnsolver.block_jacobi_preconditioner(A, block_size=1)
        np.testing.assert_allclose(M_block1(r).numpy(), M_scalar(r).numpy(), atol=1e-6)


class TestGMRES:
    def test_identity(self):
        A = torch.eye(4)
        b = torch.tensor([1.0, 2.0, 3.0, 4.0])
        x, iters, res = trnsolver.gmres(A, b)
        np.testing.assert_allclose(x.numpy(), b.numpy(), atol=1e-5)

    def test_nonsymmetric(self, random_matrix):
        n = 16
        # Make well-conditioned nonsymmetric matrix
        A = random_matrix(n, n) + n * torch.eye(n)
        b = torch.randn(n)
        x, iters, res = trnsolver.gmres(A, b, tol=1e-5)
        residual = A @ x - b
        assert torch.linalg.norm(residual).item() / torch.linalg.norm(b).item() < 1e-4
