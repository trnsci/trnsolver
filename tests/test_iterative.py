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
