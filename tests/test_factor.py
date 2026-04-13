"""Test matrix factorizations."""

import pytest
import torch
import numpy as np
import trnsolver


class TestCholesky:

    def test_identity(self):
        L = trnsolver.cholesky(torch.eye(4))
        np.testing.assert_allclose(L.numpy(), np.eye(4), atol=1e-6)

    def test_reconstruction(self, spd_matrix):
        n = 16
        A = spd_matrix(n)
        L = trnsolver.cholesky(A)
        reconstructed = L @ L.T
        np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-4)

    def test_lower_triangular(self, spd_matrix):
        L = trnsolver.cholesky(spd_matrix(8))
        upper = torch.triu(L, diagonal=1)
        np.testing.assert_allclose(upper.numpy(), np.zeros_like(upper.numpy()), atol=1e-7)

    def test_upper(self, spd_matrix):
        A = spd_matrix(8)
        U = trnsolver.cholesky(A, upper=True)
        reconstructed = U.T @ U
        np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-4)


class TestLU:

    def test_reconstruction(self, random_matrix):
        A = random_matrix(8, 8)
        P, L, U = trnsolver.lu(A)
        reconstructed = P @ L @ U
        np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-5)


class TestQR:

    def test_reconstruction(self, random_matrix):
        A = random_matrix(8, 6)
        Q, R = trnsolver.qr(A)
        reconstructed = Q @ R
        np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-5)

    def test_orthogonality(self, random_matrix):
        A = random_matrix(8, 6)
        Q, _ = trnsolver.qr(A)
        QtQ = Q.T @ Q
        np.testing.assert_allclose(QtQ.numpy(), np.eye(6), atol=1e-5)


class TestSolve:

    def test_identity(self):
        A = torch.eye(4)
        b = torch.tensor([1.0, 2.0, 3.0, 4.0])
        x = trnsolver.solve(A, b)
        np.testing.assert_allclose(x.numpy(), b.numpy(), atol=1e-6)

    def test_vs_torch(self, random_matrix):
        A = random_matrix(16, 16)
        b = torch.randn(16)
        x = trnsolver.solve(A, b)
        residual = A @ x - b
        np.testing.assert_allclose(residual.numpy(), np.zeros(16), atol=1e-4)


class TestSolveSPD:

    def test_vs_solve(self, spd_matrix):
        n = 16
        A = spd_matrix(n)
        b = torch.randn(n)
        x_spd = trnsolver.solve_spd(A, b)
        x_gen = trnsolver.solve(A, b)
        np.testing.assert_allclose(x_spd.numpy(), x_gen.numpy(), atol=1e-4)

    def test_residual(self, spd_matrix):
        n = 16
        A = spd_matrix(n)
        b = torch.randn(n)
        x = trnsolver.solve_spd(A, b)
        residual = A @ x - b
        np.testing.assert_allclose(residual.numpy(), np.zeros(n), atol=1e-4)


class TestInvSqrtSPD:

    def test_identity(self):
        A = torch.eye(4)
        A_inv_sqrt = trnsolver.inv_sqrt_spd(A)
        np.testing.assert_allclose(A_inv_sqrt.numpy(), np.eye(4), atol=1e-6)

    def test_reconstruction(self, spd_matrix):
        """A^{-1/2} @ A @ A^{-1/2} should equal I."""
        n = 8
        A = spd_matrix(n)
        A_inv_sqrt = trnsolver.inv_sqrt_spd(A)
        product = A_inv_sqrt @ A @ A_inv_sqrt
        np.testing.assert_allclose(product.numpy(), np.eye(n), atol=1e-3)


class TestInvSqrtSpdNS:

    def test_identity(self):
        A = 2.0 * torch.eye(4, dtype=torch.float64)
        X, iters, res = trnsolver.inv_sqrt_spd_ns(A)
        # Expected A^{-1/2} = (1/sqrt(2)) I
        expected = (1.0 / np.sqrt(2.0)) * np.eye(4)
        np.testing.assert_allclose(X.numpy(), expected, atol=1e-6)
        assert iters >= 1
        assert res < 1e-6

    def test_matches_eig_reference(self, spd_matrix):
        n = 32
        A = spd_matrix(n, dtype=torch.float64)
        X_ref = trnsolver.inv_sqrt_spd(A)
        X_ns, iters, res = trnsolver.inv_sqrt_spd_ns(A, tol=1e-9)
        # Allow some slack; NS is quadratically convergent but not exact.
        np.testing.assert_allclose(X_ns.numpy(), X_ref.numpy(), rtol=1e-4, atol=1e-5)
        assert iters < 20

    def test_reconstruction(self, spd_matrix):
        n = 16
        A = spd_matrix(n, dtype=torch.float64)
        X, _, _ = trnsolver.inv_sqrt_spd_ns(A, tol=1e-9)
        product = X @ A @ X
        np.testing.assert_allclose(product.numpy(), np.eye(n), atol=1e-4)

    def test_returns_iteration_count(self):
        A = torch.eye(4, dtype=torch.float64) * 3.0
        _, iters, res = trnsolver.inv_sqrt_spd_ns(A)
        assert isinstance(iters, int) and iters >= 1
        assert isinstance(res, float)
