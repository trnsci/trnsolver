"""Test eigenvalue decomposition."""

import numpy as np
import pytest
import torch

import trnsolver


class TestEigh:
    def test_identity(self):
        A = torch.eye(4)
        vals, vecs = trnsolver.eigh(A)
        np.testing.assert_allclose(vals.numpy(), np.ones(4), atol=1e-6)

    def test_diagonal(self):
        A = torch.diag(torch.tensor([4.0, 1.0, 3.0, 2.0]))
        vals, vecs = trnsolver.eigh(A)
        np.testing.assert_allclose(vals.numpy(), [1.0, 2.0, 3.0, 4.0], atol=1e-6)

    def test_vs_torch(self, sym_matrix):
        n = 16
        A = sym_matrix(n)
        vals, vecs = trnsolver.eigh(A)
        expected_vals, expected_vecs = torch.linalg.eigh(A)
        np.testing.assert_allclose(vals.numpy(), expected_vals.numpy(), atol=1e-4)

    def test_reconstruction(self, sym_matrix):
        """V @ diag(λ) @ V^T should reconstruct A."""
        n = 8
        A = sym_matrix(n)
        vals, vecs = trnsolver.eigh(A)
        reconstructed = vecs @ torch.diag(vals) @ vecs.T
        np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-4)

    def test_orthogonality(self, sym_matrix):
        """Eigenvectors should be orthonormal."""
        n = 8
        A = sym_matrix(n)
        vals, vecs = trnsolver.eigh(A)
        VtV = vecs.T @ vecs
        np.testing.assert_allclose(VtV.numpy(), np.eye(n), atol=1e-4)

    def test_sorted_ascending(self, sym_matrix):
        n = 16
        A = sym_matrix(n)
        vals, _ = trnsolver.eigh(A)
        for i in range(len(vals) - 1):
            assert vals[i].item() <= vals[i + 1].item() + 1e-10

    def test_residual_norm(self, spd_matrix):
        """Subspace rotation should give ||AV - V diag(w)||_F / (n||A||) < 1e-5."""
        n = 64
        A = spd_matrix(n)
        vals, vecs = trnsolver.eigh(A)
        R = A @ vecs - vecs * vals
        rel_res = torch.linalg.norm(R, ord="fro").item() / (
            n * torch.linalg.norm(A, ord="fro").item()
        )
        assert rel_res < 1e-5, f"Relative residual {rel_res:.2e} exceeds 1e-5"

    def test_orthogonality_tight(self, spd_matrix):
        """Subspace rotation re-orthogonalises: ||V^T V - I||_F < 1e-5 at n=64."""
        n = 64
        A = spd_matrix(n)
        _, vecs = trnsolver.eigh(A)
        VtV = vecs.T @ vecs
        err = torch.linalg.norm(VtV - torch.eye(n), ord="fro").item()
        assert err < 1e-5, f"Orthogonality error {err:.2e} exceeds 1e-5"


class TestEighGeneralized:
    def test_vs_standard(self, sym_matrix, spd_matrix):
        """Generalized with B=I should match standard eigh."""
        n = 8
        A = sym_matrix(n)
        B = torch.eye(n)
        vals_gen, vecs_gen = trnsolver.eigh_generalized(A, B)
        vals_std, vecs_std = trnsolver.eigh(A)
        np.testing.assert_allclose(vals_gen.numpy(), vals_std.numpy(), atol=1e-4)

    def test_satisfies_generalized(self, sym_matrix, spd_matrix):
        """A @ x = λ B @ x should hold for each eigenpair."""
        n = 8
        A = sym_matrix(n)
        B = spd_matrix(n)
        vals, vecs = trnsolver.eigh_generalized(A, B)
        for i in range(n):
            lhs = A @ vecs[:, i]
            rhs = vals[i] * B @ vecs[:, i]
            np.testing.assert_allclose(lhs.numpy(), rhs.numpy(), atol=1e-3)
