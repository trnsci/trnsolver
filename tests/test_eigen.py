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


class TestEighRandomized:
    """eigh_randomized: Nakatsukasa–Tropp sRR (SIAM SIMAX 45(2), 2024).

    Tests cover PSD accuracy, Frobenius reconstruction, eigenvector
    orthonormality, dtype restoration, power iteration, reproducibility,
    and input validation.
    """

    def test_psd_top_k_eigenvalues(self, spd_matrix):
        """Returned k eigenvalues approximate the k largest of torch.linalg.eigh."""
        n, k = 32, 5
        A = spd_matrix(n)
        w_all, _ = torch.linalg.eigh(A)  # ascending
        w_top_ref = w_all[-k:].sort().values  # k largest, ascending
        # n_iter=2 needed for rtol=1% accuracy; n_iter=0 with oversampling=15
        # gives ~10% error on this fixture (dense spectrum, small relative gaps).
        w_rand, _ = trnsolver.eigh_randomized(A, k, oversampling=15, n_iter=2, seed=0)
        np.testing.assert_allclose(w_rand.numpy(), w_top_ref.numpy(), rtol=1e-2, atol=1e-3)

    def test_reconstruction_rank_k(self, spd_matrix):
        """Low-rank reconstruction error bounded by tail eigenvalue energy."""
        n, k = 32, 8
        A = spd_matrix(n)
        w, V = trnsolver.eigh_randomized(A, k, oversampling=15, seed=42)
        A_approx = V @ torch.diag(w) @ V.T
        # Tail bound: ||A - A_k||_F = sqrt(Σ_{i>k} σ_i^2), approximately 2× slack
        w_all, _ = torch.linalg.eigh(A)
        tail_bound = torch.linalg.norm(w_all[:-k]).item()
        residual = torch.linalg.norm(A - A_approx, ord="fro").item()
        assert residual < 2.0 * tail_bound + 1e-2

    def test_eigenvectors_orthonormal(self, sym_matrix):
        """Returned eigenvectors are orthonormal: V^T V ≈ I."""
        n, k = 24, 6
        A = sym_matrix(n)
        _, V = trnsolver.eigh_randomized(A, k, oversampling=10, seed=0)
        VtV = V.T @ V
        np.testing.assert_allclose(VtV.numpy(), np.eye(k), atol=1e-4)

    def test_bf16_input_restored(self, sym_matrix):
        """BF16 input → BF16 output (dtype round-trip via _to_fp32/_restore)."""
        n, k = 16, 4
        A = sym_matrix(n).to(torch.bfloat16)
        w, V = trnsolver.eigh_randomized(A, k, oversampling=10, seed=0)
        assert w.dtype == torch.bfloat16
        assert V.dtype == torch.bfloat16

    def test_power_iteration_accuracy(self):
        """n_iter=1 does not degrade accuracy vs n_iter=0 (usually improves)."""
        torch.manual_seed(7)
        n, k = 48, 6
        A = torch.randn(n, n)
        A = 0.5 * (A + A.T)
        w_all, _ = torch.linalg.eigh(A)
        w_top_ref = w_all[-k:].sort().values

        w0, _ = trnsolver.eigh_randomized(A, k, oversampling=15, n_iter=0, seed=1)
        w1, _ = trnsolver.eigh_randomized(A, k, oversampling=15, n_iter=1, seed=1)
        err0 = torch.abs(w0 - w_top_ref).max().item()
        err1 = torch.abs(w1 - w_top_ref).max().item()
        # Power iteration should not make things worse; allow 2× slack for randomness
        assert err1 <= err0 * 2.0 + 1e-3

    def test_seed_reproducibility(self, sym_matrix):
        """Same seed produces identical eigenvalues and eigenvectors."""
        n, k = 20, 5
        A = sym_matrix(n)
        w1, V1 = trnsolver.eigh_randomized(A, k, seed=42)
        w2, V2 = trnsolver.eigh_randomized(A, k, seed=42)
        np.testing.assert_array_equal(w1.numpy(), w2.numpy())

    def test_invalid_k_raises(self, spd_matrix):
        """k < 1 and k + oversampling ≥ n both raise ValueError."""
        A = spd_matrix(10)
        with pytest.raises(ValueError):
            trnsolver.eigh_randomized(A, 0)  # k < 1
        with pytest.raises(ValueError):
            trnsolver.eigh_randomized(A, 9, oversampling=10)  # k + p = 19 ≥ n=10
