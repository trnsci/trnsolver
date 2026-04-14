"""Hardware-only tests for the NKI eigh path (Householder-QR, NKI 0.3.0).

Run via: `AWS_PROFILE=aws ./scripts/run_neuron_tests.sh trn1`

Silent PyTorch fallback is disabled by default in the runner script via
TRNSOLVER_REQUIRE_NKI=1 so these tests fail loudly if kernels compile but
return wrong values. Simulator parity (test_nki_sim.py) covers the same
contracts at zero AWS cost; this file owns the real-hardware signal
(MLIR verifier, NEFF compile, on-chip dispatch).
"""

import numpy as np
import pytest
import torch

import trnsolver

pytestmark = pytest.mark.neuron


class TestEighNKI:
    def test_identity(self):
        A = torch.eye(4)
        w, V = trnsolver.eigh(A)
        np.testing.assert_allclose(w.numpy(), np.ones(4), atol=1e-4)

    def test_diagonal(self):
        D = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        w, V = trnsolver.eigh(D)
        np.testing.assert_allclose(sorted(w.numpy()), [1.0, 2.0, 3.0, 4.0], atol=1e-4)

    @pytest.mark.parametrize("n", [8, 16, 32, 64, 128])
    def test_eigh_vs_torch(self, n):
        torch.manual_seed(42)
        A = torch.randn(n, n)
        A = 0.5 * (A + A.T)  # symmetric

        w_ref, _ = torch.linalg.eigh(A)
        w, V = trnsolver.eigh(A)

        np.testing.assert_allclose(w.numpy(), w_ref.numpy(), atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_eigh_reconstruction(self, n):
        """V diag(w) V^T should reconstruct A."""
        torch.manual_seed(42)
        A = torch.randn(n, n)
        A = 0.5 * (A + A.T)

        w, V = trnsolver.eigh(A)
        reconstructed = V @ torch.diag(w) @ V.T
        np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-2)

    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_eigh_orthogonality(self, n):
        """Eigenvectors should be orthonormal: V^T V ≈ I."""
        torch.manual_seed(42)
        A = torch.randn(n, n)
        A = 0.5 * (A + A.T)

        w, V = trnsolver.eigh(A)
        product = V.T @ V
        np.testing.assert_allclose(product.numpy(), np.eye(n), atol=1e-3)

    def test_eigh_generalized(self):
        """Generalized eigenproblem: A V = B V diag(w)."""
        n = 16
        torch.manual_seed(42)
        A = torch.randn(n, n)
        A = 0.5 * (A + A.T)
        B = torch.randn(n, n)
        B = B @ B.T + n * torch.eye(n)  # SPD

        w, V = trnsolver.eigh_generalized(A, B)

        lhs = A @ V
        rhs = B @ V @ torch.diag(w)
        np.testing.assert_allclose(lhs.numpy(), rhs.numpy(), atol=1e-2)
