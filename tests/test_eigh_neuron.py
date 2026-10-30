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


def _xla_path() -> bool:
    from trnsolver.nki.dispatch import HAS_NKI, _get_dispatch_path

    return HAS_NKI and _get_dispatch_path() == "xla"


# The two direct-kernel tests below call the raw kernels on torch_xla
# tensors. tests/test_nki_native.py covers the same kernels as
# trnsolver:: custom ops on the native path.
xla_only = pytest.mark.skipif(
    "not _xla_path()", reason="calls kernels on torch_xla tensors (XLA lineage only)"
)


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

    @xla_only
    def test_matmul_tensor_engine(self):
        """matmul_kernel (nisa.nc_matmul) on XLA device matches torch.matmul.

        Validates the Tensor Engine path directly: hardware NEFF compile +
        on-chip dispatch. This is the signal that #36 (free_dim constraint)
        is resolved — a square (n, k) moving operand with k=n satisfies
        the NKI 0.3.0 minimum free-dim requirement.
        """
        import torch_xla

        from trnsolver.nki.dispatch import matmul_kernel

        torch.manual_seed(42)
        n, k = 64, 64
        A = torch.randn(n, n)
        V = torch.randn(n, k)
        C_ref = (A @ V).numpy()

        xla = torch_xla.device()
        C = matmul_kernel(A.to(xla), V.to(xla))
        torch_xla.sync()
        np.testing.assert_allclose(C.cpu().numpy(), C_ref, atol=1e-3)

    @xla_only
    def test_apply_reflectors_kernel(self):
        """apply_reflectors_kernel on XLA device: Q_T_out^T matches host _apply_reflectors.

        Validates the fused Householder reflector application kernel on hardware.
        The kernel uses only Vector Engine ops (load_transpose2d + broadcast_to
        + multiply + sum axis=1), so it must compile cleanly on any trn1/trn2.
        """
        import torch_xla

        from trnsolver.eigen import _apply_reflectors, _householder_tridiag
        from trnsolver.nki.dispatch import apply_reflectors_kernel

        torch.manual_seed(42)
        n = 64
        A = torch.randn(n, n)
        A = 0.5 * (A + A.T)

        _, _, V_refs = _householder_tridiag(A)
        Q = torch.eye(n)
        Q_ref = _apply_reflectors(V_refs, Q.clone())

        xla = torch_xla.device()
        Q_T_out = apply_reflectors_kernel(V_refs.to(xla), Q.to(xla))
        torch_xla.sync()
        Q_kernel = Q_T_out.cpu().T.contiguous()
        np.testing.assert_allclose(Q_kernel.numpy(), Q_ref.numpy(), atol=1e-3)

    @pytest.mark.parametrize("n", [8, 32, 64, 128])
    def test_eigh_with_apply_reflectors_kernel(self, n):
        """end-to-end eigh eigenvectors correct with apply_reflectors_kernel."""
        torch.manual_seed(42)
        A = torch.randn(n, n)
        A = 0.5 * (A + A.T)

        w_ref, _ = torch.linalg.eigh(A)
        w, V = trnsolver.eigh(A)

        np.testing.assert_allclose(w.numpy(), w_ref.numpy(), atol=1e-3, rtol=1e-3)
        A_reconstructed = V @ torch.diag(w) @ V.T
        np.testing.assert_allclose(A_reconstructed.numpy(), A.numpy(), atol=1e-2)
