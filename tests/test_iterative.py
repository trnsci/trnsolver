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


class TestSSORPreconditioner:
    def test_reduces_iterations_vs_jacobi(self):
        # 1D Laplacian — classical test case where symmetric Gauss-Seidel (omega=1)
        # beats scalar Jacobi in CG iteration count.
        n = 64
        ones = torch.ones(n - 1)
        A = 2 * torch.eye(n) - torch.diag(ones, 1) - torch.diag(ones, -1)
        b = torch.randn(n)

        M_jac = trnsolver.jacobi_preconditioner(A)
        M_ssor = trnsolver.ssor_preconditioner(A, omega=1.0)

        _, iters_jac, _ = trnsolver.cg(A, b, tol=1e-8, maxiter=500, M=M_jac)
        _, iters_ssor, _ = trnsolver.cg(A, b, tol=1e-8, maxiter=500, M=M_ssor)

        assert iters_ssor < iters_jac

    def test_omega_1_converges(self, spd_matrix):
        n = 16
        A = spd_matrix(n)
        b = torch.randn(n)
        M = trnsolver.ssor_preconditioner(A, omega=1.0)
        x, iters, res = trnsolver.cg(A, b, tol=1e-8, maxiter=200, M=M)
        assert res < 1e-7
        assert iters > 0

    def test_omega_range(self, spd_matrix):
        n = 16
        A = spd_matrix(n)
        b = torch.randn(n)
        for omega in (0.5, 1.5):
            M = trnsolver.ssor_preconditioner(A, omega=omega)
            _, iters, res = trnsolver.cg(A, b, tol=1e-6, maxiter=200, M=M)
            assert res < 1e-5, f"omega={omega} did not converge: res={res}"

    def test_invalid_omega(self, spd_matrix):
        A = spd_matrix(8)
        with pytest.raises(ValueError, match="omega"):
            trnsolver.ssor_preconditioner(A, omega=0.0)
        with pytest.raises(ValueError, match="omega"):
            trnsolver.ssor_preconditioner(A, omega=2.0)

    def test_apply_matches_manual(self):
        # Small n=4 case: manually compute forward/backward solve and compare.
        torch.manual_seed(5)
        n = 4
        Q, _ = torch.linalg.qr(torch.randn(n, n))
        A = Q @ torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])) @ Q.T
        r = torch.randn(n)
        omega = 1.0

        d = torch.diagonal(A)
        L_factor = omega * torch.tril(A, diagonal=-1) + torch.diag(d)
        U_factor = L_factor.T
        dscale = omega * (2.0 - omega) * d

        t = torch.linalg.solve_triangular(L_factor, r.unsqueeze(-1), upper=False).squeeze(-1)
        v = dscale * t
        z_ref = torch.linalg.solve_triangular(U_factor, v.unsqueeze(-1), upper=True).squeeze(-1)

        M = trnsolver.ssor_preconditioner(A, omega=omega)
        z = M(r)

        np.testing.assert_allclose(z.numpy(), z_ref.numpy(), atol=1e-6)


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


class TestGmresIr:
    """GMRES-IR: Carson–Higham iterative refinement for general Ax = b.

    Tests cover correctness vs torch.linalg.solve, improved residual vs plain
    GMRES for ill-conditioned systems, nonsymmetric matrices, BF16 dtype
    round-trip, callable-A rejection, and outer-iteration semantics.
    """

    def test_identity(self):
        """A = I: LU is trivial; x should equal b with FP64 residual < tol."""
        n = 8
        A = torch.eye(n)
        b = torch.arange(1, n + 1, dtype=torch.float32)
        x, total_inner, res = trnsolver.gmres_ir(A, b, tol=1e-10)
        np.testing.assert_allclose(x.numpy(), b.numpy(), atol=1e-8)
        assert res < 1e-10

    def test_vs_solve(self):
        """Random well-conditioned A: solution matches torch.linalg.solve."""
        torch.manual_seed(42)
        n = 16
        A = torch.randn(n, n) + n * torch.eye(n)  # diagonally dominant
        b = torch.randn(n)
        x_ref = torch.linalg.solve(A, b)
        x, _iters, res = trnsolver.gmres_ir(A, b, tol=1e-8)
        rel_err = torch.linalg.norm(x - x_ref) / torch.linalg.norm(x_ref)
        assert rel_err.item() < 1e-5, f"Solution error {rel_err:.2e} vs torch.linalg.solve"
        # FP32 LU limits the achievable FP64 residual to ~eps32 ≈ 1.2e-7
        assert res < 1e-6

    def test_nonsymmetric(self):
        """Non-symmetric A: Ax ≈ b to high accuracy."""
        torch.manual_seed(7)
        n = 20
        # Strictly non-symmetric: upper triangle twice the lower
        A = torch.randn(n, n)
        A = A + 2 * A.T + n * torch.eye(n)
        b = torch.randn(n)
        x, _iters, res = trnsolver.gmres_ir(A, b, tol=1e-8)
        actual_res = (
            torch.linalg.norm(A.double() @ x.double() - b.double()) / torch.linalg.norm(b.double())
        ).item()
        assert actual_res < 1e-7, f"Residual {actual_res:.2e} exceeds 1e-7"

    def test_ill_conditioned_improvement(self):
        """Ill-conditioned A: GMRES-IR residual ≤ plain GMRES residual."""
        torch.manual_seed(13)
        n = 16
        # Build A with cond ≈ 1e4 via controlled singular values
        U, _, Vh = torch.linalg.svd(torch.randn(n, n))
        s = torch.logspace(0, -4, n)  # 1 → 1e-4
        A = U @ torch.diag(s) @ Vh
        b = torch.randn(n)

        _, _iters_plain, res_plain = trnsolver.gmres(A, b, tol=1e-6, maxiter=200, restart=n)
        _, _iters_ir, res_ir = trnsolver.gmres_ir(A, b, tol=1e-8)
        assert res_ir <= res_plain + 1e-8, (
            f"GMRES-IR residual {res_ir:.2e} should not exceed plain GMRES {res_plain:.2e}"
        )

    def test_bf16_input(self):
        """BF16 A and b: output is BF16 (dtype round-trip via _to_fp32/_restore)."""
        torch.manual_seed(42)
        n = 8
        A = (torch.randn(n, n) + n * torch.eye(n)).to(torch.bfloat16)
        b = torch.randn(n).to(torch.bfloat16)
        x, _iters, res = trnsolver.gmres_ir(A, b, tol=1e-4)
        assert x.dtype == torch.bfloat16, f"Expected BF16 output, got {x.dtype}"

    def test_callable_raises(self):
        """Callable A raises TypeError: LU factorization requires a concrete matrix."""
        torch.manual_seed(0)
        n = 8
        A = torch.randn(n, n) + n * torch.eye(n)
        b = torch.randn(n)
        with pytest.raises(TypeError, match="concrete matrix"):
            trnsolver.gmres_ir(lambda v: A @ v, b)

    def test_outer_iterations_semantics(self):
        """total_inner is non-negative; more outer iterations reduce residual."""
        torch.manual_seed(99)
        n = 12
        A = torch.randn(n, n) + n * torch.eye(n)
        b = torch.randn(n)

        _, iters_1, res_1 = trnsolver.gmres_ir(A, b, tol=1e-12, max_outer=1)
        _, iters_5, res_5 = trnsolver.gmres_ir(A, b, tol=1e-12, max_outer=5)

        assert iters_1 >= 0
        assert iters_5 >= 0
        # More outer iterations → residual no worse (may already converge in 1)
        assert res_5 <= res_1 + 1e-12

    def test_fp64_residual_better_than_fp32(self):
        """FP64 outer residual captures errors invisible to FP32."""
        torch.manual_seed(3)
        n = 16
        A = torch.randn(n, n) + n * torch.eye(n)
        b = torch.randn(n)
        x, _iters, res_fp64 = trnsolver.gmres_ir(A, b, tol=1e-12)

        # Recompute residual in FP32 and FP64 to show FP64 is the tighter bound
        r_fp32 = (A @ x - b).float()
        r_fp64 = A.double() @ x.double() - b.double()
        res_fp32_norm = (torch.linalg.norm(r_fp32) / torch.linalg.norm(b)).item()
        res_fp64_norm = (torch.linalg.norm(r_fp64) / torch.linalg.norm(b.double())).item()

        # The returned res is FP64; both should be small for a well-conditioned system
        assert res_fp64 < 1e-5, f"FP64 residual {res_fp64:.2e} unexpectedly large"
        # FP64 residual ≤ FP32 residual (FP64 is the tighter measurement)
        assert res_fp64_norm <= res_fp32_norm + 1e-12
