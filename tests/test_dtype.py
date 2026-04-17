"""BF16/FP16 dtype round-trip tests (#19).

Each test verifies that:
  1. The function accepts BF16 or FP16 input without raising.
  2. The output dtype matches the input dtype.
  3. The numerical result is close to the FP32 reference within a
     dtype-appropriate tolerance.

BF16 has ~3 decimal digits of precision (eps ≈ 7.8e-3); the tolerance is
set to atol=1e-1, rtol=1e-1 to accommodate the round-trip cast loss.
FP16 has ~3.3 decimal digits (eps ≈ 4.9e-4); atol=5e-2 is comfortable.

These are correctness gates, not precision benchmarks — the goal is to
confirm that results are computed in FP32 internally and cast back, not
that low-precision arithmetic itself is accurate.
"""

from __future__ import annotations

import pytest
import torch

import trnsolver


def _make_spd(n: int) -> torch.Tensor:
    """Reproducible SPD matrix in FP32."""
    torch.manual_seed(0)
    A = torch.randn(n, n)
    return A @ A.T + n * torch.eye(n)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
class TestLowPrecisionDtype:
    """Round-trip dtype tests for all public API entry points."""

    def test_cholesky(self, dtype):
        A_fp32 = _make_spd(16)
        A = A_fp32.to(dtype)
        L = trnsolver.cholesky(A)
        assert L.dtype == dtype
        L_ref = trnsolver.cholesky(A_fp32)
        assert torch.allclose(L.float(), L_ref, atol=1e-1, rtol=1e-1)

    def test_lu(self, dtype):
        torch.manual_seed(1)
        A_fp32 = torch.randn(12, 12)
        A = A_fp32.to(dtype)
        P, L, U = trnsolver.lu(A)
        assert P.dtype == dtype
        assert L.dtype == dtype
        assert U.dtype == dtype
        # Reconstruction: P^T A = L U
        recon = L.float() @ U.float()
        ref = P.float().T @ A_fp32
        assert torch.allclose(recon, ref, atol=1e-1, rtol=1e-1)

    def test_qr(self, dtype):
        torch.manual_seed(2)
        A_fp32 = torch.randn(16, 12)
        A = A_fp32.to(dtype)
        Q, R = trnsolver.qr(A)
        assert Q.dtype == dtype
        assert R.dtype == dtype
        recon = Q.float() @ R.float()
        assert torch.allclose(recon, A_fp32, atol=1e-1, rtol=1e-1)

    def test_solve(self, dtype):
        A_fp32 = _make_spd(12)
        b_fp32 = torch.randn(12)
        A = A_fp32.to(dtype)
        b = b_fp32.to(dtype)
        x = trnsolver.solve(A, b)
        assert x.dtype == dtype
        x_ref = trnsolver.solve(A_fp32, b_fp32)
        assert torch.allclose(x.float(), x_ref, atol=1e-1, rtol=1e-1)

    def test_solve_spd(self, dtype):
        A_fp32 = _make_spd(16)
        b_fp32 = torch.randn(16)
        A = A_fp32.to(dtype)
        b = b_fp32.to(dtype)
        x = trnsolver.solve_spd(A, b)
        assert x.dtype == dtype
        x_ref = trnsolver.solve_spd(A_fp32, b_fp32)
        assert torch.allclose(x.float(), x_ref, atol=1e-1, rtol=1e-1)

    def test_inv_spd(self, dtype):
        A_fp32 = _make_spd(8)
        A = A_fp32.to(dtype)
        Ainv = trnsolver.inv_spd(A)
        assert Ainv.dtype == dtype
        # A Ainv ≈ I
        eye_approx = A.float() @ Ainv.float()
        assert torch.allclose(eye_approx, torch.eye(8), atol=2e-1, rtol=1e-1)

    def test_pinv(self, dtype):
        torch.manual_seed(3)
        A_fp32 = torch.randn(10, 6)
        A = A_fp32.to(dtype)
        Ap = trnsolver.pinv(A)
        assert Ap.dtype == dtype
        Ap_ref = trnsolver.pinv(A_fp32)
        assert torch.allclose(Ap.float(), Ap_ref, atol=1e-1, rtol=1e-1)

    def test_inv_sqrt_spd(self, dtype):
        A_fp32 = _make_spd(8)
        A = A_fp32.to(dtype)
        M = trnsolver.inv_sqrt_spd(A)
        assert M.dtype == dtype
        # M A M ≈ I  (A^{-1/2} A A^{-1/2} = I)
        prod = M.float() @ A_fp32 @ M.float()
        assert torch.allclose(prod, torch.eye(8), atol=2e-1, rtol=1e-1)

    def test_inv_sqrt_spd_ns(self, dtype):
        A_fp32 = _make_spd(8)
        A = A_fp32.to(dtype)
        M, iters, res = trnsolver.inv_sqrt_spd_ns(A, tol=1e-5)
        assert M.dtype == dtype
        assert iters > 0
        prod = M.float() @ A_fp32 @ M.float()
        assert torch.allclose(prod, torch.eye(8), atol=2e-1, rtol=1e-1)

    def test_eigh(self, dtype):
        A_fp32 = _make_spd(12)
        A = A_fp32.to(dtype)
        w, V = trnsolver.eigh(A)
        assert w.dtype == dtype
        assert V.dtype == dtype
        w_ref, _ = trnsolver.eigh(A_fp32)
        assert torch.allclose(w.float(), w_ref, atol=1e-1, rtol=1e-1)

    def test_eigh_generalized(self, dtype):
        torch.manual_seed(4)
        A_fp32 = _make_spd(12)
        B_fp32 = _make_spd(12)
        A = A_fp32.to(dtype)
        B = B_fp32.to(dtype)
        w, V = trnsolver.eigh_generalized(A, B)
        assert w.dtype == dtype
        assert V.dtype == dtype
        w_ref, _ = trnsolver.eigh_generalized(A_fp32, B_fp32)
        assert torch.allclose(w.float(), w_ref, atol=1e-1, rtol=1e-1)

    def test_cg(self, dtype):
        A_fp32 = _make_spd(16)
        b_fp32 = torch.randn(16)
        A = A_fp32.to(dtype)
        b = b_fp32.to(dtype)
        x, iters, res = trnsolver.cg(A, b, tol=1e-4)
        assert x.dtype == dtype
        assert iters > 0
        x_ref, _, _ = trnsolver.cg(A_fp32, b_fp32, tol=1e-4)
        assert torch.allclose(x.float(), x_ref, atol=2e-1, rtol=1e-1)

    def test_gmres(self, dtype):
        A_fp32 = _make_spd(12)
        b_fp32 = torch.randn(12)
        A = A_fp32.to(dtype)
        b = b_fp32.to(dtype)
        x, iters, res = trnsolver.gmres(A, b, tol=1e-4)
        assert x.dtype == dtype
        x_ref, _, _ = trnsolver.gmres(A_fp32, b_fp32, tol=1e-4)
        assert torch.allclose(x.float(), x_ref, atol=2e-1, rtol=1e-1)

    def test_ssor(self, dtype):
        A_fp32 = _make_spd(16)
        b_fp32 = torch.randn(16)
        A = A_fp32.to(dtype)
        b = b_fp32.to(dtype)
        M = trnsolver.ssor_preconditioner(A)
        x, iters, res = trnsolver.cg(A, b, M=M, tol=1e-4)
        assert x.dtype == dtype
        assert iters > 0
