"""
Matrix factorizations for Trainium.

Cholesky, LU, QR — the building blocks for linear solvers.

For DF-MP2:
- Cholesky of the Coulomb metric J = L L^T
- Triangular solve L X = B for density fitting coefficients
- These feed into trnblas.trsm for the actual solve

For SCF:
- Cholesky of overlap matrix S for generalized eigenvalue reduction
"""

from __future__ import annotations

import torch


def cholesky(A: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """Cholesky factorization: A = L @ L^T (or A = U^T @ U if upper=True).

    A must be symmetric positive definite.
    """
    L = torch.linalg.cholesky(A)
    if upper:
        return L.T
    return L


def lu(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LU factorization with partial pivoting: P @ A = L @ U

    Returns:
        P: Permutation matrix (n, n)
        L: Lower triangular with unit diagonal (n, n)
        U: Upper triangular (n, n)
    """
    LU, pivots = torch.linalg.lu_factor(A)
    P, L, U = torch.lu_unpack(LU, pivots)
    return P, L, U


def qr(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """QR factorization: A = Q @ R

    Returns:
        Q: Orthogonal matrix (m, min(m,n))
        R: Upper triangular (min(m,n), n)
    """
    return torch.linalg.qr(A)


def solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve A @ X = B for X.

    Uses LU factorization internally.
    """
    return torch.linalg.solve(A, B)


def solve_spd(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve A @ X = B where A is symmetric positive definite.

    Uses Cholesky factorization (faster than LU for SPD).
    """
    L = cholesky(A)
    squeeze = B.dim() == 1
    if squeeze:
        B = B.unsqueeze(1)
    Y = torch.linalg.solve_triangular(L, B, upper=False)
    X = torch.linalg.solve_triangular(L.T, Y, upper=True)
    if squeeze:
        X = X.squeeze(1)
    return X


def inv_spd(A: torch.Tensor) -> torch.Tensor:
    """Inverse of symmetric positive definite matrix via Cholesky.

    Used for J^{-1} in density fitting (though J^{-1/2} via solve is preferred).
    """
    n = A.shape[0]
    return solve_spd(A, torch.eye(n, dtype=A.dtype, device=A.device))


def inv_sqrt_spd(A: torch.Tensor) -> torch.Tensor:
    """A^{-1/2} for symmetric positive definite A.

    Computed via eigendecomposition: A = V diag(λ) V^T → A^{-1/2} = V diag(1/√λ) V^T

    Used in density fitting for the metric contraction J^{-1/2}.
    """
    eigenvalues, V = torch.linalg.eigh(A)
    # Clamp small eigenvalues for numerical stability
    eigenvalues = torch.clamp(eigenvalues, min=1e-12)
    inv_sqrt_eig = 1.0 / torch.sqrt(eigenvalues)
    return V @ torch.diag(inv_sqrt_eig) @ V.T


def inv_sqrt_spd_ns(
    A: torch.Tensor,
    *,
    max_iters: int = 20,
    tol: float = 1e-7,
) -> tuple[torch.Tensor, int, float]:
    """A^{-1/2} for SPD A via the coupled Newton-Schulz iteration.

    Iterates:
        Y_{k+1} = Y_k T
        Z_{k+1} = T Z_k
    with T = 0.5 (3I - Z_k Y_k), starting from Y_0 = A / s, Z_0 = I where
    s = ||A||_F is a scaling factor that keeps the spectrum of Y_0 inside
    (0, 1) for convergence. At convergence Y_k → (A/s)^{1/2} and
    Z_k → (A/s)^{-1/2}, so A^{-1/2} = Z_k / sqrt(s). Convergence is tracked
    via ||Y_k Z_k - I||_F (which must go to 0).

    All operations are GEMMs — the workload shape that maps well to the
    Trainium Tensor Engine via trnblas. Prefer this over :func:`inv_sqrt_spd`
    when the NKI backend is active and A is well-conditioned. Falls back
    to the eigendecomposition path (:func:`inv_sqrt_spd`) is recommended
    for A with eigenvalue spread beyond ~1e6.

    Args:
        A: Symmetric positive definite matrix (n, n).
        max_iters: Maximum iterations.
        tol: Relative Frobenius-norm tolerance on ||Y - I||.

    Returns:
        X: Approximation of A^{-1/2}.
        iters: Number of iterations executed.
        residual: Final ||Y - I||_F relative to ||I||_F.

    Note:
        TODO: power-iteration-based spectral-norm scaling is tighter than
        Frobenius for matrices with one dominant eigenvalue. Frobenius is
        good enough for typical DF-MP2 metric matrices.
    """
    try:
        import trnblas as _tb
    except ImportError:
        _tb = None

    n = A.shape[0]
    s = torch.linalg.norm(A, ord="fro")
    eye_n = torch.eye(n, dtype=A.dtype, device=A.device)

    Y = A / s
    Z = eye_n.clone()

    norm_I = torch.linalg.norm(eye_n, ord="fro")
    residual = float("inf")
    for k in range(max_iters):
        if _tb is not None:
            # NKI-accelerated path: O(n³) GEMMs via trnblas (blocked NKI GEMM
            # on Trainium, torch.matmul fallback elsewhere). The O(n²) scalar
            # combination for T stays on the host — no systolic advantage there.
            ZY = _tb.gemm(1.0, Z, Y)  # Z @ Y
            T = 1.5 * eye_n - 0.5 * ZY  # 0.5*(3I - Z@Y), host O(n²)
            Y = _tb.gemm(1.0, Y, T)  # Y @ T
            Z = _tb.gemm(1.0, T, Z)  # T @ Z
            YZ = _tb.gemm(1.0, Y, Z)  # Y @ Z  (convergence check)
        else:
            T = 0.5 * (3.0 * eye_n - Z @ Y)
            Y = Y @ T
            Z = T @ Z
            YZ = Y @ Z
        residual = (torch.linalg.norm(YZ - eye_n, ord="fro") / norm_I).item()
        if residual < tol:
            return Z / torch.sqrt(s), k + 1, residual

    return Z / torch.sqrt(s), max_iters, residual
