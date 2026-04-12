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
from typing import Tuple, Optional


def cholesky(A: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """Cholesky factorization: A = L @ L^T (or A = U^T @ U if upper=True).

    A must be symmetric positive definite.
    """
    L = torch.linalg.cholesky(A)
    if upper:
        return L.T
    return L


def lu(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LU factorization with partial pivoting: P @ A = L @ U

    Returns:
        P: Permutation matrix (n, n)
        L: Lower triangular with unit diagonal (n, n)
        U: Upper triangular (n, n)
    """
    LU, pivots = torch.linalg.lu_factor(A)
    P, L, U = torch.lu_unpack(LU, pivots)
    return P, L, U


def qr(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
