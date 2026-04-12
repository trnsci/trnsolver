"""
Eigenvalue decomposition for Trainium.

Symmetric eigenvalue problem: A @ V = V @ diag(eigenvalues)

Methods:
- Jacobi iteration (NKI-accelerable: rotation = 2×2 matmul per sweep)
- torch.linalg.eigh fallback for non-NKI backends

The Jacobi method is the natural fit for Trainium because each rotation
is a small matmul that maps to the Tensor Engine, and the sweep over
off-diagonal elements maps to the Vector Engine for max-finding.

Primary use case: SCF eigenvalue problem FC = SCε in quantum chemistry.
The generalized eigenproblem is first reduced to standard form via
Cholesky: S = L L^T, then solve (L^{-1} F L^{-T}) C' = C' ε.
"""

from __future__ import annotations

import math
import torch
from typing import Optional, Tuple

from .nki import _use_nki


def eigh(
    A: torch.Tensor,
    max_sweeps: int = 100,
    tol: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric eigenvalue decomposition: A = V @ diag(w) @ V^T

    Args:
        A: Symmetric matrix (n, n)
        max_sweeps: Maximum Jacobi sweeps
        tol: Convergence threshold (sum of squared off-diagonal elements)

    Returns:
        eigenvalues: (n,) sorted ascending
        eigenvectors: (n, n) columns are eigenvectors
    """
    if _use_nki():
        return _jacobi_eigh(A, max_sweeps, tol)
    return _torch_eigh(A)


def eigh_generalized(
    A: torch.Tensor,
    B: torch.Tensor,
    max_sweeps: int = 100,
    tol: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized symmetric eigenvalue problem: A @ x = λ B @ x

    Reduces to standard form via Cholesky of B:
        B = L @ L^T
        (L^{-1} A L^{-T}) y = λ y
        x = L^{-T} y

    This is the SCF path: solve FC = SCε where F is the Fock matrix
    and S is the overlap matrix.
    """
    L = torch.linalg.cholesky(B)
    # L^{-1} A L^{-T}
    L_inv_A = torch.linalg.solve_triangular(L, A, upper=False)
    A_prime = torch.linalg.solve_triangular(L, L_inv_A.T, upper=False).T

    # Symmetrize (numerical stability)
    A_prime = 0.5 * (A_prime + A_prime.T)

    eigenvalues, V_prime = eigh(A_prime, max_sweeps, tol)

    # Back-transform: V = L^{-T} @ V_prime
    eigenvectors = torch.linalg.solve_triangular(L.T, V_prime, upper=True)

    return eigenvalues, eigenvectors


def _torch_eigh(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback to torch.linalg.eigh."""
    return torch.linalg.eigh(A)


def _jacobi_eigh(
    A: torch.Tensor,
    max_sweeps: int = 100,
    tol: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classical Jacobi eigenvalue algorithm.

    Each sweep applies n(n-1)/2 Givens rotations to zero off-diagonal elements.
    Each rotation is a 2×2 orthogonal transform — maps to Tensor Engine on NKI.

    Convergence: quadratic for well-separated eigenvalues.
    """
    n = A.shape[0]
    assert A.shape == (n, n), f"Expected square matrix, got {A.shape}"

    # Work on a copy
    D = A.clone()
    V = torch.eye(n, dtype=A.dtype, device=A.device)

    for sweep in range(max_sweeps):
        # Check convergence: sum of squared off-diagonal elements
        off_diag = D.clone()
        off_diag.fill_diagonal_(0.0)
        off_norm = (off_diag ** 2).sum().item()
        if off_norm < tol:
            break

        # Sweep over all off-diagonal pairs
        for p in range(n):
            for q in range(p + 1, n):
                if abs(D[p, q].item()) < tol * 0.01:
                    continue

                # Compute rotation angle
                if abs(D[p, p].item() - D[q, q].item()) < 1e-15:
                    theta = math.pi / 4.0
                else:
                    tau = (D[q, q].item() - D[p, p].item()) / (2.0 * D[p, q].item())
                    if tau >= 0:
                        t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                    else:
                        t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                    c = 1.0 / math.sqrt(1.0 + t * t)
                    s = t * c
                    theta = math.atan2(s, c)

                c = math.cos(theta)
                s = math.sin(theta)

                # Apply rotation to D: D' = G^T @ D @ G
                # Only rows/cols p and q change
                for r in range(n):
                    if r == p or r == q:
                        continue
                    d_rp = D[r, p].item()
                    d_rq = D[r, q].item()
                    D[r, p] = c * d_rp - s * d_rq
                    D[p, r] = D[r, p]
                    D[r, q] = s * d_rp + c * d_rq
                    D[q, r] = D[r, q]

                d_pp = D[p, p].item()
                d_qq = D[q, q].item()
                d_pq = D[p, q].item()
                D[p, p] = c * c * d_pp - 2 * s * c * d_pq + s * s * d_qq
                D[q, q] = s * s * d_pp + 2 * s * c * d_pq + c * c * d_qq
                D[p, q] = 0.0
                D[q, p] = 0.0

                # Accumulate eigenvectors: V' = V @ G
                for r in range(n):
                    v_rp = V[r, p].item()
                    v_rq = V[r, q].item()
                    V[r, p] = c * v_rp - s * v_rq
                    V[r, q] = s * v_rp + c * v_rq

    # Extract eigenvalues from diagonal
    eigenvalues = torch.diag(D)

    # Sort ascending
    idx = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    return eigenvalues, V
