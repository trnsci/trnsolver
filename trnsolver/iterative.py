"""
Iterative linear solvers for Trainium.

Conjugate Gradient (CG) for SPD systems, GMRES for general systems.
These are relevant for large-scale SCF with linear-scaling methods
and for CPSCF (coupled-perturbed SCF) response equations.

All inner products and matvecs map to trnblas Level 1/2 ops.
"""

from __future__ import annotations

import torch
from typing import Optional, Callable, Tuple


def cg(
    A: torch.Tensor | Callable,
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    tol: float = 1e-6,
    maxiter: int = 1000,
    M: Optional[torch.Tensor | Callable] = None,
) -> Tuple[torch.Tensor, int, float]:
    """Conjugate Gradient solver for A @ x = b (A must be SPD).

    Args:
        A: SPD matrix (n, n) or callable matvec A(x) → A@x
        b: Right-hand side (n,)
        x0: Initial guess (default: zeros)
        tol: Convergence tolerance on relative residual
        maxiter: Maximum iterations
        M: Preconditioner matrix or callable M(r) → M^{-1}@r

    Returns:
        x: Solution vector
        iters: Number of iterations
        residual: Final relative residual norm
    """
    n = b.shape[0]
    matvec = A if callable(A) else lambda x: torch.mv(A, x)
    precond = M if callable(M) else (lambda r: torch.mv(torch.linalg.inv(M), r)) if M is not None else None

    x = x0.clone() if x0 is not None else torch.zeros(n, dtype=b.dtype, device=b.device)
    r = b - matvec(x)
    b_norm = torch.linalg.norm(b).item()
    if b_norm < 1e-15:
        return x, 0, 0.0

    if precond is not None:
        z = precond(r)
    else:
        z = r.clone()

    p = z.clone()
    rz = torch.dot(r, z).item()

    for k in range(maxiter):
        Ap = matvec(p)
        pAp = torch.dot(p, Ap).item()
        if abs(pAp) < 1e-30:
            break

        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = torch.linalg.norm(r).item()
        if r_norm / b_norm < tol:
            return x, k + 1, r_norm / b_norm

        if precond is not None:
            z = precond(r)
        else:
            z = r.clone()

        rz_new = torch.dot(r, z).item()
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return x, maxiter, torch.linalg.norm(r).item() / b_norm


def gmres(
    A: torch.Tensor | Callable,
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    tol: float = 1e-6,
    maxiter: int = 100,
    restart: int = 30,
) -> Tuple[torch.Tensor, int, float]:
    """GMRES solver for A @ x = b (general, non-symmetric).

    Restarted GMRES with Arnoldi process. Each iteration builds an
    orthonormal Krylov basis via modified Gram-Schmidt, then solves
    the least-squares Hessenberg system.

    Args:
        A: Matrix (n, n) or callable matvec
        b: Right-hand side (n,)
        x0: Initial guess
        tol: Convergence tolerance
        maxiter: Maximum outer iterations (restarts)
        restart: Krylov subspace size before restart

    Returns:
        x: Solution vector
        total_iters: Total matvec count
        residual: Final relative residual norm
    """
    n = b.shape[0]
    matvec = A if callable(A) else lambda x: torch.mv(A, x)

    x = x0.clone() if x0 is not None else torch.zeros(n, dtype=b.dtype, device=b.device)
    b_norm = torch.linalg.norm(b).item()
    if b_norm < 1e-15:
        return x, 0, 0.0

    total_iters = 0

    for outer in range(maxiter):
        r = b - matvec(x)
        r_norm = torch.linalg.norm(r).item()
        if r_norm / b_norm < tol:
            return x, total_iters, r_norm / b_norm

        # Arnoldi process
        m = min(restart, n)
        V = torch.zeros(n, m + 1, dtype=b.dtype, device=b.device)
        H = torch.zeros(m + 1, m, dtype=b.dtype, device=b.device)
        V[:, 0] = r / r_norm
        g = torch.zeros(m + 1, dtype=b.dtype, device=b.device)
        g[0] = r_norm

        for j in range(m):
            total_iters += 1
            w = matvec(V[:, j])

            # Modified Gram-Schmidt
            for i in range(j + 1):
                H[i, j] = torch.dot(w, V[:, i])
                w = w - H[i, j] * V[:, i]

            H[j + 1, j] = torch.linalg.norm(w)
            if H[j + 1, j].item() < 1e-14:
                m = j + 1
                break
            V[:, j + 1] = w / H[j + 1, j]

            # Check convergence via Givens rotations on g
            # (simplified: just check residual norm)

        # Solve least-squares: min ||H @ y - g||
        y, _ = torch.linalg.lstsq(H[:m + 1, :m], g[:m + 1])[:2]
        x = x + V[:, :m] @ y[:m]

    r = b - matvec(x)
    return x, total_iters, torch.linalg.norm(r).item() / b_norm
