"""
Iterative linear solvers for Trainium.

Conjugate Gradient (CG) for SPD systems, GMRES for general systems.
These are relevant for large-scale SCF with linear-scaling methods
and for CPSCF (coupled-perturbed SCF) response equations.

All inner products and matvecs map to trnblas Level 1/2 ops.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def jacobi_preconditioner(A: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a Jacobi (diagonal) preconditioner M(r) = r / diag(A).

    Cheap and effective when A is diagonally dominant. Approximates A^{-1}
    by inverting only the diagonal.
    """
    d = torch.diagonal(A).clone()
    if torch.any(d.abs() < 1e-15):
        raise ValueError("Jacobi preconditioner: diagonal has near-zero entries")
    inv_d = 1.0 / d
    return lambda r: r * inv_d


def cg(
    A: torch.Tensor | Callable,
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    tol: float = 1e-6,
    maxiter: int = 1000,
    M: torch.Tensor | Callable | None = None,
) -> tuple[torch.Tensor, int, float]:
    """Conjugate Gradient solver for A @ x = b (A must be SPD).

    Args:
        A: SPD matrix (n, n) or callable matvec A(x) → A@x
        b: Right-hand side (n,)
        x0: Initial guess (default: zeros)
        tol: Convergence tolerance on relative residual
        maxiter: Maximum iterations
        M: Preconditioner as callable M(r) → approx A^{-1} @ r, or tensor
            (interpreted as the already-inverted preconditioner applied as M@r).
            Use `jacobi_preconditioner(A)` for the common diagonal case.

    Returns:
        x: Solution vector
        iters: Number of iterations
        residual: Final relative residual norm
    """
    n = b.shape[0]
    matvec = A if callable(A) else lambda x: torch.mv(A, x)
    if M is None:
        precond = None
    elif callable(M):
        precond = M
    else:

        def precond(r):
            return torch.mv(M, r)

    x = x0.clone() if x0 is not None else torch.zeros(n, dtype=b.dtype, device=b.device)
    r = b - matvec(x)
    b_norm = torch.linalg.norm(b).item()
    if b_norm < 1e-15:
        return x, 0, 0.0

    z = precond(r) if precond is not None else r.clone()

    p = z.clone()
    rz = float(r.to(torch.float64).dot(z.to(torch.float64)))

    for k in range(maxiter):
        Ap = matvec(p)
        pAp = float(p.to(torch.float64).dot(Ap.to(torch.float64)))
        if abs(pAp) < 1e-30:
            break

        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = torch.linalg.norm(r).item()
        if r_norm / b_norm < tol:
            return x, k + 1, r_norm / b_norm

        z = precond(r) if precond is not None else r.clone()

        rz_new = float(r.to(torch.float64).dot(z.to(torch.float64)))
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return x, maxiter, torch.linalg.norm(r).item() / b_norm


def gmres(
    A: torch.Tensor | Callable,
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    tol: float = 1e-6,
    maxiter: int = 100,
    restart: int = 30,
) -> tuple[torch.Tensor, int, float]:
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

    for _outer in range(maxiter):
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

            # Modified Gram-Schmidt — FP64 inner products for numerical stability
            for i in range(j + 1):
                H[i, j] = float(w.to(torch.float64).dot(V[:, i].to(torch.float64)))
                w = w - H[i, j] * V[:, i]

            H[j + 1, j] = torch.linalg.norm(w)
            if H[j + 1, j].item() < 1e-14:
                m = j + 1
                break
            V[:, j + 1] = w / H[j + 1, j]

            # Check convergence via Givens rotations on g
            # (simplified: just check residual norm)

        # Solve least-squares: min ||H @ y - g||
        y, _ = torch.linalg.lstsq(H[: m + 1, :m], g[: m + 1])[:2]
        x = x + V[:, :m] @ y[:m]

    r = b - matvec(x)
    return x, total_iters, torch.linalg.norm(r).item() / b_norm
