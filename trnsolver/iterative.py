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

from .factor import _restore, _to_fp32


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


def block_jacobi_preconditioner(
    A: torch.Tensor,
    block_size: int = 16,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a block-Jacobi preconditioner from diagonal blocks of A.

    Each diagonal block of size `block_size` is Cholesky-factorized
    independently. Applying the preconditioner solves the block-diagonal
    system exactly via back-substitution. More effective than scalar Jacobi
    for problems with localized coupling (FEM stiffness matrices,
    density-fitting metric matrices with nearby-basis-function coupling).

    Architecturally, per-block Cholesky solves are independent and map
    naturally to NeuronCore-parallel execution on Trainium.

    Args:
        A: Symmetric positive definite matrix (n, n).
        block_size: Diagonal block size. Last block may be smaller.
            block_size >= n reduces to a full Cholesky preconditioner.

    Returns:
        Callable M(r) → approx A^{-1} @ r for use as M= argument to `cg`.

    Raises:
        ValueError: If any diagonal block is not positive definite.
    """
    A, _ = _to_fp32(A)
    n = A.shape[0]
    blocks = []
    start = 0
    while start < n:
        end = min(start + block_size, n)
        block = A[start:end, start:end].clone()
        try:
            L = torch.linalg.cholesky(block)
        except torch.linalg.LinAlgError as exc:
            raise ValueError(
                f"block_jacobi_preconditioner: block [{start}:{end}] is not positive definite"
            ) from exc
        blocks.append((start, end, L))
        start = end

    def apply(r: torch.Tensor) -> torch.Tensor:
        r_fp32, r_orig = _to_fp32(r)
        z = torch.empty_like(r_fp32)
        for s, e, L in blocks:
            z[s:e] = torch.cholesky_solve(r_fp32[s:e].unsqueeze(1), L).squeeze(1)
        return _restore(z, r_orig)

    return apply


def ssor_preconditioner(
    A: torch.Tensor,
    omega: float = 1.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build an SSOR preconditioner for symmetric positive definite A.

    For symmetric A = D + L + L^T (D diagonal, L strictly lower triangular),
    the SSOR preconditioner with relaxation ω ∈ (0, 2) applies M^{-1} r via:
      1. Forward solve:  (D + ω L) t = r
      2. Scale:          v = ω(2-ω) * diag(A) ⊙ t
      3. Backward solve: (D + ω L^T) z = v

    At ω = 1 this reduces to the symmetric Gauss-Seidel step. M_SSOR is SPD
    when A is SPD and ω ∈ (0, 2), so it is safe as a CG preconditioner.

    SSOR outperforms scalar Jacobi on matrices with off-diagonal coupling —
    FEM stiffness matrices, Poisson-like systems, and density-fitting metrics
    with neighbour-basis coupling.

    Args:
        A: Symmetric positive definite matrix (n, n).
        omega: Relaxation parameter in (0, 2). Default 1.0 (symmetric Gauss-Seidel).

    Returns:
        Callable M(r) → approx A^{-1} @ r for use as M= argument to `cg`.

    Raises:
        ValueError: If omega is not in (0, 2) or the diagonal has near-zero entries.
    """
    if not (0.0 < omega < 2.0):
        raise ValueError(f"ssor_preconditioner: omega must be in (0, 2), got {omega}")
    A, _ = _to_fp32(A)
    d = torch.diagonal(A).clone()
    if torch.any(d.abs() < 1e-15):
        raise ValueError("ssor_preconditioner: diagonal has near-zero entries")
    # Lower triangular factor: D + ω * L_strictly_lower
    L_factor = omega * torch.tril(A, diagonal=-1) + torch.diag(d)
    U_factor = L_factor.T  # valid for symmetric A
    dscale = omega * (2.0 - omega) * d

    def apply(r: torch.Tensor) -> torch.Tensor:
        r_fp32, r_orig = _to_fp32(r)
        t = torch.linalg.solve_triangular(L_factor, r_fp32.unsqueeze(-1), upper=False).squeeze(-1)
        v = dscale * t
        z = torch.linalg.solve_triangular(U_factor, v.unsqueeze(-1), upper=True).squeeze(-1)
        return _restore(z, r_orig)

    return apply


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
    b, orig = _to_fp32(b)
    if not callable(A):
        A = A.to(b.dtype)
    if x0 is not None:
        x0 = x0.to(b.dtype)
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
        return _restore(x, orig), 0, 0.0

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
            return _restore(x, orig), k + 1, r_norm / b_norm

        z = precond(r) if precond is not None else r.clone()

        rz_new = float(r.to(torch.float64).dot(z.to(torch.float64)))
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return _restore(x, orig), maxiter, torch.linalg.norm(r).item() / b_norm


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
    b, orig = _to_fp32(b)
    if not callable(A):
        A = A.to(b.dtype)
    if x0 is not None:
        x0 = x0.to(b.dtype)
    n = b.shape[0]
    matvec = A if callable(A) else lambda x: torch.mv(A, x)

    x = x0.clone() if x0 is not None else torch.zeros(n, dtype=b.dtype, device=b.device)
    b_norm = torch.linalg.norm(b).item()
    if b_norm < 1e-15:
        return _restore(x, orig), 0, 0.0

    total_iters = 0

    for _outer in range(maxiter):
        r = b - matvec(x)
        r_norm = torch.linalg.norm(r).item()
        if r_norm / b_norm < tol:
            return _restore(x, orig), total_iters, r_norm / b_norm

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
    return _restore(x, orig), total_iters, torch.linalg.norm(r).item() / b_norm
