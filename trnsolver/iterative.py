"""
Iterative linear solvers for Trainium.

Conjugate Gradient (CG) for SPD systems, GMRES for general systems,
and GMRES-IR (GMRES with iterative refinement) for high-accuracy general solves.

CG/GMRES inner products map to trnblas Level 1/2 ops. GMRES-IR extends the
Carson–Higham two-precision regime from solve_spd (Cholesky, SPD) to GMRES
(LU, general), using FP64 outer residuals and FP32 inner factorization.
Relevant for CPSCF (coupled-perturbed SCF) response equations.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .factor import _mH, _restore, _to_fp32


def jacobi_preconditioner(A: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a Jacobi (diagonal) preconditioner M(r) = r / diag(A).

    Cheap and effective when A is diagonally dominant. Approximates A^{-1}
    by inverting only the diagonal.
    """
    d = torch.diagonal(A).clone()
    if torch.any(d.abs() < 1e-15):
        raise ValueError("Jacobi preconditioner: diagonal has near-zero entries")
    inv_d = 1.0 / d
    # r is (n,) or a block (n, k): scale rows, never columns.
    return lambda r: r * (inv_d if r.dim() == 1 else inv_d.unsqueeze(-1))


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
    U_factor = _mH(L_factor)  # D + omega L^H: valid for symmetric / Hermitian A
    dscale = omega * (2.0 - omega) * d

    def apply(r: torch.Tensor) -> torch.Tensor:
        r_fp32, r_orig = _to_fp32(r)
        t = torch.linalg.solve_triangular(L_factor, r_fp32.unsqueeze(-1), upper=False).squeeze(-1)
        v = dscale * t
        z = torch.linalg.solve_triangular(U_factor, v.unsqueeze(-1), upper=True).squeeze(-1)
        return _restore(z, r_orig)

    return apply


@torch.compiler.disable
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

    Eager-only: convergence is a data-dependent Python loop that reads
    scalars with ``.item()``, and the returned iteration count and residual
    are Python numbers. The function is wrapped in ``torch.compiler.disable``,
    so inside a compiled region it runs eagerly behind a single graph break
    (an error under ``fullgraph=True``) instead of fragmenting the graph (#41).
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
    if b_norm == 0.0:
        # A x = 0 has the solution x = 0 for any nonsingular A, whatever x0
        # is. (A threshold of 1e-15 here returned x0, or zeros, with residual
        # 0.0 for any small-scale but nonzero b.)
        return _restore(torch.zeros_like(x), orig), 0, 0.0

    z = precond(r) if precond is not None else r.clone()

    p = z.clone()
    rz = float(r.to(torch.float64).dot(z.to(torch.float64)))

    for k in range(maxiter):
        Ap = matvec(p)
        pAp = float(p.to(torch.float64).dot(Ap.to(torch.float64)))
        # Exact breakdown only: an absolute threshold (1e-30) stopped
        # small-scale problems (A, b ~ 1e-16) before the first step.
        if pAp == 0.0:
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


@torch.compiler.disable
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

    Eager-only (``torch.compiler.disable``); see :func:`cg` (#41).
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
    if b_norm == 0.0:
        # A x = 0 has the solution x = 0 for any nonsingular A, whatever x0
        # is. (A threshold of 1e-15 here returned x0, or zeros, with residual
        # 0.0 for any small-scale but nonzero b.)
        return _restore(torch.zeros_like(x), orig), 0, 0.0

    total_iters = 0

    for _outer in range(maxiter):
        r = b - matvec(x)
        r_norm = torch.linalg.norm(r).item()
        if r_norm / b_norm < tol:
            return _restore(x, orig), total_iters, r_norm / b_norm

        # Arnoldi process. The Krylov basis V lives on b's device; the small
        # Hessenberg matrix H and the RHS g live on the host. Every entry of
        # H is already read back as a Python float, so keeping H on the host
        # removes the per-element device writes and the device lstsq without
        # adding syncs (rule 8 of the native port; CPU results unchanged).
        host = torch.device("cpu")
        m = min(restart, n)
        V = torch.zeros(n, m + 1, dtype=b.dtype, device=b.device)
        H = torch.zeros(m + 1, m, dtype=b.dtype, device=host)
        V[:, 0] = r / r_norm
        g = torch.zeros(m + 1, dtype=b.dtype, device=host)
        g[0] = r_norm

        for j in range(m):
            total_iters += 1
            w = matvec(V[:, j])

            # Modified Gram-Schmidt — FP64 inner products for numerical stability
            for i in range(j + 1):
                H[i, j] = float(w.to(torch.float64).dot(V[:, i].to(torch.float64)))
                w = w - H[i, j].item() * V[:, i]

            H[j + 1, j] = torch.linalg.norm(w).item()
            h_next = H[j + 1, j].item()
            if h_next < 1e-14:
                m = j + 1
                break
            V[:, j + 1] = w / h_next

            # Check convergence via Givens rotations on g
            # (simplified: just check residual norm)

        # Solve least-squares: min ||H @ y - g|| (host, (m+1) x m)
        y, _ = torch.linalg.lstsq(H[: m + 1, :m], g[: m + 1])[:2]
        x = x + V[:, :m] @ y[:m].to(V.device)

    r = b - matvec(x)
    return _restore(x, orig), total_iters, torch.linalg.norm(r).item() / b_norm


@torch.compiler.disable
def gmres_ir(
    A: torch.Tensor,
    b: torch.Tensor,
    tol: float = 1e-10,
    max_outer: int = 5,
    inner_tol: float = 1e-1,
    inner_maxiter: int = 50,
    inner_restart: int = 20,
) -> tuple[torch.Tensor, int, float]:
    """GMRES with iterative refinement (GMRES-IR): general Ax = b.

    Solves general (non-symmetric) linear systems via left-LU-preconditioned
    GMRES inside an FP64 outer refinement loop. Implements the Carson–Higham
    two-precision framework for general systems, extending the
    ``solve_spd(iterative_refinement=True)`` path (Cholesky + one correction
    step, SPD only) to GMRES + multiple outer iterations over any invertible A.

    Precision tiers:

    - **Factorization:**  FP32 LU of A via ``torch.linalg.lu`` (computed once,
      reused for the initial guess and every outer correction step).
    - **Inner GMRES:**    FP32 Krylov vectors and Gram-Schmidt dot products.
      The inner problem is the left-preconditioned system
      ``(U⁻¹ L⁻¹ Pᵀ A) δx = U⁻¹ L⁻¹ Pᵀ r``, which is near-identity when
      LU is accurate — so the inner GMRES converges in very few steps.
    - **Outer residual:** FP64 — captures errors below FP32 machine epsilon
      (eps32 ≈ 1.2e-7). Reliable for cond(A) ≲ 1/eps32 ≈ 1e7.
    - **Correction:**     FP32 triangular solves reusing L, U.

    .. note::
        **BF16 inner path deferred (v0.16.0).** The matvec A@v in the Arnoldi
        step could use ``matmul_kernel_fp32out`` (BF16 Tensor Engine → FP32 PSUM,
        Principle 1) for a 50–100× throughput gain. However, the Gram-Schmidt
        dot products need SR (Principle 3, Connolly–Higham–Mary 2021) for BF16
        stability at n ≥ ~300. SR is deferred to v0.16.0 when ``trnrand`` SR lands.

    .. note::
        **Callable A not supported.** The LU factorization requires a concrete
        matrix. For matrix-free operators, use ``gmres`` directly.

    Args:
        A: (n, n) general square matrix. BF16/FP16 upcast to FP32 internally.
        b: (n,) right-hand side vector. Block RHS (n, k) planned for v0.16.0.
        tol: Relative residual tolerance for the outer FP64 loop. Default 1e-10.
        max_outer: Maximum outer refinement iterations. Default 5.
        inner_tol: GMRES convergence tolerance per inner solve. Default 1e-1.
                   A coarse inner solve suffices; the outer loop corrects the error.
        inner_maxiter: Maximum restart count for inner GMRES. Default 50.
        inner_restart: Krylov subspace size for inner GMRES. Default 20.

    Returns:
        x:           (n,) solution in the original input dtype.
        total_inner: Total GMRES matvec iterations summed across all outer steps.
        residual:    Final ``‖b − Ax‖₂ / ‖b‖₂`` in FP64.

    Raises:
        TypeError: if A is callable (matrix required for LU factorization).
        torch.linalg.LinAlgError: if A is singular or factorization fails.

    Eager-only (``torch.compiler.disable``); see :func:`cg` (#41).
    """
    if callable(A):
        raise TypeError(
            "gmres_ir requires a concrete matrix for LU factorization; "
            "use gmres() directly for callable matvec operators."
        )

    A, orig = _to_fp32(A)
    b_fp32, _ = _to_fp32(b)
    b_fp64 = b_fp32.to(torch.float64)
    A_fp64 = A.to(torch.float64)

    # LU factorization (FP32): torch.linalg.lu returns (P, L, U) with A = P L U.
    # Solving (PLU) x = b: x = U⁻¹ L⁻¹ Pᵀ b
    P, L, U = torch.linalg.lu(A)

    def apply_prec(v: torch.Tensor) -> torch.Tensor:
        """Left-preconditioner: solve PLU y = v  →  y = U⁻¹ L⁻¹ Pᵀ v."""
        # solve_triangular requires 2D B; unsqueeze vector → column, squeeze back.
        col = (P.T @ v).unsqueeze(1)
        y = torch.linalg.solve_triangular(L, col, upper=False)
        z = torch.linalg.solve_triangular(U, y, upper=True)
        return z.squeeze(1)

    def prec_matvec(v: torch.Tensor) -> torch.Tensor:
        """Left-preconditioned matvec: (U⁻¹ L⁻¹ Pᵀ A) v ≈ v when LU is accurate."""
        return apply_prec(A @ v)

    # Initial guess: one exact LU solve (FP32)
    x = apply_prec(b_fp32)
    total_inner = 0

    for _ in range(max_outer):
        # FP64 residual: captures rounding errors below FP32 eps
        r_fp64 = b_fp64 - A_fp64 @ x.to(torch.float64)
        rel_res = (torch.linalg.norm(r_fp64) / torch.linalg.norm(b_fp64)).item()
        if rel_res < tol:
            break

        # Preconditioned correction RHS (FP32)
        prec_rhs = apply_prec(r_fp64.to(torch.float32))

        # Inner GMRES: solve prec_matvec(δx) = prec_rhs using the callable interface
        dx, inner_iters, _ = gmres(
            prec_matvec,
            prec_rhs,
            tol=inner_tol,
            maxiter=inner_maxiter,
            restart=inner_restart,
        )
        x = x + dx
        total_inner += inner_iters

    # Final FP64 residual (captures any improvement from the last outer step)
    r_fp64 = b_fp64 - A_fp64 @ x.to(torch.float64)
    rel_res = (torch.linalg.norm(r_fp64) / torch.linalg.norm(b_fp64)).item()

    return _restore(x, orig), total_inner, rel_res
