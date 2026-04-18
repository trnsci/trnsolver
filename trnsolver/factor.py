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

# BF16 and FP16 are not supported by torch.linalg on CPU. Functions in this
# module accept these dtypes by silently upcasting to FP32 for computation,
# then restoring the original dtype on output. FP32/FP64 inputs are passed
# through unchanged. (#19)
_LOW_PRECISION = (torch.bfloat16, torch.float16)


def _to_fp32(x: torch.Tensor) -> tuple[torch.Tensor, torch.dtype]:
    """Upcast BF16/FP16 to FP32 for torch.linalg compatibility.

    Returns (x_fp32_or_unchanged, original_dtype).
    """
    if x.dtype in _LOW_PRECISION:
        return x.float(), x.dtype
    return x, x.dtype


def _restore(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast x back to dtype if it was upcast from a low-precision type."""
    return x.to(dtype) if dtype in _LOW_PRECISION else x


def cholesky(A: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """Cholesky factorization: A = L @ L^T (or A = U^T @ U if upper=True).

    A must be symmetric positive definite.
    """
    A, orig = _to_fp32(A)
    L = torch.linalg.cholesky(A)
    if upper:
        return _restore(L.T, orig)
    return _restore(L, orig)


def lu(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LU factorization with partial pivoting: P @ A = L @ U

    Returns:
        P: Permutation matrix (n, n)
        L: Lower triangular with unit diagonal (n, n)
        U: Upper triangular (n, n)
    """
    A, orig = _to_fp32(A)
    LU, pivots = torch.linalg.lu_factor(A)
    P, L, U = torch.lu_unpack(LU, pivots)
    return _restore(P, orig), _restore(L, orig), _restore(U, orig)


def qr(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """QR factorization: A = Q @ R

    Returns:
        Q: Orthogonal matrix (m, min(m,n))
        R: Upper triangular (min(m,n), n)
    """
    A, orig = _to_fp32(A)
    Q, R = torch.linalg.qr(A)
    return _restore(Q, orig), _restore(R, orig)


def svd(
    A: torch.Tensor,
    full_matrices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Singular Value Decomposition: A = U @ diag(s) @ Vh

    Args:
        A: Input matrix (m, n). Real or complex.
        full_matrices: If True, return full unitary U (m, m) and Vh (n, n).
            If False (default), return economy/thin U (m, k) and Vh (k, n)
            where k = min(m, n). Default False is memory-efficient and matches
            the usage in `pinv`.

    Returns:
        U:  Left singular vectors (m, m) or (m, k)
        s:  Singular values (k,) sorted descending
        Vh: Right singular vectors conjugate-transposed (n, n) or (k, n)
    """
    A, orig = _to_fp32(A)
    U, s, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    return _restore(U, orig), _restore(s, orig), _restore(Vh, orig)


def solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve A @ X = B for X.

    Uses LU factorization internally.
    """
    A, orig = _to_fp32(A)
    B = B.to(A.dtype)
    X = torch.linalg.solve(A, B)
    return _restore(X, orig)


def solve_spd(
    A: torch.Tensor,
    B: torch.Tensor,
    iterative_refinement: bool = False,
) -> torch.Tensor:
    """Solve A @ X = B where A is symmetric positive definite.

    Uses Cholesky factorization (faster than LU for SPD).

    Args:
        A: SPD matrix (n, n).
        B: Right-hand side (n,) or (n, k).
        iterative_refinement: If True, perform one step of iterative
            refinement: compute the residual R = B - A @ X and apply a
            second Cholesky solve for the correction dX = chol_solve(R).
            Improves accuracy for ill-conditioned A at the cost of one
            extra matvec and two triangular solves. Closes #32.

    Returns:
        X: Solution (same shape as B).
    """
    A, orig = _to_fp32(A)
    B = B.to(A.dtype)
    L = cholesky(A)
    squeeze = B.dim() == 1
    if squeeze:
        B = B.unsqueeze(1)
    Y = torch.linalg.solve_triangular(L, B, upper=False)
    X = torch.linalg.solve_triangular(L.T, Y, upper=True)
    if iterative_refinement:
        # Compute residual in FP64 to capture the low-order bits that the
        # FP32 Cholesky solve missed. The correction dX is solved in FP32
        # (reusing L), then cast back. This is standard mixed-precision
        # iterative refinement: the FP64 residual is O(eps64 * cond(A))
        # rather than O(eps32 * cond(A)), giving a reliable improvement
        # for cond(A) up to ~1/eps32 ≈ 1e7.
        R = (B.double() - A.double() @ X.double()).to(A.dtype)
        dY = torch.linalg.solve_triangular(L, R, upper=False)
        dX = torch.linalg.solve_triangular(L.T, dY, upper=True)
        X = X + dX
    if squeeze:
        X = X.squeeze(1)
    return _restore(X, orig)


def inv_spd(A: torch.Tensor) -> torch.Tensor:
    """Inverse of symmetric positive definite matrix via Cholesky.

    Used for J^{-1} in density fitting (though J^{-1/2} via solve is preferred).
    """
    A, orig = _to_fp32(A)
    n = A.shape[0]
    result = solve_spd(A, torch.eye(n, dtype=A.dtype, device=A.device))
    return _restore(result, orig)


def pinv(A: torch.Tensor, rcond: float | None = None) -> torch.Tensor:
    """Moore-Penrose pseudoinverse via truncated SVD.

    For a full-rank square matrix this equals the ordinary inverse. For
    rank-deficient or non-square matrices it gives the least-squares
    minimum-norm solution operator.

    Args:
        A: Input matrix (m, n). Real or complex.
        rcond: Singular value cutoff relative to the largest singular value.
            Singular values s[i] <= rcond * s[0] are treated as zero.
            Default: machine epsilon * max(m, n), matching numpy/scipy.

    Returns:
        A^+: Pseudoinverse (n, m).
    """
    A, orig = _to_fp32(A)
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)
    if rcond is None:
        rcond = torch.finfo(s.dtype).eps * max(A.shape)
    mask = s > rcond * s[0]
    s_inv = torch.where(mask, 1.0 / s, torch.zeros_like(s))
    result = (Vh.mH * s_inv) @ U.mH
    return _restore(result, orig)


def inv_sqrt_spd(A: torch.Tensor) -> torch.Tensor:
    """A^{-1/2} for symmetric positive definite A.

    Computed via eigendecomposition: A = V diag(λ) V^T → A^{-1/2} = V diag(1/√λ) V^T

    Used in density fitting for the metric contraction J^{-1/2}.
    """
    A, orig = _to_fp32(A)
    eigenvalues, V = torch.linalg.eigh(A)
    # Clamp small eigenvalues for numerical stability
    eigenvalues = torch.clamp(eigenvalues, min=1e-12)
    inv_sqrt_eig = 1.0 / torch.sqrt(eigenvalues)
    result = V @ torch.diag(inv_sqrt_eig) @ V.T
    return _restore(result, orig)


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
    A, orig = _to_fp32(A)

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
            return _restore(Z / torch.sqrt(s), orig), k + 1, residual

    return _restore(Z / torch.sqrt(s), orig), max_iters, residual
