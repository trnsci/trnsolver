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


# ----------------------------------------------------------------------
# torch.compile support (#41)
# ----------------------------------------------------------------------
#
# Inside a compiled region (``torch.compiler.is_compiling()``) real inputs
# take traceable implementations built only from ops the Neuron backend can
# lower to StableHLO: elementwise ops, ``where`` masks, ``mm``/``matmul``,
# ``sqrt``, ``sum``, ``argmax``, ``gather``, ``eye`` and ``triu``. Each is a
# static Python loop over the matrix dimension that unrolls at trace time,
# with no ``.item()``, no data-dependent control flow and no in-place slice
# assignment. Two reasons:
#
# - ``torch.linalg.cholesky`` / ``torch.linalg.solve`` decompose to a
#   ``*_ex`` kernel plus ``aten._linalg_check_errors(info)``. That check is
#   side-effecting (it only raises), and newer PyTorch wraps it in the
#   ``with_effects`` HOP, which TorchNeuron's FxImporter cannot import.
# - Neuron has no lowering for ``linalg_cholesky_ex``, ``linalg_solve_ex`` or
#   ``linalg_qr`` themselves.
#
# Consequence: under ``torch.compile`` a non-SPD / singular input returns NaN
# or inf instead of raising ``LinAlgError``. Eager calls are unchanged and
# keep using LAPACK / MAGMA.


def _traceable(A: torch.Tensor) -> bool:
    return torch.compiler.is_compiling() and not A.is_complex()


def _cholesky_traceable(A: torch.Tensor) -> torch.Tensor:
    """Lower Cholesky factor by right-looking column elimination.

    Reads only the lower triangle of A, like LAPACK ``potrf``. Step k takes
    ``l_k = S[:, k] / sqrt(S[k, k])`` (rows >= k) from the Schur complement S
    and subtracts the rank-1 update ``l_k l_k^T``. Supports batch dims. A
    non-SPD input gives NaN from ``sqrt`` of a negative pivot.
    """
    n = A.shape[-1]
    S = torch.tril(A) + torch.tril(A, diagonal=-1).transpose(-2, -1)
    rows = torch.arange(n, device=A.device).unsqueeze(-1)  # (n, 1)
    cols = torch.arange(n, device=A.device)  # (n,)
    L = torch.zeros_like(A)
    for k in range(n):
        d = torch.sqrt(S[..., k : k + 1, k : k + 1])  # (..., 1, 1)
        col = torch.where(rows >= k, S[..., :, k : k + 1] / d, 0.0)  # (..., n, 1)
        L = torch.where(cols == k, col, L)
        S = S - col @ col.transpose(-2, -1)
    return L


def _solve_triangular_traceable(T: torch.Tensor, B: torch.Tensor, upper: bool) -> torch.Tensor:
    """Solve T X = B by forward (lower) or back (upper) substitution. B is (..., n, k)."""
    n = T.shape[-1]
    T = torch.triu(T) if upper else torch.tril(T)
    rows = torch.arange(n, device=T.device).unsqueeze(-1)
    X = torch.zeros(
        torch.broadcast_shapes(T.shape[:-2], B.shape[:-2]) + B.shape[-2:],
        dtype=B.dtype,
        device=B.device,
    )
    for k in reversed(range(n)) if upper else range(n):
        # Rows of X not yet solved are still zero, so T[k, :] @ X only picks
        # up the already-solved entries.
        r = B[..., k : k + 1, :] - T[..., k : k + 1, :] @ X
        X = torch.where(rows == k, r / T[..., k : k + 1, k : k + 1], X)
    return X


def _solve_lu_traceable(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve A X = B by Gaussian elimination with partial pivoting (GEPP).

    Same pivot choice as LAPACK ``getrf`` (first row of maximal ``|a_ik|``,
    i >= k). The row swap is a ``gather`` with a permutation index built from
    ``argmax``; elimination runs on the augmented ``[A | B]`` so no L factor
    or pivot vector is kept, then back substitution on the upper triangle.
    B is (..., n, k). A singular A gives inf/NaN.
    """
    n = A.shape[-1]
    batch = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    M = torch.cat([A.expand(*batch, n, n), B.expand(*batch, n, B.shape[-1])], dim=-1)
    rows = torch.arange(n, device=A.device)
    for k in range(n):
        mag = torch.where(rows >= k, M[..., :, k].abs(), -1.0)  # (..., n)
        p = torch.argmax(mag, dim=-1, keepdim=True)  # (..., 1)
        perm = torch.where(rows == k, p, torch.where(rows == p, k, rows))  # (..., n)
        M = torch.gather(M, -2, perm.unsqueeze(-1).expand(M.shape))
        factors = torch.where(rows > k, M[..., :, k] / M[..., k : k + 1, k], 0.0)
        M = M - factors.unsqueeze(-1) * M[..., k : k + 1, :]
    return _solve_triangular_traceable(M[..., :n], M[..., n:], upper=True)


def _cholesky_lower(A: torch.Tensor) -> torch.Tensor:
    """Lower Cholesky factor; traceable formulation when compiling."""
    if _traceable(A):
        return _cholesky_traceable(A)
    return torch.linalg.cholesky(A)


def _solve_triangular(T: torch.Tensor, B: torch.Tensor, upper: bool) -> torch.Tensor:
    """``torch.linalg.solve_triangular``; traceable formulation when compiling."""
    if _traceable(T):
        return _solve_triangular_traceable(T, B, upper)
    return torch.linalg.solve_triangular(T, B, upper=upper)


def _solve_general(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """LU-based solve; traceable formulation when compiling."""
    if _traceable(A):
        vector = B.dim() == A.dim() - 1
        X = _solve_lu_traceable(A, B.unsqueeze(-1) if vector else B)
        return X.squeeze(-1) if vector else X
    return torch.linalg.solve(A, B)


def _householder_qr(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduced Householder QR built from elementwise ops, matmuls and ``where``.

    Used by :func:`qr` inside ``torch.compile``: ``aten.linalg_qr`` has no
    StableHLO lowering on Neuron (#41). This builds the same factorization
    LAPACK's ``geqrf`` + ``orgqr`` compute, with the same sign convention
    (``R[k, k] = -sign(A[k, k]) * ||A[k:, k]||``, reflector skipped when the
    sub-column is already zero). No data-dependent Python control flow, no
    ``.item()``, no in-place slice assignment: the column loop is a static
    Python loop over ``min(m, n)`` that unrolls at trace time, and every
    per-column restriction is a row/column mask. Supports leading batch dims.
    Real dtypes only.
    """
    m, n = A.shape[-2], A.shape[-1]
    kk = min(m, n)
    R = A
    rows = torch.arange(m, device=A.device).unsqueeze(-1)  # (m, 1)
    cols = torch.arange(n, device=A.device)  # (n,)
    one = torch.ones((), dtype=A.dtype, device=A.device)
    vs = []
    taus = []
    for k in range(kk):
        below = rows >= k  # (m, 1)
        x = torch.where(below, R[..., :, k : k + 1], 0.0)  # (..., m, 1)
        alpha = R[..., k : k + 1, k : k + 1]  # (..., 1, 1)
        tail = torch.where(rows > k, x, 0.0)
        tail_sq = (tail * tail).sum(dim=-2, keepdim=True)  # (..., 1, 1)
        norm_x = torch.sqrt(alpha * alpha + tail_sq)
        sign = torch.where(alpha >= 0, one, -one)
        beta = -sign * norm_x
        active = tail_sq > 0
        denom = torch.where(active, alpha - beta, one)
        tau = torch.where(active, (beta - alpha) / torch.where(active, beta, one), 0.0)
        # v[k] = 1, v[k+1:] = x[k+1:] / (alpha - beta), v[:k] = 0
        v = torch.where(rows == k, one, tail / denom)
        # R <- (I - tau v v^T) R, restricted to columns k: (columns < k are
        # already reduced and would only pick up rounding noise).
        w = (v.transpose(-2, -1) @ R) * (cols >= k)  # (..., 1, n)
        R = R - tau * (v @ w)
        vs.append(v)
        taus.append(tau)
    # Q = H_0 H_1 ... H_{kk-1} I[:, :kk], accumulated backward (orgqr order).
    Q = torch.eye(m, kk, dtype=A.dtype, device=A.device).expand(*A.shape[:-2], m, kk)
    for k in reversed(range(kk)):
        v, tau = vs[k], taus[k]
        Q = Q - tau * (v @ (v.transpose(-2, -1) @ Q))
    R = torch.triu(R[..., :kk, :])
    return Q, R


def cholesky(A: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """Cholesky factorization: A = L @ L^T (or A = U^T @ U if upper=True).

    A must be symmetric positive definite.

    Eager calls use ``torch.linalg.cholesky``. Under ``torch.compile`` real
    inputs use :func:`_cholesky_traceable`, a right-looking column Cholesky in
    lowerable ops, because Neuron cannot lower ``linalg_cholesky_ex`` (#41).
    There a non-SPD input yields NaN instead of raising ``LinAlgError``.
    """
    A, orig = _to_fp32(A)
    L = _cholesky_lower(A)
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

    Eager calls use ``torch.linalg.qr`` (LAPACK / MAGMA). Under
    ``torch.compile`` real inputs go through :func:`_householder_qr`, a
    Householder formulation in standard aten ops, because ``aten.linalg_qr``
    does not lower to StableHLO on Neuron (#41). Both follow LAPACK's sign
    convention; results agree to rounding.
    """
    A, orig = _to_fp32(A)
    if _traceable(A):
        Q, R = _householder_qr(A)
    else:
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


def _hessenberg_reduce(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce A to upper Hessenberg form via Householder reflectors.

    Returns (H, Q) such that A = Q @ H @ Q.T, H is upper Hessenberg.
    Operates on a copy; A is not modified.
    """
    n = A.shape[0]
    H = A.clone()
    Q = torch.eye(n, dtype=A.dtype, device=A.device)

    for k in range(n - 2):
        # Extract the column below the subdiagonal
        x = H[k + 1 :, k].clone()
        norm_x = torch.linalg.norm(x)
        if norm_x < 1e-14:
            continue

        # Build Householder vector: v = x ± ||x|| e_1
        v = x.clone()
        v[0] = v[0] + torch.sign(v[0]) * norm_x if v[0] != 0 else norm_x
        v = v / torch.linalg.norm(v)

        # Apply from left: H[k+1:, k:] -= 2 v (v^T H[k+1:, k:])
        H[k + 1 :, k:] -= 2.0 * torch.outer(v, v @ H[k + 1 :, k:])
        # Apply from right: H[:, k+1:] -= 2 (H[:, k+1:] v) v^T
        H[:, k + 1 :] -= 2.0 * torch.outer(H[:, k + 1 :] @ v, v)
        # Accumulate Q: Q[:, k+1:] -= 2 (Q[:, k+1:] v) v^T
        Q[:, k + 1 :] -= 2.0 * torch.outer(Q[:, k + 1 :] @ v, v)

        # Zero out below-subdiagonal entries for numerical cleanliness
        H[k + 2 :, k] = 0.0

    return H, Q


def _francis_qr_sweep(
    H: torch.Tensor,
    Q: torch.Tensor,
    lo: int,
    hi: int,
) -> None:
    """One Francis implicit double-shift QR sweep on H[lo:hi+1, lo:hi+1].

    Modifies H and Q in-place. Uses the trailing 2×2 block to compute the
    double shift (s, t), then chases the bulge with 3×3 (then 2×2)
    Householder reflectors applied as full-column/full-row similarity transforms.
    """
    n = H.shape[0]
    # Shift parameters from trailing 2×2
    s = H[hi - 1, hi - 1] + H[hi, hi]
    t = H[hi - 1, hi - 1] * H[hi, hi] - H[hi - 1, hi] * H[hi, hi - 1]

    # Starting vector: first column of H^2 - sH + tI (restricted to active block)
    h00 = H[lo, lo]
    h10 = H[lo + 1, lo]
    h01 = H[lo, lo + 1]
    h11 = H[lo + 1, lo + 1]

    p0 = h00 * h00 + h01 * h10 - s * h00 + t
    p1 = h10 * (h00 + h11 - s)
    if hi - lo >= 2:
        p2 = h10 * H[lo + 2, lo + 1]
    else:
        p2 = torch.zeros((), dtype=H.dtype, device=H.device)

    for k in range(lo, hi):
        nr = min(3, hi - k + 1)  # 3 normally, 2 at the last step (k = hi-1)
        p = torch.stack([p0, p1, p2] if nr == 3 else [p0, p1])
        norm_p = torch.linalg.norm(p)
        if norm_p < 1e-14:
            break

        # Householder sign convention: maximize |p[0]| after reflection
        sign_p0 = (
            torch.sign(p[0]) if p[0] != 0.0 else torch.ones((), dtype=p.dtype, device=p.device)
        )
        p = p.clone()
        p[0] = p[0] + sign_p0 * norm_p
        p = p / torch.linalg.norm(p)

        # Left: restrict to col max(lo, k-1) onwards.  Columns < k-1 are
        # zero (Hessenberg) so there's nothing to apply, but we must not
        # touch col k-1 for k > lo because those rows are already zero there
        # and applying P from the left would create nonzero fill in columns
        # strictly left of the active block that are never corrected by the
        # matching right application.
        col_start = max(lo, k - 1)
        H[k : k + nr, col_start:] -= 2.0 * torch.outer(p, p @ H[k : k + nr, col_start:])
        # Right: restrict rows to those that can be non-zero (Hessenberg structure).
        # Rows > k+nr are zero in cols k:k+nr for a Hessenberg matrix, except for
        # the bulge at row k+nr introduced by this step's left application.
        row_end = min(k + nr + 1, n)
        H[:row_end, k : k + nr] -= 2.0 * torch.outer(H[:row_end, k : k + nr] @ p, p)
        # Accumulate orthogonal factor (all rows of Q are always dense)
        Q[:, k : k + nr] -= 2.0 * torch.outer(Q[:, k : k + nr] @ p, p)

        # Read starting vector for next step from the bulge below the subdiagonal
        if k + 1 <= hi - 2:
            p0 = H[k + 1, k]
            p1 = H[k + 2, k]
            p2 = H[k + 3, k] if k + 3 <= hi else torch.zeros((), dtype=H.dtype, device=H.device)
        else:
            p0 = H[k + 1, k]
            p1 = H[k + 2, k] if k + 2 <= hi else torch.zeros((), dtype=H.dtype, device=H.device)
            p2 = torch.zeros((), dtype=H.dtype, device=H.device)


def _schur_iterate(
    H: torch.Tensor,
    Q: torch.Tensor,
    tol: float,
    max_sweeps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drive the Francis QR iteration to convergence via deflation.

    Returns (T, Q_total) where A = Q_total @ T @ Q_total.T.

    Scans from hi downward each iteration to find the top of the current
    unreduced block; zeros any small subdiagonal entry found and re-classifies
    the window size before deciding whether to apply another sweep. Correctly
    handles interior deflations by updating lo from the scan result.
    """
    n = H.shape[0]
    hi = n - 1  # active upper boundary; shrinks as blocks are deflated
    sweep = 0

    while hi >= 1 and sweep < max_sweeps:
        # Scan from hi downward to find top of current unreduced block [lo, hi]
        lo = hi
        while lo >= 1:
            sub = abs(H[lo, lo - 1].item())
            diag = abs(H[lo - 1, lo - 1].item()) + abs(H[lo, lo].item())
            if sub <= tol * diag:
                H[lo, lo - 1] = 0.0  # explicitly zero for numerical cleanliness
                break
            lo -= 1
        # lo == 0: no interior deflation found; active window is [0, hi]

        if lo >= hi:
            # H[hi, hi-1] was already small: 1×1 block deflated at hi
            hi -= 1
            continue

        if hi - lo == 1:
            # 2×2 active block — resolve based on eigenvalue type.
            a = H[lo, lo].item()
            b = H[lo, lo + 1].item()
            c = H[lo + 1, lo].item()
            d = H[lo + 1, lo + 1].item()
            disc = ((a - d) / 2) ** 2 + b * c
            if disc < 0:
                # Complex conjugate pair — leave as quasi-Schur 2×2 block.
                hi -= 2
            else:
                # Real eigenvalue pair — apply one Givens rotation to split.
                # Choose the eigenvalue of the 2×2 closest to d as the shift.
                sq = disc**0.5
                mu = (
                    (a + d) / 2 + sq
                    if abs((a + d) / 2 + sq - d) < abs((a + d) / 2 - sq - d)
                    else (a + d) / 2 - sq
                )
                # Givens rotation: zero H[lo+1, lo] in H-mu*I
                x = a - mu
                y = c
                r = (x**2 + y**2) ** 0.5
                if r > 1e-14:
                    ct = x / r
                    st = y / r
                    G = torch.tensor([[ct, st], [-st, ct]], dtype=H.dtype, device=H.device)
                    # Similarity: H[lo:lo+2, :] = G @ H[lo:lo+2, :]; H[:, lo:lo+2] = H[:, lo:lo+2] @ G.T
                    H[lo : lo + 2, :] = G @ H[lo : lo + 2, :]
                    H[:, lo : lo + 2] = H[:, lo : lo + 2] @ G.T
                    Q[:, lo : lo + 2] = Q[:, lo : lo + 2] @ G.T
                # Now H[lo+1, lo] should be near zero; deflate both as 1×1.
                if abs(H[lo + 1, lo].item()) <= tol * (
                    abs(H[lo, lo].item()) + abs(H[lo + 1, lo + 1].item())
                ):
                    H[lo + 1, lo] = 0.0
                    hi -= 1  # deflate hi as 1×1; lo is next iteration's hi
                # else: didn't converge (rare for real pair), let outer loop retry
            continue

        # Active window [lo, hi] with hi - lo >= 2: apply Francis double-shift sweep
        _francis_qr_sweep(H, Q, lo, hi)
        sweep += 1

    return H, Q


def schur(
    A: torch.Tensor,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Schur decomposition: A = Q @ T @ Q.T

    T is quasi-upper triangular — upper triangular with possible 2×2 blocks
    on the diagonal corresponding to complex-conjugate eigenvalue pairs of
    real A. For symmetric A all blocks are 1×1 (T is diagonal = eigenvalues).

    Host-only (no NKI dispatch): device inputs are copied to the CPU and the
    factors are returned on the input's device. Uses Householder Hessenberg reduction
    followed by Francis implicit double-shift QR iteration with deflation.
    NKI acceleration of the Hessenberg stage is deferred to Phase 3.

    Args:
        A: Square real matrix (n, n).
        tol: Deflation tolerance (relative to sum of adjacent diagonal magnitudes).

    Returns:
        Q: Orthogonal matrix (n, n) such that A = Q @ T @ Q.T.
        T: Quasi-upper triangular Schur form (n, n).
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"schur: A must be square, got shape {tuple(A.shape)}")
    A, orig = _to_fp32(A)
    # Hessenberg reduction and the Francis iteration are scalar host code
    # (per-element reads for deflation and shifts, small reflector updates).
    # Run them on a CPU copy so a device-resident input does not pay a launch
    # or sync per scalar, as the eigh QR iteration does (#42); move the result
    # back to A's device. CPU inputs are unchanged (`.to` is a no-op).
    device = A.device
    H, Q = _hessenberg_reduce(A.detach().to("cpu"))
    T, Q = _schur_iterate(H, Q, tol, max_sweeps=30 * A.shape[0])
    return _restore(Q.to(device), orig), _restore(T.to(device), orig)


def solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve A @ X = B for X.

    Uses LU factorization with partial pivoting. Eager calls use
    ``torch.linalg.solve``. Under ``torch.compile`` real inputs use
    :func:`_solve_lu_traceable` (Gaussian elimination with partial pivoting
    in lowerable ops), because Neuron cannot lower ``linalg_solve_ex`` (#41).
    There a singular A yields inf/NaN instead of raising ``LinAlgError``.
    """
    A, orig = _to_fp32(A)
    B = B.to(A.dtype)
    X = _solve_general(A, B)
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
    Y = _solve_triangular(L, B, upper=False)
    X = _solve_triangular(L.T, Y, upper=True)
    if iterative_refinement:
        # Compute residual in FP64 to capture the low-order bits that the
        # FP32 Cholesky solve missed. The correction dX is solved in FP32
        # (reusing L), then cast back. This is standard mixed-precision
        # iterative refinement: the FP64 residual is O(eps64 * cond(A))
        # rather than O(eps32 * cond(A)), giving a reliable improvement
        # for cond(A) up to ~1/eps32 ≈ 1e7.
        R = (B.double() - A.double() @ X.double()).to(A.dtype)
        dY = _solve_triangular(L, R, upper=False)
        dX = _solve_triangular(L.T, dY, upper=True)
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
