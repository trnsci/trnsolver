"""
Block conjugate gradient for many right-hand sides on matrix-free operators.

``cg_block`` runs R independent preconditioned CG recurrences, one per column
of an (n, R) right-hand side, driven by a single operator application A(P) on
the whole (n, R) block per iteration. For a matrix-free operator, such as a
fused kernel-matrix product, the application is the whole cost, so R solves
cost about as much as one. The Gaussian-process exemplar (trnsci/trnsci#109)
solves (K + σ²I) X = [y, z₁ … z_p] this way: the posterior-mean solve and
every log-det probe solve share each operator application.

Per-iteration math stays on device tensors: converged columns are frozen with
``torch.where`` masks (α = β = 0, X stops changing) and the block keeps its
shape, so each iteration is one fixed-shape launch. The host reads a value
back only every ``check_every`` iterations, to decide whether to stop.

Also here:

- ``cg_block_ir``: fp32 inner ``cg_block`` inside a float64 host refinement
  loop (the same shape as ``gmres_ir``), for kernel matrices with small noise
  (κ ≈ 1e6 and above), where fp32 CG stagnates near 1e-5 relative.
- ``pivoted_cholesky`` / ``pivoted_lowrank_preconditioner``: a rank-k
  partially pivoted Cholesky factor from diagonal and row access only, and
  the (L Lᵀ + shift·I)⁻¹ preconditioner applied with two (n × k) GEMMs.
- ``lanczos_tridiag_from_cg`` / ``slq_logdet``: Lanczos tridiagonals from the
  CG coefficients, and stochastic Lanczos quadrature for log det(A) (#97).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from .factor import _restore, _to_fp32

_HOST = torch.device("cpu")


def _as_block_op(A, dtype) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a callable on (n, R) blocks for a tensor or callable A."""
    if callable(A):
        return A
    A = A.to(dtype)
    return lambda V: A @ V


def _cdot(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Column-wise inner products Re(xⱼᴴ yⱼ), shape (R,)."""
    s = (X.conj() * Y).sum(dim=0) if X.is_complex() else (X * Y).sum(dim=0)
    return s.real if s.is_complex() else s


def _col_norm(X: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(X, dim=0)


@torch.compiler.disable
def cg_block(
    A: torch.Tensor | Callable,
    B: torch.Tensor,
    M: torch.Tensor | Callable | None = None,
    *,
    tol: float = 1e-6,
    maxiter: int | None = None,
    check_every: int = 10,
    X0: torch.Tensor | None = None,
    return_coeffs: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Column-batched preconditioned CG for A X = B (A SPD), one A(P) per iteration.

    Args:
        A: SPD operator. A callable on an (n, R) block returning A @ block, or
            a dense (n, n) tensor (cast to B's dtype, applied as ``A @ V``).
            The callable is always given the full (n, R) block, converged
            columns included, so every call has the same shape.
        B: Right-hand sides (n, R), or a single (n,) vector (X is then (n,)).
        M: Optional preconditioner applying ≈ A⁻¹: a callable on (n, R)
            blocks, or a tensor applied as ``M @ V``. ``pivoted_lowrank_
            preconditioner`` returns a suitable callable.
        tol: Per-column convergence ‖rⱼ‖ ≤ tol · ‖bⱼ‖ on the recurrence
            residual (unpreconditioned 2-norm).
        maxiter: Iteration cap (default 10 n, as in scipy).
        check_every: Read the convergence state back to the host (one sync)
            every this many iterations; between checks nothing is read back.
            Columns still freeze on the exact iteration they converge, so
            this only changes how many extra (all-frozen) iterations can run
            after the last column converges, never the solution.
        X0: Initial guess (n, R) or (n,). Costs one extra operator call.
        return_coeffs: Also return the per-column CG coefficients α and β
            (for ``lanczos_tridiag_from_cg`` and stochastic log-det).

    Returns:
        X: Solution, shape and dtype of B (bf16/fp16 computed in fp32).
        info: dict with

        - ``"iterations"`` (int): loop iterations run, which is the number of
          operator calls in the loop.
        - ``"col_iterations"`` (int64 tensor (R,)): iterations column j took
          to converge (or ran before ``maxiter``/breakdown).
        - ``"converged"`` (bool tensor (R,)).
        - ``"residual"`` (tensor (R,)): recurrence ‖rⱼ‖ / ‖bⱼ‖.
        - ``"true_residual"`` (tensor (R,)): ‖bⱼ − A xⱼ‖ / ‖bⱼ‖, from one
          extra operator call at the end. In fp32 the recurrence residual
          drifts below the true one, so report both.
        - ``"breakdown"`` (bool tensor (R,)): pᵀAp ≤ 0 was hit (A not SPD
          for that column, or exhausted precision); the column froze.
        - with ``return_coeffs``: ``"alpha"``, ``"beta"`` (tensors
          (iterations, R); entries after a column froze are 0, so use the
          first ``col_iterations[j]`` rows of column j) and ``"rz0"``
          (tensor (R,), r₀ᵀ M r₀, the squared norm of the whitened start
          vector).

        Zero columns of B return zero, count as converged at iteration 0,
        and report residual 0.

    Inner products run in the working dtype on B's device (no float64 on
    device). Eager-only (``torch.compiler.disable``); see :func:`cg` (#41).
    """
    B, orig = _to_fp32(B)
    vector = B.dim() == 1
    if vector:
        B = B.unsqueeze(-1)
    if B.dim() != 2:
        raise ValueError(f"cg_block: B must be (n,) or (n, R), got shape {tuple(B.shape)}")
    if check_every < 1:
        raise ValueError(f"cg_block: check_every must be >= 1, got {check_every}")
    n, R = B.shape
    if maxiter is None:
        maxiter = 10 * n
    dtype, device = B.dtype, B.device
    rdtype = B.real.dtype if B.is_complex() else dtype
    op = _as_block_op(A, dtype)
    precond = None if M is None else _as_block_op(M, dtype)

    b_norm = _col_norm(B)
    nonzero = b_norm > 0
    thresh = tol * b_norm
    if X0 is not None:
        X0 = X0.to(dtype=dtype, device=device)
        X = (X0.unsqueeze(-1) if vector else X0).clone()
        X = torch.where(nonzero, X, torch.zeros_like(X))
        Rr = B - op(X)
    else:
        X = torch.zeros_like(B)
        Rr = B.clone()
    Z = precond(Rr) if precond is not None else Rr
    P = Z.clone()
    rz = _cdot(Rr, Z)
    rz0 = rz.clone()
    active = nonzero & (_col_norm(Rr) > thresh)
    breakdown = torch.zeros(R, dtype=torch.bool, device=device)
    col_iters = torch.zeros(R, dtype=torch.int64, device=device)
    one = torch.ones((), dtype=rdtype, device=device)
    zero = torch.zeros((), dtype=rdtype, device=device)
    alphas: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []

    k = 0
    while k < maxiter:
        if k % check_every == 0 and not bool(active.any()):
            break
        AP = op(P)
        pAp = _cdot(P, AP)
        ok = pAp > 0
        breakdown = breakdown | (active & ~ok)
        active = active & ok
        alpha = torch.where(active, rz / torch.where(ok, pAp, one), zero)
        X = X + alpha * P
        Rr = Rr - alpha * AP
        Z = precond(Rr) if precond is not None else Rr
        rz_new = _cdot(Rr, Z)
        rz_ok = rz != 0
        beta = torch.where(active & rz_ok, rz_new / torch.where(rz_ok, rz, one), zero)
        P = torch.where(active, Z + beta * P, P)
        rz = torch.where(active, rz_new, rz)
        col_iters = col_iters + active.to(torch.int64)
        if return_coeffs:
            alphas.append(alpha)
            betas.append(beta)
        active = active & (_col_norm(Rr) > thresh)
        k += 1

    safe_norm = torch.where(nonzero, b_norm, torch.ones_like(b_norm))
    res = torch.where(nonzero, _col_norm(Rr) / safe_norm, torch.zeros_like(b_norm))
    true_res = torch.where(nonzero, _col_norm(B - op(X)) / safe_norm, torch.zeros_like(b_norm))
    info: dict = {
        "iterations": k,
        "col_iterations": col_iters,
        "converged": ~nonzero | ((res <= tol) & ~breakdown),
        "residual": res,
        "true_residual": true_res,
        "breakdown": breakdown,
    }
    if return_coeffs:
        empty = torch.zeros(0, R, dtype=rdtype, device=device)
        info["alpha"] = torch.stack(alphas) if alphas else empty
        info["beta"] = torch.stack(betas) if betas else empty
        info["rz0"] = rz0
    if vector:
        X = X.squeeze(-1)
    return _restore(X, orig), info


@torch.compiler.disable
def cg_block_ir(
    A: torch.Tensor | Callable,
    B: torch.Tensor,
    A_hi: torch.Tensor | Callable | None = None,
    M: torch.Tensor | Callable | None = None,
    *,
    tol: float = 1e-6,
    inner_tol: float = 1e-4,
    max_outer: int = 10,
    **cg_kwargs,
) -> tuple[torch.Tensor, dict]:
    """Mixed-precision iterative refinement around ``cg_block`` (same shape as ``gmres_ir``).

    Each outer step forms the residual R = B − A_hi(X) in float64 on the
    host, solves A D = R column-scaled in fp32 with ``cg_block`` on the fast
    operator ``A`` (to ``inner_tol``), and adds the correction, X += D, in
    float64. Columns already at ``tol`` get a zero inner right-hand side, so
    they cost nothing extra and stay put.

    Args:
        A: Fast operator for the inner solve: a callable on fp32 (n, R)
            blocks on B's device, or an (n, n) tensor (cast to fp32).
        B: Right-hand sides (n, R) or (n,).
        A_hi: Accurate operator for the outer residual: a callable taking a
            float64 (n, R) host tensor, or an (n, n) tensor (cast to float64
            on the host). Defaults to A in float64: a tensor A is cast, a
            callable A is called on the float64 host block (pass ``A_hi``
            if A cannot take float64 host input).
        M: Preconditioner for the inner solves (see ``cg_block``).
        tol: Outer per-column relative true-residual target (float64).
        inner_tol: ``cg_block`` tolerance for each correction solve.
        max_outer: Maximum number of corrections.
        **cg_kwargs: Passed to ``cg_block`` (``maxiter``, ``check_every``).

    Returns:
        X: float64 solution on the host, shape of B. (fp64 because an fp32
            X cannot hold a 1e-10 relative residual.)
        info: dict with ``"outer_iterations"`` (int, corrections applied),
        ``"inner_iterations"`` (list of int, ``cg_block`` iterations per
        correction), ``"residual_history"`` (float64 tensor (outer + 1, R):
        the true relative residual before each correction and after the
        last one), ``"true_residual"`` (final, (R,)) and ``"converged"``
        (bool (R,)).

    Eager-only (``torch.compiler.disable``); see :func:`cg` (#41).
    """
    if "return_coeffs" in cg_kwargs or "X0" in cg_kwargs:
        raise TypeError("cg_block_ir: return_coeffs and X0 are not supported")
    vector = B.dim() == 1
    Bm = B.unsqueeze(-1) if vector else B
    device = Bm.device
    inner_dtype = torch.complex64 if Bm.is_complex() else torch.float32
    hi_dtype = torch.complex128 if Bm.is_complex() else torch.float64
    B64 = Bm.to(device=_HOST, dtype=hi_dtype)

    A_lo = A if callable(A) else A.to(device=device, dtype=inner_dtype)
    if A_hi is None:
        A_hi = A
    if callable(A_hi):
        hi_fn = A_hi

        def A_hi_op(X):
            return hi_fn(X).to(device=_HOST, dtype=hi_dtype)
    else:
        A64 = A_hi.to(device=_HOST, dtype=hi_dtype)

        def A_hi_op(X):
            return A64 @ X

    b_norm = _col_norm(B64)
    nonzero = b_norm > 0
    safe_norm = torch.where(nonzero, b_norm, torch.ones_like(b_norm))
    X = torch.zeros_like(B64)
    history: list[torch.Tensor] = []
    inner_iters: list[int] = []
    for outer in range(max_outer + 1):
        Rr = B64 - A_hi_op(X) if outer > 0 else B64.clone()
        r_norm = _col_norm(Rr)
        rel = torch.where(nonzero, r_norm / safe_norm, torch.zeros_like(b_norm))
        history.append(rel)
        todo = nonzero & (rel > tol)
        if outer == max_outer or not bool(todo.any()):
            break
        scale = torch.where(todo, r_norm, torch.ones_like(r_norm))
        rhs = torch.where(todo, Rr / scale, torch.zeros_like(Rr))
        D, inner = cg_block(
            A_lo, rhs.to(device=device, dtype=inner_dtype), M, tol=inner_tol, **cg_kwargs
        )
        X = X + D.to(device=_HOST, dtype=hi_dtype) * scale
        inner_iters.append(inner["iterations"])

    info = {
        "outer_iterations": len(inner_iters),
        "inner_iterations": inner_iters,
        "residual_history": torch.stack(history),
        "true_residual": history[-1],
        "converged": ~nonzero | (history[-1] <= tol),
    }
    return (X.squeeze(-1) if vector else X), info


def pivoted_cholesky(
    diag_fn: Callable[[], torch.Tensor],
    rows_fn: Callable[[torch.Tensor], torch.Tensor],
    rank: int,
    *,
    tol: float = 0.0,
    block: int = 1,
    sub_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    relax: float = 1.0,
) -> torch.Tensor:
    """Greedy partially pivoted Cholesky, K ≈ L Lᵀ, from diagonal and row access.

    Each step adds the pivot i with the largest remaining diagonal of the
    Schur complement and the column l = (K[:, i] − L Lᵢᵀ) / √dᵢ. Total K
    access is the diagonal plus the requested rows; no n × n matrix is
    formed.

    ``block`` > 1 batches the work (#114). A round takes the ``block``
    largest remaining diagonals as candidates and runs greedy pivoted
    Cholesky on their exact c × c Schur complement, accepting a candidate
    only while its remaining diagonal is at least ``relax`` times the
    largest remaining diagonal outside the round (which can only have
    shrunk since the round began). With ``relax=1`` that is exactly the
    pivot the one-row-per-step algorithm takes, so ``block`` changes the
    number of ``rows_fn`` calls, not the pivots (up to ties and rounding).
    The accepted pivots' rows then come from one ``rows_fn`` call, their
    Schur rows from one (j × m) · (m × n) GEMM and the new columns from one
    triangular solve, so a matrix-free K (a device kernel evaluation) sees
    one large launch per round instead of one row per pivot.

    The c × c candidate block comes from ``sub_fn(idx)`` when given (cheap:
    c² kernel entries), so ``rows_fn`` is only asked for accepted pivots.
    Without it, ``rows_fn`` is asked for all c candidate rows and the
    rejected ones are wasted. On smooth kernels the largest diagonals
    cluster in space, so strict greedy (``relax=1``) accepts few of them
    per round; ``relax`` < 1 (e.g. 0.5) accepts more at a small cost in
    approximation quality per pivot.

    The arithmetic (Schur rows, the new columns and the diagonal update)
    runs in float64 on the host, whatever device ``rows_fn`` returns on, so
    the factor does not depend on how a device reduces (the LNC split).

    Args:
        diag_fn: ``diag_fn() -> (n,)``, the diagonal of K (PSD; without any
            noise / shift term).
        rows_fn: ``rows_fn(idx) -> (len(idx), n)``, rows ``idx`` of K, any
            device and floating dtype. ``idx`` is an int64 host tensor of
            at most ``block`` distinct indices.
        rank: Maximum rank k.
        tol: Stop early when the trace of the remaining Schur complement is
            at most ``tol · trace(K)``. Also stops when the largest remaining
            diagonal reaches roundoff (n · eps · max diag), so a rank-r K
            gives at most r columns.
        block: Candidates per round (default 1: one row per pivot, the
            original behaviour).
        sub_fn: Optional ``sub_fn(idx) -> (len(idx), len(idx))``, the
            submatrix K[idx][:, idx].
        relax: Acceptance factor in (0, 1] for candidates after the first
            of a round (the first is always the global greedy pivot).

    Returns:
        L: (n, k) with k ≤ rank, in the diagonal's dtype and on its device.
    """
    if int(block) < 1:
        raise ValueError(f"pivoted_cholesky: block must be >= 1, got {block}")
    if not 0 < relax <= 1:
        raise ValueError(f"pivoted_cholesky: relax must be in (0, 1], got {relax}")
    d_in = diag_fn()
    out_dtype, out_device = d_in.dtype, d_in.device
    d = d_in.detach().to(device=_HOST, dtype=torch.float64).clone()
    n = d.shape[0]
    rank = min(int(rank), n)
    # Lt[m] is column m of L: rows of Lt are contiguous, so appending a
    # column and the (j × m) · (m × n) Schur GEMM are both stride-1.
    Lt = torch.zeros(rank, n, dtype=torch.float64)
    trace0 = float(d.sum())
    floor = n * torch.finfo(out_dtype).eps * float(d.max()) if n else 0.0
    chosen = torch.zeros(n, dtype=torch.bool)

    def host64(T, *shape):
        return T.detach().reshape(*shape).to(device=_HOST, dtype=torch.float64)

    m = 0
    while m < rank:
        rem = float(d.sum())
        live = int((d > floor).sum())
        if live == 0 or rem <= tol * trace0:
            break
        c = min(int(block), live)
        if c < live:
            top = torch.topk(d, c + 1)
            cand, outside = top.indices[:c], relax * float(top.values[c])
        else:
            cand, outside = torch.topk(d, c).indices, 0.0
        Lc = Lt[:m, cand]  # (m, c): L at the candidates
        if sub_fn is not None:
            K_c = None
            Scc = host64(sub_fn(cand), c, c) - Lc.T @ Lc
        else:
            K_c = host64(rows_fn(cand), c, n)
            Scc = K_c[:, cand] - Lc.T @ Lc
        # Greedy pivoted Cholesky on the c × c Schur block of the candidates.
        Scc = (Scc + Scc.T) / 2
        Sw = Scc.clone()
        taken = torch.zeros(c, dtype=torch.bool)
        order: list[int] = []
        while len(order) < min(c, rank - m):
            rdiag = torch.diagonal(Sw)
            p = int(torch.argmax(torch.where(taken, -math.inf, rdiag)))
            dp = float(rdiag[p])
            # The first candidate is the global greedy pivot; later ones must
            # still beat (relax times) everything outside the round.
            if dp <= floor or (order and dp < outside):
                break
            col = Sw[:, p] / math.sqrt(dp)
            Sw -= torch.outer(col, col)
            taken[p] = True
            order.append(p)
        if not order:  # roundoff: the largest Schur diagonal is not positive
            break
        perm = torch.tensor(order, dtype=torch.int64)
        C, bad = torch.linalg.cholesky_ex(Scc[perm][:, perm])
        if int(bad) > 0:  # rounding: keep the leading positive-definite pivots
            perm, C = perm[: int(bad) - 1], C[: int(bad) - 1, : int(bad) - 1]
            if perm.numel() == 0:
                break
        piv = cand[perm]
        j = perm.numel()
        K_p = K_c[perm] if K_c is not None else host64(rows_fn(piv), j, n)
        S = K_p - Lt[:m, piv].T @ Lt[:m] if m else K_p  # Schur rows of the pivots
        Lnew = torch.linalg.solve_triangular(C, S, upper=False)  # (j, n)
        # Honour tol pivot by pivot: keep pivot r only if the trace before it is above tol.
        after = rem - torch.cumsum((Lnew * Lnew).sum(1), 0)
        keep = 1 + int((after[:-1] > tol * trace0).sum())
        Lnew, piv = Lnew[:keep], piv[:keep]
        chosen[piv] = True
        Lnew[:, chosen] = 0.0
        Lnew[:, piv] = C[:keep, :keep].T
        Lt[m : m + keep] = Lnew
        d = torch.clamp(d - (Lnew * Lnew).sum(0), min=0)
        d[chosen] = 0.0
        m += keep
    return Lt[:m].T.contiguous().to(dtype=out_dtype, device=out_device)


def _gram_basis(L64: torch.Tensor, G: torch.Tensor, shift: float):
    """Orthonormal Q (n × k') and λ (k',) with L Lᵀ ≈ Q diag(λ) Qᵀ, float64 host.

    From the Gram matrix G = Lᵀ L (already formed for the k × k factor):
    G = V diag(λ) Vᵀ and Q = L V λ^(-1/2), one (n × k) · (k × k) GEMM, no
    n × k QR. Directions with λ ≤ 1e-6 · shift are dropped: the
    preconditioner applies 1/(λ + s) ≈ 1/s there, as it does off L's range,
    so dropping them changes it by at most 1e-6 relative, and it bounds the
    Gram loss of orthogonality, about eps · λ_max / λ_min. The two-pass
    apply tolerates a loss far above fp32 roundoff (it enters at second
    order); above 1e-6 one more Gram pass restores Q to float64
    orthogonality and rediagonalises.
    """
    eps = torch.finfo(torch.float64).eps
    lam, V = torch.linalg.eigh(G)
    keep = lam > 1e-6 * shift
    lam, V = lam[keep], V[:, keep]
    if lam.numel() == 0:
        return L64.new_zeros(L64.shape[0], 0), lam
    Q = L64 @ (V / lam.sqrt())
    if eps * float(lam.max() / lam.min()) > 1e-6:
        lam1, V1 = torch.linalg.eigh(Q.T @ Q)
        Q = Q @ (V1 / lam1.sqrt())
        T = lam1.sqrt()[:, None] * V1.T  # Q_old = Q_new T
        lam, V2 = torch.linalg.eigh((T * lam) @ T.T)
        Q = Q @ V2
    return Q, lam


class PivotedLowRankPreconditioner:
    """Applies (L Lᵀ + shift·I)⁻¹ to (n,) vectors or (n, R) blocks.

    By the low-rank-plus-diagonal inverse identity,
    (L Lᵀ + s I)⁻¹ V = (1/s) (V − L (s I_k + Lᵀ L)⁻¹ Lᵀ V)
    = (1/s) (V − W (Wᵀ V)) with W = L C⁻ᵀ, C Cᵀ = s I_k + Lᵀ L.
    The k × k Cholesky factor C and W are built once, in float64 on the host;
    each application is two (n × k) GEMMs.

    Precision. W Wᵀ has eigenvalues λ/(λ + s), within s/λ of 1 along the
    directions L captures, so V − W (Wᵀ V) cancels: in fp32 the result errs
    by about u·√k/s against a true value of about 1/λ, i.e. the applied
    operator is wrong by a factor up to u·√k·λ/s along those directions.
    For a GP with noise s = 1e-3 that breaks preconditioned CG outright
    (no convergence in 150 iterations where fp64 needs 20). ``apply`` picks:

    - ``"host64"``: apply in float64 on the host and return V's dtype and
      device (one (n, R) round trip per application);
    - ``"native"``: W in L's dtype on L's device (fast; accurate only while
      u·√k·λ_max/s ≪ 1, e.g. float64 L);
    - ``"projected"`` (#114): in L's dtype on L's device, stable in fp32.
      With L = Q Σ Gᵀ (Q orthonormal, built once in float64 on the host),
      (L Lᵀ + s I)⁻¹ V = Q (Σ² + s I)⁻¹ Qᵀ V + (I − Q Qᵀ) V / s. The
      cancellation is all in the projection (I − Q Qᵀ) V, so it is applied
      twice (Y = V − Q (QᵀV), then Y − Q (QᵀY), folded with the diagonal
      term into four (n × k) GEMMs). One pass leaves an error of about
      u·√k·‖V‖ along Q, which the 1/s weight blows up; the second pass
      takes that out, so what is left is an error orthogonal to Q, which
      perturbs the preconditioner only to second order. No host round
      trip; the (k,) diagonal stays on the device too.
    - ``"auto"`` (default): ``"native"`` for float64 L, else ``"host64"``.

    Attributes:
        L: (n, k) low-rank factor.
        shift: s > 0.
        W: (n, k) whitened factor L C⁻ᵀ (float64 host for ``"host64"``;
            None for ``"projected"``).
        Q, g: ``"projected"`` only: the (n, k) orthonormal basis of L's
            columns and the (k,) weights s / (σ² + s), in L's dtype on L's
            device.
        apply: the resolved mode.
    """

    def __init__(self, L: torch.Tensor, shift: float, *, apply: str = "auto"):
        shift = float(shift)
        if not shift > 0:
            raise ValueError(f"PivotedLowRankPreconditioner: shift must be > 0, got {shift}")
        if apply not in ("auto", "host64", "native", "projected"):
            raise ValueError(
                f"apply must be 'auto', 'host64', 'native' or 'projected', got {apply!r}"
            )
        if apply == "auto":
            apply = "native" if L.dtype == torch.float64 else "host64"
        self.apply = apply
        self.L = L
        self.shift = shift
        L64 = L.to(device=_HOST, dtype=torch.float64)
        k = L64.shape[1]
        G = L64.T @ L64
        C = torch.linalg.cholesky(shift * torch.eye(k, dtype=torch.float64) + G)
        self._C = C
        self.Q = self.g = self.W = None
        if apply == "projected":
            Q, lam = _gram_basis(L64, G, shift)
            self.Q = Q.to(device=L.device, dtype=L.dtype)
            self.g = (shift / (lam + shift)).to(device=L.device, dtype=L.dtype)
            return
        W = torch.linalg.solve_triangular(C, L64.T, upper=False).T
        self.W = W if apply == "host64" else W.to(device=L.device, dtype=L.dtype)

    def __call__(self, V: torch.Tensor) -> torch.Tensor:
        if self.apply == "host64":
            V64 = V.detach().to(device=_HOST, dtype=torch.float64)
            out = (V64 - self.W @ (self.W.T @ V64)) / self.shift
            return out.to(device=V.device, dtype=V.dtype)
        if self.apply == "projected":
            return self._apply_projected(V)
        W = self.W if V.dtype == self.W.dtype else self.W.to(V.dtype)
        return (V - W @ (W.T @ V)) / self.shift

    def _apply_projected(self, V: torch.Tensor) -> torch.Tensor:
        vec = V.dim() == 1
        Vm = V.unsqueeze(-1) if vec else V
        Q, g = self.Q, self.g
        if Q.dtype != Vm.dtype:
            Q, g = Q.to(Vm.dtype), g.to(Vm.dtype)
        c1 = Q.T @ Vm
        Y = Vm - Q @ c1  # (I − Q Qᵀ) V, first pass
        c2 = Q.T @ Y  # what the first pass left along Q
        # z = (Y − Q c2)/s + Q (c1 + c2)/(σ² + s) = (Y − Q (c2 − g ⊙ (c1 + c2)))/s
        out = (Y - Q @ (c2 - g[:, None] * (c1 + c2))) / self.shift
        return out.squeeze(-1) if vec else out

    def logdet(self) -> float:
        """log det(L Lᵀ + shift·I), by the matrix determinant lemma (float64)."""
        n, k = self.L.shape
        return (n - k) * math.log(self.shift) + 2.0 * float(
            torch.log(torch.diagonal(self._C)).sum()
        )

    def sample(self, num: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw ``num`` columns from N(0, L Lᵀ + shift·I): L g₁ + √shift · g₂.

        Normal draws use ``generator`` on the host, in float64, then move to
        L's dtype and device. These are the probes for preconditioned SLQ.
        """
        n, k = self.L.shape
        g1 = torch.randn(k, num, generator=generator, dtype=torch.float64)
        g2 = torch.randn(n, num, generator=generator, dtype=torch.float64)
        L64 = self.L.to(device=_HOST, dtype=torch.float64)
        Z = L64 @ g1 + math.sqrt(self.shift) * g2
        return Z.to(device=self.L.device, dtype=self.L.dtype)


def pivoted_lowrank_preconditioner(
    diag_fn: Callable[[], torch.Tensor],
    rows_fn: Callable[[torch.Tensor], torch.Tensor],
    rank: int,
    shift: float,
    *,
    tol: float = 0.0,
    apply: str = "auto",
) -> PivotedLowRankPreconditioner:
    """Preconditioner (L Lᵀ + shift·I)⁻¹ for K + shift·I, L from ``pivoted_cholesky``.

    ``diag_fn`` and ``rows_fn`` describe K without the shift (see
    :func:`pivoted_cholesky`; one row of K is requested per step). For a GP,
    pass the kernel diagonal and rows, and the noise variance as ``shift``.
    Use the result as ``M=`` in ``cg_block``; its ``logdet()`` and
    ``sample()`` serve preconditioned ``slq_logdet``.
    """
    L = pivoted_cholesky(diag_fn, rows_fn, rank, tol=tol)
    return PivotedLowRankPreconditioner(L, shift, apply=apply)


def lanczos_tridiag_from_cg(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    col_iterations: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Per-column Lanczos tridiagonals T_j from CG coefficients.

    With m CG steps on column j (α₀ … α_{m−1}, β₀ … β_{m−2}),
    T[k, k] = 1/α_k + β_{k−1}/α_{k−1} (the second term absent for k = 0) and
    T[k, k+1] = T[k+1, k] = √β_k / α_k. For preconditioned CG, T_j is the
    Lanczos matrix of M^{1/2} A M^{1/2} (M ≈ A⁻¹) started from the whitened
    residual.

    Args:
        alpha, beta: (iters, R) from ``cg_block(..., return_coeffs=True)``.
        col_iterations: (R,) steps per column (default: all ``iters`` rows).

    Returns:
        List of R float64 host tensors, T_j of shape (m_j, m_j).
    """
    a = alpha.to(device=_HOST, dtype=torch.float64)
    b = beta.to(device=_HOST, dtype=torch.float64)
    iters, R = a.shape
    counts = [iters] * R if col_iterations is None else [int(c) for c in col_iterations.tolist()]
    out = []
    for j in range(R):
        m = counts[j]
        aj, bj = a[:m, j], b[:m, j]
        diag = 1.0 / aj
        if m > 1:
            diag[1:] += bj[: m - 1] / aj[: m - 1]
        T = torch.diag(diag)
        if m > 1:
            off = torch.sqrt(bj[: m - 1]) / aj[: m - 1]
            T += torch.diag(off, 1) + torch.diag(off, -1)
        out.append(T)
    return out


@torch.compiler.disable
def slq_logdet(
    A: torch.Tensor | Callable,
    n: int,
    num_probes: int = 32,
    num_iters: int = 50,
    generator: torch.Generator | None = None,
    M: PivotedLowRankPreconditioner | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    tol: float = 1e-10,
) -> tuple[float, dict]:
    """Stochastic Lanczos quadrature estimate of log det(A) for SPD A.

    Runs one ``cg_block`` over all probes (one operator call per iteration
    for the whole probe block), builds T_j from the CG coefficients, and
    averages ‖w_j‖² · e₁ᵀ log(T_j) e₁.

    - ``M=None``: Rademacher probes z (‖z‖² = n), so the estimate is
      (1/p) Σ_j n · e₁ᵀ log(T_j) e₁.
    - ``M`` a ``PivotedLowRankPreconditioner`` (P = L Lᵀ + s I): probes
      b ~ N(0, P) from ``M.sample``, preconditioned CG, and
      log det A = log det P + (1/p) Σ_j (bⱼᵀ P⁻¹ bⱼ) e₁ᵀ log(T_j) e₁.
      A good P leaves only a small, low-variance remainder to estimate.

    The standard error falls as 1/√num_probes; ``info["std_error"]`` gives
    the sample estimate of it.

    Args:
        A: SPD operator, callable on (n, p) blocks or an (n, n) tensor.
        n: Dimension.
        num_probes: p, the number of probe vectors.
        num_iters: Lanczos steps (CG iterations) per probe.
        generator: Host ``torch.Generator`` for the probes.
        M: Optional ``PivotedLowRankPreconditioner`` (needs ``logdet`` and
            ``sample``).
        dtype, device: Probe dtype and device (default fp32, A's device if A
            is a tensor, else host).
        tol: ``cg_block`` tolerance (columns that converge early end their
            quadrature there).

    Returns:
        estimate (float) and info with ``"std_error"``, ``"quadratures"``
        (float64 tensor (p,), per-probe terms, preconditioner logdet not
        included) and ``"cg_info"`` (the ``cg_block`` info).
    """
    if device is None:
        device = A.device if isinstance(A, torch.Tensor) else _HOST
    if M is None:
        Z = torch.randint(0, 2, (n, num_probes), generator=generator, dtype=torch.int64)
        Z = (2 * Z - 1).to(device=device, dtype=dtype)
        base = 0.0
    else:
        if not (hasattr(M, "logdet") and hasattr(M, "sample")):
            raise TypeError("slq_logdet: M must provide logdet() and sample() (see #97)")
        Z = M.sample(num_probes, generator=generator).to(device=device, dtype=dtype)
        base = M.logdet()
    _, info = cg_block(
        A, Z, M, tol=tol, maxiter=num_iters, check_every=num_iters, return_coeffs=True
    )
    Ts = lanczos_tridiag_from_cg(info["alpha"], info["beta"], info["col_iterations"])
    w2 = info["rz0"].to(device=_HOST, dtype=torch.float64)
    quads = torch.zeros(num_probes, dtype=torch.float64)
    for j, T in enumerate(Ts):
        if T.shape[0] == 0:
            continue
        theta, U = torch.linalg.eigh(T)
        theta = torch.clamp(theta, min=torch.finfo(torch.float64).tiny)
        quads[j] = w2[j] * (U[0] ** 2 * torch.log(theta)).sum()
    est = base + float(quads.mean())
    std = float(quads.std() / math.sqrt(num_probes)) if num_probes > 1 else float("nan")
    return est, {"std_error": std, "quadratures": quads, "cg_info": info}
