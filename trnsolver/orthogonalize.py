"""
Orthogonalizers for an overlap (Gram) matrix S.

An SCF code needs a matrix X with X^H S X = I to turn F C = S C eps into an
ordinary eigenproblem (X^H F X) C' = C' eps, C = X C'. For large or diffuse
basis sets S is SPD in exact arithmetic but numerically near-singular: some
combinations of basis functions are almost linearly dependent. A Cholesky of
S (``eigh_generalized``'s default reduction) or an unthresholded S^{-1/2}
then fails or amplifies rounding by 1/sqrt(lambda_min). The standard fix is
to drop those combinations and work in the kept m <= n dimensional subspace
(#72):

- ``canonical``: S = U diag(s) U^H, keep s_i > lindep * max(s),
  X = U_keep diag(s_keep^{-1/2}) (n x m).
- ``symmetric`` (Loewdin): X = S^{-1/2} = U diag(s^{-1/2}) U^H (n x n) when
  nothing is dropped. When the threshold drops eigenvalues there is no n x n
  X with X^H S X = I, and it returns the canonical X (PySCF's convention).
- ``cholesky``: pivoted Cholesky of S selects m basis functions P (stop when
  the largest remaining Schur-complement diagonal is <= lindep * max(diag S)),
  then S_PP = L L^H and X[P, :] = L^{-H}, zero elsewhere. No eigensolve: it is
  O(n m^2) and X is triangular on the kept functions.

X is built once per SCF in float64 on the host. Applying it, X^H F X and
X C', is two or three GEMMs per iteration and can run on the device in fp32
(``eigh_generalized(F, X=X)``).
"""

from __future__ import annotations

from typing import NamedTuple

import torch

_METHODS = ("canonical", "symmetric", "cholesky")


class Orthogonalizer(NamedTuple):
    """Result of :func:`orthogonalizer`.

    X: (..., n, m) with X^H S X = I_m (for a batch with unequal ranks, the
       columns past each matrix's rank are zero).
    rank: number of kept combinations m (an int; a LongTensor of shape
       S.shape[:-2] for a batch).
    eigenvalues: S's eigenvalues, ascending, float64 (``canonical`` and
       ``symmetric``); None for ``cholesky``.
    pivots: the kept basis functions in pivot order (``cholesky``; (..., m),
       padded with -1 in a batch); None otherwise.
    """

    X: torch.Tensor
    rank: int | torch.Tensor
    eigenvalues: torch.Tensor | None
    pivots: torch.Tensor | None


def orthogonalizer(
    S: torch.Tensor,
    method: str = "symmetric",
    lindep: float = 1e-7,
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> Orthogonalizer:
    """X with X^H S X = I on the numerically independent subspace of S.

    Args:
        S: Hermitian positive semi-definite (n, n) or batch (..., n, n),
           real or complex. Only its Hermitian part is used.
        method: ``"symmetric"`` (default; Loewdin S^{-1/2} when nothing is
           dropped, else canonical), ``"canonical"`` or ``"cholesky"``
           (pivoted Cholesky, no eigensolve).
        lindep: relative threshold. ``canonical`` / ``symmetric`` keep the
           eigenvalues s_i > lindep * max(s); ``cholesky`` stops pivoting when
           the largest remaining Schur-complement diagonal is
           <= lindep * max(diag S). The kept problem's condition number is
           then at most 1 / lindep, so X amplifies rounding by at most
           1 / sqrt(lindep). For a normalised AO basis max(s) >= 1, so this is
           at least as strict as an absolute threshold of the same value.
           ``lindep=0`` keeps every positive eigenvalue (pivot).
        dtype: dtype of the returned X (default float64, or complex128 for
           complex S). The computation is always float64 on the host.
        device: device of the returned X (default the host).

    Returns:
        :class:`Orthogonalizer` ``(X, rank, eigenvalues, pivots)``.

    Differentiable (autograd) through S: ``symmetric`` without drops and
    ``cholesky`` for a fixed pivot set give X itself; ``canonical`` X is
    defined up to the sign (phase) of each eigenvector, so only
    gauge-invariant functions of it (such as X X^H) have well-defined
    gradients, and degenerate kept eigenvalues have none.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if lindep < 0:
        raise ValueError(f"lindep must be >= 0, got {lindep}")
    if S.dim() < 2 or S.shape[-1] != S.shape[-2]:
        raise ValueError(f"S must be (..., n, n), got shape {tuple(S.shape)}")
    work = torch.complex128 if S.is_complex() else torch.float64
    S64 = S.to(device="cpu", dtype=work)
    S64 = 0.5 * (S64 + S64.mH)

    batch = S64.shape[:-2]
    if not batch:
        X, rank, w, piv = _one(S64, method, lindep)
    else:
        flat = S64.reshape(-1, *S64.shape[-2:])
        outs = [_one(Si, method, lindep) for Si in flat]
        X, rank, w, piv = _stack(outs, batch, S64.shape[-1], method)
    if dtype is not None or device is not None:
        X = X.to(device=device if device is not None else X.device, dtype=dtype or X.dtype)
    return Orthogonalizer(X, rank, w, piv)


def _one(S: torch.Tensor, method: str, lindep: float):
    if method == "cholesky":
        X, piv = _cholesky_orth(S, lindep)
        return X, int(piv.numel()), None, piv
    s, U = torch.linalg.eigh(S)
    smax = s[-1].detach()
    if not smax > 0:
        raise ValueError("S has no positive eigenvalue")
    keep = s.detach() > lindep * smax
    m = int(keep.sum())
    if method == "symmetric" and m == S.shape[-1]:
        X = (U * s.rsqrt()) @ U.mH
        X = 0.5 * (X + X.mH)
    else:
        # eigh sorts ascending, so the kept eigenvalues are the last m.
        X = U[:, -m:] * s[-m:].rsqrt()
    return X, m, s.detach(), None


def _cholesky_orth(S: torch.Tensor, lindep: float):
    """Pivoted-Cholesky orthogonalizer: (X (n, m), pivots (m,))."""
    with torch.no_grad():
        piv = _pivots(S.detach(), lindep)
    if piv.numel() == 0:
        raise ValueError("S has no positive diagonal element")
    # The pivot set is discrete; the factor of the kept block is
    # differentiable. Greedy pivot order keeps this unpivoted Cholesky stable.
    Spp = S[piv][:, piv]
    L = torch.linalg.cholesky(Spp)
    eye = torch.eye(L.shape[-1], dtype=S.dtype)
    Linv_h = torch.linalg.solve_triangular(L, eye, upper=False).mH  # L^{-H}
    X = torch.zeros(S.shape[-1], L.shape[-1], dtype=S.dtype)
    X = X.index_copy(0, piv, Linv_h)
    return X, piv


def _pivots(S: torch.Tensor, lindep: float) -> torch.Tensor:
    """Greedy (diagonal) pivots of the Cholesky of S, stopping at lindep."""
    n = S.shape[-1]
    d = S.diagonal().real.clone()
    dmax = float(d.max())
    if not dmax > 0:
        return torch.zeros(0, dtype=torch.long)
    tol = lindep * dmax
    if lindep == 0:
        tol = n * torch.finfo(torch.float64).eps * dmax
    fast = _pivots_lapack(S, tol)
    if fast is not None:
        return fast
    return _pivots_torch(S, d, tol)


def _pivots_lapack(S: torch.Tensor, tol: float) -> torch.Tensor | None:
    """LAPACK ?pstrf through SciPy (blocked, about 10x faster than eigh)."""
    try:
        from scipy.linalg import lapack
    except ImportError:
        return None
    fn = lapack.zpstrf if S.is_complex() else lapack.dpstrf
    a = S.numpy().copy(order="F")
    _, piv, rank, info = fn(a, lower=1, tol=tol)
    if info < 0:
        return None
    return torch.from_numpy(piv[:rank].astype("int64") - 1)


def _pivots_torch(S: torch.Tensor, d: torch.Tensor, tol: float) -> torch.Tensor:
    """Unblocked pivoted Cholesky: one GEMV per kept function."""
    n = S.shape[-1]
    L = torch.zeros(n, n, dtype=S.dtype)
    d = d.clone()
    taken = torch.zeros(n, dtype=torch.bool)
    piv = []
    for k in range(n):
        p = int(torch.where(taken, float("-inf"), d).argmax())
        if not d[p] > tol:
            break
        piv.append(p)
        taken[p] = True
        col = S[:, p] - L[:, :k] @ L[p, :k].conj()
        col = col / d[p].sqrt()
        col[taken] = 0
        col[p] = d[p].sqrt()
        L[:, k] = col
        d = d - col.abs() ** 2
        d[taken] = 0
    return torch.tensor(piv, dtype=torch.long)


def _stack(outs, batch, n, method):
    ranks = [o[1] for o in outs]
    m = max(ranks)
    dtype = outs[0][0].dtype
    Xs = []
    for o in outs:
        Xi = o[0]
        if Xi.shape[-1] < m:
            Xi = torch.cat([Xi, torch.zeros(n, m - Xi.shape[-1], dtype=dtype)], dim=-1)
        Xs.append(Xi)
    X = torch.stack(Xs).reshape(*batch, n, m)
    rank = torch.tensor(ranks, dtype=torch.long).reshape(batch)
    w = None if method == "cholesky" else torch.stack([o[2] for o in outs]).reshape(*batch, n)
    piv = None
    if method == "cholesky":
        piv = torch.full((len(outs), m), -1, dtype=torch.long)
        for i, o in enumerate(outs):
            piv[i, : o[3].numel()] = o[3]
        piv = piv.reshape(*batch, m)
    return X, rank, w, piv
