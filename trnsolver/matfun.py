"""GEMM-only matrix functions for SCF and DFT (#149).

Two matrix functions that need no eigenvectors, built only from matrix
products and O(n^2) elementwise work:

- :func:`trnsolver.inv_sqrt_spd_ns` (in ``factor.py``): S^{-1/2} by the
  coupled Newton–Schulz iteration, optionally with minimax first-iteration
  polynomials; the schedule helpers are here.
- :func:`density_purification`: the SCF density matrix from the Fock matrix by
  canonical (Palser–Manolopoulos) or SP2 (Niklasson) purification.

Both run a fixed number of iterations on fixed shapes, with no host
synchronisation inside the loop: the branch each purification step takes is
a ``torch.where`` on a scalar tensor, not Python control flow. A fixed
iteration count compiles into one static Tensor Engine schedule
(trnsci/trnsci#176, rule 4). The ``precision=`` hook (``_precision.py``)
routes each product, so precision can be staged per iteration
(trnsci/trnsci#172), for example fp32 products for the early iterations and
``trnblas.gemm_fp64`` for the late ones, as in the large-scale DFT
purification on TPUs (Pederson et al., arXiv 2202.01255).

References:

- Newton–Schulz for the matrix square root and its inverse (coupled form):
  N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM 2008,
  chapter 6.
- Minimax ("Polar Express") polynomials: N. Amsel, D. Persson, C. Musco and
  R. M. Gower, "The Polar Express: optimal matrix sign methods and their
  application to the Muon algorithm", arXiv 2505.16932 (2025). The step
  polynomial is the odd degree-3 minimax approximation to 1 on the current
  singular-value interval; the greedy composition is their construction.
- Canonical purification: A. H. R. Palser and D. E. Manolopoulos,
  "Canonical purification of the density matrix in electronic-structure
  theory", Phys. Rev. B 58, 12704 (1998).
- SP2: A. M. N. Niklasson, "Expansion algorithm for the density matrix",
  Phys. Rev. B 66, 155115 (2002).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import _precision
from .nki.dispatch import _fp64_host_guard

__all__ = [
    "PurificationResult",
    "density_purification",
    "ns_coefficients",
    "ns_iteration_count",
]


# ----------------------------------------------------------------------
# Newton–Schulz schedule
# ----------------------------------------------------------------------
#
# The coupled iteration Y_{k+1} = Y_k T_k, Z_{k+1} = T_k Z_k with
# T_k = c1 I + c3 Z_k Y_k keeps Y_k and Z_k polynomials in A. On an
# eigenvalue x = lambda / s of the scaled A, with sigma = sqrt(x), one step
# maps sigma to g(sigma) = c1 sigma + c3 sigma^3 (the odd cubic of the polar
# Newton–Schulz family), and Z_k converges to (A / s)^{-1/2} as sigma -> 1.
# Newton–Schulz is (c1, c3) = (3/2, -1/2). The minimax step is the odd cubic
# with the smallest max |1 - g| on the current interval [a, b] of sigma; it
# equioscillates at a, at its maximum sigma*, and at b, which gives it in
# closed form. Its image is [1 - E, 1 + E], the next step's interval.


def _minimax_cubic(a: float, b: float) -> tuple[float, float, float]:
    """(c1, c3, E): the odd cubic c1 s + c3 s^3 closest to 1 on [a, b] in max
    norm, and its error E. Equioscillation at a, s*, b with g(a) = g(b) =
    1 - E and g(s*) = 1 + E, s*^2 = (a^2 + ab + b^2) / 3."""
    q = a * a + a * b + b * b
    s_star = math.sqrt(q / 3.0)
    c3 = -2.0 / (2.0 * s_star**3 + a * b * (a + b))
    c1 = -c3 * q
    err = 1.0 - (c1 * a + c3 * a**3)
    return c1, c3, err


def _cubic_image(c1: float, c3: float, a: float, b: float) -> tuple[float, float]:
    """The image [lo, hi] of [a, b] under s -> c1 s + c3 s^3 (0 < a <= b)."""

    def g(s):
        return c1 * s + c3 * s**3

    vals = [g(a), g(b)]
    if c3 < 0:
        sc = math.sqrt(-c1 / (3.0 * c3))
        if a < sc < b:
            vals.append(g(sc))
    return min(vals), max(vals)


def ns_coefficients(
    iters: int, polynomial: str = "newton", cond: float | None = None
) -> list[tuple[float, float]]:
    """Per-iteration ``(c1, c3)`` for ``T_k = c1 I + c3 Z_k Y_k``.

    ``polynomial="newton"``: Newton–Schulz, ``(1.5, -0.5)`` every iteration.
    ``polynomial="minimax"``: the greedy minimax cubic schedule of arXiv
    2505.16932 for eigenvalues of the scaled matrix in ``[1 / cond, 1]``.
    Its coefficients tend to Newton–Schulz as the interval closes on 1.
    Eigenvalues below ``1 / cond`` still converge, only more slowly, so
    ``cond`` may be a rough upper bound on ``s / lambda_min`` (``s`` the
    scaling; see :func:`trnsolver.inv_sqrt_spd_ns`).
    """
    if iters < 0:
        raise ValueError(f"iters must be >= 0; got {iters}")
    if polynomial == "newton":
        return [(1.5, -0.5)] * iters
    if polynomial != "minimax":
        raise ValueError(f"polynomial must be 'newton' or 'minimax'; got {polynomial!r}")
    if cond is None or not cond >= 1.0:
        raise ValueError(f"polynomial='minimax' needs cond >= 1; got {cond!r}")
    a, b = cond**-0.5, 1.0
    out = []
    for _ in range(iters):
        c1, c3, _err = _minimax_cubic(a, b)
        out.append((c1, c3))
        a, b = _cubic_image(c1, c3, a, b)
    return out


def ns_iteration_count(
    cond: float, tol: float, polynomial: str = "minimax", max_iters: int = 200
) -> int:
    """Iterations the schedule needs, in exact arithmetic, to bring every
    eigenvalue of ``Z_k Y_k`` in ``[1 / cond, 1]`` within ``tol`` of 1.

    Data-independent: it depends only on ``cond``, ``tol`` and the
    polynomial, so it fixes the iteration count (and the compiled schedule)
    ahead of time. Rounding sets a floor of a few ulps times n on what the
    iteration reaches.
    """
    a, b = cond**-0.5, 1.0
    for k in range(max_iters + 1):
        if max(1.0 - a * a, b * b - 1.0) <= tol:
            return k
        if polynomial == "minimax":
            c1, c3, _ = _minimax_cubic(a, b)
        elif polynomial == "newton":
            c1, c3 = 1.5, -0.5
        else:
            raise ValueError(f"polynomial must be 'newton' or 'minimax'; got {polynomial!r}")
        a, b = _cubic_image(c1, c3, a, b)
    raise ValueError(f"no convergence to tol={tol:g} within {max_iters} iterations (cond={cond:g})")


def _ns_scale(A: torch.Tensor, scaling) -> torch.Tensor:
    """s >= lambda_max(A) from a cheap norm bound (a 0-d tensor, no host sync)."""
    if isinstance(scaling, (int, float)) and not isinstance(scaling, bool):
        if not scaling > 0:
            raise ValueError(f"scaling must be positive; got {scaling}")
        return torch.as_tensor(float(scaling), dtype=A.dtype, device=A.device)
    if scaling == "fro":
        return torch.linalg.norm(A, ord="fro")
    inf = A.abs().sum(-1).amax()  # max row sum: Gershgorin bound on |lambda|
    if scaling == "inf":
        return inf
    if scaling == "bound":
        return torch.minimum(torch.linalg.norm(A, ord="fro"), inf)
    raise ValueError(f"scaling must be 'bound', 'fro', 'inf' or a positive number; got {scaling!r}")


def _ns_step(Y, Z, c1: float, c3: float, gemm, first: bool):
    """One coupled step. On the first step Z = I, so Z Y = Y and T Z = T."""
    n = Y.shape[-1]
    eye = torch.eye(n, dtype=Y.dtype, device=Y.device)
    M = Y if first else gemm(Z, Y)
    T = c1 * eye + c3 * M
    return gemm(Y, T), (T if first else gemm(T, Z))


def _ns_fixed(A: torch.Tensor, coeffs, s: torch.Tensor, gemm_k, gemm_last):
    """Fixed-iteration coupled NS. Returns (A^{-1/2}, ||Y Z - I||_F / sqrt(n)),
    both tensors; no host synchronisation."""
    n = A.shape[-1]
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    Y, Z = A / s, eye
    for k, (c1, c3) in enumerate(coeffs):
        Y, Z = _ns_step(Y, Z, c1, c3, gemm_k(k), first=(k == 0))
    res = torch.linalg.norm(gemm_last(Y, Z) - eye, ord="fro") / math.sqrt(n)
    X = 0.5 * (Z + Z.mT) / torch.sqrt(s)
    return X, res


# ----------------------------------------------------------------------
# Density-matrix purification
# ----------------------------------------------------------------------


@dataclass
class PurificationResult:
    """Result of :func:`density_purification`.

    Attributes:
        P: density projector in the input (AO) basis, ``X P_orth X^T``. With
            S, ``P S P = P`` and ``tr(P S) = n_occ`` at convergence. It is the
            projector on the occupied space, ``C_occ C_occ^T``; the
            closed-shell density is ``2 P``.
        P_orth: the purified projector in the orthonormal basis (m, m).
        X: the orthonormalising transform used (``S^{-1/2}`` when computed
            here), or None without S and X.
        energy: ``tr(P F)``, the sum of the occupied orbital energies at
            convergence (closed shell: ``2 tr(P F)``).
        idempotency_error: ``||P_orth^2 - P_orth||_F`` after the last iteration.
        trace_error: ``|tr(P_orth) - n_occ|``.
        iters: purification iterations run (fixed).
        method: ``"canonical"`` or ``"sp2"``.
        history: ``tr(P_orth - P_orth^2)`` at the start of each iteration
            (iters,): the convergence trace, free of extra products. It tends
            to 0 for a gapped spectrum.
        ns_iters: Newton–Schulz iterations for ``S^{-1/2}`` (0 if not computed).
        ns_residual: ``||Y Z - I||_F / sqrt(n)`` of that iteration, or None.
    """

    P: torch.Tensor
    P_orth: torch.Tensor
    X: torch.Tensor | None
    energy: float
    idempotency_error: float
    trace_error: float
    iters: int
    method: str
    history: torch.Tensor
    ns_iters: int = 0
    ns_residual: float | None = None


_DEFAULT_ITERS = {"canonical": 80, "sp2": 80}


def _gershgorin(F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d = torch.diagonal(F)
    r = F.abs().sum(-1) - d.abs()
    return (d - r).amin(), (d + r).amax()


def _sym(P: torch.Tensor) -> torch.Tensor:
    return 0.5 * (P + P.mT)


def _canonical(Fo, n_occ, iters, gemm_k, emin, emax):
    """Palser–Manolopoulos canonical purification (PRB 58, 12704, 1998).

    Trace-conserving: P_0 has trace n_occ, and each step keeps it. The step
    coefficient c_k = tr(P^2 - P^3) / tr(P - P^2) picks one of two cubics;
    the pick is a torch.where on 0-d tensors (no host branch).
    """
    m = Fo.shape[-1]
    eye = torch.eye(m, dtype=Fo.dtype, device=Fo.device)
    mu = torch.diagonal(Fo).sum() / m
    tiny = torch.finfo(Fo.dtype).tiny
    lam = torch.minimum(
        n_occ / torch.clamp(emax - mu, min=tiny), (m - n_occ) / torch.clamp(mu - emin, min=tiny)
    )
    P = (lam / m) * (mu * eye - Fo) + (n_occ / m) * eye
    eps = torch.finfo(Fo.dtype).eps
    hist = []
    for k in range(iters):
        g = gemm_k(k)
        P2 = g(P, P)
        P3 = g(P2, P)
        t1 = torch.diagonal(P).sum() - torch.diagonal(P2).sum()  # tr(P - P^2)
        t2 = torch.diagonal(P2).sum() - torch.diagonal(P3).sum()  # tr(P^2 - P^3)
        hist.append(t1)
        ok = t1 > eps * m
        c = torch.where(
            ok, t2 / torch.where(ok, t1, torch.ones_like(t1)), 0.5 * torch.ones_like(t1)
        )
        c = torch.clamp(c, 0.0, 1.0)
        low = c <= 0.5
        # c <= 1/2: [(1 - 2c) P + (1 + c) P^2 - P^3] / (1 - c)
        # c >  1/2: [(1 + c) P^2 - P^3] / c
        den = torch.where(low, 1.0 - c, c)
        a1 = torch.where(low, 1.0 - 2.0 * c, torch.zeros_like(c)) / den
        a2 = (1.0 + c) / den
        a3 = -1.0 / den
        P = _sym(a1 * P + a2 * P2 + a3 * P3)
    return P, hist


def _sp2(Fo, n_occ, iters, gemm_k, emin, emax):
    """Niklasson's trace-correcting SP2 purification (PRB 66, 155115, 2002).

    Each step is P^2 or 2P - P^2, whichever trace is closer to n_occ; tr(P^2)
    = ||P||_F^2 for symmetric P, so the choice needs no extra product. The
    choice is a torch.where on 0-d tensors (no host branch).
    """
    m = Fo.shape[-1]
    eye = torch.eye(m, dtype=Fo.dtype, device=Fo.device)
    P = (emax * eye - Fo) / (emax - emin)
    hist = []
    for k in range(iters):
        P2 = gemm_k(k)(P, P)
        tr = torch.diagonal(P).sum()
        tr2 = torch.diagonal(P2).sum()
        hist.append(tr - tr2)
        square = (tr2 - n_occ).abs() < (2.0 * tr - tr2 - n_occ).abs()
        P = _sym(torch.where(square, P2, 2.0 * P - P2))
    return P, hist


def _purify_fixed(Fo, n_occ, method, iters, gemm_k, gemm_last, bounds=None):
    """Purification in the orthonormal basis. Returns tensors only (no host
    sync): (P, history, ||P^2 - P||_F, |tr P - n_occ|)."""
    if bounds is None:
        emin, emax = _gershgorin(Fo)
    else:
        emin = torch.as_tensor(float(bounds[0]), dtype=Fo.dtype, device=Fo.device)
        emax = torch.as_tensor(float(bounds[1]), dtype=Fo.dtype, device=Fo.device)
    if method == "canonical":
        P, hist = _canonical(Fo, n_occ, iters, gemm_k, emin, emax)
    elif method == "sp2":
        P, hist = _sp2(Fo, n_occ, iters, gemm_k, emin, emax)
    else:
        raise ValueError(f"method must be 'canonical' or 'sp2'; got {method!r}")
    idem = torch.linalg.norm(gemm_last(P, P) - P, ord="fro")
    trace = (torch.diagonal(P).sum() - n_occ).abs()
    history = torch.stack(hist) if hist else Fo.new_zeros(0)
    return P, history, idem, trace


@_fp64_host_guard("density_purification")
def density_purification(
    F: torch.Tensor,
    S: torch.Tensor | None = None,
    *,
    n_occ: int,
    X: torch.Tensor | None = None,
    method: str = "canonical",
    iters: int | None = None,
    precision=None,
    bounds: tuple[float, float] | None = None,
    ns_iters: int | None = None,
    ns_cond: float = 1e6,
    ns_polynomial: str = "minimax",
) -> PurificationResult:
    """SCF density matrix from the Fock matrix, without eigenvectors.

    Matrix products only, a fixed number of iterations, fixed shapes (#149).
    The orthonormal-basis Fock matrix is ``F' = X^T F X``; purification maps
    it to the projector ``P'`` on its ``n_occ`` lowest eigenvectors, and the
    AO density projector is ``P = X P' X^T``.

    Args:
        F: symmetric Fock matrix (n, n), real.
        S: overlap matrix (n, n). If given without ``X``, ``X = S^{-1/2}``
            (Löwdin) by :func:`trnsolver.inv_sqrt_spd_ns` with a fixed
            iteration count. Omit S and X for an orthonormal basis.
        n_occ: number of occupied orbitals (closed shell: electrons / 2).
        X: an orthonormalising transform (n, m) with ``X^T S X = I``, e.g.
            ``trnsolver.orthogonalizer(S).X`` (which may drop near-dependent
            directions, m < n). Takes precedence over S.
        method: ``"canonical"`` (Palser–Manolopoulos 1998; 2 products per
            iteration, trace-conserving) or ``"sp2"`` (Niklasson 2002; 1
            product per iteration, trace-correcting).
        iters: fixed purification iteration count (default 80). Both
            methods converge quadratically once eigenvalues have separated;
            the count before that grows like log(spectral width / gap).
        precision: GEMM routing per iteration (see ``trnsolver._precision``):
            None (``torch.matmul``), ``"fp32"``, ``"fp64"`` / ``"ozaki1"`` /
            ``"ozaki2"`` (``trnblas.gemm_fp64``), a dict of ``gemm_fp64``
            keywords, a callable, or a list of these per iteration. The basis
            transforms, S^{-1/2} and the final check use the last entry.
        bounds: ``(e_min, e_max)`` bounds on the spectrum of ``F'``. Default:
            Gershgorin discs of ``F'``.
        ns_iters: Newton–Schulz iterations for ``S^{-1/2}``. Default:
            :func:`ns_iteration_count` for ``ns_cond`` at the dtype's precision.
        ns_cond: assumed bound on ``s / lambda_min(S)`` for the S^{-1/2}
            schedule (``s`` the norm-bound scaling).
        ns_polynomial: ``"minimax"`` (default) or ``"newton"``.

    Returns:
        :class:`PurificationResult` with the density, the convergence report
        (idempotency and trace errors, per-iteration ``tr(P' - P'^2)``) and
        the iteration counts.
    """
    if F.is_complex():
        raise NotImplementedError("density_purification: complex F is not supported")
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError(f"F must be (n, n); got {tuple(F.shape)}")
    if iters is None:
        if method not in _DEFAULT_ITERS:
            raise ValueError(f"method must be 'canonical' or 'sp2'; got {method!r}")
        iters = _DEFAULT_ITERS[method]
    gemm_k, gemm_last = _precision.schedule(precision)

    ns_res = None
    n_ns = 0
    if X is None and S is not None:
        if S.shape != F.shape:
            raise ValueError(f"S must match F's shape {tuple(F.shape)}; got {tuple(S.shape)}")
        S = S.to(F.dtype)
        if ns_iters is None:
            tol = 50.0 * torch.finfo(F.dtype).eps
            ns_iters = ns_iteration_count(ns_cond, tol, ns_polynomial)
        n_ns = ns_iters
        coeffs = ns_coefficients(ns_iters, ns_polynomial, ns_cond)
        X, ns_res_t = _ns_fixed(S, coeffs, _ns_scale(S, "bound"), lambda k: gemm_last, gemm_last)
        ns_res = ns_res_t
    if X is not None:
        X = X.to(F.dtype)
        m = X.shape[-1]
        Fo = _sym(gemm_last(gemm_last(X.mT, F), X))
    else:
        m = F.shape[-1]
        Fo = _sym(F)
    if not 0 <= n_occ <= m:
        raise ValueError(f"n_occ must be in [0, {m}]; got {n_occ}")

    P, history, idem, trace = _purify_fixed(Fo, n_occ, method, iters, gemm_k, gemm_last, bounds)
    P_ao = _sym(gemm_last(gemm_last(X, P), X.mT)) if X is not None else P
    energy = (P * Fo).sum()  # tr(P' F') = tr(P F)
    return PurificationResult(
        P=P_ao,
        P_orth=P,
        X=X,
        energy=float(energy),
        idempotency_error=float(idem),
        trace_error=float(trace),
        iters=iters,
        method=method,
        history=history,
        ns_iters=n_ns,
        ns_residual=None if ns_res is None else float(ns_res),
    )
