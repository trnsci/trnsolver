"""
Occupied-subspace eigensolver for SCF: Chebyshev-filtered subspace iteration.

An SCF iteration needs the lowest ``k = nocc`` eigenpairs of a symmetric
Fock matrix F (in an orthonormal basis), or only the subspace they span, and
F changes little from one iteration to the next. :func:`occupied_subspace`
uses that: it keeps a block of ``kb = k + buffer`` orthonormal Ritz vectors
X from the previous iteration and refines it, per pass, with

    Y = p_m(F) X                  Chebyshev filter: m dependent GEMMs with F
                                  resident on chip
    E = Y - X (X^T Y)             the filtered correction outside span(X)
    Grams of [X, E] and F[X, E]   six kb x kb GEMM blocks
    (theta, Q) = RR on span[X, E] 2 kb x 2 kb Rayleigh-Ritz, float64, host
    X = [X, E] Q[:, :kb]          on F's device

The first three lines are one NKI launch (``trnsolver/nki/subspace.py``);
only the six Gram blocks (and one residual row) cross to the host. The
filter is the scaled Chebyshev filter of Zhou and Saad (SIAM J. Matrix Anal.
Appl. 29(3), 2007; in SCF, Zhou, Saad, Tiago and Chelikowsky, J. Comput.
Phys. 219, 2006): it damps ``[a, b] = [theta_kb, upper]`` and grows the
wanted end, normalised to 1 at the lowest Ritz value.

Why the correction form. The filter grows the bottom of the wanted interval
(core orbitals, tens of Hartree below the HOMO) faster than its top, by
``R = T_m(x(theta_1)) / T_m(x(theta_k))``; for an alkane at degree 8 that is
already 1e6. The textbook Rayleigh-Ritz on ``Y^T Y`` and ``Y^T F Y`` squares
that ratio in FP32 Gram matrices and loses the HOMO (measured: SCF diverges
at R = 1e6, and is slowed by a capped degree at R = 1e4). Splitting off the
part of Y inside span(X) on the vectors, in FP32, costs one more GEMM pair and
leaves E carrying only the correction, whose Grams are well scaled column by
column; the Rayleigh-Ritz on span[X, E] then contains span(Y) and the
previous block, like a block Davidson step with a polynomial preconditioner.
It is variational: span[X, E] contains X, so a pass never loses accuracy.
Above degree ~12 the same growth makes every column of E dominated by the
core levels' corrections and the HOMO-level corrections fall below the
Gram's resolution (measured on C10H22 Fock sequences; worse at 16 than at
8), so the default degree is 8 and accuracy comes from passes instead.

Reference path: :func:`filter_reference` is the PyTorch twin of the kernel
(same inputs, same packed output) and runs on any device and float dtype.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import NamedTuple

import torch

from .nki import _REQUIRE_NKI, HAS_NKI, _use_nki, _use_simulator

# Filter degrees the driver picks from: one compiled kernel per (n, kb, m).
DEGREES = (2, 4, 6, 8, 12, 16, 24, 32, 48, 64)
_MAX_DEGREE = 8
# Cap on the growth ratio T_m(x(theta_1)) / T_m(x(theta_k)): keeps the
# filtered block well inside the FP32 range.
_AMP_MAX = 1e12
# Rayleigh-Ritz drops directions of span[X, E] below this fraction of the
# (unit-diagonal) Gram's largest eigenvalue: dependent correction columns.
_DROP = 1e-6
# Packed kernel output: rows [E; Gxx; Hxx; Gxe; Hxe; Gee; Hee; r].
N_GRAMS = 6


@dataclass
class SubspaceState:
    """Warm start carried from one :func:`occupied_subspace` call to the next.

    ``X`` (n, kb) has orthonormal columns (Ritz vectors, ascending) on F's
    device; ``theta`` (kb,) float64 on the host are their Ritz values for
    the matrix of the call that produced them. ``upper`` is the spectral
    upper bound last used; ``degree`` and ``passes`` describe the last call,
    and ``residual`` holds ``|F x_j - theta_j x_j|`` of the block that entered
    its last pass (before that pass; None after a cold start).
    """

    X: torch.Tensor
    theta: torch.Tensor
    upper: float
    degree: int = 0
    passes: int = 0
    residual: torch.Tensor | None = None


class SubspaceResult(NamedTuple):
    eigenvalues: torch.Tensor  # (k,) ascending, F's dtype and device
    eigenvectors: torch.Tensor  # (n, k) orthonormal columns
    state: SubspaceState


def default_buffer(k: int) -> int:
    """Extra block columns beyond k: about 15%, at least 8."""
    return max(8, -(-3 * k // 20))


def chebyshev_coefficients(m: int, a0: float, a: float, b: float) -> torch.Tensor:
    """(3, m) float64 ``[alpha; beta; gamma]`` of the scaled Chebyshev filter.

    Step i of the filter is ``Y_i = alpha_i F Y_{i-1} + beta_i Y_{i-1} +
    gamma_i Y_{i-2}`` (``Y_0 = X``, ``gamma_1 = 0``), which is Zhou and
    Saad's Algorithm 4.1: damp ``[a, b]``, scale so that p_m(a0) = 1.
    """
    if not (a0 < a < b):
        raise ValueError(f"need a0 < a < b; got a0={a0}, a={a}, b={b}")
    e = 0.5 * (b - a)
    c = 0.5 * (b + a)
    sigma1 = e / (a0 - c)
    tau = 2.0 / sigma1
    out = torch.zeros(3, m, dtype=torch.float64)
    sig = sigma1
    out[0, 0] = sigma1 / e
    out[1, 0] = -c * sigma1 / e
    for i in range(1, m):
        sig_new = 1.0 / (tau - sig)
        out[0, i] = 2.0 * sig_new / e
        out[1, i] = -c * 2.0 * sig_new / e
        out[2, i] = -sig * sig_new
        sig = sig_new
    return out


def _acosh_abs_x(lam: float, a: float, b: float) -> float:
    """acosh |x(lam)| for x mapping [a, b] onto [-1, 1] (0 inside)."""
    x = abs((lam - 0.5 * (a + b)) / (0.5 * (b - a)))
    return math.acosh(x) if x > 1.0 else 0.0


def choose_degree(
    theta: torch.Tensor,
    k: int,
    upper: float,
    *,
    amp_max: float = _AMP_MAX,
    max_degree: int = _MAX_DEGREE,
) -> int:
    """Largest degree in ``DEGREES`` (<= max_degree) whose growth ratio
    ``T_m(x(theta_1)) / T_m(x(theta_k))`` stays below ``amp_max``."""
    a = float(theta[-1])
    d = _acosh_abs_x(float(theta[0]), a, upper) - _acosh_abs_x(float(theta[k - 1]), a, upper)
    m_cap = max_degree if d <= 0 else min(max_degree, math.log(amp_max) / d)
    ok = [m for m in DEGREES if m <= m_cap]
    return ok[-1] if ok else DEGREES[0]


def filter_table(coef: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """(3 m + kb,) float32 ``[alpha | beta | gamma | -theta]``: the kernel's row."""
    return torch.cat([coef.reshape(-1), -theta.to(torch.float64)]).to(torch.float32)


def filter_reference(F: torch.Tensor, X: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """PyTorch twin of ``subspace_filter_kernel``: same inputs, same packed output.

    ``table`` is :func:`filter_table` (length 3 m + kb). Returns
    (n + 6 kb + 1, kb): rows ``[E; X^T X; X^T F X; X^T E; (F X)^T E; E^T E;
    E^T F E; r]`` with ``Y = p_m(F) X``, ``E = Y - X (X^T Y)`` and
    ``r_j = |F x_j - theta_j x_j|^2`` for the input block.
    """
    n, kb = X.shape
    m = (table.numel() - kb) // 3
    t = table.to(F.dtype)
    alpha, beta, gamma, neg_theta = t[:m], t[m : 2 * m], t[2 * m : 3 * m], t[3 * m :]
    FX = F @ X
    R = FX + X * neg_theta
    r = (R * R).sum(0, keepdim=True)
    prev = X
    cur = alpha[0] * FX + beta[0] * X
    for i in range(1, m):
        prev, cur = cur, alpha[i] * (F @ cur) + beta[i] * cur + gamma[i] * prev
    E = cur - X @ (X.T @ cur)
    FE = F @ E
    return torch.cat([E, X.T @ X, X.T @ FX, X.T @ E, FX.T @ E, E.T @ E, E.T @ FE, r], dim=0)


def unpack_grams(small: torch.Tensor, kb: int):
    """Rows after E of the packed output -> (G, H (2kb, 2kb), r (kb,)).

    G and H are the Gram and F-Gram of [X, E].
    """
    blk = [small[i * kb : (i + 1) * kb] for i in range(N_GRAMS)]
    gxx, hxx, gxe, hxe, gee, hee = blk
    G = torch.cat([torch.cat([gxx, gxe], 1), torch.cat([gxe.T, gee], 1)], 0)
    H = torch.cat([torch.cat([hxx, hxe], 1), torch.cat([hxe.T, hee], 1)], 0)
    return G, H, small[N_GRAMS * kb]


def _orth_basis(G: torch.Tensor, drop: float) -> torch.Tensor:
    """B (d, r) with B^T G B = I over G's directions above ``drop`` (unit-diagonal scale)."""
    dg = torch.diagonal(G)
    d = torch.where(dg > 0, torch.rsqrt(torch.clamp(dg, min=1e-300)), torch.zeros_like(dg))
    s, U = torch.linalg.eigh(d[:, None] * G * d[None, :])
    keep = s > drop * s[-1]
    return d[:, None] * U[:, keep] * torch.rsqrt(s[keep])[None, :]


def _augmented_basis(G: torch.Tensor, kb: int, drop: float) -> torch.Tensor | None:
    """G-orthonormal basis of span[X, E] from the X block first (None: fall back).

    X is nearly orthonormal, so its block takes a Cholesky; E is then
    orthogonalised against X at the Gram level (a Schur complement with no
    cancellation, as E is already nearly orthogonal to X) and only E's
    kb x kb block needs the rank-revealing eigendecomposition.
    """
    gxx, gxe, gee = G[:kb, :kb], G[:kb, kb:], G[kb:, kb:]
    L, info = torch.linalg.cholesky_ex(gxx)
    if int(info) != 0:
        return None
    eye = torch.eye(kb, dtype=G.dtype)
    Bx = torch.linalg.solve_triangular(L.T, eye, upper=True)  # Bx^T gxx Bx = I
    C = torch.cholesky_solve(gxe, L)  # gxx^{-1} gxe
    S = gee - gxe.T @ C
    Be = _orth_basis(0.5 * (S + S.T), drop)
    top = torch.cat([Bx, -C @ Be], 1)
    bot = torch.cat([torch.zeros(gee.shape[0], kb, dtype=G.dtype), Be], 1)
    return torch.cat([top, bot], 0)


def rayleigh_ritz(
    G: torch.Tensor, H: torch.Tensor, drop: float = _DROP, *, kb: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ritz pairs of the pencil (H, G) in float64: (theta (r,), Q (d, r)), Q^T G Q = I.

    G is scaled to unit diagonal (zero columns stay zero) and its directions
    below ``drop`` times the largest eigenvalue are dropped, so r <= d. With
    ``kb`` (G, H of [X, E], X its first kb columns), the X block is
    orthonormalised by Cholesky and only E's block by an eigendecomposition,
    which halves that step's host cost.
    """
    G = G.to(torch.float64)
    H = H.to(torch.float64)
    G = 0.5 * (G + G.T)
    H = 0.5 * (H + H.T)
    B = _augmented_basis(G, kb, drop) if kb is not None else None
    if B is None:
        B = _orth_basis(G, drop)
    Hb = B.T @ H @ B
    theta, W = torch.linalg.eigh(0.5 * (Hb + Hb.T))
    return theta, B @ W


def _subspace_device_ok(F: torch.Tensor, kb: int) -> bool:
    """True iff the filter runs as the NKI kernel for F and a kb-column block."""
    if F.dtype != torch.float32 or not (_use_nki() and HAS_NKI):
        return False
    if os.environ.get("TRNSOLVER_SUBSPACE_KERNEL", "1").strip() == "0":
        return False
    from .nki.subspace import kernel_fits

    if not kernel_fits(F.shape[0], kb):
        return False
    if _use_simulator():
        return True
    if F.device.type != "neuron":
        return False
    from .nki.dispatch import _get_dispatch_path

    return _get_dispatch_path() == "native"


def _run_filter(F: torch.Tensor, X: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """One filter launch: the kernel where it applies, else the reference."""
    if _subspace_device_ok(F, X.shape[1]):
        try:
            from .nki.subspace import run_filter_kernel

            return run_filter_kernel(F, X, table)
        except Exception:
            if _REQUIRE_NKI:
                raise
    return filter_reference(F, X, table.to(F.device))


def _to(t: torch.Tensor, device, dtype) -> torch.Tensor:
    """``t.to(device, dtype)``, casting on the host first for a host ``t``.

    On TorchNeuron a float64 host tensor sent with ``.to("neuron",
    torch.float32)`` is cast after the copy, which took 6.3 ms for the
    (372, 186) Ritz block on Trn2; cast and made contiguous on the host it
    is 0.1 ms.
    """
    if t.device.type == "cpu" and torch.device(device).type != "cpu":
        return t.to(dtype).contiguous().to(device)
    return t.to(device, dtype)


def _gershgorin_upper(F: torch.Tensor) -> float:
    """Upper bound on the spectrum: max_i (f_ii + sum_{j != i} |f_ij|)."""
    d = torch.diagonal(F)
    return float((d + F.abs().sum(1) - d.abs()).max())


def _cold_start(F: torch.Tensor, kb: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """(X, theta, top eigenvalue) from one full float64 host eigh of F."""
    w, V = torch.linalg.eigh(F.detach().to("cpu", torch.float64))
    return _to(V[:, :kb], F.device, F.dtype), w[:kb].clone(), float(w[-1])


def _complete(X: torch.Tensor, theta: torch.Tensor, kb: int, seed: int):
    """Pad (X (n, r), theta (r,)) to kb columns with random unit columns.

    Only for a rank-deficient block; the next pass's Rayleigh-Ritz
    orthogonalises them. Their Ritz values repeat the largest one.
    """
    r = X.shape[1]
    if r >= kb:
        return X[:, :kb], theta[:kb]
    g = torch.Generator().manual_seed(seed)
    extra = torch.randn(X.shape[0], kb - r, generator=g, dtype=torch.float64)
    extra = extra / extra.norm(dim=0, keepdim=True)
    X = torch.cat([X, _to(extra, X.device, X.dtype)], dim=1)
    return X, torch.cat([theta, theta[-1:].expand(kb - r)])


def _rr_block(F: torch.Tensor, X: torch.Tensor, kb: int):
    """Rayleigh-Ritz of an arbitrary (n, <= kb) block on F's device: (X', theta)."""
    G = (X.T @ X).to("cpu", torch.float64)
    H = (X.T @ (F @ X)).to("cpu", torch.float64)
    theta, Q = rayleigh_ritz(G, H)
    return _complete(X @ _to(Q, X.device, X.dtype), theta, kb, seed=kb)


@torch.compiler.disable
def occupied_subspace(
    F: torch.Tensor,
    k: int,
    *,
    guess: SubspaceState | torch.Tensor | None = None,
    buffer: int | None = None,
    passes: int = 2,
    tol: float | None = None,
    degree: int | None = None,
    max_degree: int = _MAX_DEGREE,
    amp_max: float = _AMP_MAX,
    upper: float | str = "gershgorin",
    refresh: float | None = 3e-2,
) -> SubspaceResult:
    """Lowest k eigenpairs of a symmetric F by Chebyshev-filtered subspace iteration.

    Args:
        F: (n, n) symmetric matrix (FP32 or FP64) on any device. In SCF, the
           Fock matrix in an orthonormal basis (``X^T F X``).
        k: number of eigenpairs (``nocc``).
        guess: the ``state`` of the previous call (warm start), an (n, j)
           block of starting vectors, or None for a cold start (one full
           float64 host ``eigh`` of F, returned exactly).
        buffer: extra block columns beyond k (default :func:`default_buffer`).
        passes: at most this many filter + Rayleigh-Ritz passes.
        tol: stop before a pass once every wanted residual
           ``|F x_j - theta_j x_j|`` of the block entering it is below
           ``tol * max(|theta_1|, |upper|)`` (that block is returned). Note
           the first pass checks the previous call's Ritz values against
           the new F, an upper bound. None: always run ``passes`` passes.
        degree: fixed filter degree (from ``DEGREES``); default: chosen per
           pass by :func:`choose_degree` (``max_degree``, ``amp_max``).
        refresh: if the first pass finds a wanted residual above
           ``refresh * max(|theta_1|, |upper|)`` (the block is far from F, as
           in the first SCF iterations), return a full host eigh instead,
           which also resets the warm start. None: never. The defaults
           (``passes=2``, ``refresh=3e-2``) are the measured SCF setting:
           DF-RHF in FP32 then needs the same iterations as with a full
           eigh (n_ao = 250 / 490: 16 / 16 against 16 / 15).
        upper: spectral upper bound: a number, ``"gershgorin"`` (default;
           one reduction and one host sync per call), or ``"cached"`` (the
           state's bound: the cold start's top eigenvalue plus 5% of the
           spread; valid only while F's top end stays below it).

    Returns:
        ``(eigenvalues (k,), eigenvectors (n, k), state)``. Pass ``state`` as
        ``guess`` to the next call.

    The filter, the correction and the Gram blocks run as one NKI launch for
    FP32 F on the native ``neuron`` device (and under
    ``TRNSOLVER_USE_SIMULATOR=1``); elsewhere as :func:`filter_reference`.
    The Rayleigh-Ritz is float64 on the host.
    """
    if F.dim() != 2 or F.shape[0] != F.shape[1]:
        raise ValueError(f"occupied_subspace expects (n, n); got {tuple(F.shape)}")
    n = F.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n; got k={k}, n={n}")
    if F.dtype not in (torch.float32, torch.float64):
        F = F.float()
    kb = min(n, k + (default_buffer(k) if buffer is None else max(0, buffer)))

    def cold():
        X, theta, top = _cold_start(F, kb)
        return X, theta, top + 0.05 * (top - float(theta[0]))

    if kb >= n or guess is None:
        # A full eigh is the answer (and the warm start for the next call).
        X, theta, bound = cold()
        state = SubspaceState(X=X, theta=theta, upper=bound)
        return SubspaceResult(_to(theta[:k], F.device, F.dtype), X[:, :k], state)

    cached = None
    if isinstance(guess, SubspaceState):
        X, theta, cached = guess.X.to(F.device, F.dtype), guess.theta, guess.upper
        if X.shape != (n, kb):
            X, theta = _rr_block(F, X[:, :kb], kb)
    else:
        X, theta = _rr_block(F, guess.to(F.device, F.dtype)[:, :kb], kb)

    if isinstance(upper, str):
        if upper not in ("gershgorin", "cached"):
            raise ValueError(f"upper must be a number, 'gershgorin' or 'cached'; got {upper!r}")
        b = cached if (upper == "cached" and cached is not None) else _gershgorin_upper(F)
    else:
        b = float(upper)

    resid = None
    w_dev = None  # theta[:k] on F's device, from the last pass's copy
    m = 0
    done = 0
    for _ in range(max(1, passes)):
        a0, a = float(theta[0]), float(theta[-1])
        if not (a0 < a < b):
            # Degenerate block, or a bound below the block: refresh exactly.
            X, theta, b = cold()
            resid = w_dev = None
            break
        m = degree or choose_degree(theta, k, b, amp_max=amp_max, max_degree=max_degree)
        table = filter_table(chebyshev_coefficients(m, a0, a, b), theta)
        out = _run_filter(F, X, table)
        G, H, r = unpack_grams(out[n:].to("cpu", torch.float64), kb)
        resid = torch.sqrt(torch.clamp(r, min=0.0))
        done += 1
        if os.environ.get("TRNSOLVER_SUBSPACE_DEBUG"):
            rel = float(resid[:k].max()) / max(abs(a0), abs(b))
            print(f"occupied_subspace: pass {done} m={m} resid/scale={rel:.2e}")
        worst = float(resid[:k].max()) / max(abs(a0), abs(b))
        if done == 1 and refresh is not None and worst > refresh:
            # The block is far from the new F (early SCF): solve exactly.
            X, theta, b = cold()
            m = done = 0
            w_dev = None
            break
        if tol is not None and worst <= tol:
            break
        theta, Q = rayleigh_ritz(G, H, kb=kb)
        # One H2D copy for the new block's coefficients and its Ritz values:
        # on Trn2 a second small copy right after the update took up to
        # 3.8 ms (a (41,) vector at n = 250), the packed one ~0.3 ms.
        w = min(kb, Q.shape[1])
        pk = _to(torch.cat([Q[:, :w], theta[:w].reshape(1, w)], 0), F.device, F.dtype)
        Q, w_dev = pk[:-1], (pk[-1, :k] if w >= k else None)
        X, theta = _complete(X @ Q[:kb] + out[:n] @ Q[kb:], theta[:kb], kb, seed=done)
    state = SubspaceState(X=X, theta=theta, upper=b, degree=m, passes=done, residual=resid)
    if w_dev is None:
        w_dev = _to(theta[:k], F.device, F.dtype)
    return SubspaceResult(w_dev, X[:, :k], state)
