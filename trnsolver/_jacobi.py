"""
Batched parallel-ordered (round-robin) Jacobi for symmetric ``eigh`` (#48).

A cyclic Jacobi sweep in round-robin order has n - 1 steps. Each step
rotates n / 2 disjoint pairs ``(p, q)`` at once, so the whole step is one
orthogonal matrix ``J = diag(c) + diag(s) @ Pi`` where ``Pi`` is the
step's partner permutation (an involution, ``Pi[p, q] = Pi[q, p] = 1``).
The update ``A <- J^T A J`` and the eigenvector update ``V <- V J`` are
dense matmuls, so over a batch of B matrices a step is a batched matmul.
That keeps the Tensor Engine busy, where the Householder reduction of #49
is a chain of n - 2 dependent BLAS-2 steps.

This module holds the device-agnostic parts: the round-robin schedule,
the per-step rotation angles (written so that both members of a pair
derive the same rotation from their own row, with no gather and no
divide), and a torch implementation that runs on any device. The NKI
kernel in ``trnsolver/nki/jacobi.py`` follows the same math step for step.

Angles. For a pair with ``h = a_qq - a_pp`` and ``g = a_pq``, the inner
rotation (|theta| <= pi/4) that zeros ``a_pq`` has

    t = sgn(h) * 2 g / (|h| + sqrt(h^2 + 4 g^2)),   c = 1/sqrt(1 + t^2),  s = t c

with ``J[p, p] = J[q, q] = c``, ``J[p, q] = s``, ``J[q, p] = -s``. Row p
computes ``t`` from ``h_p = d[partner(p)] - d[p]``; row q sees ``h_q = -h_p``
and gets ``-t``, which is exactly the ``J[q, p] = -s`` entry. The sign of
``h`` is taken in the pair's canonical orientation (``sigma = +1`` for the
lower index), so a tie ``h = 0`` still gives an antisymmetric pair.

Odd n is padded to n + 1 with a decoupled zero row/column; its rotation
angle is exactly zero, and the extra eigenpair is dropped at the end.
"""

from __future__ import annotations

import functools

import torch

# Below this denominator the pair is already diagonal (h = g = 0): t = 0.
_TINY = 1e-30


@functools.lru_cache(maxsize=32)
def round_robin_schedule(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-robin (circle-method) pairing for even ``n``.

    Returns:
        partner : (n - 1, n) int64. ``partner[r, p]`` is p's partner in step r.
        sigma   : (n - 1, n) float32. +1 if p is the lower index of its
                  pair in step r, -1 otherwise.

    Every unordered pair (p, q), p != q, occurs in exactly one step.
    """
    if n < 2 or n % 2:
        raise ValueError(f"round_robin_schedule needs an even n >= 2, got {n}")
    players = list(range(n))
    partner = torch.empty((n - 1, n), dtype=torch.int64)
    for r in range(n - 1):
        for i in range(n // 2):
            p, q = players[i], players[n - 1 - i]
            partner[r, p] = q
            partner[r, q] = p
        players = [players[0], players[-1], *players[1:-1]]
    idx = torch.arange(n).expand(n - 1, n)
    sigma = torch.where(idx < partner, 1.0, -1.0).to(torch.float32)
    return partner, sigma


def step_masks(n: int, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense per-step partner matrices and signs, the kernel's constants.

    Returns:
        Pi    : (n - 1, n, n). ``Pi[r]`` is the step-r partner permutation.
        sigma : (n - 1, n, 1).
    """
    partner, sigma = round_robin_schedule(n)
    Pi = torch.zeros((n - 1, n, n), dtype=dtype)
    Pi.scatter_(2, partner.unsqueeze(-1), 1.0)
    return Pi, sigma.to(dtype).unsqueeze(-1)


def rotation_csk(
    d: torch.Tensor, dq: torch.Tensor, g: torch.Tensor, sigma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row rotation of one step: (c, s, c - 1).

    Args:
        d     : (..., n) diagonal a_pp.
        dq    : (..., n) partner diagonal a_qq.
        g     : (..., n) a_pq.
        sigma : (n,) +1 / -1 canonical orientation.

    ``c - 1 = -s^2 / (1 + c)`` is formed directly: the updates are applied
    as ``X + X K`` with ``K = J - I`` (see :func:`jacobi_step`), so a small
    rotation keeps its O(t^2) diagonal shrinkage instead of rounding
    ``c`` to 1 and growing every column norm by t^2.
    """
    h = dq - d
    sg = sigma * torch.where(sigma * h >= 0, 1.0, -1.0).to(d.dtype)
    r = torch.sqrt(h * h + 4.0 * g * g)
    t = sg * 2.0 * g * torch.reciprocal(torch.clamp(h.abs() + r, min=_TINY))
    c = torch.rsqrt(1.0 + t * t)
    s = t * c
    return c, s, -(s * s) * torch.reciprocal(1.0 + c)


def jacobi_step(
    A: torch.Tensor, W: torch.Tensor, P: torch.Tensor, sigma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One round-robin step on a (B, n, n) batch with partner matrix P.

    ``W`` holds the eigenvectors transposed (``W = V^T``), the layout the
    NKI kernel keeps: ``V <- V J`` becomes ``W <- J^T W``. With ``K = J - I``:

        T = A + A K,    A <- T + K^T T,    W <- W + K^T W.
    """
    n = A.shape[-1]
    d = torch.diagonal(A, dim1=-2, dim2=-1)
    dq = d @ P
    # Symmetrised a_pq: rows p and q must see the same g, or s_p != -s_q and
    # the rounding asymmetry of A is amplified by a_qq / (a_qq - a_pp) into
    # an off(A) floor near sqrt(eps) on clustered spectra.
    g = 0.5 * ((A + A.transpose(-2, -1)) * P).sum(-1)
    _, s, k = rotation_csk(d, dq, g, sigma)
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    K = k.unsqueeze(-1) * eye + s.unsqueeze(-1) * P
    Kt = K.transpose(-2, -1)
    T = A + torch.bmm(A, K)
    A = T + torch.bmm(Kt, T)
    W = W + torch.bmm(Kt, W)
    return A, W


def off_norm(A: torch.Tensor) -> torch.Tensor:
    """Relative off-diagonal Frobenius norm per matrix: off(A) / ||A||_F.

    Summed directly over the off-diagonal entries; ``||A||^2 - ||diag||^2``
    cancels down to a floor of sqrt(eps) and never reaches the tolerance.
    """
    n = A.shape[-1]
    mask = 1.0 - torch.eye(n, dtype=A.dtype, device=A.device)
    off2 = (A * A * mask).sum(dim=(-2, -1))
    fro2 = (A * A).sum(dim=(-2, -1))
    return torch.sqrt(off2) / torch.sqrt(torch.clamp(fro2, min=_TINY))


def _pad_even(A: torch.Tensor) -> tuple[torch.Tensor, int]:
    n = A.shape[-1]
    if n % 2 == 0:
        return A, n
    Ap = torch.zeros((*A.shape[:-2], n + 1, n + 1), dtype=A.dtype, device=A.device)
    Ap[..., :n, :n] = A
    return Ap, n


def jacobi_sweeps_torch(
    A: torch.Tensor,
    W: torch.Tensor,
    sweeps: int,
    step=jacobi_step,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``sweeps`` round-robin sweeps on a (B, n, n) batch (n even).

    ``W = V^T``. Every op is a plain aten op (``torch.bmm`` for the
    updates), so this runs unchanged on CPU or the ``neuron`` device.
    ``step`` may be a ``torch.compile``'d :func:`jacobi_step`.
    """
    n = A.shape[-1]
    Pi, sigma = step_masks(n, A.dtype)
    Pi = Pi.to(A.device)
    sigma = sigma.squeeze(-1).to(A.device)
    for _ in range(sweeps):
        for r in range(n - 1):
            A, W = step(A, W, Pi[r], sigma[r])
    return A, W


def finish(A: torch.Tensor, W: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigenpairs from the converged (A, W = V^T): sorted ascending, padding dropped."""
    w = A.diagonal(dim1=-2, dim2=-1)
    V = W.transpose(-2, -1)
    if V.shape[-1] != n:
        # The padded index (last) is decoupled: its eigenvector is e_last.
        w = w[..., :n]
        V = V[..., :n, :n]
    w, order = torch.sort(w, dim=-1)
    V = torch.gather(V, -1, order.unsqueeze(-2).expand_as(V))
    return w, V


def jacobi_eigh_torch(
    A: torch.Tensor,
    *,
    tol: float = 1e-6,
    max_sweeps: int = 15,
    check_every: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Batched Jacobi eigh in torch: returns (w, V, sweeps_run).

    Converged when every matrix has ``off(A) / ||A||_F < tol``, checked every
    ``check_every`` sweeps (each check is one host sync).
    """
    squeeze = A.dim() == 2
    if squeeze:
        A = A.unsqueeze(0)
    Ap, n = _pad_even(A)
    np_ = Ap.shape[-1]
    W = torch.eye(np_, dtype=Ap.dtype, device=Ap.device).expand_as(Ap).contiguous()
    done = 0
    while done < max_sweeps:
        k = min(check_every, max_sweeps - done)
        Ap, W = jacobi_sweeps_torch(Ap, W, k)
        done += k
        if float(off_norm(Ap).max()) < tol:
            break
    w, V = finish(Ap, W, n)
    if squeeze:
        w, V = w[0], V[0]
    return w, V, done
