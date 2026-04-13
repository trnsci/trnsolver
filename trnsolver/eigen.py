"""
Eigenvalue decomposition for Trainium.

Symmetric eigenvalue problem: A @ V = V @ diag(eigenvalues)

Methods:
- NKI path: Brent-Luk parallel Jacobi. Host precomputes the n-1 round
  permutations and applies each round in a fixed strided-pair layout so the
  NKI kernel (trnsolver/nki/dispatch.py::jacobi_round_kernel) compiles once
  and is reused across all rounds and all sweeps.
- PyTorch path: torch.linalg.eigh (LAPACK / MAGMA depending on device).

Primary use case: SCF eigenvalue problem FC = SCε in quantum chemistry.
The generalized eigenproblem reduces to standard form via Cholesky of S.
"""

from __future__ import annotations

import math
import torch
from typing import Optional, Tuple

from .nki import _use_nki, _REQUIRE_NKI, HAS_NKI
from ._brent_luk import brent_luk_permutations


def eigh(
    A: torch.Tensor,
    max_sweeps: int = 100,
    tol: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric eigenvalue decomposition: A = V @ diag(w) @ V^T

    Args:
        A: Symmetric matrix (n, n)
        max_sweeps: Maximum Jacobi sweeps (NKI path only)
        tol: Convergence threshold — sum of squared off-diagonal elements

    Returns:
        eigenvalues: (n,) sorted ascending
        eigenvectors: (n, n) columns are eigenvectors
    """
    if _use_nki():
        try:
            return _jacobi_eigh_nki(A, max_sweeps, tol)
        except Exception:
            if _REQUIRE_NKI:
                raise
            return _torch_eigh(A)
    return _torch_eigh(A)


def eigh_generalized(
    A: torch.Tensor,
    B: torch.Tensor,
    max_sweeps: int = 100,
    tol: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized symmetric eigenvalue problem: A @ x = λ B @ x

    Reduces to standard form via Cholesky of B:
        B = L @ L^T
        (L^{-1} A L^{-T}) y = λ y
        x = L^{-T} y

    The Cholesky + triangular solves stay on the PyTorch path; the inner
    eigh call dispatches per backend.
    """
    L = torch.linalg.cholesky(B)
    L_inv_A = torch.linalg.solve_triangular(L, A, upper=False)
    A_prime = torch.linalg.solve_triangular(L, L_inv_A.T, upper=False).T
    A_prime = 0.5 * (A_prime + A_prime.T)

    eigenvalues, V_prime = eigh(A_prime, max_sweeps, tol)
    eigenvectors = torch.linalg.solve_triangular(L.T, V_prime, upper=True)
    return eigenvalues, eigenvectors


def _torch_eigh(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch backend path — delegates to torch.linalg.eigh (LAPACK)."""
    return torch.linalg.eigh(A)


def _rotation_angles_strided(D: torch.Tensor) -> torch.Tensor:
    """Compute (c, s) for each strided pair (2i, 2i+1) of the current D.

    Classical Jacobi rotation angle:
        if |D[p,p] - D[q,q]| ≈ 0:   θ = π/4
        else:
            τ = (D[q,q] - D[p,p]) / (2 D[p,q])
            t = sign(τ) / (|τ| + √(1 + τ²))
            c = 1 / √(1 + t²)
            s = t c

    Works element-wise over half pairs. Returns a (half, 2) tensor on the
    same device as D.
    """
    n = D.shape[0]
    half = n // 2

    idx_p = torch.arange(0, n, 2, device=D.device)        # 0, 2, 4, ...
    idx_q = idx_p + 1                                      # 1, 3, 5, ...

    d_pp = D[idx_p, idx_p]                                 # (half,)
    d_qq = D[idx_q, idx_q]
    d_pq = D[idx_p, idx_q]

    # Guard zeros: if d_pq ≈ 0 the rotation is unnecessary; pick c=1, s=0.
    abs_pq = d_pq.abs()
    safe = abs_pq > 1e-30

    diff = d_qq - d_pp
    # τ = (d_qq - d_pp) / (2 d_pq); handle d_pq ≈ 0 with a safe divisor
    tau_denom = torch.where(safe, 2.0 * d_pq, torch.ones_like(d_pq))
    tau = diff / tau_denom
    t = torch.where(
        tau >= 0,
        1.0 / (tau + torch.sqrt(1.0 + tau * tau)),
        -1.0 / (-tau + torch.sqrt(1.0 + tau * tau)),
    )
    c = 1.0 / torch.sqrt(1.0 + t * t)
    s = t * c

    # Special case: when d_pp ≈ d_qq exactly, the formula above is fine
    # (tau is large but stable). When d_pq ≈ 0, set c=1, s=0.
    c = torch.where(safe, c, torch.ones_like(c))
    s = torch.where(safe, s, torch.zeros_like(s))

    return torch.stack([c, s], dim=1).to(D.dtype)          # (half, 2)


def _diag_block_fixup_strided(D: torch.Tensor, cs: torch.Tensor, d_pp_old: torch.Tensor, d_qq_old: torch.Tensor, d_pq_old: torch.Tensor) -> torch.Tensor:
    """Overwrite the 2×2 diagonal blocks of D at strided pairs (2i, 2i+1).

    The NKI kernel rotates rows and columns independently, which produces
    incorrect values at the intersection (2i:2i+2, 2i:2i+2) blocks. Replace
    them with the analytically correct post-rotation values:

        D[p,p] = c² d_pp - 2cs d_pq + s² d_qq
        D[q,q] = s² d_pp + 2cs d_pq + c² d_qq
        D[p,q] = D[q,p] = 0
    """
    n = D.shape[0]
    c = cs[:, 0]
    s = cs[:, 1]

    new_pp = c * c * d_pp_old - 2.0 * c * s * d_pq_old + s * s * d_qq_old
    new_qq = s * s * d_pp_old + 2.0 * c * s * d_pq_old + c * c * d_qq_old

    idx_p = torch.arange(0, n, 2, device=D.device)
    idx_q = idx_p + 1

    D = D.clone()
    D[idx_p, idx_p] = new_pp
    D[idx_q, idx_q] = new_qq
    D[idx_p, idx_q] = 0.0
    D[idx_q, idx_p] = 0.0
    return D


def _jacobi_eigh_nki(
    A: torch.Tensor,
    max_sweeps: int,
    tol: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Brent-Luk parallel Jacobi via the NKI batched-round kernel.

    Requires even n. Pads A with a zero-off-diagonal identity block if odd
    (not yet implemented; Phase 1 requires even n).
    """
    if not HAS_NKI:
        raise RuntimeError("NKI backend requested but neuronxcc is not available")
    from .nki.dispatch import rotate_pairs_kernel
    import torch_neuronx  # noqa: F401 — registers the Neuron PJRT plugin
    import torch_xla

    n = A.shape[0]
    assert A.shape == (n, n), f"Expected square matrix, got {A.shape}"
    if n % 2 != 0:
        raise NotImplementedError(
            f"Phase 1 NKI Jacobi requires even n; got n={n}. Pad to even or use backend='pytorch'."
        )

    half = n // 2
    orig_device = A.device
    xla_device = torch_xla.device()

    D = A.clone().to(xla_device)
    V = torch.eye(n, dtype=A.dtype, device=xla_device)

    perms_host = brent_luk_permutations(n)
    perms = perms_host.to(xla_device)
    cum_perm = torch.arange(n, dtype=torch.int64, device=xla_device)

    idx_p = torch.arange(0, n, 2, device=xla_device)
    idx_q = idx_p + 1

    for sweep in range(max_sweeps):
        diag_sq = (torch.diagonal(D) ** 2).sum()
        off_sq = (D * D).sum() - diag_sq
        if off_sq.item() < tol:
            break

        for r in range(n - 1):
            perm = perms[r]
            D = D[perm][:, perm]
            V = V[:, perm]
            cum_perm = cum_perm[perm]

            d_pp_old = D[idx_p, idx_p].clone()
            d_qq_old = D[idx_q, idx_q].clone()
            d_pq_old = D[idx_p, idx_q].clone()

            cs = _rotation_angles_strided(D)              # (half, 2)
            c_col = cs[:, 0:1].contiguous()               # (half, 1)
            s_col = cs[:, 1:2].contiguous()

            # --- Rotate D's rows: even rows (0, 2, 4, ...) with odd rows (1, 3, 5, ...) ---
            D_even = D[idx_p, :]                          # (half, n)
            D_odd  = D[idx_q, :]
            D_even_new, D_odd_new = rotate_pairs_kernel(D_even, D_odd, c_col, s_col)
            D = D.clone()
            D[idx_p, :] = D_even_new
            D[idx_q, :] = D_odd_new

            # --- Rotate D's cols: even cols with odd cols ---
            # Transpose-view: cols (n, half) → tile (half, n) by taking D^T rows
            Dc_even = D[:, idx_p].t().contiguous()        # (half, n)
            Dc_odd  = D[:, idx_q].t().contiguous()
            Dc_even_new, Dc_odd_new = rotate_pairs_kernel(Dc_even, Dc_odd, c_col, s_col)
            D[:, idx_p] = Dc_even_new.t()
            D[:, idx_q] = Dc_odd_new.t()

            # --- Rotate V's cols: even cols with odd cols ---
            Vc_even = V[:, idx_p].t().contiguous()
            Vc_odd  = V[:, idx_q].t().contiguous()
            Vc_even_new, Vc_odd_new = rotate_pairs_kernel(Vc_even, Vc_odd, c_col, s_col)
            V = V.clone()
            V[:, idx_p] = Vc_even_new.t()
            V[:, idx_q] = Vc_odd_new.t()

            # --- Diagonal block fixup ---
            D = _diag_block_fixup_strided(D, cs, d_pp_old, d_qq_old, d_pq_old)

    # Un-permute to original index order.
    inv_perm = torch.argsort(cum_perm)
    D = D[inv_perm][:, inv_perm]
    V = V[:, inv_perm]

    eigenvalues = torch.diagonal(D).clone()
    idx = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    return eigenvalues.to(orig_device), V.to(orig_device)
