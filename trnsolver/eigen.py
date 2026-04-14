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

import torch

from ._brent_luk import brent_luk_permutations
from .nki import _REQUIRE_NKI, HAS_NKI, _use_nki, _use_simulator


def eigh(
    A: torch.Tensor,
    max_sweeps: int = 100,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
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
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _torch_eigh(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    idx_p = torch.arange(0, n, 2, device=D.device)  # 0, 2, 4, ...
    idx_q = idx_p + 1  # 1, 3, 5, ...

    d_pp = D[idx_p, idx_p]  # (half,)
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

    return torch.stack([c, s], dim=1).to(D.dtype)  # (half, 2)


def _diag_block_fixup_strided(
    D: torch.Tensor,
    cs: torch.Tensor,
    d_pp_old: torch.Tensor,
    d_qq_old: torch.Tensor,
    d_pq_old: torch.Tensor,
) -> torch.Tensor:
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


def _call_matvec(A, v):
    """Dispatch matvec_kernel through torch_xla or the CPU simulator."""
    from .nki.dispatch import matvec_kernel

    if _use_simulator():
        import nki

        w = nki.simulate(matvec_kernel)(A.detach().cpu().numpy(), v.detach().cpu().numpy())
        return torch.from_numpy(w).to(A.device)
    return matvec_kernel(A, v)


def _call_rank2_update(A, u, v):
    """Dispatch rank2_update_kernel through torch_xla or the CPU simulator."""
    from .nki.dispatch import rank2_update_kernel

    if _use_simulator():
        import nki

        out = nki.simulate(rank2_update_kernel)(
            A.detach().cpu().numpy(), u.detach().cpu().numpy(), v.detach().cpu().numpy()
        )
        return torch.from_numpy(out).to(A.device)
    return rank2_update_kernel(A, u, v)


def _householder_tridiag(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric Householder tridiagonalization via NKI kernels.

    Returns (diag, subdiag, V_reflectors). V[:, k] stores the k-th
    Householder vector (zero-padded above row k+1), normalized so that
    ||V[:, k]||₂ = 1. Q₁ can be reconstructed on the host as the product
    of (I − 2 V[:, k] V[:, k]ᵀ) for k = 0..n-2.

    Host drives the outer loop; kernels (matvec_kernel, rank2_update_kernel)
    handle the hot Tensor-Engine-shaped ops. Under TRNSOLVER_USE_SIMULATOR=1
    the kernels route through nki.simulate on CPU; otherwise through the
    XLA path on a Neuron device.

    Correctness MVP — the kernels are Vector-Engine-only for now. Tensor
    Engine conversion is a pure-perf refactor (issue #36) that keeps the
    same correctness test contract.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI backend requested but the nki package isn't importable")

    n = A.shape[0]
    assert A.shape == (n, n), f"Expected square matrix, got {A.shape}"
    if n > 128:
        raise NotImplementedError(
            f"Phase 1 Householder tridiag requires n ≤ 128 (single-tile); got n={n}."
        )

    A_work = A.clone()
    V_refs = torch.zeros((n, max(n - 1, 1)), dtype=A.dtype, device=A.device)

    for k in range(n - 2):
        # Householder on column k, entries k+1..n-1
        x = A_work[k + 1 :, k]
        x_norm = torch.linalg.norm(x).item()
        if x_norm < 1e-20:
            # Already zero below diag in this column; skip step.
            continue

        sign_x0 = torch.sign(x[0]).item() if x[0].item() != 0 else 1.0
        alpha = -sign_x0 * x_norm

        # v_sub = x - alpha * e1, expanded to full n with zeros above row k+1
        v = torch.zeros(n, dtype=A.dtype, device=A.device)
        v[k + 1 :] = x
        v[k + 1] -= alpha

        vv = torch.dot(v, v).item()
        if vv < 1e-30:
            continue
        beta = 2.0 / vv

        # w = beta * A @ v (via kernel)
        v_col = v.unsqueeze(1)  # (n, 1)
        w_col = _call_matvec(A_work, v_col)  # (n, 1)
        w = w_col.squeeze(1) * beta

        # gamma = beta * v^T w / 2
        gamma = beta * torch.dot(v, w).item() / 2.0

        # u = w - gamma * v
        u = w - gamma * v

        # A <- A - u v^T - v u^T (via kernel; u, v are zero above row k+1)
        u_col = u.unsqueeze(1)
        A_work = _call_rank2_update(A_work, u_col, v_col)

        # Enforce the tridiagonal structure at col/row k: the kernel update
        # leaves (i ≤ k, j ≤ k) untouched and correctly modifies (k+1:, k+1:);
        # the new (k+1, k) entry should be α and everything below that in col k
        # should be zero.
        A_work[k + 1, k] = alpha
        A_work[k, k + 1] = alpha
        if k + 2 < n:
            A_work[k + 2 :, k] = 0.0
            A_work[k, k + 2 :] = 0.0

        # Store normalized reflector for eventual Q reconstruction.
        v_norm = v.norm()
        if v_norm.item() > 0:
            V_refs[:, k] = v / v_norm

    diag = torch.diagonal(A_work).clone()
    subdiag = torch.diagonal(A_work, offset=1).clone()
    return diag, subdiag, V_refs


def _call_rotate_pairs(even, odd, c, s):
    """Dispatch rotate_pairs_kernel through torch_xla or the CPU simulator.

    When `TRNSOLVER_USE_SIMULATOR=1` is set and the `nki` package is
    importable, inputs are converted torch → numpy and routed through
    `nki.simulate(kernel)(np_args)`; outputs come back as torch tensors.
    Otherwise the kernel is invoked with its XLA inputs directly.
    """
    from .nki.dispatch import rotate_pairs_kernel

    if _use_simulator():
        import nki

        even_np = even.detach().cpu().numpy()
        odd_np = odd.detach().cpu().numpy()
        c_np = c.detach().cpu().numpy()
        s_np = s.detach().cpu().numpy()
        ne, no = nki.simulate(rotate_pairs_kernel)(even_np, odd_np, c_np, s_np)
        return torch.from_numpy(ne).to(even.device), torch.from_numpy(no).to(even.device)

    return rotate_pairs_kernel(even, odd, c, s)


def _jacobi_eigh_nki(
    A: torch.Tensor,
    max_sweeps: int,
    tol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Brent-Luk parallel Jacobi via the NKI batched-round kernel.

    Requires even n. Pads A with a zero-off-diagonal identity block if odd
    (not yet implemented; Phase 1 requires even n).

    Dispatch paths:
      * `TRNSOLVER_USE_SIMULATOR=1` → CPU, kernels routed through
        `nki.simulate(...)`. No torch_xla dependency.
      * Otherwise → Neuron hardware via torch_xla / torch_neuronx.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI backend requested but the nki package isn't importable")

    n = A.shape[0]
    assert A.shape == (n, n), f"Expected square matrix, got {A.shape}"
    if n % 2 != 0:
        raise NotImplementedError(
            f"Phase 1 NKI Jacobi requires even n; got n={n}. Pad to even or use backend='pytorch'."
        )

    orig_device = A.device

    if _use_simulator():
        # Stay on CPU — the simulator takes numpy, no XLA device needed.
        work_device = A.device
    else:
        import torch_neuronx  # noqa: F401 — registers the Neuron PJRT plugin
        import torch_xla

        work_device = torch_xla.device()

    D = A.clone().to(work_device)
    V = torch.eye(n, dtype=A.dtype, device=work_device)

    perms_host = brent_luk_permutations(n)
    perms = perms_host.to(work_device)
    cum_perm = torch.arange(n, dtype=torch.int64, device=work_device)

    idx_p = torch.arange(0, n, 2, device=work_device)
    idx_q = idx_p + 1

    for _sweep in range(max_sweeps):
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

            cs = _rotation_angles_strided(D)  # (half, 2)
            c_col = cs[:, 0:1].contiguous()  # (half, 1)
            s_col = cs[:, 1:2].contiguous()

            # --- Rotate D's rows: even rows (0, 2, 4, ...) with odd rows (1, 3, 5, ...) ---
            D_even = D[idx_p, :]  # (half, n)
            D_odd = D[idx_q, :]
            D_even_new, D_odd_new = _call_rotate_pairs(D_even, D_odd, c_col, s_col)
            D = D.clone()
            D[idx_p, :] = D_even_new
            D[idx_q, :] = D_odd_new

            # --- Rotate D's cols: even cols with odd cols ---
            # Transpose-view: cols (n, half) → tile (half, n) by taking D^T rows
            Dc_even = D[:, idx_p].t().contiguous()  # (half, n)
            Dc_odd = D[:, idx_q].t().contiguous()
            Dc_even_new, Dc_odd_new = _call_rotate_pairs(Dc_even, Dc_odd, c_col, s_col)
            D[:, idx_p] = Dc_even_new.t()
            D[:, idx_q] = Dc_odd_new.t()

            # --- Rotate V's cols: even cols with odd cols ---
            Vc_even = V[:, idx_p].t().contiguous()
            Vc_odd = V[:, idx_q].t().contiguous()
            Vc_even_new, Vc_odd_new = _call_rotate_pairs(Vc_even, Vc_odd, c_col, s_col)
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
