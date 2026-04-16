"""
Eigenvalue decomposition for Trainium.

Symmetric eigenvalue problem: A @ V = V @ diag(eigenvalues)

Methods:
- NKI path: Householder tridiagonalization via NKI kernels (matvec +
  rank-2 update on the Tensor Engine) followed by pure-host implicit-
  shift QR iteration on the tridiagonal. Eigenvector matrix is assembled
  on host from stored Householder reflectors and accumulated Givens
  rotations.
- PyTorch path: torch.linalg.eigh (LAPACK / MAGMA depending on device).

Primary use case: SCF eigenvalue problem FC = SCε in quantum chemistry.
The generalized eigenproblem reduces to standard form via Cholesky of S.

Design background: the classical-Jacobi path (pre-2026-04-14) fought the
NKI compile cache. See #9 post-mortem and #38's architecture decision
for the Householder-QR pivot rationale.
"""

from __future__ import annotations

import math

import torch

from .nki import _REQUIRE_NKI, HAS_NKI, _use_nki, _use_simulator


def eigh(
    A: torch.Tensor,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric eigenvalue decomposition: A = V @ diag(w) @ V^T

    Args:
        A: Symmetric matrix (n, n)
        tol: Convergence tolerance for QR iteration (NKI path only)

    Returns:
        eigenvalues: (n,) sorted ascending
        eigenvectors: (n, n) columns are eigenvectors
    """
    if _use_nki():
        try:
            return _householder_qr_eigh(A, tol)
        except Exception:
            if _REQUIRE_NKI:
                raise
            return _torch_eigh(A)
    return _torch_eigh(A)


def eigh_generalized(
    A: torch.Tensor,
    B: torch.Tensor,
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

    eigenvalues, V_prime = eigh(A_prime, tol)
    eigenvectors = torch.linalg.solve_triangular(L.T, V_prime, upper=True)
    return eigenvalues, eigenvectors


def _torch_eigh(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch backend path — delegates to torch.linalg.eigh (LAPACK)."""
    return torch.linalg.eigh(A)


# ----------------------------------------------------------------------
# NKI kernel dispatch helpers
# ----------------------------------------------------------------------


def _call_matvec(A, v):
    """Dispatch matvec_kernel through torch_xla or the CPU simulator.

    On hardware: matvec_kernel (Tensor Engine via nisa.nc_matmul).
    On simulator: matvec_kernel_sim (Vector Engine broadcast+sum). The NKI
    0.3.0 simulator context wrapper strips the stationary argument from
    nc_matmul calls, so the Tensor Engine kernel cannot run under nki.simulate.
    Both kernels compute the same result.
    """
    if _use_simulator():
        import nki

        from .nki.dispatch import matvec_kernel_sim

        w = nki.simulate(matvec_kernel_sim)(A.detach().cpu().numpy(), v.detach().cpu().numpy())
        return torch.from_numpy(w).to(A.device)
    from .nki.dispatch import matvec_kernel

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


# ----------------------------------------------------------------------
# Householder tridiagonalization (stage 1, #38 option B)
# ----------------------------------------------------------------------


def _householder_tridiag(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce symmetric A to tridiagonal form via Householder reflections.

    Returns (diag, subdiag, V_reflectors). V[:, k] stores the k-th
    Householder vector (zero-padded above row k+1), normalized so that
    ||V[:, k]||₂ = 1. Q₁ = product of (I − 2 V[:, k] V[:, k]ᵀ) satisfies
    Q₁ᵀ A Q₁ = T.

    Host drives the outer loop; kernels (matvec_kernel, rank2_update_kernel
    in trnsolver/nki/dispatch.py) handle the hot ops. Under
    TRNSOLVER_USE_SIMULATOR=1 the kernels route through nki.simulate on CPU;
    otherwise through the XLA path on a Neuron device.

    Current kernels are Vector-Engine-only — the Tensor Engine lift (via
    nisa.nc_matmul + PSUM accumulation) is a pure-perf refactor tracked
    in #36 that keeps the same correctness test contract.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI backend requested but the nki package isn't importable")

    n = A.shape[0]
    assert A.shape == (n, n), f"Expected square matrix, got {A.shape}"
    if n > 128:
        raise NotImplementedError(
            f"Phase 1 Householder tridiag requires n ≤ 128 (single-tile); got n={n}."
        )

    # On hardware (HAS_NKI and not _use_simulator) the NKI kernel expects
    # tensors on the XLA (Neuron) device. The simulator path takes numpy,
    # so CPU tensors are fine there.
    orig_device = A.device
    if _use_simulator():
        work_device = orig_device
    else:
        import torch_neuronx  # noqa: F401 — registers the Neuron PJRT plugin
        import torch_xla

        work_device = torch_xla.device()

    A_work = A.clone().to(work_device)
    V_refs = torch.zeros((n, max(n - 1, 1)), dtype=A.dtype, device=work_device)

    for k in range(n - 2):
        x = A_work[k + 1 :, k]
        x_norm = torch.linalg.norm(x).item()
        if x_norm < 1e-20:
            continue

        sign_x0 = torch.sign(x[0]).item() if x[0].item() != 0 else 1.0
        alpha = -sign_x0 * x_norm

        v = torch.zeros(n, dtype=A.dtype, device=work_device)
        v[k + 1 :] = x
        v[k + 1] -= alpha

        vv = torch.dot(v, v).item()
        if vv < 1e-30:
            continue
        beta = 2.0 / vv

        v_col = v.unsqueeze(1)
        w_col = _call_matvec(A_work, v_col)
        w = w_col.squeeze(1) * beta

        gamma = beta * torch.dot(v, w).item() / 2.0
        u = w - gamma * v

        u_col = u.unsqueeze(1)
        A_work = _call_rank2_update(A_work, u_col, v_col)

        A_work[k + 1, k] = alpha
        A_work[k, k + 1] = alpha
        if k + 2 < n:
            A_work[k + 2 :, k] = 0.0
            A_work[k, k + 2 :] = 0.0

        v_norm_val = v.norm()
        if v_norm_val.item() > 0:
            V_refs[:, k] = v / v_norm_val

        # Commit all pending XLA ops before the next iteration.
        # A_work is the output of rank2_update_kernel, so it carries a
        # computation history that makes the traced XLA graph unique per
        # step — causing a fresh NEFF compile on every outer-loop call.
        # torch_xla.sync() materialises A_work into a concrete HBM buffer;
        # the next _call_matvec then sees a plain leaf tensor and reuses
        # the cached NEFF. Skipped on the simulator path (no XLA layer).
        if not _use_simulator():
            import torch_xla

            torch_xla.sync()

    diag = torch.diagonal(A_work).clone().to(orig_device)
    subdiag = torch.diagonal(A_work, offset=1).clone().to(orig_device)
    V_refs = V_refs.to(orig_device)
    return diag, subdiag, V_refs


# ----------------------------------------------------------------------
# Implicit-shift QR iteration (stage 2, pure host)
# ----------------------------------------------------------------------


def _wilkinson_shift(diag_tail: torch.Tensor, subdiag_tail: torch.Tensor) -> float:
    """Wilkinson shift for the trailing 2×2 block of the active tridiagonal.

    diag_tail    : (2,) — [d[n-2], d[n-1]]
    subdiag_tail : (1,) — [e[n-2]]

    Picks the eigenvalue of the 2×2 block closer to d[n-1]. Gives cubic
    convergence of implicit QR near simple eigenvalues.
    """
    d0 = diag_tail[0].item()
    d1 = diag_tail[1].item()
    e = subdiag_tail[0].item()
    delta = (d0 - d1) / 2.0
    sign = 1.0 if delta >= 0 else -1.0
    denom = abs(delta) + math.sqrt(delta * delta + e * e)
    if denom < 1e-30:
        return d1
    return d1 - sign * e * e / denom


def _qr_sweep(diag: torch.Tensor, subdiag: torch.Tensor, Q: torch.Tensor, lo: int, hi: int):
    """One implicit-shift QR sweep on diag[lo:hi+1], subdiag[lo:hi].

    Bulge-chase with Givens rotations. Modifies diag, subdiag, Q in-place.
    Active block endpoints are inclusive: processes diag[lo], ..., diag[hi].
    """
    n_active = hi - lo + 1
    if n_active < 2:
        return

    shift = _wilkinson_shift(diag[hi - 1 : hi + 1], subdiag[hi - 1 : hi])

    # Initial Givens to annihilate the first subdiagonal after shift.
    x = diag[lo].item() - shift
    y = subdiag[lo].item()

    for k in range(lo, hi):
        # Compute Givens (c, s) that zeroes y using x.
        r = math.hypot(x, y)
        if r < 1e-30:
            c, s = 1.0, 0.0
        else:
            c = x / r
            s = y / r

        # Apply the rotation to rows/cols k and k+1 of the active tridiagonal.
        if k > lo:
            subdiag[k - 1] = r

        d_k = diag[k].item()
        d_k1 = diag[k + 1].item()
        e_k = subdiag[k].item()

        new_d_k = c * c * d_k + 2.0 * c * s * e_k + s * s * d_k1
        new_d_k1 = s * s * d_k - 2.0 * c * s * e_k + c * c * d_k1
        new_e_k = (c * c - s * s) * e_k + c * s * (d_k1 - d_k)

        diag[k] = new_d_k
        diag[k + 1] = new_d_k1
        subdiag[k] = new_e_k

        # Bulge chase — the rotation creates a bulge at position (k+2, k) that
        # needs to be chased down in the next iteration.
        if k + 1 < hi:
            e_k1_old = subdiag[k + 1].item()
            new_e_k1 = c * e_k1_old
            bulge = s * e_k1_old
            subdiag[k + 1] = new_e_k1
            x = new_e_k
            y = bulge

        # Accumulate the Givens into Q (columns k and k+1).
        q_k = Q[:, k].clone()
        q_k1 = Q[:, k + 1].clone()
        Q[:, k] = c * q_k + s * q_k1
        Q[:, k + 1] = -s * q_k + c * q_k1


def _qr_iterate(
    diag: torch.Tensor,
    subdiag: torch.Tensor,
    tol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Implicit-shift QR with deflation until off-diagonals vanish.

    Returns (eigenvalues, Q_right) where eigenvalues = final diag values
    (unsorted), and Q_right is the product of all Givens rotations.
    """
    n = diag.shape[0]
    Q = torch.eye(n, dtype=diag.dtype, device=diag.device)

    lo = 0
    hi = n - 1
    max_iters = 60 * n
    iters = 0

    while lo < hi and iters < max_iters:
        iters += 1

        # Deflation from the top: collapse lo up if subdiag[lo] is tiny.
        while lo < hi and abs(subdiag[lo].item()) < tol * (
            abs(diag[lo].item()) + abs(diag[lo + 1].item())
        ):
            lo += 1

        if lo >= hi:
            break

        # Deflation from the bottom: peel off the last eigenvalue once it's isolated.
        sub_hi = hi
        while sub_hi > lo and abs(subdiag[sub_hi - 1].item()) < tol * (
            abs(diag[sub_hi - 1].item()) + abs(diag[sub_hi].item())
        ):
            sub_hi -= 1

        if sub_hi == lo:
            hi = lo  # top isolated — done
            break

        # If we peeled from the bottom, run the sweep on the interior block.
        _qr_sweep(diag, subdiag, Q, lo, sub_hi)

        # Check if the last subdiag converged enough to shrink hi on the next pass.
        if sub_hi < hi:
            hi = sub_hi

    return diag, Q


# ----------------------------------------------------------------------
# Eigenvector assembly (apply reflectors to accumulated Givens rotations)
# ----------------------------------------------------------------------


def _apply_reflectors(V_refs: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Left-multiply Q by the product of Householder reflectors in V_refs.

    Each column V_refs[:, k] is a unit-norm Householder vector. The combined
    transform is Q_left = ∏_{k=n-3..0} (I - 2 v_k v_kᵀ), applied to Q as

        Q ← (I - 2 v_k v_kᵀ) Q  for k = n-3, n-4, ..., 0

    (reverse order matches the order in which the reflectors were applied
    during tridiagonalization). Returns the updated Q.
    """
    n = Q.shape[0]
    for k in range(n - 3, -1, -1):
        v = V_refs[:, k]
        if v.norm().item() < 1e-20:
            continue
        # Q ← Q − 2 v (vᵀ Q)
        Q = Q - 2.0 * torch.outer(v, v @ Q)
    return Q


# ----------------------------------------------------------------------
# Public NKI eigh path
# ----------------------------------------------------------------------


def _householder_qr_eigh(
    A: torch.Tensor,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecomposition via Householder tridiagonalization + implicit-shift QR.

    Two stages:
      1. NKI-kernel-driven reduction A → T (symmetric tridiagonal) via
         Householder reflections. Reflectors stored in V_refs.
      2. Pure-host implicit-shift QR iteration on (diag, subdiag) with
         deflation. Accumulates Givens rotations into Q_right.

    Eigenvectors = Q_left · Q_right where Q_left is built by applying the
    stored reflectors to the identity from the right.

    Returns (eigenvalues_sorted_ascending, eigenvectors_columns).
    """
    n = A.shape[0]
    if n == 1:
        return A[0].clone().unsqueeze(0), torch.eye(1, dtype=A.dtype, device=A.device)

    diag, subdiag, V_refs = _householder_tridiag(A)

    eigenvalues, Q_right = _qr_iterate(diag.clone(), subdiag.clone(), tol)
    V = _apply_reflectors(V_refs, Q_right)

    idx = torch.argsort(eigenvalues)
    return eigenvalues[idx], V[:, idx]
