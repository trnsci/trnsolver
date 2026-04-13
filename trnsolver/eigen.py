"""
Eigenvalue decomposition for Trainium.

Symmetric eigenvalue problem: A @ V = V @ diag(eigenvalues)

Methods:
- NKI path: classical Jacobi sweeps with the `jacobi_rotation_kernel`.
  Targets correctness on trn1/trn2 for n ≤ 512. Per-rotation dispatch;
  batched within-sweep parallelism is a later perf phase (#10).
- PyTorch path: torch.linalg.eigh (LAPACK / MAGMA depending on device).

Primary use case: SCF eigenvalue problem FC = SCε in quantum chemistry.
The generalized eigenproblem reduces to standard form via Cholesky of S.
"""

from __future__ import annotations

import math
import torch
from typing import Optional, Tuple

from .nki import _use_nki, _REQUIRE_NKI, HAS_NKI


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
            # Silent fallback — keeps the public API robust on non-Neuron hosts
            # where HAS_NKI is True but the runtime fails for other reasons.
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
    # Symmetrize for numerical stability
    A_prime = 0.5 * (A_prime + A_prime.T)

    eigenvalues, V_prime = eigh(A_prime, max_sweeps, tol)
    eigenvectors = torch.linalg.solve_triangular(L.T, V_prime, upper=True)
    return eigenvalues, eigenvectors


def _torch_eigh(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch backend path — delegates to torch.linalg.eigh (LAPACK)."""
    return torch.linalg.eigh(A)


def _jacobi_eigh_nki(
    A: torch.Tensor,
    max_sweeps: int,
    tol: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classical Jacobi sweeps via the NKI rotation kernel.

    Computes rotation cosines/sines on the host; dispatches each rotation to
    the `jacobi_rotation_kernel`. The kernel returns new D and V tensors
    (NKI kernels don't mutate inputs). After each rotation we fix up the two
    diagonal entries D[p,p], D[q,q] and the zeroed pivot D[p,q] = D[q,p] = 0
    on the host, since the kernel rotates rows and columns independently and
    the diagonal is the intersection.

    Correctness-only at this stage: per-rotation dispatch overhead dominates,
    batched parallelism is the Phase 3 perf deliverable.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI backend requested but neuronxcc is not available")
    # Import inside the function so module import doesn't fail on non-NKI hosts.
    from .nki.dispatch import jacobi_rotation_kernel

    n = A.shape[0]
    assert A.shape == (n, n), f"Expected square matrix, got {A.shape}"

    D = A.clone()
    V = torch.eye(n, dtype=A.dtype, device=A.device)

    skip_thresh = tol * 0.01

    for sweep in range(max_sweeps):
        # Convergence: sum of squared off-diagonal elements
        off_sq = (D * D).sum().item() - (torch.diagonal(D) ** 2).sum().item()
        if off_sq < tol:
            break

        for p in range(n):
            for q in range(p + 1, n):
                d_pq = D[p, q].item()
                if abs(d_pq) < skip_thresh:
                    continue

                d_pp = D[p, p].item()
                d_qq = D[q, q].item()

                # Rotation that zeroes D[p, q]
                if abs(d_pp - d_qq) < 1e-15:
                    theta = math.pi / 4.0
                    c = math.cos(theta)
                    s = math.sin(theta) if d_pq >= 0 else -math.sin(theta)
                else:
                    tau = (d_qq - d_pp) / (2.0 * d_pq)
                    if tau >= 0:
                        t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                    else:
                        t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                    c = 1.0 / math.sqrt(1.0 + t * t)
                    s = t * c

                # Kernel: rotate rows p,q and cols p,q of D; cols p,q of V
                D, V = jacobi_rotation_kernel(D, V, p, q, c, s)

                # Host-side diagonal + pivot fixup (the kernel rotates rows
                # and columns independently, which double-counts the 2x2 block
                # at (p,q). We replace that block with the correct values.)
                new_pp = c * c * d_pp - 2.0 * c * s * d_pq + s * s * d_qq
                new_qq = s * s * d_pp + 2.0 * c * s * d_pq + c * c * d_qq
                D[p, p] = new_pp
                D[q, q] = new_qq
                D[p, q] = 0.0
                D[q, p] = 0.0

    eigenvalues = torch.diagonal(D).clone()
    idx = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    return eigenvalues, V
