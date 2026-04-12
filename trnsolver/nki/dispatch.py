"""
NKI dispatch for solver operations.

The Jacobi eigensolver is the primary NKI acceleration target:
each Givens rotation is a 2×2 matmul on the Tensor Engine,
and the off-diagonal max-finding maps to the Vector Engine.
"""

from __future__ import annotations

import os

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    HAS_NKI = True
except ImportError:
    HAS_NKI = False

# When set, kernel-path failures re-raise instead of falling back to PyTorch.
# Used by the validation suite to catch silent kernel breakage during iteration.
_REQUIRE_NKI = os.environ.get("TRNSOLVER_REQUIRE_NKI", "").lower() in ("1", "true", "yes")

_backend = "auto"


def set_backend(backend: str):
    global _backend
    assert backend in ("auto", "pytorch", "nki")
    if backend == "nki" and not HAS_NKI:
        raise RuntimeError("NKI backend requires neuronxcc")
    _backend = backend


def get_backend() -> str:
    return _backend


def _use_nki() -> bool:
    if _backend == "nki":
        return True
    if _backend == "pytorch":
        return False
    return HAS_NKI


if HAS_NKI:

    @nki.jit
    def jacobi_rotation_kernel(
        D_ref, V_ref, p: int, q: int, c: float, s: float, n: int
    ):
        """Apply single Jacobi rotation to D and V matrices.

        D' = G^T @ D @ G  (updates rows/cols p, q of D)
        V' = V @ G        (accumulates eigenvectors)

        G is the Givens rotation matrix with cos(θ)=c, sin(θ)=s
        at positions (p,q).

        On Tensor Engine: the 2-row/col update is a rank-2 operation.
        On Vector Engine: the element-wise updates along rows p, q.

        STUB: Scaffolded for on-hardware validation.
        """
        # Load rows p and q of D
        d_p = nl.load(D_ref[p, :])
        d_q = nl.load(D_ref[q, :])

        # Rotate rows
        new_p = nl.add(nl.multiply(d_p, c), nl.multiply(d_q, -s))
        new_q = nl.add(nl.multiply(d_p, s), nl.multiply(d_q, c))
        nl.store(D_ref[p, :], new_p)
        nl.store(D_ref[q, :], new_q)

        # Rotate columns (D is symmetric, so update cols too)
        d_col_p = nl.load(D_ref[:, p])
        d_col_q = nl.load(D_ref[:, q])
        nl.store(D_ref[:, p], nl.add(nl.multiply(d_col_p, c), nl.multiply(d_col_q, -s)))
        nl.store(D_ref[:, q], nl.add(nl.multiply(d_col_p, s), nl.multiply(d_col_q, c)))

        # Zero the (p,q) and (q,p) entries
        nl.store(D_ref[p, q], 0.0)
        nl.store(D_ref[q, p], 0.0)

        # Accumulate eigenvectors: V' = V @ G
        v_p = nl.load(V_ref[:, p])
        v_q = nl.load(V_ref[:, q])
        nl.store(V_ref[:, p], nl.add(nl.multiply(v_p, c), nl.multiply(v_q, -s)))
        nl.store(V_ref[:, q], nl.add(nl.multiply(v_p, s), nl.multiply(v_q, c)))
