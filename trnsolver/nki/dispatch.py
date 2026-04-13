"""
NKI dispatch for solver operations.

Phase 1 target: a correctness-validated Jacobi rotation kernel on trn1/trn2.
The kernel is intentionally simple (single rotation per call, Vector-Engine
element-wise math only) — performance tuning (batched within-sweep
parallelism, Tensor-Engine matmul, cross-core sharding) lands in later
phases per trnsci.dev/roadmap/.
"""

from __future__ import annotations

import os

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
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
    def jacobi_rotation_kernel(D, V, p: int, q: int, c: float, s: float):
        """Apply one Givens rotation at (p, q) with cos=c, sin=s.

        Rotates rows p and q of D; columns p and q of D (D is symmetric and
        stays so); columns p and q of V (eigenvector accumulator).

        Math (per rotation):
            D_new[p, :] = c * D[p, :] - s * D[q, :]
            D_new[q, :] = s * D[p, :] + c * D[q, :]
            (same for cols p, q by symmetry of D)
            V_new[:, p] = c * V[:, p] - s * V[:, q]
            V_new[:, q] = s * V[:, p] + c * V[:, q]
            D_new[p, q] = D_new[q, p] = 0    (by construction of c, s)

        Phase 1 design (correctness MVP):
        - Partition dim = 1 (load one row at a time; n rows sit in free dim)
        - Free dim = n (the matrix width)
        - Vector-Engine only: nl.multiply, nl.add — no nc_matmul yet
        - Full copy of D and V to output HBM, then overwrite the 4 changed
          rows/cols. O(n^2) per rotation; dispatch overhead dominates.
          Batched within-sweep parallelism (#10) is the Phase 3 perf follow-up.

        Target size: n ≤ 512. Larger n needs tiling over the free dim.
        """
        n = D.shape[0]

        D_out = nl.ndarray((n, n), dtype=D.dtype, buffer=nl.shared_hbm)
        V_out = nl.ndarray((n, n), dtype=V.dtype, buffer=nl.shared_hbm)

        # Copy D and V verbatim first; we'll overwrite the affected rows/cols below.
        D_full = nl.load(D[0:n, 0:n])
        V_full = nl.load(V[0:n, 0:n])
        nl.store(D_out[0:n, 0:n], value=D_full)
        nl.store(V_out[0:n, 0:n], value=V_full)

        # ---- D row rotation: D[p,:] and D[q,:] ----
        # Load the two rows as two separate (1, n) tiles.
        row_p = nl.load(D[p:p+1, 0:n])        # shape (1, n), partition=1
        row_q = nl.load(D[q:q+1, 0:n])

        new_row_p = nl.add(nl.multiply(row_p, c), nl.multiply(row_q, -s))
        new_row_q = nl.add(nl.multiply(row_p, s), nl.multiply(row_q, c))

        nl.store(D_out[p:p+1, 0:n], value=new_row_p)
        nl.store(D_out[q:q+1, 0:n], value=new_row_q)

        # ---- D column rotation: D[:,p] and D[:,q] ----
        # D is symmetric, so columns p and q of the *updated* D equal the
        # rotated rows we just computed — but we also need to mix columns
        # using the rows we haven't touched yet. The cleanest way is to
        # rotate the columns from the original D_full, then overwrite.
        col_p = nl.load(D[0:n, p:p+1])        # (n, 1)
        col_q = nl.load(D[0:n, q:q+1])

        new_col_p = nl.add(nl.multiply(col_p, c), nl.multiply(col_q, -s))
        new_col_q = nl.add(nl.multiply(col_p, s), nl.multiply(col_q, c))

        nl.store(D_out[0:n, p:p+1], value=new_col_p)
        nl.store(D_out[0:n, q:q+1], value=new_col_q)

        # Diagonal entries D[p,p], D[q,q] participate in both row and col
        # rotations. The correct post-rotation values are:
        #   D[p,p] = c^2 * D[p,p] - 2*c*s * D[p,q] + s^2 * D[q,q]
        #   D[q,q] = s^2 * D[p,p] + 2*c*s * D[p,q] + c^2 * D[q,q]
        # (and D[p,q] = D[q,p] = 0 by construction of c, s).
        # Compute these on the host (as scalars) and store via small loads.
        # Actually the host already has D[p,p], D[q,q], D[p,q] as Python
        # floats (from eigen.py before dispatch); we'll inject the correct
        # diagonal values via a post-call fixup on the host side rather
        # than inside the kernel. See _jacobi_eigh_nki in eigen.py.

        # ---- V column rotation: V[:,p] and V[:,q] ----
        v_p = nl.load(V[0:n, p:p+1])
        v_q = nl.load(V[0:n, q:q+1])

        new_v_p = nl.add(nl.multiply(v_p, c), nl.multiply(v_q, -s))
        new_v_q = nl.add(nl.multiply(v_p, s), nl.multiply(v_q, c))

        nl.store(V_out[0:n, p:p+1], value=new_v_p)
        nl.store(V_out[0:n, q:q+1], value=new_v_q)

        return D_out, V_out
