"""
NKI dispatch for solver operations.

Phase 1 kernel: batched-round Jacobi. One NKI call rotates n/2 disjoint
pivot pairs at fixed strided positions (0,1), (2,3), ..., (n-2, n-1).
The host (trnsolver/eigen.py::_jacobi_eigh_nki) permutes D and V into this
strided layout via a Brent-Luk schedule before each call, then un-permutes
after the sweep.

All kernel indices are compile-time constants → the traced graph is stable
across calls → NKI caches after the first compile. This is what makes the
Phase 1 design runnable on trn1/trn2 in the first place; the per-rotation
design hit a recompile-per-pivot wall.

Phase 3 work will reformulate this as a batched 2×2 matmul on the Tensor
Engine (stationary rotation matrix, moving row/column pairs); see #36.
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

# Max partition dimension for NKI 2.24 SBUF tiles.
PMAX = 128


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
    def jacobi_round_kernel(D, V, cs):
        """Apply one Brent-Luk round: n/2 rotations on strided pairs.

        Pairs are fixed at (2i, 2i+1) for i in 0..n/2-1 — this is what makes
        the traced graph stable across calls.

        Args:
            D  : (n, n) symmetric matrix in permuted layout.
            V  : (n, n) eigenvector accumulator.
            cs : (n/2, 2) per-pair (c, s). Values vary per call; shape is stable.

        Returns:
            D_out, V_out — (n, n) tensors with the round's rotations applied.

        Math per pair i with (c, s) = cs[i]:
            Let P_i = diag(1, ..., [c, -s; s, c] at rows (2i, 2i+1), ..., 1).
            D_out = (⊕ P_i)^T · D · (⊕ P_i)
            V_out = V · (⊕ P_i)
        Since each P_i only mixes rows/cols 2i and 2i+1 and different P_i are
        disjoint, we can compute all n/2 in parallel without interaction.

        Vector-Engine-only (nl.multiply + nl.add). Tensor Engine reformulation
        is Phase 3 work per #36.
        """
        n = D.shape[0]

        D_out = nl.ndarray((n, n), dtype=D.dtype, buffer=nl.shared_hbm)
        V_out = nl.ndarray((n, n), dtype=V.dtype, buffer=nl.shared_hbm)

        # Load the whole matrices once; overwrite the affected rows/cols below.
        # For n ≤ 128 this is one partition tile; for larger n NKI will tile
        # automatically since all indices are compile-time constants.
        D_full = nl.load(D[0:n, 0:n])
        V_full = nl.load(V[0:n, 0:n])
        nl.store(D_out[0:n, 0:n], value=D_full)
        nl.store(V_out[0:n, 0:n], value=V_full)

        half = n // 2

        # Load the (c, s) values once — shape (half, 2), partition=half.
        cs_tile = nl.load(cs[0:half, 0:2])

        # For each pair i, rotate D's rows (2i, 2i+1) and cols (2i, 2i+1),
        # and V's cols (2i, 2i+1). All offsets are Python-time constants
        # inside the affine loop, so the graph is stable.
        for i in nl.affine_range(half):
            # Scalars for this pair
            c_tile = nl.load(cs[i:i+1, 0:1])  # (1, 1)
            s_tile = nl.load(cs[i:i+1, 1:2])  # (1, 1)

            # ---- D row rotation (rows 2i and 2i+1) ----
            row_p = nl.load(D[2*i:2*i+1, 0:n])
            row_q = nl.load(D[2*i+1:2*i+2, 0:n])

            new_row_p = nl.add(nl.multiply(row_p, c_tile), nl.multiply(row_q, -s_tile))
            new_row_q = nl.add(nl.multiply(row_p, s_tile), nl.multiply(row_q, c_tile))

            nl.store(D_out[2*i:2*i+1, 0:n], value=new_row_p)
            nl.store(D_out[2*i+1:2*i+2, 0:n], value=new_row_q)

            # ---- D column rotation (cols 2i and 2i+1) ----
            col_p = nl.load(D[0:n, 2*i:2*i+1])
            col_q = nl.load(D[0:n, 2*i+1:2*i+2])

            new_col_p = nl.add(nl.multiply(col_p, c_tile), nl.multiply(col_q, -s_tile))
            new_col_q = nl.add(nl.multiply(col_p, s_tile), nl.multiply(col_q, c_tile))

            nl.store(D_out[0:n, 2*i:2*i+1], value=new_col_p)
            nl.store(D_out[0:n, 2*i+1:2*i+2], value=new_col_q)

            # ---- V column rotation (cols 2i and 2i+1) ----
            v_p = nl.load(V[0:n, 2*i:2*i+1])
            v_q = nl.load(V[0:n, 2*i+1:2*i+2])

            new_v_p = nl.add(nl.multiply(v_p, c_tile), nl.multiply(v_q, -s_tile))
            new_v_q = nl.add(nl.multiply(v_p, s_tile), nl.multiply(v_q, c_tile))

            nl.store(V_out[0:n, 2*i:2*i+1], value=new_v_p)
            nl.store(V_out[0:n, 2*i+1:2*i+2], value=new_v_q)

        # Host-side diagonal fixup for the 2x2 blocks at (2i, 2i+1) is done
        # in eigen.py after the kernel returns — the kernel's row/col rotations
        # double-touch those cells with inconsistent values.

        return D_out, V_out
