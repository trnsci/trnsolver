"""
NKI dispatch for solver operations.

Phase 1 kernel: `rotate_pairs_kernel` does the core Jacobi primitive —
rotate two stacked tiles of shape (half, n) against each other with a
per-row (c, s) pair. Called 3× per sweep-round by the host:

    1. Rotate D's even rows against its odd rows  → D with rotated rows
    2. Rotate D's even cols against its odd cols  → D with rotated cols too
    3. Rotate V's even cols against its odd cols  → V accumulator

Each call is ~5 NKI ops (2 loads, 2 multiplies + 1 add each for 2 outputs,
2 stores). Compile graph is trivial; NKI caches per (half, n, dtype) shape.
Contrast the earlier `affine_range(half)`-with-inline-body design that
unrolled to 12 ops × half iterations and never finished compiling on trn1.

Phase 3 work (#36) will reformulate this as stationary 2×2 matmuls on the
Tensor Engine; the Vector-Engine path here is the correctness MVP.
"""

from __future__ import annotations

import os

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    HAS_NKI = True
except ImportError:
    HAS_NKI = False

_REQUIRE_NKI = os.environ.get("TRNSOLVER_REQUIRE_NKI", "").lower() in ("1", "true", "yes")

_backend = "auto"

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
    def rotate_pairs_kernel(even, odd, c, s):
        """Mix two stacked tiles by a per-row Givens rotation.

        Args:
            even, odd : (half, n) tiles — partition dim = half (≤ PMAX)
            c, s      : (half, 1) tiles — per-pair cosine and sine

        Returns:
            new_even, new_odd : (half, n) tiles
                new_even[i, :] = c[i] * even[i, :] − s[i] * odd[i, :]
                new_odd[i, :]  = s[i] * even[i, :] + c[i] * odd[i, :]

        The (c, s) values broadcast across the free dim (length n).
        """
        half, n = even.shape

        new_even = nl.ndarray((half, n), dtype=even.dtype, buffer=nl.shared_hbm)
        new_odd = nl.ndarray((half, n), dtype=even.dtype, buffer=nl.shared_hbm)

        e = nl.load(even[0:half, 0:n])
        o = nl.load(odd[0:half, 0:n])
        c_tile = nl.load(c[0:half, 0:1])
        s_tile = nl.load(s[0:half, 0:1])
        neg_s_tile = nl.negative(s_tile)

        ne = nl.add(nl.multiply(e, c_tile), nl.multiply(o, neg_s_tile))
        no = nl.add(nl.multiply(e, s_tile), nl.multiply(o, c_tile))

        nl.store(new_even[0:half, 0:n], value=ne)
        nl.store(new_odd[0:half, 0:n], value=no)

        return new_even, new_odd
