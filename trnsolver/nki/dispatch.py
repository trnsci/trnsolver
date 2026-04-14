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
    import nki
    import nki.isa as nisa  # noqa: F401 — used by kernels if Tensor-Engine refactor lands
    import nki.language as nl

    HAS_NKI = True
except ImportError:
    HAS_NKI = False

# When set, kernel-path failures re-raise instead of silently falling back
# to PyTorch. Used by the validation suite to catch silent kernel breakage.
_REQUIRE_NKI = os.environ.get("TRNSOLVER_REQUIRE_NKI", "").lower() in ("1", "true", "yes")

# When set, dispatch bypasses torch_xla and runs kernels through
# `nki.simulate(kernel)(np_args)` on CPU. Lets us iterate kernels on any
# x86_64 Linux box without paying the NEFF compile + hardware dispatch
# cost. Semantics follow NKI 0.3.0's simulator: no NEFF compile, no
# SBUF/PSUM capacity checks, no latency/parallelism modelling. Correctness
# iteration only; hardware still owns perf numbers.
_USE_SIMULATOR = os.environ.get("TRNSOLVER_USE_SIMULATOR", "").lower() in (
    "1",
    "true",
    "yes",
)

_backend = "auto"

PMAX = 128


def set_backend(backend: str):
    global _backend
    assert backend in ("auto", "pytorch", "nki")
    if backend == "nki" and not HAS_NKI:
        raise RuntimeError("NKI backend requires the nki package (NKI 0.3.0+)")
    _backend = backend


def get_backend() -> str:
    return _backend


def _use_nki() -> bool:
    if _backend == "nki":
        return True
    if _backend == "pytorch":
        return False
    return HAS_NKI


def _use_simulator() -> bool:
    """True iff dispatch should route through `nki.simulate` on CPU.

    Requires both `TRNSOLVER_USE_SIMULATOR=1` in the env and `nki` to be
    importable; otherwise kernels go through torch_xla (or the PyTorch
    fallback if `_use_nki()` is False).
    """
    return _USE_SIMULATOR and HAS_NKI


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

    # ------------------------------------------------------------------
    # Householder-QR building blocks (Phase 1 option B from #38)
    #
    # Two small kernels compose the per-step work of classical
    # tridiagonalization A ← H_k A H_kᵀ with H_k = I − β v vᵀ. The host
    # driver (_householder_tridiag in eigen.py) computes the Householder
    # vector v and the scaled u vector, then invokes:
    #
    #   1. matvec_kernel(A, v)  →  w = A @ v
    #   2. rank2_update_kernel  →  A − u vᵀ − v uᵀ  (symmetric rank-2 update)
    #
    # **Correctness-first cut**: both kernels are implemented via
    # Vector-Engine broadcast + sum (the same idiom that works for
    # rotate_pairs_kernel). The architectural win from option B — rank-1
    # outer products on the Tensor Engine via `nisa.nc_matmul` + FP32
    # PSUM accumulation — is a pure-perf refactor layered on later once
    # the simulator is validating correctness. See #38 and #36.
    #
    # Shape contract: n ≤ PMAX (128) for the single-tile implementation.
    # Blocked variants for n > 128 are Phase 3 work.
    # ------------------------------------------------------------------

    @nki.jit
    def matvec_kernel(A, v):
        """Matrix-vector product w = A @ v.

        Args:
            A : (n, n) FP32 in HBM. n ≤ 128.
            v : (n, 1) FP32 in HBM.

        Returns:
            w : (n, 1) FP32 in HBM.

        Correctness via element-wise: load A (partition=n, free=n), load v
        transposed to (partition=1, free=n), broadcast to (n, n), multiply,
        sum along free dim. Result shape (n, 1).
        """
        n = A.shape[0]

        w = nl.ndarray((n, 1), dtype=A.dtype, buffer=nl.shared_hbm)

        a_tile = nl.load(A[0:n, 0:n])  # (n, n), partition=n
        # Load v transposed so partition=1, free=n; broadcast to match A.
        v_row = nl.load_transpose2d(v[0:n, 0:1])  # (1, n), partition=1
        v_bc = nl.broadcast_to(v_row, (n, n))  # (n, n), partition=n

        prod = nl.multiply(a_tile, v_bc)  # (n, n)
        w_tile = nl.sum(prod, axis=1, keepdims=True)  # (n, 1) — free-dim reduce

        nl.store(w[0:n, 0:1], value=w_tile)

        return w

    @nki.jit
    def rank2_update_kernel(A, u, v):
        """Symmetric rank-2 update: A_new = A − u vᵀ − v uᵀ.

        Core Householder two-sided-transform primitive. u and v are
        expected to be zero above the active row k+1 (padded by the host)
        so entries of A at (i, j) with i ≤ k or j ≤ k are left unchanged.

        Args:
            A : (n, n) FP32 in HBM. n ≤ 128.
            u : (n, 1) FP32 in HBM.
            v : (n, 1) FP32 in HBM.

        Returns:
            A_new : (n, n) FP32 in HBM.

        Implementation: element-wise compute of
            out[i, j] = A[i, j] − u[i]*v[j] − v[i]*u[j]
        via two broadcasts (column u and row vᵀ) and two subtractions.
        """
        n = A.shape[0]

        out = nl.ndarray((n, n), dtype=A.dtype, buffer=nl.shared_hbm)

        a_tile = nl.load(A[0:n, 0:n])  # (n, n)

        u_col = nl.load(u[0:n, 0:1])  # (n, 1), partition=n
        v_col = nl.load(v[0:n, 0:1])  # (n, 1)
        u_row = nl.load_transpose2d(u[0:n, 0:1])  # (1, n), partition=1
        v_row = nl.load_transpose2d(v[0:n, 0:1])  # (1, n)

        u_col_bc = nl.broadcast_to(u_col, (n, n))  # u[i] replicated across cols
        v_col_bc = nl.broadcast_to(v_col, (n, n))
        u_row_bc = nl.broadcast_to(u_row, (n, n))  # u[j] replicated across rows
        v_row_bc = nl.broadcast_to(v_row, (n, n))

        uv = nl.multiply(u_col_bc, v_row_bc)  # u[i] * v[j]
        vu = nl.multiply(v_col_bc, u_row_bc)  # v[i] * u[j]

        out_tile = nl.subtract(a_tile, uv)
        out_tile = nl.subtract(out_tile, vu)

        nl.store(out[0:n, 0:n], value=out_tile)

        return out
