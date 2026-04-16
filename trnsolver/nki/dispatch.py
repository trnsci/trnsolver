"""
NKI dispatch for solver operations.

Phase 3 kernels (option B from #38 — Householder-QR path for symmetric eigh):

    matvec_kernel(A, v)             → w = A @ v  [Tensor Engine via nc_matmul]
    rank2_update_kernel(A, u, v)    → A − u vᵀ − v uᵀ  [Vector Engine]

Together they implement the per-step work of Householder tridiagonalization,
driven by `_householder_tridiag` in `trnsolver/eigen.py`. Eigenvalues +
eigenvectors are finished by pure-host implicit-shift QR on the resulting
tridiagonal (`_householder_qr_eigh`).

matvec_kernel uses nisa.nc_matmul (Tensor Engine, PSUM FP32 accumulation).
A is symmetric throughout tridiagonalization so A^T @ v = A @ v; the
partition dim n is the contracting dim — exactly the case where the Tensor
Engine wins over the Vector Engine.

rank2_update_kernel stays on the Vector Engine: the outer products u @ vᵀ
have contraction rank 1 (the partition dim = 1 path), which gives no
systolic-array advantage. A future optimisation packs [u|v] into a (n, 2)
tile and uses nc_matmul(X^T, Y^T) with partition=2 to compute the full
rank-2 update in one call (tracked in #36).

Design-history note: the Jacobi path (`rotate_pairs_kernel`) that lived
here before 2026-04-14 was replaced after the #9 post-mortem (classical
Jacobi fought the NKI compile cache with per-rotation dispatch). See #38
for the architecture decision and #36 for the underlying directive.
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
    # (Deleted: rotate_pairs_kernel — the Jacobi-era primitive replaced by
    # matvec_kernel + rank2_update_kernel below for the Householder-QR path.
    # See #9 post-mortem and #38 decision.)

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
    # Vector-Engine broadcast + sum (a well-exercised NKI 0.3.0 idiom).
    # The architectural win from option B — rank-1
    # outer products on the Tensor Engine via `nisa.nc_matmul` + FP32
    # PSUM accumulation — is a pure-perf refactor layered on later once
    # the simulator is validating correctness. See #38 and #36.
    #
    # Shape contract: n ≤ PMAX (128) for the single-tile implementation.
    # Blocked variants for n > 128 are Phase 3 work.
    # ------------------------------------------------------------------

    @nki.jit
    def matvec_kernel(A, v):
        """Matrix-vector product w = A @ v via the Tensor Engine.

        nisa.nc_matmul(stationary, moving) computes stationary^T @ moving
        using PSUM FP32 accumulation on the systolic array. The contraction
        dimension is the partition dim of both tiles (n here), which is exactly
        the case where the Tensor Engine wins over the Vector Engine.

        A is symmetric throughout Householder tridiagonalization, so
        A^T @ v = A @ v. We pass A as the stationary operand and v as the
        moving operand; nc_matmul returns A^T @ v = A @ v directly.

        Hardware path only. The NKI 0.3.0 simulator's context wrapper strips
        the stationary operand before forwarding to the underlying nc_matmul
        implementation, causing a TypeError. Use matvec_kernel_sim for the
        nki.simulate path (see _call_matvec in eigen.py).

        Args:
            A : (n, n) FP32 in HBM. n ≤ 128 (single-tile).
            v : (n, 1) FP32 in HBM.

        Returns:
            w : (n, 1) FP32 in HBM.
        """
        n = A.shape[0]

        w = nl.ndarray((n, 1), dtype=A.dtype, buffer=nl.shared_hbm)

        a_tile = nl.load(A[0:n, 0:n])  # (n, n): partition=n, free=n
        v_tile = nl.load(v[0:n, 0:1])  # (n, 1): partition=n, free=1

        # Tensor Engine: A^T @ v = A @ v (A symmetric). PSUM FP32 accumulation.
        w_psum = nisa.nc_matmul(a_tile, v_tile)  # (n, 1)

        nl.store(w[0:n, 0:1], value=w_psum)

        return w

    @nki.jit
    def matvec_kernel_sim(A, v):
        """Matrix-vector product w = A @ v via the Vector Engine.

        Simulator-compatible path used by nki.simulate in _call_matvec.
        The NKI 0.3.0 simulator context wrapper does not support
        nisa.nc_matmul (it strips the stationary argument), so this kernel
        uses the Vector-Engine broadcast+sum idiom that the simulator handles
        correctly.  The result is numerically identical to matvec_kernel.

        Args:
            A : (n, n) FP32 in HBM. n ≤ 128.
            v : (n, 1) FP32 in HBM.

        Returns:
            w : (n, 1) FP32 in HBM.
        """
        n = A.shape[0]

        w = nl.ndarray((n, 1), dtype=A.dtype, buffer=nl.shared_hbm)

        a_tile = nl.load(A[0:n, 0:n])  # (n, n), partition=n
        v_row = nl.load_transpose2d(v[0:n, 0:1])  # (1, n), partition=1
        v_bc = nl.broadcast_to(v_row, (n, n))  # broadcast to match A

        prod = nl.multiply(a_tile, v_bc)  # element-wise A[i,j] * v[j]
        w_tile = nl.sum(prod, axis=1, keepdims=True)  # sum over j → (n, 1)

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
