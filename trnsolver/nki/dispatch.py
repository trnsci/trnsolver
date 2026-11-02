"""
NKI dispatch for solver operations.

Phase 1 kernels (option B from #38 — Householder-QR path for symmetric eigh):

    matvec_kernel(A, v)             → w = A @ v  [Vector Engine]
    rank2_update_kernel(A, u, v)    → A − u vᵀ − v uᵀ  [Vector Engine]

Together they implement the per-step work of Householder tridiagonalization,
driven by `_householder_tridiag` in `trnsolver/eigen.py`. Eigenvalues +
eigenvectors are finished by pure-host implicit-shift QR on the resulting
tridiagonal (`_householder_qr_eigh`).

Both kernels use the Vector Engine broadcast+sum idiom, validated on
hardware (trn1.2xlarge, NKI 0.3.0) in v0.4.0: 14/14 @pytest.mark.neuron
tests pass at rtol=1e-3 for n ∈ {4, 8, 16, 32, 64, 128}.

Phase 3 kernels (closes #36 — Tensor Engine matmul for Rayleigh-Ritz):

    matmul_kernel(A, V)             → C = A @ V  [Tensor Engine, nisa.nc_matmul]
    matmul_kernel_sim(A, V)         → C = A @ V  [Vector Engine, simulator-safe]

Phase 4 kernels (v0.14.0 — BF16-native sketch for eigh_randomized, Principle 1):

    matmul_kernel_fp32out(A, V)     → C = A @ V, output FP32  [Tensor Engine, BF16 inputs]
    matmul_kernel_fp32out_sim(A, V) → C = A @ V, output FP32  [Vector Engine, simulator-safe]

matmul_kernel_fp32out is the key Principle 1 primitive: A and V are BF16 (loaded as
BF16 tiles in SBUF), nc_matmul accumulates BF16→FP32 via PSUM for free, and the
result is stored as FP32. This gives 50–100× BF16 throughput on the Tensor Engine
while keeping downstream QR/eigh ops at FP32 quality — the accumulation is free.

matmul_kernel uses nisa.nc_matmul with A^T stationary in SBUF and V streaming
as the moving operand. The moving operand shape (n, k) with k > 1 satisfies
the NKI 0.3.0 free-dim constraint that blocked the earlier vector matvec
attempt: a matrix moving operand has free_dim = k > 1. Closes #36.

matmul_kernel is NOT simulator-safe: nki.simulate strips the nc_matmul
stationary operand. matmul_kernel_sim (Vector Engine per-column loop) is used
under TRNSOLVER_USE_SIMULATOR=1. Both compute identical results.

Both matmul kernels are wired into the Rayleigh-Ritz refinement step of
`_householder_qr_eigh` (eigen.py) via `_call_matmul`.

Tensor Engine investigation history (#36): nisa.nc_matmul was attempted for
matvec_kernel but rejected by both the simulator and neuronxcc-2.24 with
"nc_matmul() missing value for required argument 'moving'". Root cause:
NKI 0.3.0 requires the moving operand's free dimension to meet a minimum
tile width (>> 1). A vector v of shape (n, 1) has free_dim=1, which is
below that threshold. rank2_update outer products (rank-1, partition=1) also
offer no Tensor Engine advantage. The fix for the Rayleigh-Ritz step is to
batch the k columns of V into a single (n, k) moving operand — k=n in that
context, so free_dim=n >> 1 on any non-trivial matrix.

Phase 2 kernel (v0.12.0 — fused Householder reflector application):

    apply_reflectors_kernel(V_refs, Q_in)  → Q_T_out = (∏ Hₖ @ Q)ᵀ  [Vector Engine]

Applies n-2 Householder reflectors stored in V_refs to Q in reverse order (the
eigenvector assembly step of `_householder_qr_eigh`). Previously a pure-host
Python loop with n-2 PyTorch rank-1 outer-product updates; the fused kernel
does all n-2 steps in a single NKI dispatch with Q^T resident in SBUF.

SBUF layout trick: the kernel maintains Q^T in SBUF (not Q) because the
required reduction `p = v^T Q` reduces along the partition axis (axis=0 of Q),
which NKI 0.3.0 does not support. With Q^T in SBUF, the same reduction becomes
a free-dim sum (axis=1), matching the supported broadcast+sum idiom. The kernel
returns Q^T; `_call_apply_reflectors` transposes the result back to Q before
handing it to `_householder_qr_eigh`.

All ops are Vector Engine only: load_transpose2d + broadcast_to + multiply +
sum(axis=1). The kernel is therefore simulator-safe and has no separate _sim
variant; `apply_reflectors_kernel_sim = apply_reflectors_kernel`.

Design-history note: the Jacobi path (`rotate_pairs_kernel`) that lived
here before 2026-04-14 was replaced after the #9 post-mortem (classical
Jacobi fought the NKI compile cache with per-rotation dispatch). See #38
for the architecture decision and #36 for the matmul resolution.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import warnings

import torch

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


# ── Device dispatch path: native TorchNeuron vs XLA ──────────────────────────
#
# Two Neuron PyTorch lineages coexist:
#   native  torch-neuronx 2.14+ (TorchNeuron, Trn2/Trn3): Trainium is a
#           PyTorch `neuron` device; kernels are custom ops registered via
#           `torch_neuronx.nki_op` (see `_native.py`).
#   xla     torch-neuronx 2.9 lineage (PyTorch/XLA; Trn1 and Inf2 stay
#           here): kernels run on `torch_xla.device()` and are materialised
#           with `torch_xla.sync()`, exactly as before the native path existed.
# Anything else is "cpu": no device path, so a kernel launch outside the
# simulator raises from the torch_xla import, as it did before. Detection
# runs on the first launch, not at import, and the native probe never
# imports torch_xla.

_NEURON_DEVICE = "neuron"
_LAUNCH_LOCK = threading.Lock()
_dispatch_path: str | None = None
_native_ops: dict | None = None
_xla_deprecation_warned = False

_XLA_DEPRECATION_MESSAGE = (
    "the PyTorch/XLA path is deprecated on Trn2/Trn3; use TorchNeuron (native). "
    "Trn1/Inf2 remain supported in maintenance mode."
)


def _native_available() -> bool:
    try:
        import torch_neuronx
    except Exception:
        return False
    # The XLA-lineage torch_neuronx has no `nki_op`.
    if not hasattr(torch_neuronx, "nki_op"):
        return False
    try:
        torch.device(_NEURON_DEVICE)  # torch_neuronx renames PrivateUse1 to "neuron"
        return torch_neuronx.device_count() > 0
    except Exception:
        return False


def _detect_dispatch_path() -> str:
    # TRNSCI_NKI_DISPATCH=auto|native|xla|cpu is shared by every trnsci
    # library, so one variable pins the whole suite on hosts that have both
    # lineages.
    forced = os.environ.get("TRNSCI_NKI_DISPATCH", "auto").strip().lower()
    if forced == "native":
        if not _native_available():
            raise RuntimeError(
                "TRNSCI_NKI_DISPATCH=native, but no TorchNeuron backend with a "
                "'neuron' device is available"
            )
        return "native"
    if forced == "xla":
        if importlib.util.find_spec("torch_xla") is None:
            raise RuntimeError("TRNSCI_NKI_DISPATCH=xla, but torch_xla is not installed")
        return "xla"
    if forced == "cpu":
        return "cpu"
    if forced != "auto":
        raise ValueError(f"TRNSCI_NKI_DISPATCH must be auto, native, xla, or cpu; got {forced!r}")
    if _native_available():
        return "native"
    if importlib.util.find_spec("torch_xla") is not None:
        return "xla"
    return "cpu"


def _maybe_warn_xla_deprecated() -> None:
    """Dormant hook: one DeprecationWarning on the XLA path, opt-in only.

    Off unless ``TRNSCI_XLA_DEPRECATION=1``; the suite turns it on at
    TorchNeuron GA (trnsci/trnsci#35).
    """
    global _xla_deprecation_warned
    if _xla_deprecation_warned:
        return
    if os.environ.get("TRNSCI_XLA_DEPRECATION", "").strip() != "1":
        return
    _xla_deprecation_warned = True
    warnings.warn(_XLA_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)


def _get_dispatch_path() -> str:
    """Return "native", "xla", or "cpu" (cached after the first call)."""
    global _dispatch_path
    if _dispatch_path is None:
        _dispatch_path = _detect_dispatch_path()
    if _dispatch_path == "xla":
        _maybe_warn_xla_deprecated()
    return _dispatch_path


def _get_native_ops() -> dict:
    """Import `_native` (registering the trnsolver:: custom ops) on first use."""
    global _native_ops
    if _native_ops is None:
        from . import _native

        _native_ops = _native.OPS
    return _native_ops


def _work_device() -> torch.device:
    """Device a host driver should keep kernel operands on between launches.

    ``neuron`` on the native path; ``torch_xla.device()`` otherwise (which
    raises ImportError without torch_xla, the pre-native behaviour).
    """
    if _get_dispatch_path() == "native":
        return torch.device(_NEURON_DEVICE)
    import torch_neuronx  # noqa: F401 — registers the Neuron PJRT plugin
    import torch_xla

    return torch_xla.device()


def _step_sync() -> None:
    """Materialise pending device work between host-driven steps.

    XLA path only: ``torch_xla.sync()`` turns a kernel output into a leaf
    buffer so the next step reuses the cached NEFF. Native (eager) launches
    need nothing here.
    """
    if _get_dispatch_path() == "native":
        return
    import torch_xla

    torch_xla.sync()


def _launch(
    kernel,
    op_name: str,
    *tensors: torch.Tensor,
    device: torch.device | None = None,
    sync: bool = True,
    static: tuple = (),
) -> torch.Tensor:
    """Run one kernel launch on the Neuron device.

    ``op_name`` is the ``trnsolver::<op_name>`` custom op used on the native
    path; the XLA path calls ``kernel`` directly on torch_xla tensors and,
    when ``sync`` is set, materialises the result with ``torch_xla.sync()``.
    The result is returned on ``device`` (default: the first input's device).
    ``static`` are trailing Python scalars passed after the tensors (static
    ``nki_op`` arguments on the native path). A value that specialises a
    kernel must arrive this way, not through a global or closure, so the NKI
    trace cache keys on it (aws-neuron/aws-neuron-sdk#1454).
    """
    out_device = tensors[0].device if device is None else device
    if _get_dispatch_path() == "native":
        op = _get_native_ops()[op_name]
        # Launch on the output's NeuronCore when it is one (``neuron:k``, so
        # a multi-core driver can keep each chunk on its own core), else on
        # the current ``neuron`` device.
        run_on = out_device if out_device.type == _NEURON_DEVICE else _NEURON_DEVICE
        args = [t.to(run_on) for t in tensors]
        # NKI kernel tracing (on a dispatch-cache miss) is not thread-safe:
        # concurrent first calls from per-core threads fail inside the
        # tracer. The call itself only enqueues work, so serialising it
        # keeps the cores' execution concurrent.
        with _LAUNCH_LOCK:
            out = op(*args, *static)
        return out.to(out_device)
    import torch_neuronx  # noqa: F401
    import torch_xla

    work_device = torch_xla.device()
    out = kernel(*(t.to(work_device) for t in tensors), *static)
    if sync:
        torch_xla.sync()
    return out.to(out_device)


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
        """Matrix-vector product w = A @ v.

        Args:
            A : (n, n) FP32 in HBM. n ≤ 128.
            v : (n, 1) FP32 in HBM.

        Returns:
            w : (n, 1) FP32 in HBM.

        Vector Engine broadcast+sum: load A (partition=n, free=n), load v
        transposed to (partition=1, free=n), broadcast to (n, n), multiply
        element-wise, sum along the free dim. Result shape (n, 1).

        Tensor Engine note: nisa.nc_matmul is not used here because NKI 0.3.0
        requires the moving operand's free dimension to meet a minimum tile
        width (>> 1). A vector v of shape (n, 1) has free_dim=1, which falls
        below that threshold — the compiler rejects the call on both the
        simulator and hardware. Tensor Engine matvec is revisited in a later
        phase when the SDK documents the minimum free-dim constraint or when
        the algorithm is restructured to batch multiple vectors per call.
        See #36 for the investigation record.
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

    # matvec_kernel_sim is a simulator-compatible alias for nki.simulate dispatch.
    # The NKI 0.3.0 simulator context wrapper cannot route calls through
    # nisa.nc_matmul (it strips the stationary operand), so _call_matvec routes
    # to this function under TRNSOLVER_USE_SIMULATOR=1. Both kernels are
    # identical; the split keeps the dispatch logic explicit.
    matvec_kernel_sim = matvec_kernel

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

    @nki.jit
    def matmul_kernel(A, V):
        """Matrix-matrix product C = A @ V using the Tensor Engine.

        Uses nisa.nc_matmul with A^T stationary in SBUF and V streaming as the
        moving operand. The moving operand shape (n, k) with k > 1 satisfies
        the NKI 0.3.0 free-dim constraint that blocked the earlier vector matvec
        attempt (free_dim=1 for a (n,1) vector). Closes #36.

        Args:
            A : (n, n) FP32 in HBM. n ≤ 128.
            V : (n, k) FP32 in HBM. k > 1, k ≤ 128.

        Returns:
            C : (n, k) FP32 in HBM.

        nc_matmul convention: stationary stored as (K, M) = A^T, moving is
        (K, N) = V, result is (M, N) = A @ V. load_transpose2d(A) gives A^T
        in SBUF.

        NOT simulator-safe: nki.simulate strips the nc_matmul stationary
        operand. Use matmul_kernel_sim via _call_matmul under
        TRNSOLVER_USE_SIMULATOR=1.
        """
        n = A.shape[0]
        k = V.shape[1]

        C = nl.ndarray((n, k), dtype=A.dtype, buffer=nl.shared_hbm)

        A_t = nl.load_transpose2d(A[0:n, 0:n])  # stationary: A^T in SBUF, shape (n,n)
        V_tile = nl.load(V[0:n, 0:k])  # moving: V, shape (n,k)

        psum = nl.zeros((n, k), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=psum, stationary=A_t, moving=V_tile, accumulate=False)

        C_sbuf = nl.copy(psum, dtype=A.dtype)
        nl.store(C[0:n, 0:k], value=C_sbuf)

        return C

    @nki.jit
    def matmul_kernel_sim(A, V):
        """Simulator-compatible matrix-matrix product C = A @ V.

        Uses Vector Engine broadcast+sum in a per-column loop. Same output
        contract as matmul_kernel; used under TRNSOLVER_USE_SIMULATOR=1
        because nki.simulate cannot route nisa.nc_matmul calls (it strips
        the stationary operand).

        Args:
            A : (n, n) FP32 in HBM. n ≤ 128.
            V : (n, k) FP32 in HBM. k ≥ 1, k ≤ 128.

        Returns:
            C : (n, k) FP32 in HBM.
        """
        n = A.shape[0]
        k = V.shape[1]

        C = nl.ndarray((n, k), dtype=A.dtype, buffer=nl.shared_hbm)

        a_tile = nl.load(A[0:n, 0:n])

        for j in nl.affine_range(k):
            v_row = nl.load_transpose2d(V[0:n, j : j + 1])  # (1, n) partition=1, free=n
            v_bc = nl.broadcast_to(v_row, (n, n))
            prod = nl.multiply(a_tile, v_bc)
            c_col = nl.sum(prod, axis=1, keepdims=True)  # (n, 1)
            nl.store(C[0:n, j : j + 1], value=c_col)

        return C

    @nki.jit
    def matmul_kernel_fp32out(A, V):
        """Tensor Engine C = A @ V, output always FP32 (BF16 PSUM principle).

        Designed for the BF16-native sketch path in eigh_randomized (v0.14.0).
        A and V are expected to be BF16; nc_matmul accumulates BF16 products
        to FP32 via PSUM (Principle 1: PSUM as free FP32 accumulator). The
        result is stored as FP32 — no precision is lost to the BF16→BF16
        downcast that matmul_kernel performs.

        Args:
            A : (n, n) BF16 in HBM. n ≤ 128.
            V : (n, k) BF16 in HBM. k > 1, k ≤ 128.

        Returns:
            C : (n, k) **FP32** in HBM.  A @ V with FP32-accumulation quality.

        NOT simulator-safe: nki.simulate strips the nc_matmul stationary
        operand. Use matmul_kernel_fp32out_sim under TRNSOLVER_USE_SIMULATOR=1.
        """
        n = A.shape[0]
        k = V.shape[1]

        C = nl.ndarray((n, k), dtype=nl.float32, buffer=nl.shared_hbm)

        A_t = nl.load_transpose2d(A[0:n, 0:n])  # stationary: A^T (BF16) in SBUF
        V_tile = nl.load(V[0:n, 0:k])  # moving: V (BF16)

        psum = nl.zeros((n, k), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=psum, stationary=A_t, moving=V_tile, accumulate=False)

        # Keep PSUM result as FP32 — do not downcast to BF16.
        C_sbuf = nl.copy(psum, dtype=nl.float32)
        nl.store(C[0:n, 0:k], value=C_sbuf)

        return C

    @nki.jit
    def matmul_kernel_fp32out_sim(A, V):
        """Simulator-compatible C = A @ V, output FP32.

        Vector Engine per-column loop; same output contract as
        matmul_kernel_fp32out but safe under nki.simulate. The simulator
        receives FP32-upcast numpy arrays (BF16 is not a numpy dtype), so
        the accumulation is exact FP32 rather than BF16→FP32 PSUM.

        Args:
            A : (n, n) FP32 in HBM (upcast from BF16 by _call_matmul_fp32out).
            V : (n, k) FP32 in HBM.

        Returns:
            C : (n, k) FP32 in HBM.
        """
        n = A.shape[0]
        k = V.shape[1]

        C = nl.ndarray((n, k), dtype=nl.float32, buffer=nl.shared_hbm)

        a_tile = nl.load(A[0:n, 0:n])

        for j in nl.affine_range(k):
            v_row = nl.load_transpose2d(V[0:n, j : j + 1])  # (1, n)
            v_bc = nl.broadcast_to(v_row, (n, n))
            prod = nl.multiply(a_tile, v_bc)
            c_col = nl.sum(prod, axis=1, keepdims=True)  # (n, 1) FP32
            nl.store(C[0:n, j : j + 1], value=c_col)

        return C

    @nki.jit
    def apply_reflectors_kernel(V_refs, Q_in):
        """Fused Householder reflector application: Q_out = (∏ₖ Hₖ) @ Q_in.

        Applies the n-2 Householder reflectors stored in V_refs to Q_in in
        reverse order (k = n-3, n-4, …, 0), implementing the eigenvector
        assembly step of _householder_qr_eigh without host round-trips.

        Args:
            V_refs : (n, n-2) FP32 in HBM. Column k is the unit-norm
                     Householder vector vₖ used at step k of tridiagonalization.
            Q_in   : (n, n) FP32 in HBM. Accumulated Givens-rotation matrix.

        Returns:
            Q_T_out : (n, n) FP32 in HBM. The TRANSPOSE of the updated Q.
                      Caller transposes back with .T.contiguous().

        SBUF layout:  Q^T is maintained in SBUF (partition=n, free=n) so that
        the key reduction `p[j] = Σᵢ vₖ[i]·Q[i,j]` reduces along the free
        axis (axis=1, supported), not the partition axis (axis=0, unsupported
        in NKI 0.3.0). Each Householder vector vₖ is loaded per step with
        load_transpose2d(V_refs[:, k:k+1]) so it lands as a (1, n) row at
        partition 0 (see #40 for why it is not sliced out of an SBUF V^T).

        All operations are Vector Engine (load_transpose2d, broadcast_to,
        multiply, sum axis=1). The kernel is therefore simulator-safe; no
        separate _sim variant is needed.

        Shape constraint: n ≤ PMAX (128). SBUF usage: Q_T (n,n) ≈ 65 KB at
        n=128 plus one (1, n) row per step, well within SBUF per NeuronCore.
        """
        n = Q_in.shape[0]
        n_refs = V_refs.shape[1]  # = n - 2

        Q_T_out = nl.ndarray((n, n), dtype=Q_in.dtype, buffer=nl.shared_hbm)

        # Load Q transposed: Q_T[j, i] = Q[i, j].
        # Free-dim (axis=1) reduction of Q_T gives p[j] = Σᵢ v[i]*Q[i,j]. ✓
        Q_T = nl.load_transpose2d(Q_in[0:n, 0:n])  # (n, n)

        # Apply reflectors in reverse order: Hₙ₋₃ · Hₙ₋₄ · … · H₀ @ Q_in.
        # k = n_refs-1 down to 0.  nl.affine_range unrolls compile-time.
        for k_rev in nl.affine_range(n_refs):
            k = n_refs - 1 - k_rev

            # v_k as row vector (1, n) at partition 0: v_row[0, i] = vₖ[i].
            # Loaded straight from HBM rather than sliced out of an SBUF
            # V^T (V_T[k:k+1, :]): NKI 0.7.0 only allows a 1-partition
            # access starting at partition 0/32/64/96, and k is arbitrary
            # (#40). Same values, so the update is bit-identical.
            v_row = nl.load_transpose2d(V_refs[0:n, k : k + 1])  # (1, n)

            # Broadcast v_k across partition dim: v_bc[j, i] = vₖ[i]
            v_bc = nl.broadcast_to(v_row, (n, n))

            # p[j] = Σᵢ vₖ[i] · Q[i,j] = Σᵢ vₖ[i] · Q_T[j,i]
            prod = nl.multiply(Q_T, v_bc)  # prod[j,i] = Q_T[j,i] · vₖ[i]
            p_col = nl.sum(prod, axis=1, keepdims=True)  # (n,1): p[j]

            # outer[j,i] = vₖ[i] · p[j]  →  Q_T ← Q_T − 2·outer
            p_bc = nl.broadcast_to(p_col, (n, n))  # p_bc[j,i] = p[j]
            outer = nl.multiply(v_bc, p_bc)  # (n,n): vₖ[i]·p[j]
            outer_2 = nl.add(outer, outer)  # 2·outer
            Q_T = nl.subtract(Q_T, outer_2)

        nl.store(Q_T_out[0:n, 0:n], value=Q_T)
        return Q_T_out

    # apply_reflectors_kernel_sim: all ops are Vector Engine, so the hardware
    # kernel is already simulator-safe. The alias keeps the dispatch logic
    # symmetric with the matmul pair.
    apply_reflectors_kernel_sim = apply_reflectors_kernel
