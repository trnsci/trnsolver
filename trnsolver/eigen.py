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
import os

import torch

from .factor import _cholesky_lower, _restore, _to_fp32
from .nki import _REQUIRE_NKI, HAS_NKI, _use_nki, _use_simulator


def eigh(
    A: torch.Tensor,
    tol: float = 1e-10,
    *,
    method: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric eigenvalue decomposition: A = V @ diag(w) @ V^T

    Args:
        A: Symmetric matrix (n, n). With ``method="jacobi"``, also a batch
           (..., n, n).
        tol: Convergence tolerance for QR iteration (NKI path only)
        method: ``"auto"`` (default) or ``"jacobi"``. ``None`` reads
           ``TRNSOLVER_EIGH_METHOD`` (default ``auto``).

    Returns:
        eigenvalues: (n,) sorted ascending ((..., n) for a batch)
        eigenvectors: (n, n) columns are eigenvectors ((..., n, n))

    ``method="jacobi"`` (#48) runs batched one-sided Jacobi. On the native
    ``neuron`` device with NKI, n <= 128, it is the SBUF-resident kernel in
    ``trnsolver/nki/jacobi.py`` (one launch per sweep per group of up to
    128 tiles of 128 / n matrices); elsewhere it is the torch implementation in
    ``trnsolver/_jacobi.py`` (batched matmuls, any device). It does not
    change the default ``auto`` dispatch below, except in the measured
    region of ``_JACOBI_AUTO_REGION``: FP32 batches on the native ``neuron``
    device where the Jacobi kernel beats the aten cutover (see
    :func:`_jacobi_auto`; ``TRNSOLVER_EIGH_JACOBI_AUTO`` overrides it).
    ``TRNSOLVER_EIGH_CORES=<k>|auto`` splits the kernel's batch across
    logical NeuronCores (:func:`_jacobi_devices`); unset, it uses one.
    With ``TRNSOLVER_EIGH_POOL=proc`` a host batch is split over persistent
    per-core worker processes instead of threads
    (``trnsolver/nki/eigh_pool.py``).

    The NKI path (Householder tridiagonalization + implicit-shift QR) is
    eager-only: deflation and shift selection are data-dependent Python
    control flow on ``.item()`` scalars. :func:`_householder_qr_eigh` is
    wrapped in ``torch.compiler.disable``, so under ``torch.compile`` it runs
    eagerly behind a single graph break (an error under ``fullgraph=True``)
    rather than splitting into dozens of subgraphs (#41). The PyTorch path
    traces as one ``torch.linalg.eigh`` call.

    On the native ``neuron`` device, inputs with n below
    ``TRNSOLVER_EIGH_NKI_MIN_N`` (default 512) skip the NKI path and run
    ``torch.linalg.eigh`` on the input's device (#42). The NKI path is
    host-driven and launch-bound: on Trn2 it took 0.4 s at n=16 and 21 s at
    n=128, against about 1 ms for aten at every measured size. Setting
    ``TRNSOLVER_REQUIRE_NKI=1`` disables the cutover so the kernels still run.
    """
    resolved = _eigh_method(method)
    if resolved == "jacobi" or (resolved == "auto" and _jacobi_auto(A)):
        return _jacobi_eigh(A)
    A, orig = _to_fp32(A)
    if _use_nki() and _eigh_aten_cutover(A):
        w, V = _aten_eigh_on_device(A)
    elif _use_nki():
        try:
            w, V = _householder_qr_eigh(A, tol)
        except Exception:
            if _REQUIRE_NKI:
                raise
            w, V = _torch_eigh(A)
    else:
        w, V = _torch_eigh(A)
    return _restore(w, orig), _restore(V, orig)


def eigh_generalized(
    A: torch.Tensor,
    B: torch.Tensor,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized symmetric eigenvalue problem: A @ x = λ B @ x

    Reduces to standard form via Cholesky of B:
        B = L @ L^H
        (L^{-1} A L^{-H}) y = λ y
        x = L^{-H} y

    Real symmetric or complex Hermitian; leading batch dims are allowed.

    The triangular solves use trnblas.trsm (NKI-accelerated blocked path)
    when trnblas is installed; they fall back to torch.linalg.solve_triangular
    otherwise. The inner eigh call dispatches to NKI per backend setting.
    Closes #11.
    """
    A, orig = _to_fp32(A)
    B = B.to(A.dtype)

    try:
        import trnblas as _tb
    except ImportError:
        _tb = None

    L = _cholesky_lower(B)

    # trnblas.trsm takes one real (n, n) matrix. Complex (Hermitian) input
    # and batches take torch.linalg.solve_triangular with conjugate
    # transposes of the last two dims (``.mH``): the reduction is
    # L^{-1} A L^{-H}, and ``.T`` neither conjugated nor kept batch dims.
    use_tb = _tb is not None and A.dim() == 2 and not A.is_complex()

    if use_tb:
        # NKI path: trnblas.trsm dispatches to the blocked NKI GEMM path on
        # Trainium; falls back to torch.linalg.solve_triangular on CPU.
        L_inv_A = _tb.trsm(1.0, L, A, side="left", uplo="lower", trans=False)
        A_prime = _tb.trsm(1.0, L, L_inv_A.T, side="left", uplo="lower", trans=False).T
    else:
        L_inv_A = torch.linalg.solve_triangular(L, A, upper=False)
        A_prime = torch.linalg.solve_triangular(L, L_inv_A.mH, upper=False).mH

    A_prime = 0.5 * (A_prime + A_prime.mH)
    eigenvalues, V_prime = eigh(A_prime, tol)

    if use_tb:
        eigenvectors = _tb.trsm(1.0, L, V_prime, side="left", uplo="lower", trans=True)
    else:
        eigenvectors = torch.linalg.solve_triangular(L.mH, V_prime, upper=True)

    return _restore(eigenvalues, orig), _restore(eigenvectors, orig)


def eigh_randomized(
    A: torch.Tensor,
    k: int,
    *,
    oversampling: int = 10,
    n_iter: int = 0,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomized symmetric eigensolver: k dominant eigenvalues/eigenvectors.

    Implements the Nakatsukasa–Tropp sketch-and-Rayleigh-Ritz (sRR) algorithm
    (SIAM SIMAX 45(2), 2024). Returns the k eigenvalues of largest absolute
    magnitude and their corresponding eigenvectors. Complexity O(n²k) vs
    eigh's O(n³), making it efficient when k ≪ n.

    **BF16-native sketch path (v0.14.0, Principle 1):** when A is BF16 and
    the NKI backend is active, the sketch (A @ Ω) and projection (A @ Q) route
    through matmul_kernel_fp32out — nc_matmul (Tensor Engine) with BF16 inputs
    accumulating to FP32 via PSUM for free. A stays in BF16 throughout; only
    the sketch outputs (Y, Q, B) are FP32. This avoids the O(n²) upcast and
    delivers the 50–100× BF16 throughput advantage on the Tensor Engine.
    FP16 and FP32 inputs use the standard _to_fp32/restore path.

    Args:
        A: (n, n) symmetric matrix. BF16 uses the BF16-native sketch path on
           NKI hardware; FP16/FP32 upcast to FP32 via _to_fp32.
        k: Number of eigenvalues/eigenvectors to return. Must satisfy k ≥ 1
           and k + oversampling < n.
        oversampling: Oversampling parameter p; sketch size is ℓ = k + p.
                      Larger values improve accuracy at O(n²p) extra cost.
                      Default 10.
        n_iter: Power-iteration rounds for improved accuracy when the spectral
                gap between eigenvalues k and k+1 is small. Default 0 (pure
                sRR). Each round adds one A@ call with re-orthogonalization.
        seed: RNG seed for the sketch matrix Ω. None = unseeded.

    Returns:
        eigenvalues: (k,) Rayleigh quotients sorted ascending. These approximate
                     the k eigenvalues of A with largest absolute value.
        eigenvectors: (n, k) corresponding eigenvectors (columns), orthonormal.

    Note:
        This algorithm targets eigenvalues of **largest magnitude**, not the
        smallest. For SCF's k occupied orbitals (k smallest eigenvalues of the
        Fock matrix), use ``eigh`` instead. A shift strategy for smallest
        eigenvalues is planned for a future release.

    Raises:
        ValueError: if k < 1 or k + oversampling ≥ n.
    """
    # BF16-native path (Principle 1): keep A in BF16 and accumulate sketch/
    # projection matmuls to FP32 via nc_matmul PSUM. All post-sketch ops
    # (QR, eigh(B)) run in FP32. Triggered when A is BF16 and NKI is active.
    # Falls back to the standard FP32-upcast path otherwise.
    _bf16_native = A.dtype == torch.bfloat16 and _use_nki()

    if _bf16_native:
        orig = torch.bfloat16  # for _restore at return
        # A stays BF16; Omega will be BF16 for the Tensor Engine
    else:
        A, orig = _to_fp32(A)

    n = A.shape[0]
    ell = k + oversampling  # sketch size

    if k < 1:
        raise ValueError(f"k must be ≥ 1, got {k}")
    if ell >= n:
        raise ValueError(f"k + oversampling ({ell}) must be < n ({n}); reduce k or oversampling")

    # Random sketch matrix Ω. BF16-native path: BF16 Omega feeds the Tensor Engine
    # directly. Standard path: FP32 Omega matches upcast A.
    sketch_dtype = torch.bfloat16 if _bf16_native else A.dtype

    try:
        import trnrand as _tr

        _gen = _tr.Generator(seed=seed) if seed is not None else None
        # trnrand.standard_normal(*size, dtype, generator) has no `device`
        # argument: draw the sketch, then move it to A's device.
        Omega = _tr.standard_normal(n, ell, dtype=sketch_dtype, generator=_gen).to(A.device)
    except ImportError:
        _gen = torch.Generator(device=A.device)
        if seed is not None:
            _gen.manual_seed(seed)
        Omega = torch.randn(n, ell, dtype=sketch_dtype, device=A.device, generator=_gen)

    if _bf16_native:
        # BF16 A ⊗ BF16 Ω → FP32  (Tensor Engine + PSUM, Principle 1).
        Y = _call_matmul_fp32out(A, Omega)
        # Power iteration: Q_i is FP32 (from QR); _call_matmul_fp32out casts
        # it to BF16 internally for nc_matmul. Acceptable: we are in the
        # randomized sketch — approximation quality is what matters.
        for _ in range(n_iter):
            Q_i, _ = torch.linalg.qr(Y, mode="reduced")
            Y = _call_matmul_fp32out(A, Q_i)
        Q, _ = torch.linalg.qr(Y, mode="reduced")  # FP32 (n, ell)
        AQ = _call_matmul_fp32out(A, Q)  # BF16 A @ BF16(Q) → FP32
    else:
        # Standard FP32 path.
        Y = _call_matmul(A, Omega)
        for _ in range(n_iter):
            Q_i, _ = torch.linalg.qr(Y, mode="reduced")
            Y = _call_matmul(A, Q_i)
        Q, _ = torch.linalg.qr(Y, mode="reduced")
        AQ = _call_matmul(A, Q)

    # Project: B = Q^H A Q  (ℓ × ℓ, always FP32 in both paths). ``.mH``
    # conjugates complex Hermitian A; for real A it is ``.T``.
    B = Q.mH @ AQ
    B = 0.5 * (B + B.mH)  # symmetrize for FP32 safety

    # Small eigenproblem: exact LAPACK eigh on (ℓ × ℓ) — always FP32
    eig_vals, V_small = torch.linalg.eigh(B)  # (ℓ,) ascending, (ℓ, ℓ)

    # Select k eigenvalues of largest |value|. Sort ascending within those k.
    idx = torch.argsort(eig_vals.abs(), descending=True)[:k]
    idx_sorted = idx.sort().values
    eigenvalues_k = eig_vals[idx_sorted]
    V_k = Q @ V_small[:, idx_sorted]  # (n, k) FP32

    return _restore(eigenvalues_k, orig), _restore(V_k, orig)


_EIGH_METHODS = ("auto", "jacobi")


def _eigh_method(method: str | None) -> str:
    """Resolve ``eigh(method=...)``; ``None`` reads ``TRNSOLVER_EIGH_METHOD``."""
    if method is None:
        method = os.environ.get("TRNSOLVER_EIGH_METHOD", "").strip().lower() or "auto"
    if method not in _EIGH_METHODS:
        raise ValueError(f"eigh method must be one of {_EIGH_METHODS}; got {method!r}")
    return method


# method="auto" -> Jacobi kernel (#48): rows (n_min, n_max, B_min) of the
# measured region where the kernel beats the default aten cutover on the
# native neuron device, FP32. A row matches when n_min <= n <= n_max and the
# batch has at least B_min matrices. Measured on Trn2, one logical
# NeuronCore, end to end from host input to host output, warm medians in ms
# (kernel / aten cutover; full table on #48):
#
#   n \ B     64            256           1024
#   32        9.3 / 4.5     12.1 / 16.0   28.3 / 78.3
#   40        11.0 / 6.5    20.5 / 26.9   73.2 / 99.0
#   64        11.0 / 11.7   22.1 / 47.9   75.2 / 198
#   80        27.1 / 19.3   77.9 / 74.6   291 / 361
#   96        28.1 / 30.3   76.8 / 94.5   301 / 442
#   128       27.1 / 50.5   79.0 / 207    321 / 904
#
# The kernel's cost follows the padded size (32, 64 or 128), aten's follows
# n, so each row starts where the kernel's padding stops costing more than
# it saves. n <= 32 keeps the B >= 512 threshold of #50 (B = 256 wins at
# n = 32 by 1.3x, but smaller n pad less favourably and were not measured).
# Outside these rows the default cutover is unchanged.
_JACOBI_AUTO_REGION: tuple[tuple[int, int, int], ...] = (
    (1, 32, 512),
    (33, 64, 256),
    (65, 128, 1024),
    (96, 128, 256),
    (128, 128, 64),
)


def _jacobi_auto_region() -> tuple[tuple[int, int, int], ...]:
    """The auto region, or ``TRNSOLVER_EIGH_JACOBI_AUTO`` if set.

    The override is ``off`` (never) or comma-separated ``n_min-n_max:B_min``
    rows, e.g. ``1-32:256,33-64:1024``. Unparseable values fall back to the
    table.
    """
    raw = os.environ.get("TRNSOLVER_EIGH_JACOBI_AUTO", "").strip().lower()
    if not raw:
        return _JACOBI_AUTO_REGION
    if raw in ("off", "0", "none", "false"):
        return ()
    rows = []
    try:
        for item in raw.split(","):
            span, b_min = item.split(":")
            n_min, n_max = span.split("-")
            rows.append((int(n_min), int(n_max), int(b_min)))
    except ValueError:
        return _JACOBI_AUTO_REGION
    return tuple(rows)


def _jacobi_auto(A: torch.Tensor) -> bool:
    """True iff ``method="auto"`` should run the Jacobi kernel for A.

    Only FP32 batches on the native ``neuron`` device whose (n, batch) lies
    in `_jacobi_auto_region`, and only where the kernel can run.
    """
    if A.device.type != _NATIVE_DEVICE_TYPE or A.dtype != torch.float32:
        return False
    if A.dim() < 3 or A.shape[-1] != A.shape[-2]:
        return False
    n = A.shape[-1]
    batch = A.numel() // max(n * n, 1)
    region = _jacobi_auto_region()
    if not any(lo <= n <= hi and batch >= b_min for lo, hi, b_min in region):
        return False
    return _jacobi_device_ok(n, A.dtype)


def _jacobi_device_ok(n: int, dtype: torch.dtype) -> bool:
    """True iff the FP32 Jacobi NKI kernel can run: simulator, or the native path."""
    if not (_use_nki() and HAS_NKI and 1 <= n <= 128 and dtype == torch.float32):
        return False
    if _use_simulator():
        return True
    from .nki.dispatch import _get_dispatch_path

    # The kernel uses the NKI 0.7 ISA (TorchNeuron lineage); the XLA lineage
    # (Trn1, NKI 0.3) takes the torch implementation.
    return _get_dispatch_path() == "native"


def _eigh_cores() -> int | str:
    """``TRNSOLVER_EIGH_CORES``: a core count, ``"auto"``, or 1 (the default).

    Unparseable or non-positive values mean 1.
    """
    raw = os.environ.get("TRNSOLVER_EIGH_CORES", "").strip().lower()
    if raw == "auto":
        return "auto"
    try:
        return max(1, int(raw)) if raw else 1
    except ValueError:
        return 1


# TRNSOLVER_EIGH_CORES=auto: use as many cores as keep at least this many
# matrices each, by padded size. Measured on Trn2 (#48), device input, host
# output, 8 host CPUs, warm medians in ms on 1 / 2 / 4 logical cores:
#
#   n \ B   64               256              1024             4096
#   32      9.3 / 9.6 / 16   11 / 14 / 20     27 / 22 / 23     113 / 60 / 42
#   64      11 / 12 / 18     22 / 21 / 22     72 / 46 / 33     276 / 183 / 122
#   128     27 / 23 / 26     79 / 48 / 41     309 / 202 / 127  1219 / 769 / 533
#
# A split costs a few ms of thread and host-staging overhead per core, so
# it pays once each core has a few hundred small (or ~100 n = 128) matrices.
_MULTICORE_MIN_PER_CORE = {32: 512, 64: 256, 128: 128}


def _jacobi_devices(X: torch.Tensor) -> list[torch.device]:
    """NeuronCores for the Jacobi kernel on X: X's own first, then the others.

    One core (X's, or the default one for a host X) unless
    ``TRNSOLVER_EIGH_CORES`` asks for more. ``auto`` uses every visible core
    while each keeps at least ``_MULTICORE_MIN_PER_CORE`` matrices.
    """
    want = _eigh_cores()
    if want == 1 or X.device.type not in (_NATIVE_DEVICE_TYPE, "cpu") or _use_simulator():
        return [X.device]
    import torch_neuronx

    from .nki.jacobi import block_layout

    count = torch_neuronx.device_count()
    if want == "auto":
        ne = block_layout(X.shape[-1])[0]
        per = _MULTICORE_MIN_PER_CORE.get(ne, _MULTICORE_MIN_PER_CORE[32])
        want = max(1, min(count, X.shape[0] // per))
    want = min(want, count)
    if want == 1:
        return [X.device]
    first = X.device.index or 0
    return [torch.device(_NATIVE_DEVICE_TYPE, (first + i) % count) for i in range(want)]


def _jacobi_pool_split(X: torch.Tensor) -> tuple[list[int], int]:
    """(pool core ids, cores to split over) for ``TRNSOLVER_EIGH_POOL=proc``.

    ``([], 1)`` when the pool does not apply. The pool always starts one
    worker per pool core, so a call that splits over fewer reuses them.

    The pool needs host input: a tensor on a NeuronCore means this process
    holds the cores, which the workers then cannot open. Once requested,
    every host call goes through the pool, even one that splits over a
    single core, so this process never opens a core the workers hold. The
    count follows ``TRNSOLVER_EIGH_CORES`` as for the thread split (unset
    means 1), over the pool's cores (``TRNSOLVER_EIGH_POOL_CORES``, else
    ``NEURON_RT_VISIBLE_CORES``).
    """
    from .nki.eigh_pool import pool_cores, pool_requested

    if not pool_requested() or X.device.type != "cpu" or _use_simulator():
        return [], 1
    cores = pool_cores()
    if not cores:
        return [], 1
    want = _eigh_cores()
    if want == "auto":
        from .nki.jacobi import block_layout

        ne = block_layout(X.shape[-1])[0]
        per = _MULTICORE_MIN_PER_CORE.get(ne, _MULTICORE_MIN_PER_CORE[32])
        want = max(1, min(len(cores), X.shape[0] // per))
    return cores, min(want, len(cores))


@torch.compiler.disable
def _jacobi_eigh(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``eigh(method="jacobi")`` for (n, n) or (..., n, n) input (#48)."""
    from . import _jacobi

    A, orig = _to_fp32(A)
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"eigh expects (..., n, n); got {tuple(A.shape)}")
    batch_shape = A.shape[:-2]
    n = A.shape[-1]
    X = A.reshape(-1, n, n)
    if X.shape[0] == 0:
        w = X.new_empty((0, n))
        V = X.new_empty((0, n, n))
    elif _jacobi_device_ok(n, X.dtype):
        try:
            from .nki.jacobi import jacobi_eigh_device, jacobi_eigh_multicore

            pool_cores, parts = _jacobi_pool_split(X)
            devices = [X.device] if pool_cores else _jacobi_devices(X)
            if pool_cores:
                from .nki.eigh_pool import jacobi_eigh_pool

                try:
                    w, V = jacobi_eigh_pool(X, pool_cores, parts)
                except Exception:
                    if _REQUIRE_NKI:
                        raise
                    w, V = jacobi_eigh_device(X)
            elif len(devices) > 1:
                try:
                    w, V = jacobi_eigh_multicore(X, devices)
                except Exception:
                    if _REQUIRE_NKI:
                        raise
                    w, V = jacobi_eigh_device(X)
            else:
                w, V = jacobi_eigh_device(X)
        except Exception:
            if _REQUIRE_NKI:
                raise
            w, V, _ = _jacobi.jacobi_eigh_torch(X)
    else:
        w, V, _ = _jacobi.jacobi_eigh_torch(X)
    w = w.reshape(*batch_shape, n)
    V = V.reshape(*batch_shape, n, n)
    return _restore(w, orig), _restore(V, orig)


def _torch_eigh(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch backend path — delegates to torch.linalg.eigh (LAPACK)."""
    return torch.linalg.eigh(A)


# Native-device cutover (#42). The NKI eigh path costs O(n) host-synchronised
# steps in tridiagonalization plus O(n^2) scalar steps in the QR iteration,
# about 0.4 s at n=16 and 21 s at n=128 on Trn2. aten eigh on the neuron
# device takes 0.4-1.1 ms over the same range. The NKI path is single-tile
# (n <= 128), so any threshold above 128 routes every size it supports to
# aten; 512 leaves room for a blocked kernel to be measured against it.
_EIGH_NKI_MIN_N_DEFAULT = 512
_NATIVE_DEVICE_TYPE = "neuron"


def _eigh_nki_min_n() -> int:
    """Smallest n for which eigh on the ``neuron`` device uses the NKI path.

    Read from ``TRNSOLVER_EIGH_NKI_MIN_N`` on every call (default 512).
    ``0`` disables the cutover. Unparseable values fall back to the default.
    """
    raw = os.environ.get("TRNSOLVER_EIGH_NKI_MIN_N", "").strip()
    if not raw:
        return _EIGH_NKI_MIN_N_DEFAULT
    try:
        return int(raw)
    except ValueError:
        return _EIGH_NKI_MIN_N_DEFAULT


def _eigh_aten_cutover(A: torch.Tensor) -> bool:
    """True iff eigh should use aten instead of the NKI kernels for A.

    Applies to tensors on the native ``neuron`` device, and to host tensors
    when the kernels would run on that device (:func:`_host_input_on_native_path`),
    with n < ``_eigh_nki_min_n()``. ``TRNSOLVER_REQUIRE_NKI=1`` disables it, so
    the hardware suite keeps exercising the kernels.
    """
    if _REQUIRE_NKI:
        return False
    if A.device.type != _NATIVE_DEVICE_TYPE and not _host_input_on_native_path(A):
        return False
    return A.shape[-1] < _eigh_nki_min_n()


def _host_input_on_native_path(A: torch.Tensor) -> bool:
    """True iff A is a host tensor and the NKI kernels would run on TorchNeuron.

    The NKI path copies a host input to the ``neuron`` device and drives the
    same launch-bound tridiagonalization as for a device input, so on a
    Neuron host the default backend turned ``eigh`` of a CPU tensor into
    minutes of kernel launches and compiles (265 s at n = 64 on Trn2, first
    call, against 0.15 ms for LAPACK; #15). Such inputs take the cutover
    too, which for a host tensor is host LAPACK. The simulator keeps the
    kernel path.
    """
    if A.device.type != "cpu" or _use_simulator():
        return False
    from .nki.dispatch import _get_dispatch_path

    return _get_dispatch_path() == "native"


@torch.compiler.disable
def _aten_eigh_on_device(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.linalg.eigh with the result on A's device.

    Eager-only, like the NKI path it replaces, so the traced graph under
    ``torch.compile`` is unchanged (one graph break). If the backend computes
    eigh on the host, the result is moved back to A's device.
    """
    w, V = torch.linalg.eigh(A)
    if w.device != A.device:
        w = w.to(A.device)
    if V.device != A.device:
        V = V.to(A.device)
    return w, V


# ----------------------------------------------------------------------
# NKI kernel dispatch helpers
# ----------------------------------------------------------------------


def _call_matvec(A, v):
    """Dispatch matvec_kernel (native op, torch_xla, or the CPU simulator).

    No sync on the XLA path: the tridiagonalization driver syncs once per
    step. The result is on A's device.
    """
    from .nki.dispatch import _launch, matvec_kernel

    if _use_simulator():
        import nki

        w = nki.simulate(matvec_kernel)(A.detach().cpu().numpy(), v.detach().cpu().numpy())
        return torch.from_numpy(w).to(A.device)
    return _launch(matvec_kernel, "matvec", A, v, sync=False)


def _call_rank2_update(A, u, v):
    """Dispatch rank2_update_kernel (native op, torch_xla, or the CPU simulator).

    No sync on the XLA path (see :func:`_call_matvec`). The result is on A's
    device.
    """
    from .nki.dispatch import _launch, rank2_update_kernel

    if _use_simulator():
        import nki

        out = nki.simulate(rank2_update_kernel)(
            A.detach().cpu().numpy(), u.detach().cpu().numpy(), v.detach().cpu().numpy()
        )
        return torch.from_numpy(out).to(A.device)
    return _launch(rank2_update_kernel, "rank2_update", A, u, v, sync=False)


def _call_matmul(A, V):
    """Dispatch matmul_kernel (Tensor Engine) or matmul_kernel_sim (Vector Engine).

    Computes C = A @ V.  A: (n, n), V: (n, k), k > 1.

    Simulator path routes through matmul_kernel_sim (nisa.nc_matmul is not
    simulator-safe — nki.simulate strips the stationary operand). The hardware
    path goes through ``dispatch._launch``: the ``trnsolver::matmul`` custom
    op on the native ``neuron`` device, or matmul_kernel on the XLA device
    followed by a sync. The result returns to orig_device.

    To compute B.T @ C: call _call_matmul(B.T.contiguous(), C). The
    `.contiguous()` is required because load_transpose2d in the kernel expects
    a row-major HBM tensor; a transposed view has wrong strides.

    Falls back to torch.matmul when: NKI is not available (_use_nki() is False),
    n > 128, or k > 128 (single-tile limits). This makes _call_matmul safe for
    callers such as eigh_randomized that call it unconditionally — on non-Neuron
    hardware the NKI guard ensures torch.matmul is used transparently.
    """
    # The kernels are real; complex operands always take torch.matmul.
    if not _use_nki() or A.is_complex() or V.is_complex() or A.shape[0] > 128 or V.shape[1] > 128:
        return torch.matmul(A, V)

    orig_device = A.device

    if _use_simulator():
        import nki

        from .nki.dispatch import matmul_kernel_sim

        C = nki.simulate(matmul_kernel_sim)(A.detach().cpu().numpy(), V.detach().cpu().numpy())
        return torch.from_numpy(C).to(orig_device)

    from .nki.dispatch import _launch, matmul_kernel

    return _launch(matmul_kernel, "matmul", A, V, device=orig_device)


def _call_matmul_fp32out(A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """C = A @ V with FP32 output. Inputs may be BF16 (BF16-native sketch path).

    On NKI hardware (n ≤ 128, k ≤ 128): casts A and V to BF16, then calls
    matmul_kernel_fp32out (Tensor Engine). nc_matmul accumulates BF16 products
    to FP32 via PSUM for free (Principle 1: PSUM as free FP32 accumulator).
    Output is always FP32 regardless of input dtype.

    On the NKI simulator: upcasts inputs to FP32 for numpy compatibility
    (BF16 is not a numpy dtype), then calls matmul_kernel_fp32out_sim. Same
    FP32 output contract; code path is exercised even though the accumulation
    is exact FP32 rather than BF16→FP32 PSUM.

    Falls back to torch.matmul(A.float(), V.float()) when NKI is unavailable
    or the matrix exceeds single-tile bounds.
    """
    if not _use_nki() or A.shape[0] > 128 or V.shape[1] > 128:
        return torch.matmul(A.float(), V.float())

    orig_device = A.device

    if _use_simulator():
        import nki

        from .nki.dispatch import matmul_kernel_fp32out_sim

        # Upcast to FP32 for numpy: the simulator can't represent BF16 arrays.
        C = nki.simulate(matmul_kernel_fp32out_sim)(
            A.float().detach().cpu().numpy(),
            V.float().detach().cpu().numpy(),
        )
        return torch.from_numpy(C).to(orig_device)

    from .nki.dispatch import _launch, matmul_kernel_fp32out

    # Cast both operands to BF16: nc_matmul uses the Tensor Engine in BF16 mode
    # and accumulates to FP32 PSUM without any extra cost (Principle 1).
    return _launch(
        matmul_kernel_fp32out,
        "matmul_fp32out",
        A.to(torch.bfloat16),
        V.to(torch.bfloat16),
        device=orig_device,
    )


def _call_apply_reflectors(V_refs: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Dispatch apply_reflectors_kernel for fused Householder eigenvector assembly.

    Applies n-2 Householder reflectors (stored in V_refs) to Q in reverse order:

        Q_out = H_{n-3} @ H_{n-4} @ … @ H_0 @ Q

    where Hₖ = I − 2 vₖ vₖᵀ.

    The kernel returns Q^T (SBUF layout trick — see dispatch.py docstring).
    This function transposes the result back before returning.

    Args:
        V_refs : (n, n-2) unit-norm Householder vectors, one per column.
        Q      : (n, n) matrix to left-multiply (Givens accumulation from QR).

    Returns:
        Q_out : (n, n) updated eigenvector matrix Q.
    """
    n_refs = V_refs.shape[1]
    if n_refs == 0:
        return Q  # n ≤ 2: no reflectors to apply

    orig_device = Q.device

    if _use_simulator():
        import nki

        from .nki.dispatch import apply_reflectors_kernel_sim

        Q_T_out = nki.simulate(apply_reflectors_kernel_sim)(
            V_refs.detach().cpu().numpy(), Q.detach().cpu().numpy()
        )
        return torch.from_numpy(Q_T_out).T.contiguous().to(orig_device)

    from .nki.dispatch import _launch, apply_reflectors_kernel

    Q_T_out = _launch(apply_reflectors_kernel, "apply_reflectors", V_refs, Q, device=orig_device)
    return Q_T_out.T.contiguous()


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
    otherwise through ``dispatch._launch`` (native ``neuron`` custom ops or
    the XLA device). The scalar control flow runs on host copies.

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

    # On hardware (HAS_NKI and not _use_simulator) the NKI kernels expect
    # A_work on the Neuron device (native `neuron` or the XLA device). The
    # simulator path takes numpy, so CPU tensors are fine there.
    from .nki.dispatch import _step_sync, _work_device

    orig_device = A.device
    work_device = orig_device if _use_simulator() else _work_device()

    # Only A_work and the two kernels live on the device. The per-step scalar
    # work (norms, signs, dot products, building v and u) runs on host copies:
    # one column read and one kernel-output read per step instead of about six
    # device-side `.item()` syncs and element writes (#40, rule 8 of the
    # native port). The arithmetic is the same; on CPU it is bit-identical.
    host = torch.device("cpu")
    A_work = A.clone().to(work_device)
    V_refs = torch.zeros((n, max(n - 1, 1)), dtype=A.dtype, device=host)

    for k in range(n - 2):
        x = A_work[k + 1 :, k].detach().to(host)
        x_norm = torch.linalg.norm(x).item()
        if x_norm < 1e-20:
            continue

        sign_x0 = torch.sign(x[0]).item() if x[0].item() != 0 else 1.0
        alpha = -sign_x0 * x_norm

        v = torch.zeros(n, dtype=A.dtype, device=host)
        v[k + 1 :] = x
        v[k + 1] -= alpha

        vv = torch.dot(v, v).item()
        if vv < 1e-30:
            continue
        beta = 2.0 / vv

        v_col = v.unsqueeze(1)
        w_col = _call_matvec(A_work, v_col)
        w = w_col.squeeze(1).to(host) * beta

        gamma = beta * torch.dot(v, w).item() / 2.0
        u = w - gamma * v

        u_col = u.unsqueeze(1)
        A_work = _call_rank2_update(A_work, u_col, v_col)

        # Column k below the diagonal and row k right of it become
        # (alpha, 0, ..., 0): two slice writes instead of four element and
        # slice writes, same values.
        edge = torch.zeros(n - k - 1, dtype=A.dtype, device=host)
        edge[0] = alpha
        edge = edge.to(A_work.device)
        A_work[k + 1 :, k] = edge
        A_work[k, k + 1 :] = edge

        v_norm_val = v.norm()
        if v_norm_val.item() > 0:
            V_refs[:, k] = v / v_norm_val

        # Commit all pending XLA ops before the next iteration.
        # A_work is the output of rank2_update_kernel, so it carries a
        # computation history that makes the traced XLA graph unique per
        # step — causing a fresh NEFF compile on every outer-loop call.
        # torch_xla.sync() materialises A_work into a concrete HBM buffer;
        # the next _call_matvec then sees a plain leaf tensor and reuses
        # the cached NEFF. Skipped on the simulator path (no XLA layer) and
        # a no-op on the native path (eager launches).
        if not _use_simulator():
            _step_sync()

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


@torch.compiler.disable
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

    # The QR iteration is scalar host code: about 20 element reads/writes per
    # Givens rotation and O(n^2) rotations per call. Run it on CPU tensors so
    # a device-resident input does not pay one launch or sync per element
    # (#42). CPU inputs already ran here, so their results are unchanged.
    host = torch.device("cpu")
    eigenvalues, Q_right = _qr_iterate(
        diag.detach().to(host, copy=True), subdiag.detach().to(host, copy=True), tol
    )
    Q_right = Q_right.to(A.device)
    V = _call_apply_reflectors(V_refs, Q_right)

    # Subspace rotation refinement (#31): one step of Rayleigh-Ritz on the
    # current approximate eigenbasis V. Compute the n×n projected matrix
    # H = V^T A V, re-diagonalize via LAPACK eigh (exact for this small
    # problem), and rotate V to the eigenbasis of H. The eigh eigenvalues of
    # H are Rayleigh quotients — quadratically convergent from the already-
    # good V — and supersede the raw QR-iteration estimates. The rotation
    # Q_corr also corrects FP32 rounding that accumulated in the Givens-
    # rotation and Householder-application steps, reducing
    # ||A V - V diag(w)||_F by ~1–2 orders of magnitude for n ≥ 64.
    # Cost: one extra n×n GEMM (V^T @ AV, reuses AV) + one n×n LAPACK call.
    AV = _call_matmul(A, V)
    H = _call_matmul(V.T.contiguous(), AV)
    H = 0.5 * (H + H.T)  # symmetrize for FP32 safety
    eigenvalues, Q_corr = torch.linalg.eigh(H)
    V = _call_matmul(V.contiguous(), Q_corr)
    # torch.linalg.eigh returns sorted ascending — no argsort needed.
    return eigenvalues, V
