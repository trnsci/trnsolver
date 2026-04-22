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

from .factor import _restore, _to_fp32
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
    A, orig = _to_fp32(A)
    if _use_nki():
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
        B = L @ L^T
        (L^{-1} A L^{-T}) y = λ y
        x = L^{-T} y

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

    L = torch.linalg.cholesky(B)

    if _tb is not None:
        # NKI path: trnblas.trsm dispatches to the blocked NKI GEMM path on
        # Trainium; falls back to torch.linalg.solve_triangular on CPU.
        L_inv_A = _tb.trsm(1.0, L, A, side="left", uplo="lower", trans=False)
        A_prime = _tb.trsm(1.0, L, L_inv_A.T, side="left", uplo="lower", trans=False).T
    else:
        L_inv_A = torch.linalg.solve_triangular(L, A, upper=False)
        A_prime = torch.linalg.solve_triangular(L, L_inv_A.T, upper=False).T

    A_prime = 0.5 * (A_prime + A_prime.T)
    eigenvalues, V_prime = eigh(A_prime, tol)

    if _tb is not None:
        eigenvectors = _tb.trsm(1.0, L, V_prime, side="left", uplo="lower", trans=True)
    else:
        eigenvectors = torch.linalg.solve_triangular(L.T, V_prime, upper=True)

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
        Omega = _tr.standard_normal(n, ell, dtype=sketch_dtype, device=A.device, generator=_gen)
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

    # Project: B = Q^T A Q  (ℓ × ℓ, always FP32 in both paths)
    B = Q.T @ AQ
    B = 0.5 * (B + B.T)  # symmetrize for FP32 safety

    # Small eigenproblem: exact LAPACK eigh on (ℓ × ℓ) — always FP32
    eig_vals, V_small = torch.linalg.eigh(B)  # (ℓ,) ascending, (ℓ, ℓ)

    # Select k eigenvalues of largest |value|. Sort ascending within those k.
    idx = torch.argsort(eig_vals.abs(), descending=True)[:k]
    idx_sorted = idx.sort().values
    eigenvalues_k = eig_vals[idx_sorted]
    V_k = Q @ V_small[:, idx_sorted]  # (n, k) FP32

    return _restore(eigenvalues_k, orig), _restore(V_k, orig)


def _torch_eigh(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch backend path — delegates to torch.linalg.eigh (LAPACK)."""
    return torch.linalg.eigh(A)


# ----------------------------------------------------------------------
# NKI kernel dispatch helpers
# ----------------------------------------------------------------------


def _call_matvec(A, v):
    """Dispatch matvec_kernel through torch_xla or the CPU simulator."""
    from .nki.dispatch import matvec_kernel

    if _use_simulator():
        import nki

        w = nki.simulate(matvec_kernel)(A.detach().cpu().numpy(), v.detach().cpu().numpy())
        return torch.from_numpy(w).to(A.device)
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


def _call_matmul(A, V):
    """Dispatch matmul_kernel (Tensor Engine) or matmul_kernel_sim (Vector Engine).

    Computes C = A @ V.  A: (n, n), V: (n, k), k > 1.

    Simulator path routes through matmul_kernel_sim (nisa.nc_matmul is not
    simulator-safe — nki.simulate strips the stationary operand). Hardware path
    moves tensors to the XLA device, calls matmul_kernel, syncs to materialise
    the result, and returns to orig_device.

    To compute B.T @ C: call _call_matmul(B.T.contiguous(), C). The
    `.contiguous()` is required because load_transpose2d in the kernel expects
    a row-major HBM tensor; a transposed view has wrong strides.

    Falls back to torch.matmul when: NKI is not available (_use_nki() is False),
    n > 128, or k > 128 (single-tile limits). This makes _call_matmul safe for
    callers such as eigh_randomized that call it unconditionally — on non-Neuron
    hardware the NKI guard ensures torch.matmul is used transparently.
    """
    if not _use_nki() or A.shape[0] > 128 or V.shape[1] > 128:
        return torch.matmul(A, V)

    orig_device = A.device

    if _use_simulator():
        import nki

        from .nki.dispatch import matmul_kernel_sim

        C = nki.simulate(matmul_kernel_sim)(A.detach().cpu().numpy(), V.detach().cpu().numpy())
        return torch.from_numpy(C).to(orig_device)

    import torch_neuronx  # noqa: F401
    import torch_xla

    from .nki.dispatch import matmul_kernel

    work_device = torch_xla.device()
    C = matmul_kernel(A.to(work_device), V.to(work_device))
    torch_xla.sync()
    return C.to(orig_device)


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

    import torch_neuronx  # noqa: F401
    import torch_xla

    from .nki.dispatch import matmul_kernel_fp32out

    work_device = torch_xla.device()
    # Cast both operands to BF16: nc_matmul uses the Tensor Engine in BF16 mode
    # and accumulates to FP32 PSUM without any extra cost (Principle 1).
    C = matmul_kernel_fp32out(
        A.to(torch.bfloat16).to(work_device),
        V.to(torch.bfloat16).to(work_device),
    )
    torch_xla.sync()
    return C.to(orig_device)


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

    import torch_neuronx  # noqa: F401
    import torch_xla

    from .nki.dispatch import apply_reflectors_kernel

    work_device = torch_xla.device()
    Q_T_out = apply_reflectors_kernel(V_refs.to(work_device), Q.to(work_device))
    torch_xla.sync()
    return Q_T_out.to(orig_device).T.contiguous()


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
