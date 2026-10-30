"""
Batched small ``eigh`` on device: many n <= 128 problems per launch (#48).

A single n = 128 ``eigh`` on the all-device path is a latency benchmark:
about 7.9 ms on Trn2, almost all of it fixed cost (launches, copies, host
LAPACK) rather than compute. Calling it once per matrix pays that cost B
times. This module runs B matrices through the same three device stages
with a fixed, small number of launches:

    stage                       kernel                              launches
    Householder reduction       `householder_tridiag_batched_kernel`   1 / chunk
    host tridiagonal solve      one pass over the batch (eigen.py)      0
    reflector application       `apply_reflectors_batched_kernel`      1 / chunk
    Rayleigh-Ritz H = V^T A V   `rr_project_batched_kernel`            1 / chunk
    host eigh(H)                one batched torch.linalg.eigh           0
    Rayleigh-Ritz V <- V Q      `rr_rotate_batched_kernel`             1 / chunk

so a batch of B matrices costs 4 * ceil(B / C) launches for chunk size C,
against 5 * B for B single calls (tridiag, apply, three matmuls).

Design: fixed-size chunks (option B of #48)
-------------------------------------------
Every kernel takes a leading chunk dimension that is a static shape, and
the Python loops over the chunk are unrolled at trace time. Within each
Householder step the chunk's matrices are processed in lockstep: one step
for matrix 0, then the same step for matrix 1, and so on. The per-step
work is a long chain of small dependent Vector/Scalar/Tensor Engine
instructions; C independent chains in one instruction stream let the
scheduler overlap them across engines, which a loop of C single launches
cannot. The step masks (``[j > k]``, the one-hot selectors) are shared by
the whole chunk.

The chunk size is a power of two, at most ``TRNSOLVER_EIGH_BATCH_CHUNK``
(default 16). The batch is split into full chunks plus one tail chunk
rounded up to the next power of two and padded with identity matrices
(whose reduction is a no-op), so each n needs at most
log2(C) + 1 compiled kernels whatever the batch sizes a caller uses.
Keeping the whole batch in one launch (option A) would need one NEFF per
distinct B, and an unrolled kernel whose size grows with B: at n = 128 one
matrix is already about 126 * 35 traced instructions.

NKI 0.7 rules, as for `tridiag.py` (checked by the MLIR verifier in
``tests/test_eigh_batched.py``; the CPU simulator does not check them):

* every SBUF tile spans all n partitions from partition 0, so partition
  counts match and there are no single-partition copies;
* (P, 1) arithmetic is float-only and divide-free (``rsqrt`` refined by
  Newton steps; there is no divide anywhere);
* the batch index only appears in HBM access patterns (``A[g, 0:n, 0:n]``),
  never in SBUF partition offsets.
"""

from __future__ import annotations

import os

import torch

from .dispatch import HAS_NKI, PMAX, _use_simulator
from .tridiag import _FLOOR, _SQRT_HALF, n_out_cols, tridiag_constants

_CHUNK_ENV = "TRNSOLVER_EIGH_BATCH_CHUNK"
_CHUNK_DEFAULT = 16


def batch_chunk_max() -> int:
    """Largest chunk (matrices per launch), a power of two >= 1.

    Read from ``TRNSOLVER_EIGH_BATCH_CHUNK`` on every call (default 16).
    Values are rounded down to a power of two; unparseable or non-positive
    values fall back to the default.
    """
    raw = os.environ.get(_CHUNK_ENV, "").strip()
    try:
        c = int(raw) if raw else _CHUNK_DEFAULT
    except ValueError:
        c = _CHUNK_DEFAULT
    if c < 1:
        c = _CHUNK_DEFAULT
    return 1 << (c.bit_length() - 1)


def batch_chunks(batch: int, cmax: int | None = None) -> list[tuple[int, int, int]]:
    """Split ``batch`` matrices into launches: ``[(start, count, padded), ...]``.

    Full chunks of ``cmax``, then one tail rounded up to a power of two.
    ``padded`` is the static chunk size the kernel is compiled for.
    """
    if cmax is None:
        cmax = batch_chunk_max()
    out = []
    start = 0
    while start < batch:
        count = min(cmax, batch - start)
        padded = 1 << (count - 1).bit_length()
        out.append((start, count, padded))
        start += count
    return out


def n_launches(batch: int, cmax: int | None = None) -> int:
    """Kernel launches for one batched device eigh of ``batch`` matrices."""
    return 4 * len(batch_chunks(batch, cmax))


if HAS_NKI:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def householder_tridiag_batched_kernel(A, consts):
        """Householder tridiagonalization of a chunk of symmetric matrices.

        Same per-matrix algorithm, scaling and output layout as
        `tridiag.householder_tridiag_kernel`; see that module's docstring.

        Args:
            A      : (C, n, n) FP32 symmetric, in HBM. 2 <= n <= 128.
            consts : (n, 2n) FP32 from `tridiag_constants` (identity, J).

        Returns:
            out : (C, n, 2 + max(n - 1, 1)) FP32 in HBM, per matrix: column 0
                  = diag, column 1 = subdiag (last entry 0), columns 2.. =
                  unit-norm Householder vectors, zero-padded.
        """
        nb = A.shape[0]
        n = A.shape[1]
        ncol = max(n - 1, 1)
        f32 = nl.float32

        out = nl.ndarray((nb, n, 2 + ncol), dtype=A.dtype, buffer=nl.shared_hbm)

        ident = nl.load(consts[0:n, 0:n])
        jidx = nl.load(consts[0:n, n : 2 * n])
        ones = nl.add(nl.multiply(ident, 0.0), 1.0)
        pidx = nl.sum(nl.multiply(ident, jidx), axis=1, keepdims=True)
        zeros_col = nl.multiply(pidx, 0.0)

        def rsqrt_refined(x):
            y = nl.rsqrt(x)
            for _ in range(2):
                y = nl.multiply(
                    y, nl.add(nl.multiply(nl.multiply(nl.multiply(y, y), x), -0.5), 1.5)
                )
            return y

        def row_broadcast(col):
            diag_t = nl.multiply(ident, nl.broadcast_to(col, (n, n)))
            ps = nl.zeros((n, n), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps, stationary=ones, moving=diag_t, accumulate=False)
            return nl.copy(ps, dtype=f32)

        npad = ncol - (n - 2)
        a_s = []
        gmax_s = []
        sub_s = []
        for g in range(nb):
            a = nl.load(A[g, 0:n, 0:n])
            rowmax = nl.max(nl.abs(a), axis=1, keepdims=True)
            gmax = nl.maximum(nl.max(row_broadcast(rowmax), axis=1, keepdims=True), _FLOOR)
            ginv = rsqrt_refined(gmax)
            ginv = nl.multiply(ginv, ginv)
            a_s.append(nl.multiply(a, nl.broadcast_to(ginv, (n, n))))
            gmax_s.append(gmax)
            sub_s.append(zeros_col)
            vzero = nl.multiply(nl.broadcast_to(zeros_col, (n, npad)), 1.0)
            nl.store(out[g, 0:n, 2 + n - 2 : 2 + ncol], value=vzero)

        for k in range(n - 2):
            # Step masks, shared by every matrix in the chunk.
            ek_col = nl.equal(pidx, float(k), dtype=f32)
            sel = nl.broadcast_to(ek_col, (n, n))
            mrow = nl.greater(jidx, float(k), dtype=f32)
            erow = nl.equal(jidx, float(k + 1), dtype=f32)
            mcol = nl.greater(pidx, float(k), dtype=f32)

            for g in range(nb):
                a = a_s[g]
                ps_r = nl.zeros((n, n), dtype=f32, buffer=nl.psum)
                nisa.nc_matmul(dst=ps_r, stationary=sel, moving=a, accumulate=False)
                r = nl.copy(ps_r, dtype=f32)

                xr = nl.multiply(r, mrow)
                sigma = nl.sum(nl.multiply(xr, xr), axis=1, keepdims=True)
                x0 = nl.sum(nl.multiply(r, erow), axis=1, keepdims=True)
                norm = nl.multiply(sigma, rsqrt_refined(nl.maximum(sigma, _FLOOR)))

                sgn = nl.add(nl.multiply(nl.greater_equal(x0, 0.0, dtype=f32), 2.0), -1.0)
                snorm = nl.multiply(sgn, norm)

                half_vv = nl.add(
                    nl.multiply(nl.add(sigma, nl.multiply(norm, norm)), 0.5),
                    nl.multiply(nl.abs(x0), norm),
                )
                y = rsqrt_refined(nl.maximum(half_vv, _FLOOR))
                beta = nl.multiply(y, y)

                vr = nl.add(xr, nl.multiply(erow, nl.broadcast_to(snorm, (n, n))))
                vc = nl.sum(nl.multiply(vr, ident), axis=1, keepdims=True)

                wc = nl.multiply(nl.sum(nl.multiply(a, vr), axis=1, keepdims=True), beta)
                wr = row_broadcast(wc)

                vw = nl.sum(nl.multiply(vr, wr), axis=1, keepdims=True)
                gam = nl.multiply(nl.multiply(beta, vw), 0.5)

                ur = nl.multiply(
                    nl.subtract(wr, nl.multiply(vr, nl.broadcast_to(gam, (n, n)))), mrow
                )
                uc = nl.multiply(nl.subtract(wc, nl.multiply(vc, gam)), mcol)

                uv = nl.multiply(nl.broadcast_to(uc, (n, n)), vr)
                vu = nl.multiply(nl.broadcast_to(vc, (n, n)), ur)
                a_s[g] = nl.subtract(nl.subtract(a, uv), vu)

                vunit = nl.multiply(vc, nl.multiply(y, _SQRT_HALF))
                nl.store(out[g, 0:n, 2 + k : 3 + k], value=vunit)

                sub_s[g] = nl.subtract(sub_s[g], nl.multiply(ek_col, snorm))

        sup_mask = nl.equal(jidx, nl.broadcast_to(nl.add(pidx, 1.0), (n, n)), dtype=f32)
        e_last = nl.equal(pidx, float(n - 2), dtype=f32)
        for g in range(nb):
            a = a_s[g]
            diag = nl.sum(nl.multiply(a, ident), axis=1, keepdims=True)
            sup = nl.sum(nl.multiply(a, sup_mask), axis=1, keepdims=True)
            sub = nl.add(sub_s[g], nl.multiply(sup, e_last))
            nl.store(out[g, 0:n, 0:1], value=nl.multiply(diag, gmax_s[g]))
            nl.store(out[g, 0:n, 1:2], value=nl.multiply(sub, gmax_s[g]))
        return out

    @nki.jit
    def apply_reflectors_batched_kernel(V_refs, Q_in):
        """Per matrix: Q_T_out = ((H_0 H_1 ... H_{m-1}) Q_in)^T, H_k = I - 2 v_k v_k^T.

        Batched form of `dispatch.apply_reflectors_kernel` with the same
        Q^T-in-SBUF layout, reflectors applied in reverse order, matrices in
        lockstep per reflector.

        Args:
            V_refs : (C, n, m) FP32 in HBM, m >= 1 unit-norm reflectors per matrix.
            Q_in   : (C, n, n) FP32 in HBM.

        Returns:
            Q_T_out : (C, n, n) FP32 in HBM, the TRANSPOSE of the updated Q.
        """
        nb = Q_in.shape[0]
        n = Q_in.shape[1]
        n_refs = V_refs.shape[2]

        Q_T_out = nl.ndarray((nb, n, n), dtype=Q_in.dtype, buffer=nl.shared_hbm)

        qt_s = [nl.load_transpose2d(Q_in[g, 0:n, 0:n]) for g in range(nb)]
        for k_rev in range(n_refs):
            k = n_refs - 1 - k_rev
            for g in range(nb):
                # (1, n) row at partition 0, loaded from HBM (see #40).
                v_row = nl.load_transpose2d(V_refs[g, 0:n, k : k + 1])
                v_bc = nl.broadcast_to(v_row, (n, n))
                p_col = nl.sum(nl.multiply(qt_s[g], v_bc), axis=1, keepdims=True)
                outer = nl.multiply(v_bc, nl.broadcast_to(p_col, (n, n)))
                qt_s[g] = nl.subtract(qt_s[g], nl.add(outer, outer))
        for g in range(nb):
            nl.store(Q_T_out[g, 0:n, 0:n], value=qt_s[g])
        return Q_T_out

    @nki.jit
    def rr_project_batched_kernel(A, V_T):
        """Per matrix: H = V^T A V on the Tensor Engine (Rayleigh-Ritz projection).

        Takes V^T, which is what `apply_reflectors_batched_kernel` returns, so
        no host transpose is needed. A V uses stationary A^T (as
        `dispatch.matmul_kernel` does); V^T (A V) uses stationary V.

        Args:
            A   : (C, n, n) FP32 in HBM.
            V_T : (C, n, n) FP32 in HBM, the transposed eigenvector estimate.

        Returns:
            H : (C, n, n) FP32 in HBM.
        """
        nb = A.shape[0]
        n = A.shape[1]
        f32 = nl.float32
        H = nl.ndarray((nb, n, n), dtype=A.dtype, buffer=nl.shared_hbm)
        for g in range(nb):
            a_t = nl.load_transpose2d(A[g, 0:n, 0:n])
            v = nl.load_transpose2d(V_T[g, 0:n, 0:n])
            ps_av = nl.zeros((n, n), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps_av, stationary=a_t, moving=v, accumulate=False)
            av = nl.copy(ps_av, dtype=f32)
            ps_h = nl.zeros((n, n), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps_h, stationary=v, moving=av, accumulate=False)
            nl.store(H[g, 0:n, 0:n], value=nl.copy(ps_h, dtype=f32))
        return H

    @nki.jit
    def rr_rotate_batched_kernel(V_T, Q):
        """Per matrix: V Q on the Tensor Engine (stationary V^T, moving Q).

        Args:
            V_T : (C, n, n) FP32 in HBM.
            Q   : (C, n, n) FP32 in HBM.

        Returns:
            VQ : (C, n, n) FP32 in HBM.
        """
        nb = Q.shape[0]
        n = Q.shape[1]
        f32 = nl.float32
        VQ = nl.ndarray((nb, n, n), dtype=Q.dtype, buffer=nl.shared_hbm)
        for g in range(nb):
            v_t = nl.load(V_T[g, 0:n, 0:n])
            q = nl.load(Q[g, 0:n, 0:n])
            ps = nl.zeros((n, n), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps, stationary=v_t, moving=q, accumulate=False)
            nl.store(VQ[g, 0:n, 0:n], value=nl.copy(ps, dtype=f32))
        return VQ


# ----------------------------------------------------------------------
# Host launch helpers
# ----------------------------------------------------------------------

_KERNELS = {
    "householder_tridiag_batched": "householder_tridiag_batched_kernel",
    "apply_reflectors_batched": "apply_reflectors_batched_kernel",
    "rr_project_batched": "rr_project_batched_kernel",
    "rr_rotate_batched": "rr_rotate_batched_kernel",
}


def _run(op_name: str, *tensors: torch.Tensor, device) -> torch.Tensor:
    """One launch of a batched kernel: simulator, native op, or XLA."""
    import sys

    kernel = getattr(sys.modules[__name__], _KERNELS[op_name])
    if _use_simulator():
        import nki as _nki

        out = _nki.simulate(kernel)(*(t.detach().cpu().contiguous().numpy() for t in tensors))
        return torch.from_numpy(out).to(device)
    from .dispatch import _launch

    return _launch(kernel, op_name, *(t.detach().contiguous() for t in tensors), device=device)


def _pad(X: torch.Tensor, padded: int, eye: bool) -> torch.Tensor:
    """Pad a (count, n, m) chunk to (padded, n, m) with identity or zero matrices."""
    count = X.shape[0]
    if count == padded:
        return X
    fill = torch.zeros((padded - count,) + tuple(X.shape[1:]), dtype=X.dtype, device=X.device)
    if eye:
        fill = fill + torch.eye(X.shape[1], X.shape[2], dtype=X.dtype, device=X.device)
    return torch.cat([X, fill], dim=0)


def _chunked(op_name: str, inputs, eyes, extra=(), device=None, cmax=None) -> torch.Tensor:
    """Run ``op_name`` over the batch in chunks; returns the concatenated output.

    ``inputs`` are (B, ...) tensors padded per chunk (identity where the
    matching ``eyes`` entry is True, else zeros); ``extra`` are passed to
    every launch unchanged.
    """
    batch = inputs[0].shape[0]
    outs = []
    for start, count, padded in batch_chunks(batch, cmax):
        args = [
            _pad(X[start : start + count], padded, e) for X, e in zip(inputs, eyes, strict=True)
        ]
        out = _run(op_name, *args, *extra, device=device)
        outs.append(out[:count])
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)


def householder_tridiag_batched_device(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched `tridiag.householder_tridiag_device`.

    Args:
        A : (B, n, n) FP32 symmetric, 2 <= n <= 128.

    Returns:
        (diag (B, n), subdiag (B, n - 1), V_refs (B, n, n - 2)) on A's
        device. V_refs drops the zero padding column the single-matrix
        kernel carries, so n = 2 gives an empty (B, 2, 0).
    """
    if not HAS_NKI:
        raise RuntimeError("NKI backend requested but the nki package isn't importable")
    batch, n = A.shape[0], A.shape[-1]
    assert A.shape == (batch, n, n), f"Expected (B, n, n), got {tuple(A.shape)}"
    if not 2 <= n <= PMAX:
        raise NotImplementedError(f"device tridiagonalization requires 2 <= n <= 128; got n={n}")

    if _use_simulator():
        consts = tridiag_constants(n)
    else:
        from .dispatch import _work_device
        from .tridiag import _consts_on

        consts = _consts_on(n, _work_device())
    packed = _chunked("householder_tridiag_batched", [A], [True], extra=(consts,), device=A.device)
    assert packed.shape == (batch, n, n_out_cols(n))
    diag = packed[:, :, 0].contiguous()
    subdiag = packed[:, : n - 1, 1].contiguous()
    V_refs = packed[:, :, 2 : 2 + n - 2].contiguous()
    return diag, subdiag, V_refs


def apply_reflectors_batched(V_refs: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Batched `eigen._call_apply_reflectors`, but returns V^T, not V.

    V_refs (B, n, m), Q (B, n, n) -> (B, n, n) = ((prod H_k) Q)^T. The
    transposed result feeds `rr_project_batched` / `rr_rotate_batched`
    directly. With m = 0 (n = 2) it returns Q^T without a launch.
    """
    if V_refs.shape[2] == 0:
        return Q.transpose(1, 2).contiguous()
    return _chunked("apply_reflectors_batched", [V_refs, Q], [False, True], device=Q.device)


def rr_project_batched(A: torch.Tensor, V_T: torch.Tensor) -> torch.Tensor:
    """H = V^T A V per matrix, from A (B, n, n) and V^T (B, n, n)."""
    return _chunked("rr_project_batched", [A, V_T], [True, True], device=A.device)


def rr_rotate_batched(V_T: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """V Q per matrix, from V^T (B, n, n) and Q (B, n, n)."""
    return _chunked("rr_rotate_batched", [V_T, Q], [True, True], device=Q.device)
