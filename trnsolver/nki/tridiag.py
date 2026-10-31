"""
On-device Householder tridiagonalization for the NKI ``eigh`` path (#42).

One kernel launch reduces a symmetric (n, n) matrix, n <= 128, to tridiagonal
form. The matrix stays resident in SBUF for all n - 2 Householder steps. The
reflector vector and its scalars (norm, sign, beta, the v.w correction) are
computed on chip, so the reduction needs no ``.item()`` and no host round
trip. The host-driven `_householder_tridiag` in ``eigen.py`` needs
2(n - 2) launches and about 6(n - 2) syncs for the same work.

Selected with ``TRNSOLVER_EIGH_TRIDIAG=device`` (default off); see
`use_device_tridiag`.

Layout
------
``A`` lives in SBUF as (partition = row i, free = column j). Every per-step
vector is held in one of two n-partition forms, so partition counts always
match and nothing is ever read from a single partition:

* "column" (n, 1): entry i on partition i.
* "row-broadcast" (n, n): entry j on every partition, along the free axis.

Moving data across partitions is done on the Tensor Engine with matmuls whose
sums have a single nonzero term, so they are exact:

* row k of A, broadcast:  ``R = Sel_k^T @ A`` with ``Sel_k[i, m] = [i == k]``
* row-broadcast of a column w:  ``W = ones^T @ (I * w)``

Scalars are (n, 1) columns holding the same value on every partition, and are
built from free-axis reductions of row-broadcast tiles. There is no divide:
the norm, beta and the reflector normalisation all come from one
Scalar-Engine ``nl.rsqrt`` each, refined with two Newton steps on the Vector
Engine because the activation is an approximation. The input is first scaled
to max|a| = 1 (undone on the outputs) so every rsqrt argument sits in the
Scalar Engine's accurate range whatever the input's magnitude.

Step k (k = 0 .. n - 3), with m_k = [j > k] and e = e_{k+1}:

    x      = A[k, :] * m_k                     (row k, below-diagonal part)
    sigma  = x . x,   x0 = x[k + 1],   norm = sigma * rsqrt(sigma)
    s      = +1 if x0 >= 0 else -1,    alpha = -s * norm
    v      = x + s * norm * e
    h      = (v . v) / 2 = (sigma + norm^2) / 2 + |x0| * norm
    beta   = 2 / (v . v) = rsqrt(h)^2,   v_unit = v * rsqrt(h) / sqrt(2)
    w      = beta * A v,   g = beta * (v . w) / 2,   u = (w - g v) * m_k
    A     <- A - u v^T - v u^T

Masking u to rows > k leaves rows and columns <= k untouched, so the trailing
block is the exact two-sided update while column k keeps its old contents.
Those stale entries are never read again: later reflectors are zero on rows
<= k + 1 and their u is masked the same way. The subdiagonal entry is
recorded directly as alpha, the diagonal is read off at the end.

Output is packed into one (n, 2 + ncol) HBM tensor to keep the launch to a
single output: column 0 is the diagonal, column 1 the subdiagonal (n - 1
valid entries, last row 0), and columns 2.. the unit-norm reflectors in the
same layout as the host path's ``V_refs`` (n, max(n - 1, 1)).

NKI 0.7 rules this kernel is written against (the CPU simulator does not
check them; ``tests/test_eigh_tridiag_device.py`` runs the MLIR verifier):

* every tile spans all n partitions starting at partition 0, so partition
  counts always match and there are no single-partition copies;
* (P, 1) arithmetic is float-only and divide-free;
* ``nl.broadcast_to`` is only applied to freshly computed tiles, never to a
  sliced view.
"""

from __future__ import annotations

import os

import torch

from .dispatch import HAS_NKI, PMAX, _use_simulator

# Floor for rsqrt arguments, 2^-100: inside the Scalar Engine's accurate
# range and far below any value that matters once A is scaled to max|a| = 1.
# Only reached when the column being reduced is already (numerically) zero,
# in which case v = 0 and the step is a no-op.
_FLOOR = 2.0**-100
_SQRT_HALF = 0.7071067811865476

_ENV = "TRNSOLVER_EIGH_TRIDIAG"


def use_device_tridiag(n: int) -> bool:
    """True iff eigh should tridiagonalize with the single-launch kernel.

    ``TRNSOLVER_EIGH_TRIDIAG=device`` turns it on (read on every call). Only
    single-tile sizes, 2 <= n <= 128, are supported; other sizes keep the
    host-driven reduction.
    """
    if os.environ.get(_ENV, "").strip().lower() != "device":
        return False
    return 2 <= n <= PMAX


def n_out_cols(n: int) -> int:
    """Packed output width: diag, subdiag, then max(n - 1, 1) reflectors."""
    return 2 + max(n - 1, 1)


def tridiag_constants(n: int, dtype=torch.float32) -> torch.Tensor:
    """(n, 2n) constant input: identity, then column-index tile J[i, j] = j."""
    eye = torch.eye(n, dtype=dtype)
    cols = torch.arange(n, dtype=dtype).unsqueeze(0).expand(n, n)
    return torch.cat([eye, cols], dim=1).contiguous()


if HAS_NKI:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def householder_tridiag_kernel(A, consts):
        """Householder reduction of symmetric A to tridiagonal, one launch.

        Args:
            A      : (n, n) FP32 symmetric, in HBM. 2 <= n <= 128.
            consts : (n, 2n) FP32 from `tridiag_constants` (identity, J).

        Returns:
            out : (n, 2 + max(n - 1, 1)) FP32 in HBM. Column 0 = diag,
                  column 1 = subdiag (last entry 0), columns 2.. = unit-norm
                  Householder vectors, one per step, zero-padded.
        """
        n = A.shape[0]
        ncol = max(n - 1, 1)
        f32 = nl.float32

        out = nl.ndarray((n, 2 + ncol), dtype=A.dtype, buffer=nl.shared_hbm)

        a = nl.load(A[0:n, 0:n])
        ident = nl.load(consts[0:n, 0:n])  # I[i, j] = [i == j]
        jidx = nl.load(consts[0:n, n : 2 * n])  # J[i, j] = j
        ones = nl.add(nl.multiply(ident, 0.0), 1.0)  # (n, n) all ones
        # P[i] = i, as a column.
        pidx = nl.sum(nl.multiply(ident, jidx), axis=1, keepdims=True)

        def rsqrt_refined(x):
            # Scalar Engine rsqrt is an approximation; two Newton steps on the
            # Vector Engine (y <- y * (1.5 - 0.5 x y^2)) bring it to FP32.
            y = nl.rsqrt(x)
            for _ in range(2):
                y = nl.multiply(
                    y, nl.add(nl.multiply(nl.multiply(nl.multiply(y, y), x), -0.5), 1.5)
                )
            return y

        def row_broadcast(col):
            # (n, 1) column -> (n, n) tile with col[j] on every partition.
            diag_t = nl.multiply(ident, nl.broadcast_to(col, (n, n)))
            ps = nl.zeros((n, n), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps, stationary=ones, moving=diag_t, accumulate=False)
            return nl.copy(ps, dtype=f32)

        # Scale to max|a| = 1 so every rsqrt argument below is O(1)..O(n)
        # whatever the input's magnitude; undone on diag/subdiag at the end.
        rowmax = nl.max(nl.abs(a), axis=1, keepdims=True)
        gmax = nl.maximum(nl.max(row_broadcast(rowmax), axis=1, keepdims=True), _FLOOR)
        ginv = rsqrt_refined(gmax)
        ginv = nl.multiply(ginv, ginv)
        a = nl.multiply(a, nl.broadcast_to(ginv, (n, n)))

        zeros_col = nl.multiply(pidx, 0.0)
        sub = zeros_col  # subdiagonal accumulator, (n, 1)
        # Zero only the padding reflector column(s) the loop never writes, so
        # no two stores target the same HBM region.
        npad = ncol - (n - 2)
        vzero = nl.multiply(nl.broadcast_to(zeros_col, (n, npad)), 1.0)
        nl.store(out[0:n, 2 + n - 2 : 2 + ncol], value=vzero)

        for k in range(n - 2):
            # One-hot column e_k (partition k) and its free-axis broadcast.
            ek_col = nl.equal(pidx, float(k), dtype=f32)
            sel = nl.broadcast_to(ek_col, (n, n))

            # Row k of A on every partition: R[m, j] = A[k, j].
            ps_r = nl.zeros((n, n), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps_r, stationary=sel, moving=a, accumulate=False)
            r = nl.copy(ps_r, dtype=f32)

            mrow = nl.greater(jidx, float(k), dtype=f32)  # [j > k]
            erow = nl.equal(jidx, float(k + 1), dtype=f32)  # [j == k + 1]
            mcol = nl.greater(pidx, float(k), dtype=f32)  # [i > k]

            xr = nl.multiply(r, mrow)
            sigma = nl.sum(nl.multiply(xr, xr), axis=1, keepdims=True)
            x0 = nl.sum(nl.multiply(r, erow), axis=1, keepdims=True)
            norm = nl.multiply(sigma, rsqrt_refined(nl.maximum(sigma, _FLOOR)))

            # s = +1 for x0 >= 0 (including 0, as the host path does), else -1.
            sgn = nl.add(nl.multiply(nl.greater_equal(x0, 0.0, dtype=f32), 2.0), -1.0)
            snorm = nl.multiply(sgn, norm)  # = -alpha

            # half_vv = (v . v) / 2 = (sigma + norm^2) / 2 + |x0| * norm, written
            # out in full so H = I - beta v v^T stays orthogonal even if norm
            # carries rounding. beta = 2 / (v . v) = y^2 with y = rsqrt(half_vv).
            half_vv = nl.add(
                nl.multiply(nl.add(sigma, nl.multiply(norm, norm)), 0.5),
                nl.multiply(nl.abs(x0), norm),
            )
            y = rsqrt_refined(nl.maximum(half_vv, _FLOOR))
            beta = nl.multiply(y, y)

            # v as row-broadcast and as column (exact diagonal extraction).
            vr = nl.add(xr, nl.multiply(erow, nl.broadcast_to(snorm, (n, n))))
            vc = nl.sum(nl.multiply(vr, ident), axis=1, keepdims=True)

            # w = beta * A v (column), then its row-broadcast on the PE.
            wc = nl.multiply(nl.sum(nl.multiply(a, vr), axis=1, keepdims=True), beta)
            wr = row_broadcast(wc)

            vw = nl.sum(nl.multiply(vr, wr), axis=1, keepdims=True)
            g = nl.multiply(nl.multiply(beta, vw), 0.5)

            ur = nl.multiply(nl.subtract(wr, nl.multiply(vr, nl.broadcast_to(g, (n, n)))), mrow)
            uc = nl.multiply(nl.subtract(wc, nl.multiply(vc, g)), mcol)

            uv = nl.multiply(nl.broadcast_to(uc, (n, n)), vr)  # u_i v_j
            vu = nl.multiply(nl.broadcast_to(vc, (n, n)), ur)  # v_i u_j
            a = nl.subtract(nl.subtract(a, uv), vu)

            # Unit-norm reflector: 1 / ||v|| = y / sqrt(2).
            vunit = nl.multiply(vc, nl.multiply(y, _SQRT_HALF))
            nl.store(out[0:n, 2 + k : 3 + k], value=vunit)

            # subdiag[k] = alpha = -snorm.
            sub = nl.subtract(sub, nl.multiply(ek_col, snorm))

        diag = nl.sum(nl.multiply(a, ident), axis=1, keepdims=True)

        # Last subdiagonal entry A[n-2, n-1] comes from the final 2x2 block.
        sup_mask = nl.equal(jidx, nl.broadcast_to(nl.add(pidx, 1.0), (n, n)), dtype=f32)
        sup = nl.sum(nl.multiply(a, sup_mask), axis=1, keepdims=True)
        e_last = nl.equal(pidx, float(n - 2), dtype=f32)
        sub = nl.add(sub, nl.multiply(sup, e_last))

        diag = nl.multiply(diag, gmax)
        sub = nl.multiply(sub, gmax)

        nl.store(out[0:n, 0:1], value=diag)
        nl.store(out[0:n, 1:2], value=sub)
        return out


_CONSTS_CACHE: dict = {}


def _consts_on(n: int, device) -> torch.Tensor:
    key = (n, str(device))
    c = _CONSTS_CACHE.get(key)
    if c is None:
        c = tridiag_constants(n).to(device)
        _CONSTS_CACHE[key] = c
    return c


def householder_tridiag_device(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in for ``eigen._householder_tridiag`` using one kernel launch.

    Returns (diag (n,), subdiag (n-1,), V_refs (n, max(n-1, 1))) on A's
    device, matching the host-driven reduction's contract.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI backend requested but the nki package isn't importable")
    n = A.shape[0]
    assert A.shape == (n, n), f"Expected square matrix, got {A.shape}"
    if not 2 <= n <= PMAX:
        raise NotImplementedError(f"device tridiagonalization requires 2 <= n <= 128; got n={n}")

    orig_device = A.device
    if _use_simulator():
        import nki as _nki

        packed = _nki.simulate(householder_tridiag_kernel)(
            A.detach().cpu().contiguous().numpy(), tridiag_constants(n).numpy()
        )
        packed = torch.from_numpy(packed).to(orig_device)
    else:
        # Native TorchNeuron: the trnsolver::householder_tridiag custom op on
        # the `neuron` device. XLA: the raw kernel on torch_xla tensors, then
        # one torch_xla.sync(). The constants are cached on the work device.
        from .dispatch import _launch, _work_device

        packed = _launch(
            householder_tridiag_kernel,
            "householder_tridiag",
            A.detach().contiguous(),
            _consts_on(n, _work_device()),
            device=orig_device,
        )

    diag = packed[:, 0].contiguous()
    subdiag = packed[: n - 1, 1].contiguous()
    V_refs = packed[:, 2:].contiguous()
    return diag, subdiag, V_refs
