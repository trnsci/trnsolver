"""
One-launch Chebyshev filter for `trnsolver.occupied_subspace` (SCF eigh).

``subspace_filter_kernel(F, X, table, sched)`` keeps the whole symmetric
F (n x n, FP32) resident in SBUF and, for a block X (n x kb, kb <= 512):

    FX, r_j = |F x_j - theta_j x_j|^2     step 0, also the residual check
    Y_1 = a_1 FX + b_1 X                  scaled Chebyshev recurrence,
    Y_i = a_i F Y_{i-1} + b_i Y_{i-1}     m - 1 more F-applies
          + g_i Y_{i-2}
    M = X^T Y,  E = Y - X M               correction outside span(X)
    FE                                    one more F-apply
    X^T X, X^T F X, X^T E, (FX)^T E,      the six kb x kb Gram blocks of
    E^T E, E^T F E                        span[X, E] (Tensor Engine)

and returns them packed as (n + 6 kb + 1, kb): ``[E; six Grams; r]``.

Layout. ``Fs[p, j, :] = F[j * 128 + p, :]`` (row tile j on the partitions,
zero-padded to nt * 128 rows and columns), and every block of vectors is
``(128, nt, kb)`` the same way. An F-apply is, for each output row tile i,
``nt`` accumulating matmuls ``F[jb, ib]^T @ Y[jb]`` into one (128, kb) PSUM
tile: the stationary operand is F's (j, i) block (it equals F[ib, jb]^T, as
F is symmetric) and the moving operand is the j-th row tile of the block,
kb <= 512 FP32 columns, which is exactly one 2 KiB PSUM bank. The F-apply
is the only O(n^2 kb) work; there are m + 1 of them per launch, all on the
Tensor Engine, with the recurrence's combine on the Scalar Engine
(``g Y_{i-2}``) and Vector Engine (the rest) overlapping the next tile's
matmuls. Nothing leaves the chip between steps.

The scalars a_i, b_i, g_i and theta come in ``table`` (128 rows of the same
3 m + kb values, so each is a (128, 1) column or a (128, kb) row), never as
trace-time constants, so one compiled kernel serves every call with the same
(n, kb, m). ``sched`` is shape-only: ``sched.shape[0] = m``.

NKI 0.7 rules followed (``tests/test_subspace.py`` runs the MLIR verifier at
lnc = 1 and 2): every SBUF/PSUM tile starts at partition 0; PSUM
accumulators are created zeroed (``nl.zeros``) and then only accumulated
into, one whole tile per accumulation group; (128, 1) arithmetic is float
multiply / add only (no divide, no reciprocal); no transposing DMA
(``nisa.nc_transpose`` on the Tensor Engine instead).

Under LNC=2 both programs run the whole launch and write identical results
(the F-applies need every row tile of the previous step, and NKI 0.7 has no
cross-core barrier inside a launch).
"""

from __future__ import annotations

import functools

import torch

from .dispatch import HAS_NKI, PMAX, _use_simulator

N_GRAMS = 6
# Per-partition SBUF budget for the resident state (of 192 KiB on Trn2),
# leaving the rest to the compiler's temporaries.
_SBUF_BUDGET = 150 * 1024
_KB_MAX = 512


def layout(n: int, kb: int) -> tuple[int, int]:
    """(nt, nck): row tiles of F and 128-column chunks of the block."""
    return -(-n // PMAX), -(-kb // PMAX)


def sbuf_bytes(n: int, kb: int, m: int = 8) -> int:
    """Per-partition SBUF bytes of the kernel's resident tiles."""
    nt, nck = layout(n, kb)
    words = nt * nt * PMAX  # F
    words += 4 * nt * kb  # X, FX and the two recurrence buffers
    words += nck * nt * PMAX + nck * kb  # X^T tiles and X^T Y
    words += 3 * m + kb + 8 * kb  # table and temporaries
    return 4 * words


def kernel_fits(n: int, kb: int) -> bool:
    """True iff (n, kb) fits the kernel (kb <= 512, F resident in SBUF)."""
    return 1 <= kb <= min(n, _KB_MAX) and sbuf_bytes(n, kb) <= _SBUF_BUDGET


if HAS_NKI:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def subspace_filter_kernel(F, X, table, sched):
        """Chebyshev filter + correction + Gram blocks; see the module docstring.

        Args:
            F     : (n, n) FP32 symmetric.
            X     : (n, kb) FP32, kb <= 512.
            table : (128, 3 m + kb) FP32, every row ``[a | b | g | -theta]``.
            sched : (m, 1) FP32, shape only (the filter degree m >= 1).

        Returns:
            (n + 6 kb + 1, kb) FP32: ``E`` (n rows), then ``X^T X``,
            ``X^T F X``, ``X^T E``, ``(F X)^T E``, ``E^T E``, ``E^T F E``
            (kb rows each), then the residual row ``r``.
        """
        n = F.shape[0]
        kb = X.shape[1]
        m = sched.shape[0]
        P = PMAX
        nt = (n + P - 1) // P
        npad = nt * P
        f32 = nl.float32
        mul = nl.multiply
        add = nl.add
        tw = 3 * m + kb
        chunks = [(c0, min(P, kb - c0)) for c0 in range(0, kb, P)]

        out = nl.ndarray((n + N_GRAMS * kb + 1, kb), dtype=f32, buffer=nl.shared_hbm)

        tab = nl.ndarray((P, tw), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=tab, src=table[0:P, 0:tw])
        ones = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ones, value=1.0)

        def col(c):
            return tab[0:P, c : c + 1]

        def rows_of(i):
            return min(P, n - i * P)

        # F resident: Fs[p, j, :] = F[j P + p, :], zero-padded.
        Fs = nl.ndarray((P, nt, npad), dtype=f32, buffer=nl.sbuf)
        for j in range(nt):
            if rows_of(j) < P or npad != n:
                nisa.memset(dst=Fs[0:P, j, 0:npad], value=0.0)
            nisa.dma_copy(dst=Fs[0 : rows_of(j), j, 0:n], src=F[j * P : j * P + rows_of(j), 0:n])

        def block():
            return nl.ndarray((P, nt, kb), dtype=f32, buffer=nl.sbuf)

        Xs = block()
        for i in range(nt):
            if rows_of(i) < P:
                nisa.memset(dst=Xs[0:P, i, 0:kb], value=0.0)
            nisa.dma_copy(dst=Xs[0 : rows_of(i), i, 0:kb], src=X[i * P : i * P + rows_of(i), 0:kb])
        FXs = block()
        B1 = block()
        B2 = block()

        def apply_f(src, emit):
            """emit(i, ps) with ps = (F src)[row tile i] in PSUM, for every i."""
            for i in range(nt):
                ps = nl.zeros((P, kb), dtype=f32, buffer=nl.psum)
                for j in range(nt):
                    nisa.nc_matmul(
                        dst=ps,
                        stationary=Fs[0:P, j, i * P : (i + 1) * P],
                        moving=src[0:P, j, 0:kb],
                        accumulate=True,
                    )
                emit(i, ps)

        # Step 0: FX, the input residual, and Y_1 = a_1 FX + b_1 X.
        rn = nl.zeros((1, kb), dtype=f32, buffer=nl.psum)

        def emit0(i, ps):
            nisa.tensor_copy(dst=FXs[0:P, i, 0:kb], src=ps)
            rt = nl.ndarray((P, kb), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=rt, data1=Xs[0:P, i, 0:kb], data2=tab[0:P, 3 * m : 3 * m + kb], op=mul
            )
            nisa.tensor_tensor(dst=rt, data1=rt, data2=FXs[0:P, i, 0:kb], op=add)
            sq = nl.ndarray((P, kb), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=rt, data2=rt, op=mul)
            nisa.nc_matmul(dst=rn, stationary=ones, moving=sq, accumulate=True)
            t = nl.ndarray((P, kb), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=t, data=Xs[0:P, i, 0:kb], op0=mul, operand0=col(m))
            nisa.scalar_tensor_tensor(
                dst=B1[0:P, i, 0:kb],
                data=FXs[0:P, i, 0:kb],
                op0=mul,
                operand0=col(0),
                op1=add,
                operand1=t,
            )

        apply_f(Xs, emit0)

        # Steps 1 .. m-1. Step 1 writes B2 (X must survive); later steps
        # overwrite Y_{i-2} in place, tile by tile.
        cur, prev = B1, Xs
        for s in range(1, m):
            dst = B2 if prev is Xs else prev

            def emit(i, ps, s=s, cur=cur, prev=prev, dst=dst):
                t = nl.ndarray((P, kb), dtype=f32, buffer=nl.sbuf)
                nisa.activation(dst=t, op=nl.copy, data=prev[0:P, i, 0:kb], scale=col(2 * m + s))
                nisa.scalar_tensor_tensor(
                    dst=t,
                    data=cur[0:P, i, 0:kb],
                    op0=mul,
                    operand0=col(m + s),
                    op1=add,
                    operand1=t,
                )
                nisa.scalar_tensor_tensor(
                    dst=dst[0:P, i, 0:kb], data=ps, op0=mul, operand0=col(s), op1=add, operand1=t
                )

            apply_f(cur, emit)
            prev, cur = cur, dst
        Y = cur
        Eb = B2 if Y is B1 else B1

        def gram(A, Bv, emit):
            """emit(ci, c0, cw, ps) with ps = A[:, c-chunk]^T Bv in PSUM (cw, kb)."""
            for ci, (c0, cw) in enumerate(chunks):
                ps = nl.zeros((cw, kb), dtype=f32, buffer=nl.psum)
                for j in range(nt):
                    nisa.nc_matmul(
                        dst=ps,
                        stationary=A[0:P, j, c0 : c0 + cw],
                        moving=Bv[0:P, j, 0:kb],
                        accumulate=True,
                    )
                emit(ci, c0, cw, ps)

        # M = X^T Y (chunks of rows on the partitions), and X^T tiles.
        Ms = nl.ndarray((P, len(chunks), kb), dtype=f32, buffer=nl.sbuf)

        def keep_m(ci, c0, cw, ps):
            nisa.tensor_copy(dst=Ms[0:cw, ci, 0:kb], src=ps)

        gram(Xs, Y, keep_m)
        XT = nl.ndarray((P, len(chunks), nt, P), dtype=f32, buffer=nl.sbuf)
        for ci, (c0, cw) in enumerate(chunks):
            for i in range(nt):
                pt = nl.ndarray((cw, P), dtype=f32, buffer=nl.psum)
                nisa.nc_transpose(dst=pt, data=Xs[0:P, i, c0 : c0 + cw], engine=nisa.engine.tensor)
                nisa.tensor_copy(dst=XT[0:cw, ci, i, 0:P], src=pt)

        # E = Y - X M, row tile by row tile.
        for i in range(nt):
            ps = nl.zeros((P, kb), dtype=f32, buffer=nl.psum)
            for ci, (_c0, cw) in enumerate(chunks):
                nisa.nc_matmul(
                    dst=ps,
                    stationary=XT[0:cw, ci, i, 0:P],
                    moving=Ms[0:cw, ci, 0:kb],
                    accumulate=True,
                )
            nisa.tensor_tensor(
                dst=Eb[0:P, i, 0:kb], data1=Y[0:P, i, 0:kb], data2=ps, op=nl.subtract
            )

        # FE overwrites Y (no longer needed).
        FE = Y

        def emit_fe(i, ps):
            nisa.tensor_copy(dst=FE[0:P, i, 0:kb], src=ps)

        apply_f(Eb, emit_fe)

        for i in range(nt):
            nisa.dma_copy(
                dst=out[i * P : i * P + rows_of(i), 0:kb], src=Eb[0 : rows_of(i), i, 0:kb]
            )

        def store_at(row0):
            def emit_g(ci, c0, cw, ps):
                g = nl.ndarray((cw, kb), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=g, src=ps)
                nisa.dma_copy(dst=out[row0 + c0 : row0 + c0 + cw, 0:kb], src=g)

            return emit_g

        pairs = ((Xs, Xs), (Xs, FXs), (Xs, Eb), (FXs, Eb), (Eb, Eb), (Eb, FE))
        for g_i, (A, Bv) in enumerate(pairs):
            gram(A, Bv, store_at(n + g_i * kb))

        rs = nl.ndarray((1, kb), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rs, src=rn)
        nisa.dma_copy(dst=out[n + N_GRAMS * kb : n + N_GRAMS * kb + 1, 0:kb], src=rs)
        return out


@functools.lru_cache(maxsize=32)
def _sched(m: int, device: str) -> torch.Tensor:
    return torch.zeros((m, 1), dtype=torch.float32, device=device)


def kernel_inputs(F: torch.Tensor, X: torch.Tensor, table: torch.Tensor):
    """(F, X, table (128, w), sched) as the kernel takes them, on F's device."""
    kb = X.shape[1]
    m = (table.numel() - kb) // 3
    tab = table.to(F.device, torch.float32).reshape(1, -1).expand(PMAX, -1).contiguous()
    return (
        F.contiguous(),
        X.to(F.device, torch.float32).contiguous(),
        tab,
        _sched(m, str(F.device)),
    )


def run_filter_kernel(F: torch.Tensor, X: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """One kernel launch (simulator or native device); same output as the reference."""
    if not HAS_NKI:
        raise RuntimeError("the subspace filter kernel needs nki")
    args = kernel_inputs(F, X, table)
    if _use_simulator():
        import nki as _nki

        res = _nki.simulate(subspace_filter_kernel)(
            *(a.detach().cpu().contiguous().numpy() for a in args)
        )
        return torch.from_numpy(res).to(F.device)
    from .dispatch import _launch

    return _launch(subspace_filter_kernel, "subspace_filter", *args, device=F.device)
