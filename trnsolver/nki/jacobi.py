"""
Batched parallel-ordered Jacobi ``eigh`` on device, SBUF-resident (#48).

One launch runs whole Jacobi sweeps on a group of tiles. Each tile is one
128-partition SBUF tile holding ``b = P // n`` matrices block-diagonally
(``P = b * n <= 128``): one n = 128 matrix, two n = 64, four n = 32, and so
on. A and the transposed eigenvector matrix ``W = V^T`` stay in SBUF from
the first step of the first sweep to the last step of the last sweep; HBM
is touched once on the way in and once on the way out.

One step (n / 2 disjoint rotations per matrix, all blocks of all tiles):

    Pi    = (Q == v_r)                         partner matrix of step r
    M     = Pi^T B                 [TE]        M[p, :] = B[partner(p), :]
    d     = rowsum(B o I),  dq = rowsum(M o Pi)          a_pp, a_qq
    g     = (rowsum(B o Pi) + rowsum(M o I)) / 2         symmetrised a_pq
    (c, s, c - 1)  per row, column math on (P, 1)        see trnsolver/_jacobi.py
    K     = (c - 1) o I + s o Pi                         K = J - I
    T     = B + B^T K              [TE + add on PSUM evacuation]
    B     = T + K^T T              [TE + add]
    W     = W + K^T W              [TE + add]

so four Tensor Engine matmuls per step, three of them the update itself.
The ``X + X K`` form keeps a small rotation's O(t^2) diagonal shrinkage
(``c - 1 = -s^2 / (1 + c)``) instead of rounding ``c`` to 1, which in FP32
grows every column norm by t^2 per step. The block-diagonal zeros stay
exactly zero (products with exact zeros), so blocks never mix.

Step masks come from one constant tile ``Q``: within a block, with
``m = n - 1``, ``Q[i, j] = (i + j) mod m`` for ``i != j < m``,
``Q[i, m] = Q[m, i] = 2 i mod m``, and ``-1`` on the diagonal and off-block.
Step r pairs exactly the entries with ``Q == 2 r mod m`` (circle-method
round robin; m is odd so every index is paired once per step).

NKI 0.7 rules (checked by the MLIR verifier in tests/test_jacobi.py):
every tile spans partitions 0..P-1 of a full tile, (P, 1) column math is
float-only and divide-free (``rsqrt`` / ``reciprocal`` refined by Newton
steps), and every matmul writes a whole PSUM tile with ``accumulate=False``
(no column slices of a shared PSUM tile).
"""

from __future__ import annotations

import functools

import torch

from .dispatch import HAS_NKI, PMAX, _use_simulator

# Clamp for rsqrt arguments: the Scalar Engine rsqrt table is valid on
# [2^-87, 2^97], so the floor must sit inside it.
_TINY = 2.0**-80


def block_layout(n: int) -> tuple[int, int, int]:
    """(ne, b, P): padded block size, matrices per tile, tile partitions.

    ``ne`` is n rounded up to a power of two (at least 2), so blocks tile
    the 128 partitions and the Tensor Engine quadrants exactly. Padding rows
    and columns are zero and stay decoupled (their rotation angle is exactly
    zero), so the padded eigenpairs sit at indices >= n and are dropped.
    """
    if n > PMAX:
        raise NotImplementedError(f"device Jacobi needs n <= {PMAX}; got n={n}")
    ne = max(2, 1 << (n - 1).bit_length())
    b = PMAX // ne
    return ne, b, b * ne


@functools.lru_cache(maxsize=16)
def jacobi_constants(n: int) -> torch.Tensor:
    """(P, 2P + ne - 1) FP32: [I | Q | sigma table], for block size n.

    ``sigma[p, r]`` is +1 if p is the lower index of its step-r pair, else -1
    (the canonical orientation of the rotation-sign tie break).
    """
    ne, b, P = block_layout(n)
    m = ne - 1
    i = torch.arange(ne).unsqueeze(1)
    j = torch.arange(ne).unsqueeze(0)
    q = torch.remainder(i + j, m).to(torch.float32)
    q[:m, m] = torch.remainder(2 * torch.arange(m), m).float()
    q[m, :m] = torch.remainder(2 * torch.arange(m), m).float()
    q.fill_diagonal_(-1.0)
    Q = torch.full((P, P), -1.0)
    for k in range(b):
        Q[k * ne : (k + 1) * ne, k * ne : (k + 1) * ne] = q
    sig = torch.empty((P, m))
    cols = torch.arange(P, dtype=torch.float32)
    for r, v in enumerate(step_targets(ne)):
        partner = ((Q == v).float() * cols).sum(1)
        sig[:, r] = torch.where(partner > cols, 1.0, -1.0)
    return torch.cat([torch.eye(P), Q, sig], dim=1).contiguous()


def step_targets(n: int) -> list[float]:
    """``v_r = 2 r mod (n - 1)`` for the n - 1 steps of a sweep (n even)."""
    m = n - 1
    return [float((2 * r) % m) for r in range(m)]


if HAS_NKI:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def jacobi_sweeps_kernel(A, W, consts, sched):
        """Run ``sched.shape[0]`` Jacobi sweeps on a group of packed tiles.

        Args:
            A      : (G, P, P) FP32, block-diagonal symmetric tiles.
            W      : (G, P, P) FP32, eigenvectors transposed so far (I to start).
            consts : (P, 2P + n - 1) FP32 from `jacobi_constants`.
            sched  : (sweeps, n, chains) FP32, shape only: sweep count, block size
                     and the number of independent column-math chains.

        Returns:
            out : (2, G, P, P) FP32: [0] = A after the sweeps, [1] = W.

        Per tile, A and W live side by side in one (P, 2P) SBUF tile ``X``
        so the two updates with stationary K are one matmul:
        ``K^T [T | W]``. The (P, 1) column math of a step is done once for
        all G tiles on (P, G) tiles.
        """
        G = A.shape[0]
        P = A.shape[1]
        sweeps = sched.shape[0]
        n = sched.shape[1]
        nchain = sched.shape[2]
        m = n - 1
        f32 = nl.float32
        mul = nl.multiply
        add = nl.add

        out = nl.ndarray((2, G, P, P), dtype=A.dtype, buffer=nl.shared_hbm)

        ident = nl.load(consts[0:P, 0:P])
        qcode = nl.load(consts[0:P, P : 2 * P])
        sigt = nl.load(consts[0:P, 2 * P : 2 * P + m])
        ones_col = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ones_col, value=1.0)

        X = []
        for gi in range(G):
            x = nl.ndarray((P, 2 * P), dtype=f32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x[0:P, 0:P], src=A[gi, 0:P, 0:P])
            nisa.dma_copy(dst=x[0:P, P : 2 * P], src=W[gi, 0:P, 0:P])
            X.append(x)

        def col(width):
            return nl.ndarray((P, width), dtype=f32, buffer=nl.sbuf)

        def step(tiles, pi, sig):
            """One Jacobi step for the tiles in ``tiles`` (one shared column chain)."""
            ng = len(tiles)
            dcol = col(ng)
            gpq = col(ng)
            gs = col(ng)
            for j, gi in enumerate(tiles):
                b = X[gi][0:P, 0:P]
                junk = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(
                    dst=junk, data=b, op0=mul, operand0=1.0, op1=mul, operand1=ident,
                    reduce_op=add, reduce_res=dcol[0:P, j : j + 1],
                )
                bp = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(
                    dst=bp, data=b, op0=mul, operand0=1.0, op1=mul, operand1=pi,
                    reduce_op=add, reduce_res=gpq[0:P, j : j + 1],
                )
                # gqp[p] = B[partner(p), p] = sum_k (B o Pi)[k, p]
                ps_q = nl.ndarray((P, 1), dtype=f32, buffer=nl.psum)
                nisa.nc_matmul(dst=ps_q, stationary=bp, moving=ones_col, accumulate=False)
                # gs = a_pq + a_qp = 2 g (symmetrised)
                nisa.tensor_tensor(
                    dst=gs[0:P, j : j + 1], data1=ps_q, data2=gpq[0:P, j : j + 1], op=add
                )
            # dq[p, j] = d[partner(p), j], one matmul for the chain.
            ps_dq = nl.ndarray((P, ng), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps_dq, stationary=pi, moving=dcol, accumulate=False)

            # Column math on (P, ng); see trnsolver/_jacobi.rotation_csk.
            h = col(ng)
            nisa.tensor_tensor(dst=h, data1=ps_dq, data2=dcol, op=nl.subtract)
            hs = col(ng)
            nisa.tensor_scalar(dst=hs, data=h, op0=mul, operand0=sig)
            pos2 = col(ng)
            nisa.tensor_scalar(
                dst=pos2, data=hs, op0=nl.greater_equal, operand0=0.0, op1=mul, operand1=2.0
            )
            sg = col(ng)
            nisa.tensor_scalar(dst=sg, data=pos2, op0=nl.subtract, operand0=1.0, op1=mul, operand1=sig)
            hh = col(ng)
            nisa.tensor_tensor(dst=hh, data1=h, data2=h, op=mul)
            xx = col(ng)
            nisa.tensor_tensor(dst=xx, data1=gs, data2=gs, op=mul)
            nisa.tensor_tensor(dst=xx, data1=xx, data2=hh, op=add)
            # r = sqrt(x) as x * rsqrt(x): the Scalar Engine then only ever
            # needs the rsqrt table (each table switch costs ~1.3 us).
            nisa.tensor_scalar(dst=xx, data=xx, op0=nl.maximum, operand0=_TINY)
            rr = col(ng)
            nisa.activation(dst=rr, op=nl.rsqrt, data=xx)
            nisa.tensor_tensor(dst=rr, data1=rr, data2=xx, op=mul)
            den = col(ng)
            # |h| = max(-h, h)  (abs_max is NeuronCore-v4+ only)
            nisa.scalar_tensor_tensor(
                dst=den, data=h, op0=mul, operand0=-1.0, op1=nl.maximum, operand1=h
            )
            nisa.scalar_tensor_tensor(
                dst=den, data=den, op0=nl.maximum, operand0=0.0, op1=add, operand1=rr
            )
            nisa.tensor_scalar(dst=den, data=den, op0=nl.maximum, operand0=_TINY)
            u = col(ng)
            nisa.activation(dst=u, op=nl.rsqrt, data=den)
            t = col(ng)
            nisa.tensor_tensor(dst=t, data1=gs, data2=u, op=mul)
            nisa.tensor_tensor(dst=t, data1=t, data2=u, op=mul)
            nisa.tensor_tensor(dst=t, data1=t, data2=sg, op=mul)
            t2 = col(ng)
            nisa.tensor_tensor(dst=t2, data1=t, data2=t, op=mul)
            c = col(ng)
            nisa.activation(dst=c, op=nl.rsqrt, data=t2, bias=1.0)
            s = col(ng)
            nisa.tensor_tensor(dst=s, data1=t, data2=c, op=mul)
            w = col(ng)
            nisa.activation(dst=w, op=nl.rsqrt, data=c, bias=1.0)
            ww = col(ng)
            nisa.tensor_tensor(dst=ww, data1=w, data2=w, op=mul)
            km1 = col(ng)
            nisa.scalar_tensor_tensor(dst=km1, data=s, op0=mul, operand0=-1.0, op1=mul, operand1=s)
            nisa.tensor_tensor(dst=km1, data1=km1, data2=ww, op=mul)

            for j, gi in enumerate(tiles):
                x = X[gi]
                b = x[0:P, 0:P]
                sp = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=sp, data=pi, op0=mul, operand0=s[0:P, j : j + 1])
                kmat = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(
                    dst=kmat, data=ident, op0=mul, operand0=km1[0:P, j : j + 1],
                    op1=add, operand1=sp,
                )
                # T = B + B^T K   (overwrites B in place)
                ps_t = nl.ndarray((P, P), dtype=f32, buffer=nl.psum)
                nisa.nc_matmul(dst=ps_t, stationary=b, moving=kmat, accumulate=False)
                nisa.tensor_tensor(dst=b, data1=ps_t, data2=b, op=add)
                # [B | W] = [T | W] + K^T [T | W]
                ps_x = nl.ndarray((P, 2 * P), dtype=f32, buffer=nl.psum)
                nisa.nc_matmul(dst=ps_x, stationary=kmat, moving=x, accumulate=False)
                nisa.tensor_tensor(dst=x, data1=ps_x, data2=x, op=add)

        # Independent chains: while one chain does its (P, 1) math the other's
        # matmuls and evacuations keep the Tensor and Vector Engines busy.
        chains = [list(range(k, G, nchain)) for k in range(min(nchain, G))]
        for _sw in range(sweeps):
            for r in range(m):
                pi = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=pi, data=qcode, op0=nl.equal, operand0=float((2 * r) % m))
                sig = sigt[0:P, r : r + 1]
                for tiles in chains:
                    step(tiles, pi, sig)

        for gi in range(G):
            nisa.dma_copy(dst=out[0, gi, 0:P, 0:P], src=X[gi][0:P, 0:P])
            nisa.dma_copy(dst=out[1, gi, 0:P, 0:P], src=X[gi][0:P, P : 2 * P])
        return out


def _run_kernel(
    A: torch.Tensor, W: torch.Tensor, n: int, sweeps: int, chains: int = 2
) -> torch.Tensor:
    """One launch: simulator or device. A, W: (G, P, P)."""
    consts = jacobi_constants(n)
    sched = torch.zeros((sweeps, n, chains), dtype=torch.float32)
    if _use_simulator():
        import nki as _nki

        res = _nki.simulate(jacobi_sweeps_kernel)(
            A.detach().cpu().contiguous().numpy(),
            W.detach().cpu().contiguous().numpy(),
            consts.numpy(),
            sched.numpy(),
        )
        return torch.from_numpy(res).to(A.device)
    from .dispatch import _launch

    dev = A.device
    return _launch(
        jacobi_sweeps_kernel, "jacobi_sweeps", A, W, consts.to(dev), sched.to(dev), device=dev
    )


def pack_blocks(X: torch.Tensor, n: int) -> torch.Tensor:
    """(B, n, n) -> (T, P, P) block-diagonal tiles; pads n to even and B to T*b.

    Padding matrices are identities (already diagonal: zero rotations). The
    odd-n padding row/column is zero. Elementwise ops only, any device.
    """
    ne, b, P = block_layout(n)
    Bsz = X.shape[0]
    T = -(-Bsz // b)
    if ne != n:
        X = torch.nn.functional.pad(X, (0, ne - n, 0, ne - n))
    if T * b != Bsz:
        eye = torch.eye(ne, dtype=X.dtype, device=X.device).expand(T * b - Bsz, ne, ne)
        X = torch.cat([X, eye], dim=0)
    eb = torch.eye(b, dtype=X.dtype, device=X.device)
    tiles = X.reshape(T, b, ne, 1, ne) * eb.reshape(1, b, 1, b, 1)
    return tiles.reshape(T, P, P).contiguous()


def unpack_blocks(tiles: torch.Tensor, n: int, Bsz: int) -> torch.Tensor:
    """Inverse of `pack_blocks`: (T, P, P) -> (Bsz, ne, ne) diagonal blocks."""
    ne, b, P = block_layout(n)
    T = tiles.shape[0]
    blocks = torch.diagonal(tiles.reshape(T, b, ne, b, ne), dim1=1, dim2=3)  # (T, ne, ne, b)
    return blocks.permute(0, 3, 1, 2).reshape(T * b, ne, ne)[:Bsz]


# ----------------------------------------------------------------------
# Compact layout (v3): a tile is (P, ne) -- block b's rows on partitions
# b*ne .. b*ne + ne - 1, its ne columns on the free axis -- and every
# Vector Engine op covers all G tiles of the group at once.
# ----------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def compact_constants(n: int) -> torch.Tensor:
    """(P, 2 ne + (ne - 1) + P + qs) FP32: [I_c | Q_c | sigma | ones_bd | qmask].

    I_c[(b, i), j] = [i == j]; Q_c the block-local step codes; sigma as in
    `jacobi_constants`; ones_bd the (P, P) block-diagonal all-ones matrix;
    qmask the block-diagonal mask within one Tensor Engine quadrant.
    """
    ne, b, P = block_layout(n)
    full = jacobi_constants(n)
    m = ne - 1
    eye = full[:, :P]
    Q = full[:, P : 2 * P]
    sig = full[:, 2 * P : 2 * P + m]
    rows = torch.arange(P)
    blk = rows // ne
    cols_local = torch.arange(ne)
    gather = (blk * ne).unsqueeze(1) + cols_local.unsqueeze(0)  # (P, ne) global columns
    Ic = torch.gather(eye, 1, gather)
    Qc = torch.gather(Q, 1, gather)
    ones_bd = (blk.unsqueeze(1) == blk.unsqueeze(0)).float()
    qs = _quadrant(ne)
    local = rows % qs
    qmask = ((local // ne).unsqueeze(1) == (torch.arange(qs) // ne).unsqueeze(0)).float()
    return torch.cat([Ic, Qc, sig, ones_bd, qmask], dim=1).contiguous()


def _quadrant(ne: int) -> int:
    """Tensor Engine tile edge used for block size ne: 32, 64 or 128."""
    return max(32, ne)


def pack_compact(X: torch.Tensor, n: int) -> torch.Tensor:
    """(B, n, n) -> (T, P, ne): pad n to even (zero row/col) and B with identities."""
    ne, b, P = block_layout(n)
    Bsz = X.shape[0]
    T = -(-Bsz // b)
    if ne != n:
        X = torch.nn.functional.pad(X, (0, ne - n, 0, ne - n))
    if T * b != Bsz:
        eye = torch.eye(ne, dtype=X.dtype, device=X.device).expand(T * b - Bsz, ne, ne)
        X = torch.cat([X, eye], dim=0)
    return X.reshape(T, P, ne).contiguous()


def unpack_compact(tiles: torch.Tensor, n: int, Bsz: int) -> torch.Tensor:
    """(T, P, ne) -> (Bsz, ne, ne)."""
    ne, b, P = block_layout(n)
    return tiles.reshape(-1, ne, ne)[:Bsz]


if HAS_NKI:

    @nki.jit
    def jacobi_compact_kernel(A, W, consts, sched):
        """Run ``sched.shape[0]`` Jacobi sweeps on G compact tiles.

        Args:
            A, W   : (G, P, ne) FP32 compact tiles (see `pack_compact`).
            consts : from `compact_constants`.
            sched  : (sweeps, ne) FP32, shape only.

        Returns:
            out : (2, G, P, ne) FP32: A and W after the sweeps.
        """
        G = A.shape[0]
        P = A.shape[1]
        ne = A.shape[2]
        nb = P // ne
        sweeps = sched.shape[0]
        m = ne - 1
        f32 = nl.float32
        mul = nl.multiply
        add = nl.add
        sub = nl.subtract

        out = nl.ndarray((2, G, P, ne), dtype=A.dtype, buffer=nl.shared_hbm)

        # Constants.
        m3 = nl.ndarray((P, 3, ne), dtype=f32, buffer=nl.sbuf)  # [I | Pi_r | I]
        nisa.dma_copy(dst=m3[0:P, 0, 0:ne], src=consts[0:P, 0:ne])
        nisa.dma_copy(dst=m3[0:P, 2, 0:ne], src=consts[0:P, 0:ne])
        qc = nl.ndarray((P, ne), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=qc, src=consts[0:P, ne : 2 * ne])
        sigt = nl.ndarray((P, m), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sigt, src=consts[0:P, 2 * ne : 2 * ne + m])
        ones_bd = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ones_bd, src=consts[0:P, 2 * ne + m : 2 * ne + m + P])
        zst = nl.ndarray((1, P), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=zst, value=0.0)
        zmv = nl.ndarray((1, 512), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=zmv, value=0.0)
        # Tensor Engine quadrants: qs x qs tiles (32, 64 or 128), qb blocks each.
        qs = max(32, ne)
        nq = P // qs
        qb = qs // ne
        if qb > 1:
            # Block-diagonal mask within a quadrant, in compact (P, qb, ne) form.
            qmask_t = nl.ndarray((P, qs), dtype=f32, buffer=nl.sbuf)
            c0 = 2 * ne + m + P
            nisa.dma_copy(dst=qmask_t, src=consts[0:P, c0 : c0 + qs])
            qmask = qmask_t.reshape((P, qb, ne))

        def tile_kw(p0):
            if nq == 1:
                return {}
            return {"tile_position": (p0, p0), "tile_size": (qs, qs)}

        # State: X[:, g] = [B_g | W_g].
        X = nl.ndarray((P, G, 2 * ne), dtype=f32, buffer=nl.sbuf)
        for gi in range(G):
            nisa.dma_copy(dst=X[0:P, gi, 0:ne], src=A[gi, 0:P, 0:ne])
            nisa.dma_copy(dst=X[0:P, gi, ne : 2 * ne], src=W[gi, 0:P, 0:ne])
        Bv = X[0:P, 0:G, 0:ne]

        def bcast_g(t):  # (P, ...) -> (P, G, ...)
            return t.expand_dim(1).broadcast(1, G)

        def bcast_last(t, size):  # (P, G) -> (P, G, size)
            return t.expand_dim(2).broadcast(2, size)

        def zero_psum(ps2d, width):
            for c0 in range(0, width, 512):
                w = min(512, width - c0)
                nisa.nc_matmul(
                    dst=ps2d[0:P, c0 : c0 + w], stationary=zst, moving=zmv[0:1, 0:w],
                    accumulate=False,
                )

        def col():
            return nl.ndarray((P, G), dtype=f32, buffer=nl.sbuf)

        for _sw in range(sweeps):
            for r in range(m):
                nisa.tensor_scalar(
                    dst=m3[0:P, 1, 0:ne], data=qc, op0=nl.equal, operand0=float((2 * r) % m)
                )
                pic = m3[0:P, 1, 0:ne]
                ic = m3[0:P, 0, 0:ne]
                sig = sigt[0:P, r : r + 1]

                # [d, gpq] = rowsums of B o [I | Pi]
                prod_a = nl.ndarray((P, G, 2, ne), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=prod_a, data1=Bv.expand_dim(2).broadcast(2, 2),
                    data2=bcast_g(m3[0:P, 0:2, 0:ne]), op=mul,
                )
                dg = nl.ndarray((P, G, 2), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=dg, op=add, data=prod_a, axis=3)
                # Block column sums: ps_r[(b, i), g, 0, j] = d_(b, j),
                # ps_r[(b, i), g, 1, j] = B[(b, partner(j)), j].
                width = G * 2 * ne
                ps_r = nl.ndarray((P, width), dtype=f32, buffer=nl.psum)
                flat_a = prod_a.reshape((P, width))
                for c0 in range(0, width, 512):
                    w = min(512, width - c0)
                    nisa.nc_matmul(
                        dst=ps_r[0:P, c0 : c0 + w], stationary=ones_bd,
                        moving=flat_a[0:P, c0 : c0 + w], accumulate=False,
                    )
                # [dq, gqp]: pick j = partner(i) from the d row, j = i from the other.
                prod_b = nl.ndarray((P, G, 2, ne), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=prod_b, data1=ps_r.reshape((P, G, 2, ne)),
                    data2=bcast_g(m3[0:P, 1:3, 0:ne]), op=mul,
                )
                qg = nl.ndarray((P, G, 2), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=qg, op=add, data=prod_b, axis=3)

                d = dg[0:P, 0:G, 0]
                gpq = dg[0:P, 0:G, 1]
                dq = qg[0:P, 0:G, 0]
                gqp = qg[0:P, 0:G, 1]

                h = col()
                nisa.tensor_tensor(dst=h, data1=dq, data2=d, op=sub)
                gs = col()
                nisa.tensor_tensor(dst=gs, data1=gpq, data2=gqp, op=add)
                hs = col()
                nisa.tensor_scalar(dst=hs, data=h, op0=mul, operand0=sig)
                pos2 = col()
                nisa.tensor_scalar(
                    dst=pos2, data=hs, op0=nl.greater_equal, operand0=0.0, op1=mul, operand1=2.0
                )
                sg = col()
                nisa.tensor_scalar(dst=sg, data=pos2, op0=sub, operand0=1.0, op1=mul, operand1=sig)
                hh = col()
                nisa.tensor_tensor(dst=hh, data1=h, data2=h, op=mul)
                xx = col()
                nisa.tensor_tensor(dst=xx, data1=gs, data2=gs, op=mul)
                nisa.tensor_tensor(dst=xx, data1=xx, data2=hh, op=add)
                nisa.tensor_scalar(dst=xx, data=xx, op0=nl.maximum, operand0=_TINY)
                rr = col()
                nisa.activation(dst=rr, op=nl.rsqrt, data=xx)
                nisa.tensor_tensor(dst=rr, data1=rr, data2=xx, op=mul)
                den = col()
                nisa.scalar_tensor_tensor(
                    dst=den, data=h, op0=mul, operand0=-1.0, op1=nl.maximum, operand1=h
                )
                nisa.tensor_tensor(dst=den, data1=den, data2=rr, op=add)
                nisa.tensor_scalar(dst=den, data=den, op0=nl.maximum, operand0=_TINY)
                u = col()
                nisa.activation(dst=u, op=nl.rsqrt, data=den)
                t = col()
                nisa.tensor_tensor(dst=t, data1=gs, data2=u, op=mul)
                nisa.tensor_tensor(dst=t, data1=t, data2=u, op=mul)
                nisa.tensor_tensor(dst=t, data1=t, data2=sg, op=mul)
                t2 = col()
                nisa.tensor_tensor(dst=t2, data1=t, data2=t, op=mul)
                c = col()
                nisa.activation(dst=c, op=nl.rsqrt, data=t2, bias=1.0)
                s = col()
                nisa.tensor_tensor(dst=s, data1=t, data2=c, op=mul)
                w_ = col()
                nisa.activation(dst=w_, op=nl.rsqrt, data=c, bias=1.0)
                nisa.tensor_tensor(dst=w_, data1=w_, data2=w_, op=mul)
                km1 = col()
                nisa.scalar_tensor_tensor(
                    dst=km1, data=s, op0=mul, operand0=-1.0, op1=mul, operand1=s
                )
                nisa.tensor_tensor(dst=km1, data1=km1, data2=w_, op=mul)

                # K = (c - 1) o I + s o Pi, compact, all tiles.
                kc = nl.ndarray((P, G, ne), dtype=f32, buffer=nl.sbuf)
                ks = nl.ndarray((P, G, ne), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=ks, data1=bcast_g(pic), data2=bcast_last(s, ne), op=mul)
                nisa.tensor_tensor(dst=kc, data1=bcast_g(ic), data2=bcast_last(km1, ne), op=mul)
                nisa.tensor_tensor(dst=kc, data1=kc, data2=ks, op=add)

                # T = B + B^T K per block, then [B | W] = [T | W] + K^T [T | W].
                # Each product is one Tensor Engine tile of qs x qs per quadrant
                # (row/column tiling mode: the quadrants run concurrently),
                # with the compact rows of the quadrant's blocks as stationary.
                if qb > 1:
                    bbd = nl.ndarray((P, G, qb, ne), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=bbd, data1=Bv.expand_dim(2).broadcast(2, qb),
                        data2=bcast_g(qmask), op=mul,
                    )
                ps_t = nl.ndarray((P, G * ne), dtype=f32, buffer=nl.psum)
                zero_psum(ps_t, G * ne)
                for gi in range(G):
                    for q in range(nq):
                        p0 = q * qs
                        if qb > 1:
                            st = bbd[p0 : p0 + qs, gi, 0:qb, 0:ne].reshape((qs, qs))
                        else:
                            st = X[p0 : p0 + qs, gi, 0:ne]
                        nisa.nc_matmul(
                            dst=ps_t[p0 : p0 + qs, gi * ne : (gi + 1) * ne], stationary=st,
                            moving=kc[p0 : p0 + qs, gi, 0:ne], accumulate=True, **tile_kw(p0),
                        )
                nisa.tensor_tensor(dst=Bv, data1=ps_t.reshape((P, G, ne)), data2=Bv, op=add)

                if qb > 1:
                    kbd = nl.ndarray((P, G, qb, ne), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=kbd, data1=kc.expand_dim(2).broadcast(2, qb),
                        data2=bcast_g(qmask), op=mul,
                    )
                ps_x = nl.ndarray((P, G * 2 * ne), dtype=f32, buffer=nl.psum)
                zero_psum(ps_x, G * 2 * ne)
                for gi in range(G):
                    for q in range(nq):
                        p0 = q * qs
                        if qb > 1:
                            st = kbd[p0 : p0 + qs, gi, 0:qb, 0:ne].reshape((qs, qs))
                        else:
                            st = kc[p0 : p0 + qs, gi, 0:ne]
                        nisa.nc_matmul(
                            dst=ps_x[p0 : p0 + qs, gi * 2 * ne : (gi + 1) * 2 * ne],
                            stationary=st, moving=X[p0 : p0 + qs, gi, 0 : 2 * ne],
                            accumulate=True, **tile_kw(p0),
                        )
                nisa.tensor_tensor(
                    dst=X, data1=ps_x.reshape((P, G, 2 * ne)), data2=X, op=add
                )

        for gi in range(G):
            nisa.dma_copy(dst=out[0, gi, 0:P, 0:ne], src=X[0:P, gi, 0:ne])
            nisa.dma_copy(dst=out[1, gi, 0:P, 0:ne], src=X[0:P, gi, ne : 2 * ne])
        return out


def _run_compact(A: torch.Tensor, W: torch.Tensor, n: int, sweeps: int) -> torch.Tensor:
    """One launch of `jacobi_compact_kernel`. A, W: (G, P, ne)."""
    consts = compact_constants(n)
    sched = torch.zeros((sweeps, n), dtype=torch.float32)
    if _use_simulator():
        import nki as _nki

        res = _nki.simulate(jacobi_compact_kernel)(
            A.detach().cpu().contiguous().numpy(),
            W.detach().cpu().contiguous().numpy(),
            consts.numpy(),
            sched.numpy(),
        )
        return torch.from_numpy(res).to(A.device)
    from .dispatch import _launch

    dev = A.device
    return _launch(
        jacobi_compact_kernel, "jacobi_compact", A, W, consts.to(dev), sched.to(dev), device=dev
    )


# ----------------------------------------------------------------------
# One-sided (Hestenes) Jacobi, compact layout (v4).
#
# Rotate columns only: Z <- Z J and V <- V J with Z_0 = A + sigma I
# (sigma = Gershgorin bound, so Z_0 is positive semidefinite and the
# squared spectrum keeps A's ordering). The rotation of pair (p, q) uses
# the Gram entries g_pp = |z_p|^2, g_qq = |z_q|^2, g_pq = z_p . z_q, with
# the same angle formula as the two-sided step. Columns converge to
# orthogonal, Z = A V + sigma V, and the eigenvalues are the Rayleigh
# quotients w_j = v_j . z_j - sigma.
#
# The kernel keeps rows of [Z^T | V^T] on partitions, so a step is one
# update [Z^T | V^T] += K^T [Z^T | V^T] per Tensor Engine quadrant (half the
# matmuls of the two-sided T / X pair), and the partner rows z_q come from
# one matmul with the block-diagonal step permutation as stationary, shared
# by every tile of the group.
# ----------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def onesided_constants(n: int) -> torch.Tensor:
    """(P, ne - 1 + P) FP32: [sigma table | block-diagonal step codes Q]."""
    ne, b, P = block_layout(n)
    full = jacobi_constants(n)
    return torch.cat([full[:, 2 * P :], full[:, P : 2 * P]], dim=1).contiguous()


if HAS_NKI:

    @nki.jit
    def jacobi_onesided_kernel(Z, V, consts, sched):
        """Run ``sched.shape[0]`` one-sided Jacobi sweeps on G compact tiles.

        Args:
            Z, V   : (G, P, ne) FP32 compact tiles of Z^T and V^T.
            consts : from `onesided_constants`.
            sched  : (sweeps, ne, chains) FP32, shape only: sweep count, padded
                     block size and the number of independent dependency chains.

        Returns:
            out : (2, G, P, ne) FP32: Z^T and V^T after the sweeps.
        """
        G = Z.shape[0]
        P = Z.shape[1]
        ne = Z.shape[2]
        sweeps = sched.shape[0]
        nchain = sched.shape[2] if len(sched.shape) > 2 else 1
        m = ne - 1
        f32 = nl.float32
        mul = nl.multiply
        add = nl.add
        sub = nl.subtract

        out = nl.ndarray((2, G, P, ne), dtype=Z.dtype, buffer=nl.shared_hbm)

        sigt = nl.ndarray((P, m), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sigt, src=consts[0:P, 0:m])
        qfull = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=qfull, src=consts[0:P, m : m + P])

        # Z^T and V^T of tile g side by side: X[:, g] = [Z^T_g | V^T_g].
        X = nl.ndarray((P, G, 2 * ne), dtype=f32, buffer=nl.sbuf)
        for gi in range(G):
            nisa.dma_copy(dst=X[0:P, gi, 0:ne], src=Z[gi, 0:P, 0:ne])
            nisa.dma_copy(dst=X[0:P, gi, ne : 2 * ne], src=V[gi, 0:P, 0:ne])
        per_chunk = max(1, 512 // (2 * ne))

        def step(g0, g1, pibd, sig):
            """One step for tiles g0..g1-1 (one dependency chain)."""
            ng = g1 - g0
            Xc = X[0:P, g0:g1, 0 : 2 * ne]
            Zc = X[0:P, g0:g1, 0:ne]

            def col():
                return nl.ndarray((P, ng), dtype=f32, buffer=nl.sbuf)

            def bcast_last(t, size):
                return t.expand_dim(2).broadcast(2, size)

            # Partner rows: ps_m[p, g, :] = X[partner(p), g, :] (a permutation).
            ps_m = nl.ndarray((P, ng, 2 * ne), dtype=f32, buffer=nl.psum)
            for c0 in range(0, ng, per_chunk):
                c1 = min(ng, c0 + per_chunk)
                nisa.nc_matmul(
                    dst=ps_m[0:P, c0:c1, 0 : 2 * ne], stationary=pibd,
                    moving=X[0:P, g0 + c0 : g0 + c1, 0 : 2 * ne], accumulate=False,
                )
            mz = ps_m[0:P, 0:ng, 0:ne]

            # Gram entries g_pp = |z_p|^2, g_pq = z_p . z_q, g_qq = g_pp[partner].
            pr = nl.ndarray((P, ng, ne), dtype=f32, buffer=nl.sbuf)
            gpp = col()
            nisa.tensor_tensor(dst=pr, data1=Zc, data2=Zc, op=mul)
            nisa.tensor_reduce(dst=gpp, op=add, data=pr, axis=2)
            pr2 = nl.ndarray((P, ng, ne), dtype=f32, buffer=nl.sbuf)
            gpq = col()
            nisa.tensor_tensor(dst=pr2, data1=mz, data2=Zc, op=mul)
            nisa.tensor_reduce(dst=gpq, op=add, data=pr2, axis=2)
            ps_q = nl.ndarray((P, ng), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps_q, stationary=pibd, moving=gpp, accumulate=False)

            # (c, s, c - 1) per row; see trnsolver/_jacobi.rotation_csk.
            h = col()
            nisa.tensor_tensor(dst=h, data1=ps_q, data2=gpp, op=sub)
            gs = col()
            nisa.tensor_scalar(dst=gs, data=gpq, op0=mul, operand0=2.0)
            hs = col()
            nisa.tensor_scalar(dst=hs, data=h, op0=mul, operand0=sig)
            pos2 = col()
            nisa.tensor_scalar(
                dst=pos2, data=hs, op0=nl.greater_equal, operand0=0.0, op1=mul, operand1=2.0
            )
            sg = col()
            nisa.tensor_scalar(dst=sg, data=pos2, op0=sub, operand0=1.0, op1=mul, operand1=sig)
            hh = col()
            nisa.tensor_tensor(dst=hh, data1=h, data2=h, op=mul)
            xx = col()
            nisa.tensor_tensor(dst=xx, data1=gs, data2=gs, op=mul)
            nisa.tensor_tensor(dst=xx, data1=xx, data2=hh, op=add)
            nisa.tensor_scalar(dst=xx, data=xx, op0=nl.maximum, operand0=_TINY)
            rr = col()
            # sqrt(x) as x * rsqrt(x): the Scalar Engine keeps one table loaded.
            nisa.activation(dst=rr, op=nl.rsqrt, data=xx)
            nisa.tensor_tensor(dst=rr, data1=rr, data2=xx, op=mul)
            den = col()
            # |h| = max(-h, h)  (abs_max is NeuronCore-v4+ only)
            nisa.scalar_tensor_tensor(
                dst=den, data=h, op0=mul, operand0=-1.0, op1=nl.maximum, operand1=h
            )
            nisa.tensor_tensor(dst=den, data1=den, data2=rr, op=add)
            nisa.tensor_scalar(dst=den, data=den, op0=nl.maximum, operand0=_TINY)
            u = col()
            # 1 / den = rsqrt(den)^2 (rsqrt is accurate; reciprocal is ~1e-3).
            nisa.activation(dst=u, op=nl.rsqrt, data=den)
            t = col()
            nisa.tensor_tensor(dst=t, data1=gs, data2=u, op=mul)
            nisa.tensor_tensor(dst=t, data1=t, data2=u, op=mul)
            nisa.tensor_tensor(dst=t, data1=t, data2=sg, op=mul)
            t2 = col()
            nisa.tensor_tensor(dst=t2, data1=t, data2=t, op=mul)
            c = col()
            nisa.activation(dst=c, op=nl.rsqrt, data=t2, bias=1.0)
            negs = col()
            nisa.scalar_tensor_tensor(dst=negs, data=t, op0=mul, operand0=-1.0, op1=mul, operand1=c)
            w_ = col()
            nisa.activation(dst=w_, op=nl.rsqrt, data=c, bias=1.0)
            nisa.tensor_tensor(dst=w_, data1=w_, data2=w_, op=mul)
            km1 = col()
            nisa.scalar_tensor_tensor(
                dst=km1, data=negs, op0=mul, operand0=-1.0, op1=mul, operand1=negs
            )
            nisa.tensor_tensor(dst=km1, data1=km1, data2=w_, op=mul)

            # Rotation on the Vector Engine, in the X + K^T X form:
            #   (K^T X)[p, :] = (c - 1)_p X[p, :] - s_p X[partner(p), :]
            # (s_partner(p) = -s_p exactly), with X[partner(p), :] = ps_m.
            # (Folding both terms into PSUM with an identity matmul was
            # measured slower on Trn2: the Vector Engine work only moves.)
            t1 = nl.ndarray((P, ng, 2 * ne), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=t1, data1=ps_m, data2=bcast_last(negs, 2 * ne), op=mul)
            t2x = nl.ndarray((P, ng, 2 * ne), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=t2x, data1=Xc, data2=bcast_last(km1, 2 * ne), op=mul)
            nisa.tensor_tensor(dst=t1, data1=t1, data2=t2x, op=add)
            nisa.tensor_tensor(dst=Xc, data1=Xc, data2=t1, op=add)

        # Independent chains over disjoint tile ranges: one chain's (P, 1)
        # math overlaps the other's big Vector Engine and matmul work.
        bounds = [(G * k) // nchain for k in range(nchain + 1)]
        for _sw in range(sweeps):
            for r in range(m):
                pibd = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=pibd, data=qfull, op0=nl.equal, operand0=float((2 * r) % m))
                sig = sigt[0:P, r : r + 1]
                for k in range(nchain):
                    if bounds[k + 1] > bounds[k]:
                        step(bounds[k], bounds[k + 1], pibd, sig)

        for gi in range(G):
            nisa.dma_copy(dst=out[0, gi, 0:P, 0:ne], src=X[0:P, gi, 0:ne])
            nisa.dma_copy(dst=out[1, gi, 0:P, 0:ne], src=X[0:P, gi, ne : 2 * ne])
        return out


def _run_onesided(
    Z: torch.Tensor, V: torch.Tensor, n: int, sweeps: int, chains: int = 1
) -> torch.Tensor:
    """One launch of `jacobi_onesided_kernel`. Z, V: (G, P, ne)."""
    consts = onesided_constants(n)
    ne = block_layout(n)[0]
    sched = torch.zeros((sweeps, ne, chains), dtype=torch.float32)
    if _use_simulator():
        import nki as _nki

        res = _nki.simulate(jacobi_onesided_kernel)(
            Z.detach().cpu().contiguous().numpy(),
            V.detach().cpu().contiguous().numpy(),
            consts.numpy(),
            sched.numpy(),
        )
        return torch.from_numpy(res).to(Z.device)
    from .dispatch import _launch

    dev = Z.device
    return _launch(
        jacobi_onesided_kernel, "jacobi_onesided", Z, V, consts.to(dev), sched.to(dev), device=dev
    )
