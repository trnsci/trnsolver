"""
Batched one-sided Jacobi ``eigh`` on device, SBUF-resident (#48).

``trnsolver.eigh(A, method="jacobi")`` on the native ``neuron`` device runs
here for (B, n, n) batches with n <= 128.

Algorithm: one-sided (Hestenes) Jacobi on Z_0 = A + sigma I, where sigma is
the Gershgorin bound, so Z_0 is positive semidefinite and its squared
spectrum keeps A's ordering. A step rotates n/2 disjoint column pairs
(round-robin order, n - 1 steps per sweep):

    Z <- Z J,   V <- V J,    J = I + K,  K = (c - 1) o I + s o Pi_r

with the angle of pair (p, q) from the Gram entries g_pp = |z_p|^2,
g_qq = |z_q|^2, g_pq = z_p . z_q (the same formula as the two-sided
step in `trnsolver/_jacobi.py`). When the columns are orthogonal,
Z = (A + sigma I) V and the eigenvalues are the Rayleigh quotients
w_j = v_j . z_j - sigma.

Layout ("compact"): the kernel holds X = [Z^T | V^T] for G tiles in SBUF,
X[(b, i), g, :] = row i of matrix b of tile g, so b = 128 / ne matrices
share a tile's partitions (ne = n rounded up to a power of two). The state
never leaves SBUF during a launch. Per step, for all G tiles at once:

    M   = Pi_bd^T X          [Tensor Engine]  partner rows X[partner(p)];
                                              Pi_bd (block-diagonal step
                                              permutation) is the stationary
                                              operand, shared by every tile
    g_pp, g_pq               [Vector Engine]  rowsums of Z o Z, Z o M_Z
    g_qq = Pi_bd^T g_pp      [Tensor Engine]
    (c, s, c - 1)            (P, G) column math, divide-free (rsqrt only)
    X  += (c - 1) o X - s o M                 [Vector Engine]

using s_partner(p) = -s_p, which holds exactly because both rows of a pair
compute the same Gram entries in the same order. The Tensor Engine is the
partition shuffle (an exact permutation matmul); the rotation itself is
elementwise. ``X + K^T X`` keeps a small rotation's O(s^2) diagonal
shrinkage (``c - 1 = -s^2 / (1 + c)``) instead of rounding c to 1.

Under LNC=2 the kernel is traced once per physical core (``program_id``
0 and 1) and each core takes half of the group's tiles.

Measured on Trn2 (one logical NeuronCore, FP32), this formulation was the
fastest of four: two-sided Jacobi with block-diagonal 128x128 tiles,
two-sided compact with Tensor Engine quadrant tiling, one-sided with
per-tile update matmuls, and this one (see #48 for the table).

Block Jacobi (`jacobi_block_kernel`, padded n = 64 and 128). The scalar
step above is Vector Engine bound for n >= 64 (Trn2 profile, one sweep:
Vector Engine 90-94% active, Tensor Engine about 50%, DMA under 10%), since
every step rotates rows of width 2 n. The block kernel splits the columns
into blocks of 8 and pairs blocks with a circle-method block shuffle. Per
step the Tensor Engine forms each matrix's Gram matrix Z^T Z, compacts the
16 x 16 Gram block of every block pair, and applies the pair rotations as
one orthogonal matmul on [Z^T | V^T]; the Vector Engine runs one-sided
Jacobi on the 16 x 16 Gram blocks only, so its rows are 32 wide instead of
2 n. A sweep's first step runs the full round robin inside each pair (which
also covers the pairs within a block), later steps only the 8 cross-pair
steps, so a sweep is n - 1 Vector Engine steps as before but on narrow rows.
The profile becomes Vector 72-74%, Tensor 61-67%, Scalar 57-63% active, and
a sweep costs 2.7x (n = 64) and 4.7x (n = 128) less (see #48).

NKI 0.7 rules (checked by the MLIR verifier in tests/test_jacobi.py):
every SBUF tile spans partitions 0..P-1; the (P, G) column math is
float-only and divide-free (``rsqrt`` arguments clamped to its table range
[2^-87, 2^97]); ``abs_max`` (NeuronCore-v4+) is not used; every matmul
writes its own PSUM region with ``accumulate=False``.
"""

from __future__ import annotations

import functools
import os

import torch

from .dispatch import HAS_NKI, PMAX, _use_simulator

# Clamp for rsqrt arguments: the Scalar Engine rsqrt table is valid on
# [2^-87, 2^97], so the floor must sit inside it.
_TINY = 2.0**-80

# Tiles per launch and independent dependency chains per step, by padded
# block size ne. Measured on Trn2: the per-step (P, G) column math is a
# latency chain, so more tiles per launch amortise it until the Vector
# Engine saturates; extra chains overlap one chain's latency with another's
# bulk work.
_GROUP_MAX = {2: 128, 4: 128, 8: 128, 16: 128, 32: 128, 64: 64, 128: 16}
_CHAINS_MAX = {2: 8, 4: 8, 8: 8, 16: 8, 32: 8, 64: 4, 128: 2}
# Sweeps run before the first convergence check (random symmetric matrices
# converge to off < 1e-6 in this many; see _jacobi.py for the measurement).
_SWEEPS0 = {2: 3, 4: 5, 8: 6, 16: 6, 32: 7, 64: 7, 128: 8}
_TOL = 1e-6
_MAX_SWEEPS = 15

# Block Jacobi (`jacobi_block_kernel`) for these padded sizes: block size
# (a pair spans S = 2 * block columns), tiles per launch, chains and the
# sweeps before the first check. ``TRNSOLVER_JACOBI_BLOCK=0`` turns it off.
_BLOCK = {64: 8, 128: 8}
_BLOCK_GROUP_MAX = {64: 128, 128: 128}
_BLOCK_CHAINS_MAX = {64: 2, 128: 2}
_BLOCK_SWEEPS0 = {64: 7, 128: 8}
_INNER_SWEEPS = 1
# Tell the Tensor Engine the step permutation is 0/1 (half the FP32 passes).
# Read once at import and passed to the kernels as the ``onezero`` argument,
# never read from the kernel body: the NKI trace cache (on under
# torch_neuronx, persistent on disk) keys on kernel arguments but not on
# module globals, so a global would let another process's setting win
# (aws-neuron/aws-neuron-sdk#1454; tests/test_trace_cache_safety.py).
_ONEZERO = os.environ.get("TRNSOLVER_JACOBI_ONEZERO", "1").strip() != "0"

# Sweep count of the last `jacobi_eigh_device` call (benchmarks report it).
last_sweeps = 0


def block_layout(n: int) -> tuple[int, int, int]:
    """(ne, b, P): padded block size, matrices per tile, tile partitions.

    ``ne`` is n rounded up to a power of two (at least 2), so blocks tile
    the 128 partitions exactly. Padding rows and columns are zero and stay
    decoupled (their rotation angle is exactly zero), so the padded
    eigenpairs sit at indices >= n and are dropped.
    """
    if not 1 <= n <= PMAX:
        raise NotImplementedError(f"device Jacobi needs 1 <= n <= {PMAX}; got n={n}")
    ne = max(2, 1 << (n - 1).bit_length())
    b = PMAX // ne
    return ne, b, b * ne


def step_targets(ne: int) -> list[float]:
    """``v_r = 2 r mod (ne - 1)``: step r pairs the entries with code v_r."""
    m = ne - 1
    return [float((2 * r) % m) for r in range(m)]


@functools.lru_cache(maxsize=16)
def jacobi_constants(n: int) -> torch.Tensor:
    """(P, ne - 1 + P) FP32: [sigma table | block-diagonal step codes Q].

    Within a block, with m = ne - 1: Q[i, j] = (i + j) mod m for i != j < m,
    Q[i, m] = Q[m, i] = 2 i mod m, and -1 on the diagonal and off-block.
    Step r pairs exactly the entries with Q == 2 r mod m (circle-method
    round robin; m is odd, so every index has one partner per step).
    ``sigma[p, r]`` is +1 if p is the lower index of its step-r pair, else
    -1: the canonical orientation of the rotation-sign tie break.
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
        partner = (torch.eq(Q, v).float() * cols).sum(1)
        sig[:, r] = torch.where(partner > cols, 1.0, -1.0)
    return torch.cat([sig, Q], dim=1).contiguous()


def circle_block_perm(nb: int) -> list[int]:
    """Block shuffle of the block sweep: ``perm[new] = old`` position.

    Positions ``(2t, 2t + 1)`` form pair t. Position 0 stays; the other
    nb - 1 move one place round the circle-method ring (top row left to
    right, bottom row right to left), so nb - 1 steps meet every pair of
    blocks once and the shuffle has order nb - 1.
    """
    if nb < 2 or nb % 2:
        raise ValueError(f"circle_block_perm needs an even nb >= 2, got {nb}")
    k = nb // 2
    ring = [2 * t for t in range(1, k)] + [2 * t + 1 for t in reversed(range(k))]
    perm = list(range(nb))
    for i, p in enumerate(ring):
        perm[ring[(i + 1) % len(ring)]] = p
    return perm


@functools.lru_cache(maxsize=16)
def block_constants(n: int, bs: int) -> torch.Tensor:
    """(P, 4 P + 2 S) FP32 for `jacobi_block_kernel`, S = 2 bs.

    ``[sig | Q | mask | R | Pi_blk | sig_x | Q_x]``: the inner round robin
    of `jacobi_constants` at width S; ``mask[i, j] = 1`` when rows i and j
    are in the same pair (superblock); ``R[i, j] = (i mod S == j)``, the
    compact identity; ``Pi_blk[old(a), a] = 1`` the block shuffle of
    `circle_block_perm`, per matrix; and the cross schedule, whose step r
    (of bs) pairs local index i < bs with ``bs + (i + r) mod bs``:
    ``Q_x[i, j] = r`` for those partners, -1 elsewhere, and ``sig_x`` +1 on
    the first block of each pair.
    """
    ne, b, P = block_layout(n)
    S = 2 * bs
    if ne % S or ne // bs < 2:
        raise ValueError(f"block size {bs} does not tile ne={ne}")
    inner = jacobi_constants(S)
    idx = torch.arange(P)
    mask = (idx[:, None] // S == idx[None, :] // S).float()
    R = (idx[:, None] % S == torch.arange(S)[None, :]).float()
    perm = circle_block_perm(ne // bs)
    pi = torch.zeros((P, P))
    for a in range(P):
        mat, col = divmod(a, ne)
        old = perm[col // bs] * bs + col % bs
        pi[mat * ne + old, a] = 1.0
    # Cross steps: step r pairs i < bs with bs + (i + r) mod bs in each pair.
    loc = idx % S
    qc = torch.full((P, P), -1.0)
    for r in range(bs):
        p_loc = torch.where(loc < bs, bs + (loc + r) % bs, (loc - bs - r) % bs)
        qc[idx, idx - loc + p_loc] = float(r)
    sigc = torch.where(loc < bs, 1.0, -1.0).unsqueeze(1)
    return torch.cat([inner, mask, R, pi, sigc, qc], dim=1).contiguous()


def _use_block(ne: int) -> bool:
    """True iff padded size ``ne`` runs the block kernel."""
    if ne not in _BLOCK:
        return False
    return os.environ.get("TRNSOLVER_JACOBI_BLOCK", "1").strip() != "0"


def pack_compact(X: torch.Tensor, n: int) -> torch.Tensor:
    """(B, n, n) -> (T, P, ne): zero-pad n to ne, pad B with identities."""
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
    ne = block_layout(n)[0]
    return tiles.reshape(-1, ne, ne)[:Bsz]


if HAS_NKI:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    def _step(X, g0, g1, w, pibd, sig, per_tile, onezero, delta=True):
        """One round-robin step on tiles g0..g1-1 of the state X (one chain).

        X is (P, G, 2 w) FP32 in SBUF: per tile, ``[Y^T | U^T]`` for the
        compact w-blocks (``Y`` the columns being orthogonalised, ``U`` the
        accumulated rotations). ``pibd`` is the (P, P) block-diagonal step
        permutation, ``sig`` the (P, 1) canonical orientation.
        ``onezero`` tells the Tensor Engine that ``pibd`` is 0/1 (half the
        passes for an FP32 stationary). ``delta=False`` applies the rotation
        as ``c o X - s o M`` (three Vector Engine passes instead of four); a
        rounded c only scales the pair's two rows by the same factor, so it
        suits the block kernel, whose driver renormalises the columns.
        """
        P = X.shape[0]
        f32 = nl.float32
        mul = nl.multiply
        add = nl.add
        sub = nl.subtract
        ng = g1 - g0
        Xc = X[0:P, g0:g1, 0 : 2 * w]
        Zc = X[0:P, g0:g1, 0:w]
        per_chunk = max(1, 512 // (2 * w))

        def col():
            return nl.ndarray((P, ng), dtype=f32, buffer=nl.sbuf)

        def bcast_last(t, size):
            return t.expand_dim(2).broadcast(2, size)

        # Partner rows: ps_m[p, g, :] = X[partner(p), g, :] (a permutation).
        # Each matmul writes a whole 2 KiB PSUM bank (512 FP32 columns) or
        # the bank-aligned tail, never a sub-bank column slice next to
        # another matmul's output (the accumulate=False slice hazard).
        ps_m = nl.ndarray((P, ng, 2 * w), dtype=f32, buffer=nl.psum)
        for c0 in range(0, ng, per_chunk):
            c1 = min(ng, c0 + per_chunk)
            nisa.nc_matmul(
                dst=ps_m[0:P, c0:c1, 0 : 2 * w],
                stationary=pibd,
                moving=X[0:P, g0 + c0 : g0 + c1, 0 : 2 * w],
                is_stationary_onezero=onezero,
                accumulate=False,
            )
        mz = ps_m[0:P, 0:ng, 0:w]

        # Gram entries g_pp = |z_p|^2, g_pq = z_p . z_q, g_qq = g_pp[partner].
        gpp = col()
        gpq = col()
        if per_tile:
            # Large blocks: fused multiply + free-axis reduce per tile
            # (one Vector Engine pass instead of two).
            for j in range(ng):
                junk = nl.ndarray((P, w), dtype=f32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(
                    dst=junk,
                    data=Zc[0:P, j, 0:w],
                    op0=mul,
                    operand0=1.0,
                    op1=mul,
                    operand1=Zc[0:P, j, 0:w],
                    reduce_op=add,
                    reduce_res=gpp[0:P, j : j + 1],
                )
                junk2 = nl.ndarray((P, w), dtype=f32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(
                    dst=junk2,
                    data=Zc[0:P, j, 0:w],
                    op0=mul,
                    operand0=1.0,
                    op1=mul,
                    operand1=mz[0:P, j, 0:w],
                    reduce_op=add,
                    reduce_res=gpq[0:P, j : j + 1],
                )
        else:
            # Small blocks: one instruction over all tiles (many tiles).
            pr = nl.ndarray((P, ng, w), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pr, data1=Zc, data2=Zc, op=mul)
            nisa.tensor_reduce(dst=gpp, op=add, data=pr, axis=2)
            pr2 = nl.ndarray((P, ng, w), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pr2, data1=mz, data2=Zc, op=mul)
            nisa.tensor_reduce(dst=gpq, op=add, data=pr2, axis=2)
        ps_q = nl.ndarray((P, ng), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=ps_q, stationary=pibd, moving=gpp, is_stationary_onezero=onezero, accumulate=False
        )

        # (c, s, c - 1) per row; see trnsolver/_jacobi.rotation_csk.
        h = col()
        nisa.tensor_tensor(dst=h, data1=ps_q, data2=gpp, op=sub)
        gs = col()
        nisa.tensor_scalar(dst=gs, data=gpq, op0=mul, operand0=2.0)

        # (Placing this SBUF-only chain on the GpSimd engine failed at
        # run time on Trn2; it stays on the engines the compiler picks.)
        def ts(dst, data, op0, operand0, op1=None, operand1=None):
            nisa.tensor_scalar(
                dst=dst, data=data, op0=op0, operand0=operand0, op1=op1, operand1=operand1
            )

        def tt(dst, a, b_, op):
            nisa.tensor_tensor(dst=dst, data1=a, data2=b_, op=op)

        hs = col()
        ts(hs, h, mul, sig)
        pos2 = col()
        ts(pos2, hs, nl.greater_equal, 0.0, mul, 2.0)
        sg = col()
        ts(sg, pos2, sub, 1.0, mul, sig)
        hh = col()
        tt(hh, h, h, mul)
        xx = col()
        tt(xx, gs, gs, mul)
        tt(xx, xx, hh, add)
        ts(xx, xx, nl.maximum, _TINY)
        rr = col()
        # sqrt(x) as x * rsqrt(x): the Scalar Engine keeps one table loaded.
        nisa.activation(dst=rr, op=nl.rsqrt, data=xx)
        tt(rr, rr, xx, mul)
        den = col()
        # |h| = max(-h, h)  (abs_max is NeuronCore-v4+ only)
        ts(den, h, mul, -1.0)
        tt(den, den, h, nl.maximum)
        tt(den, den, rr, add)
        ts(den, den, nl.maximum, _TINY)
        u = col()
        # 1 / den = rsqrt(den)^2 (den is clamped into the rsqrt table range;
        # ScalarE reciprocal is ~1.5e-6 relative in range but silently
        # returns 1e7 or 0 outside about |x| in [2^-42, 2^41]).
        nisa.activation(dst=u, op=nl.rsqrt, data=den)
        t = col()
        tt(t, gs, u, mul)
        tt(t, t, u, mul)
        tt(t, t, sg, mul)
        t2 = col()
        tt(t2, t, t, mul)
        c = col()
        nisa.activation(dst=c, op=nl.rsqrt, data=t2, bias=1.0)
        negs = col()
        ts(negs, t, mul, -1.0)
        tt(negs, negs, c, mul)
        w_ = col()
        nisa.activation(dst=w_, op=nl.rsqrt, data=c, bias=1.0)
        tt(w_, w_, w_, mul)
        km1 = col()
        tt(km1, negs, negs, mul)
        ts(km1, km1, mul, -1.0)
        tt(km1, km1, w_, mul)

        # Rotation on the Vector Engine, in the X + K^T X form:
        #   (K^T X)[p, :] = (c - 1)_p X[p, :] - s_p X[partner(p), :]
        # (s_partner(p) = -s_p exactly), with X[partner(p), :] = ps_m.
        # (Folding both terms into PSUM with an identity matmul was
        # measured slower on Trn2: the Vector Engine work only moves.)
        t1 = nl.ndarray((P, ng, 2 * w), dtype=f32, buffer=nl.sbuf)
        if per_tile:
            # -s o M on the Scalar Engine, (c - 1) o X + that on the Vector
            # Engine, per tile (per-partition scalars, one pass each).
            for j in range(ng):
                nisa.activation(
                    dst=t1[0:P, j, 0 : 2 * w],
                    op=nl.copy,
                    data=ps_m[0:P, j, 0 : 2 * w],
                    scale=negs[0:P, j : j + 1],
                )
                nisa.scalar_tensor_tensor(
                    dst=t1[0:P, j, 0 : 2 * w],
                    data=Xc[0:P, j, 0 : 2 * w],
                    op0=mul,
                    operand0=km1[0:P, j : j + 1],
                    op1=add,
                    operand1=t1[0:P, j, 0 : 2 * w],
                )
        elif not delta:
            cc = col()
            ts(cc, km1, add, 1.0)
            nisa.tensor_tensor(dst=t1, data1=ps_m, data2=bcast_last(negs, 2 * w), op=mul)
            t2x = nl.ndarray((P, ng, 2 * w), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=t2x, data1=Xc, data2=bcast_last(cc, 2 * w), op=mul)
            nisa.tensor_tensor(dst=Xc, data1=t1, data2=t2x, op=add)
            return
        else:
            nisa.tensor_tensor(dst=t1, data1=ps_m, data2=bcast_last(negs, 2 * w), op=mul)
            t2x = nl.ndarray((P, ng, 2 * w), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=t2x, data1=Xc, data2=bcast_last(km1, 2 * w), op=mul)
            nisa.tensor_tensor(dst=t1, data1=t1, data2=t2x, op=add)
        nisa.tensor_tensor(dst=Xc, data1=Xc, data2=t1, op=add)

    def _split_programs(G):
        """(tiles per program, first tile) for the SPMD programs (LNC=2 cores).

        Program k owns tiles [k * G / nprog, (k + 1) * G / nprog). A group too
        small to split runs whole on every program (identical results,
        identical writes).
        """
        nprog = int(nl.num_programs(0)) if nl.program_ndim() > 0 else 1
        if nprog > 1 and G % nprog == 0:
            return G // nprog, nl.program_id(0) * (G // nprog)
        return G, 0

    @nki.jit
    def jacobi_sweeps_kernel(Z, V, consts, sched, onezero):
        """Run ``sched.shape[0]`` one-sided Jacobi sweeps on G compact tiles.

        Args:
            Z, V   : (G, P, ne) FP32 compact tiles of Z^T and V^T.
            consts : from `jacobi_constants`.
            sched  : (sweeps, ne, chains) FP32, shape only: sweep count, padded
                     block size and the number of independent dependency chains.
            onezero: int, nonzero to mark the step permutation 0/1 for the
                     Tensor Engine (``TRNSOLVER_JACOBI_ONEZERO``, `_ONEZERO`).

        Returns:
            out : (2, G, P, ne) FP32: Z^T and V^T after the sweeps.
        """
        P = Z.shape[1]
        ne = Z.shape[2]
        sweeps = sched.shape[0]
        nchain = sched.shape[2] if len(sched.shape) > 2 else 1
        m = ne - 1
        f32 = nl.float32

        out = nl.ndarray((2, Z.shape[0], P, ne), dtype=Z.dtype, buffer=nl.shared_hbm)

        sigt = nl.ndarray((P, m), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sigt, src=consts[0:P, 0:m])
        qfull = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=qfull, src=consts[0:P, m : m + P])

        G, base = _split_programs(Z.shape[0])

        # Z^T and V^T of tile g side by side: X[:, g] = [Z^T_g | V^T_g].
        X = nl.ndarray((P, G, 2 * ne), dtype=f32, buffer=nl.sbuf)
        for gi in range(G):
            nisa.dma_copy(dst=X[0:P, gi, 0:ne], src=Z[base + gi, 0:P, 0:ne])
            nisa.dma_copy(dst=X[0:P, gi, ne : 2 * ne], src=V[base + gi, 0:P, 0:ne])
        # Per-tile instructions pay off once a tile row is long (ne >= 64);
        # below that the group has many tiles and batched instructions win.
        per_tile = ne >= 64

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
                        _step(X, bounds[k], bounds[k + 1], ne, pibd, sig, per_tile, onezero != 0)

        for gi in range(G):
            nisa.dma_copy(dst=out[0, base + gi, 0:P, 0:ne], src=X[0:P, gi, 0:ne])
            nisa.dma_copy(dst=out[1, base + gi, 0:P, 0:ne], src=X[0:P, gi, ne : 2 * ne])
        return out

    @nki.jit
    def jacobi_block_kernel(Z, V, consts, sched, onezero):
        """One block one-sided Jacobi sweep on G compact tiles (n >= 64).

        Args:
            Z, V   : (G, P, ne) FP32 compact tiles of Z^T and V^T.
            consts : from `block_constants`.
            sched  : (inner_sweeps, S, chains) FP32, shape only: inner sweeps
                     per block pair, pair width S = 2 * block size, chains.
            onezero: as for `jacobi_sweeps_kernel`.

        Returns:
            out : (2, G, P, ne) FP32: Z^T and V^T after the sweep.

        The ne columns of each matrix form ne / (S / 2) blocks, and superblock
        t (columns [t S, (t + 1) S)) pairs blocks 2t and 2t + 1. An outer
        step, per tile:

            Z      = (Z^T)^T                       [Tensor Engine transpose]
            G      = Z^T Z,  masked to the S x S   [Tensor Engine]
                     superblock diagonal
            Gc     = G_masked R                    [Tensor Engine] compact
                     (R stacks identities), so Gc holds each pair's S x S Gram
            U      = one-sided Jacobi on Gc        [Vector Engine, `_step` at
                     (inner_sweeps sweeps; the      width S on all tiles]
                     sweep's first step runs the
                     full round robin, the others
                     only the cross pairs)
            W      = U_blockdiag Pi_blk            [Tensor Engine transpose
                                                    into the block shuffle]
            X      = W^T X                         [Tensor Engine]

        ``Pi_blk`` is the circle-method block shuffle: it moves the blocks so
        that the next step pairs new partners, and it has order
        ne / (S / 2) - 1, so after the sweep's last step every column is
        back in place. The Vector Engine works on rows of width 2 S instead
        of 2 ne, and the Tensor Engine carries the O(ne^2) update.
        """
        P = Z.shape[1]
        ne = Z.shape[2]
        s_in = sched.shape[0]
        S = sched.shape[1]
        nchain = sched.shape[2]
        m = S - 1
        nsteps = ne // (S // 2) - 1
        if len(sched.shape) > 3:  # tests: a partial sweep
            nsteps = sched.shape[3]
        f32 = nl.float32
        mul = nl.multiply

        out = nl.ndarray((2, Z.shape[0], P, ne), dtype=Z.dtype, buffer=nl.shared_hbm)

        # consts = [sig (m) | Q (P) | mask (P) | R (S) | Pi_blk (P) | sig_x (1) | Q_x (P)]
        sigt = nl.ndarray((P, m), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sigt, src=consts[0:P, 0:m])
        qfull = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=qfull, src=consts[0:P, m : m + P])
        mask = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=mask, src=consts[0:P, m + P : m + 2 * P])
        rid = nl.ndarray((P, S), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=rid, src=consts[0:P, m + 2 * P : m + 2 * P + S])
        pblk = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=pblk, src=consts[0:P, m + 2 * P + S : m + 3 * P + S])
        c0 = m + 3 * P + S
        sigx = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sigx, src=consts[0:P, c0 : c0 + 1])
        qx = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=qx, src=consts[0:P, c0 + 1 : c0 + 1 + P])
        # The inner step permutations, built once per launch: the m steps of
        # the full round robin, then the bs cross steps.
        bsz = S // 2
        pib = nl.ndarray((P, m + bsz, P), dtype=f32, buffer=nl.sbuf)
        for r in range(m):
            nisa.tensor_scalar(
                dst=pib[0:P, r, 0:P], data=qfull, op0=nl.equal, operand0=float((2 * r) % m)
            )
        for r in range(bsz):
            nisa.tensor_scalar(dst=pib[0:P, m + r, 0:P], data=qx, op0=nl.equal, operand0=float(r))

        G, base = _split_programs(Z.shape[0])

        X = nl.ndarray((P, G, 2 * ne), dtype=f32, buffer=nl.sbuf)
        for gi in range(G):
            nisa.dma_copy(dst=X[0:P, gi, 0:ne], src=Z[base + gi, 0:P, 0:ne])
            nisa.dma_copy(dst=X[0:P, gi, ne : 2 * ne], src=V[base + gi, 0:P, 0:ne])
        # Inner state per tile: [Gc | U^T] for the P / S compact pairs.
        Y = nl.ndarray((P, G, 2 * S), dtype=f32, buffer=nl.sbuf)
        # Block-diagonal U^T per tile, the stationary of the W transpose.
        E = nl.ndarray((P, G, P), dtype=f32, buffer=nl.sbuf)

        def gram(g):
            """Y[:, g] = [Gc | I]: the pairs' Gram blocks, rotations reset."""
            ps_t = nl.ndarray((ne, P), dtype=f32, buffer=nl.psum)
            nisa.nc_transpose(dst=ps_t, data=X[0:P, g, 0:ne], engine=nisa.engine.tensor)
            zm = nl.ndarray((ne, P), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=zm, op=nl.copy, data=ps_t)
            ps_g = nl.ndarray((P, P), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps_g, stationary=zm, moving=zm, accumulate=False)
            gm = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=gm, data1=ps_g, data2=mask, op=mul)
            ps_c = nl.ndarray((P, S), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=ps_c, stationary=gm, moving=rid, is_moving_onezero=True, accumulate=False
            )
            nisa.activation(dst=Y[0:P, g, 0:S], op=nl.copy, data=ps_c)
            nisa.activation(dst=Y[0:P, g, S : 2 * S], op=nl.copy, data=rid)

        def update(g0, g1):
            """X <- W^T X for tiles g0..g1-1, W = U_blockdiag Pi_blk."""
            ng = g1 - g0
            nisa.tensor_tensor(
                dst=E[0:P, g0:g1, 0:P].reshape_dim(2, (P // S, S)),
                data1=Y[0:P, g0:g1, S : 2 * S].expand_dim(2).broadcast(2, P // S),
                data2=mask.reshape_dim(1, (P // S, S)).expand_dim(1).broadcast(1, ng),
                op=mul,
            )
            for g in range(g0, g1):
                ps_w = nl.ndarray((P, P), dtype=f32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=ps_w,
                    stationary=E[0:P, g, 0:P],
                    moving=pblk,
                    is_moving_onezero=True,
                    accumulate=False,
                )
                wsb = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
                nisa.activation(dst=wsb, op=nl.copy, data=ps_w)
                ps_x = nl.ndarray((P, 2 * ne), dtype=f32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=ps_x, stationary=wsb, moving=X[0:P, g, 0 : 2 * ne], accumulate=False
                )
                nisa.activation(dst=X[0:P, g, 0 : 2 * ne], op=nl.copy, data=ps_x)

        bounds = [(G * k) // nchain for k in range(nchain + 1)]
        chains = [(bounds[k], bounds[k + 1]) for k in range(nchain) if bounds[k + 1] > bounds[k]]
        for st in range(nsteps):
            for g0, g1 in chains:
                for g in range(g0, g1):
                    gram(g)
            # The first step's pairs run the full round robin (it also covers
            # the pairs inside each block); later steps only rotate the
            # bs x bs cross pairs, so a sweep rotates each pair once.
            if st == 0:
                plan = [(r, sigt[0:P, r : r + 1]) for r in range(m)]
            else:
                plan = [(m + r, sigx) for r in range(bsz)]
            for _sw in range(s_in):
                for r, sig in plan:
                    for g0, g1 in chains:
                        _step(Y, g0, g1, S, pib[0:P, r, 0:P], sig, False, onezero != 0, False)
            for g0, g1 in chains:
                update(g0, g1)

        for gi in range(G):
            nisa.dma_copy(dst=out[0, base + gi, 0:P, 0:ne], src=X[0:P, gi, 0:ne])
            nisa.dma_copy(dst=out[1, base + gi, 0:P, 0:ne], src=X[0:P, gi, ne : 2 * ne])
        return out


_DEVICE_CONSTS: dict = {}


def _consts_on(
    n: int, chains: int, device: torch.device, sweeps: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """(consts, sched) for one launch, cached per device (no copy per launch)."""
    ne = block_layout(n)[0]
    key = (ne, chains, str(device), sweeps)
    if key not in _DEVICE_CONSTS:
        consts = jacobi_constants(n).to(device)
        sched = torch.zeros((sweeps, ne, chains), dtype=torch.float32, device=device)
        _DEVICE_CONSTS[key] = (consts, sched)
    return _DEVICE_CONSTS[key]


def _run_kernel(
    Z: torch.Tensor, V: torch.Tensor, n: int, chains: int, sweeps: int = 1
) -> torch.Tensor:
    """One launch (``sweeps`` sweeps) on a group: simulator or device. Z, V: (G, P, ne)."""
    if _use_simulator():
        import nki as _nki

        consts, sched = _consts_on(n, chains, torch.device("cpu"), sweeps)
        res = _nki.simulate(jacobi_sweeps_kernel)(
            Z.detach().cpu().contiguous().numpy(),
            V.detach().cpu().contiguous().numpy(),
            consts.numpy(),
            sched.numpy(),
            int(_ONEZERO),
        )
        return torch.from_numpy(res).to(Z.device)
    from .dispatch import _launch

    consts, sched = _consts_on(n, chains, Z.device, sweeps)
    return _launch(
        jacobi_sweeps_kernel,
        "jacobi_sweeps",
        Z,
        V,
        consts,
        sched,
        device=Z.device,
        static=(int(_ONEZERO),),
    )


def _block_consts_on(n: int, chains: int, device: torch.device):
    ne = block_layout(n)[0]
    bs = _BLOCK[ne]
    key = ("block", ne, bs, chains, str(device))
    if key not in _DEVICE_CONSTS:
        consts = block_constants(n, bs).to(device)
        sched = torch.zeros((_INNER_SWEEPS, 2 * bs, chains), dtype=torch.float32, device=device)
        _DEVICE_CONSTS[key] = (consts, sched)
    return _DEVICE_CONSTS[key]


def _run_block_kernel(Z: torch.Tensor, V: torch.Tensor, n: int, chains: int) -> torch.Tensor:
    """One block-sweep launch on a group: simulator or device. Z, V: (G, P, ne)."""
    if _use_simulator():
        import nki as _nki

        consts, sched = _block_consts_on(n, chains, torch.device("cpu"))
        res = _nki.simulate(jacobi_block_kernel)(
            Z.detach().cpu().contiguous().numpy(),
            V.detach().cpu().contiguous().numpy(),
            consts.numpy(),
            sched.numpy(),
            int(_ONEZERO),
        )
        return torch.from_numpy(res).to(Z.device)
    from .dispatch import _launch

    consts, sched = _block_consts_on(n, chains, Z.device)
    return _launch(
        jacobi_block_kernel,
        "jacobi_block",
        Z,
        V,
        consts,
        sched,
        device=Z.device,
        static=(int(_ONEZERO),),
    )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def group_plan(tiles: int, ne: int) -> list[tuple[int, int, int]]:
    """Split ``tiles`` into launches: ``[(start, count, padded), ...]``.

    Full groups of ``_GROUP_MAX[ne]`` (``_BLOCK_GROUP_MAX`` for the block
    kernel; ``TRNSOLVER_JACOBI_GROUP`` overrides), then one tail rounded up
    to a power of two, so each ne compiles at most log2(group) + 1 shapes
    whatever the batch size.
    """
    table = _BLOCK_GROUP_MAX if _use_block(ne) else _GROUP_MAX
    gmax = _env_int("TRNSOLVER_JACOBI_GROUP", table[ne])
    gmax = max(1, 1 << (max(gmax, 1).bit_length() - 1))
    out = []
    start = 0
    while start < tiles:
        count = min(gmax, tiles - start)
        padded = 1 << (count - 1).bit_length()
        out.append((start, count, padded))
        start += count
    return out


def gram_off(Zt: torch.Tensor) -> torch.Tensor:
    """Relative off-diagonal norm of the Gram matrix Z^T Z per matrix (Zt = Z^T).

    Summed directly over the off-diagonal entries (``|G|^2 - |diag G|^2``
    cancels to a sqrt(eps) floor).
    """
    G = torch.bmm(Zt, Zt.transpose(1, 2))
    ne = G.shape[-1]
    mask = 1.0 - torch.eye(ne, dtype=G.dtype, device=G.device)
    off2 = (G * G * mask).sum(dim=(1, 2))
    d = torch.diagonal(G, dim1=1, dim2=2)
    return torch.sqrt(off2) / torch.sqrt(torch.clamp((d * d).sum(-1), min=1e-30))


def jacobi_eigh_device(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched symmetric eigh by the one-sided Jacobi kernel; see `_solve`."""
    global last_sweeps
    w, V, last_sweeps = _solve(A)
    return w, V


def _solve(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Batched symmetric eigh by the one-sided Jacobi kernel.

    Args:
        A : (B, n, n) FP32, 1 <= n <= 128, any device (the lower triangle is
            used, as in ``torch.linalg.eigh``).

    Returns:
        (w (B, n) ascending, V (B, n, n)) on A's device, and the sweep
        count. The work stays on A's NeuronCore when A is on one.

    Runs ``_SWEEPS0[ne]`` sweeps (one launch per sweep and group of tiles),
    then one more sweep at a time while any matrix has a Gram off-diagonal
    norm above 1e-6, up to 15 sweeps. Padded sizes in ``_BLOCK`` (ne = 64,
    128) run `jacobi_block_kernel` instead, with ``_BLOCK_SWEEPS0[ne]``
    sweeps first. Preparation (symmetrise, scale, shift, pack) and the final
    Rayleigh quotients and sort run on A's device, so a host input costs
    one copy in and one copy out.

    The eigenvectors are returned with unit columns and the eigenvalues as
    the Rayleigh quotients ``v . z / v . v - sigma``: rounding in the
    accumulated rotations (the block kernel's FP32 products of rotations
    especially) shows up mostly as column-norm drift, which this removes.
    """
    Bsz, n = A.shape[0], A.shape[-1]
    if n == 1:
        return A.reshape(Bsz, 1).clone(), torch.ones_like(A), 0
    ne, b, P = block_layout(n)
    if _use_simulator():
        work = torch.device("cpu")
    elif A.device.type == "neuron":
        # Stay on A's NeuronCore (``neuron:k``): the multi-core split in
        # `jacobi_eigh_multicore` hands each core its own chunk.
        work = A.device
    else:
        from .dispatch import _work_device

        work = _work_device()

    # Symmetrise from the lower triangle (torch.linalg.eigh semantics).
    lower = torch.ones(n, n, dtype=torch.bool, device=A.device).tril()
    X = torch.where(lower, A, A.transpose(1, 2))
    # Exact power-of-two scaling so the infinity norm (a Gershgorin bound on
    # |lambda|, and >= max |a_ij|) lands in (1/2, 1]; sigma = that bound.
    nrm = torch.clamp(torch.linalg.matrix_norm(X, ord=float("inf")), min=1e-30)
    scale = torch.exp2(-torch.ceil(torch.log2(nrm)))
    sigma = nrm * scale
    X = X * scale[:, None, None]
    X.diagonal(dim1=1, dim2=2).add_(sigma[:, None])  # Z_0 = A + sigma I (symmetric)
    Z = pack_compact(X, n)
    # V^T starts as identity tiles, built on the work device (no copy).
    T = Z.shape[0]
    Vt = torch.eye(ne, dtype=X.dtype, device=work).repeat(T * b, 1, 1).reshape(T, P, ne)
    plan = group_plan(T, ne)
    block = _use_block(ne)
    run_kernel = _run_block_kernel if block else _run_kernel
    chains_max = _env_int(
        "TRNSOLVER_JACOBI_CHAINS", (_BLOCK_CHAINS_MAX if block else _CHAINS_MAX)[ne]
    )
    sweeps0 = (_BLOCK_SWEEPS0 if block else _SWEEPS0)[ne]

    def pad_tiles(t, count, padded):
        if padded == count:
            return t
        fill = torch.eye(ne, dtype=t.dtype, device=t.device).repeat(padded - count, b, 1)
        return torch.cat([t, fill.reshape(padded - count, P, ne)], dim=0)

    # Per group: (Z, V) on the work device, padded to the launch shape.
    Zw = Z.to(work)
    groups = [
        (
            pad_tiles(Zw[s : s + c], c, p),
            pad_tiles(Vt[s : s + c], c, p),
            min(chains_max, p),
        )
        for s, c, p in plan
    ]

    def sweep(count=1):
        for k, (zg, vg, ch) in enumerate(groups):
            res = run_kernel(zg, vg, n, ch) if count == 1 else run_kernel(zg, vg, n, ch, count)
            groups[k] = (res[0], res[1], ch)

    def gathered(i):
        parts = [g[i][:c] for g, (_, c, _) in zip(groups, plan, strict=True)]
        return parts[0] if len(parts) == 1 else torch.cat(parts, 0)

    # TRNSOLVER_JACOBI_SWEEPS_PER_LAUNCH=k (scalar kernel only) runs the
    # first sweeps k per launch: fewer launches and HBM round trips of the
    # state, more kernel shapes to compile.
    per_launch = 1 if block else max(1, _env_int("TRNSOLVER_JACOBI_SWEEPS_PER_LAUNCH", 1))
    done = 0
    while done < sweeps0:
        k = min(per_launch, sweeps0 - done)
        sweep(k)
        done += k
    while done < _MAX_SWEEPS:
        if float(gram_off(unpack_compact(gathered(0), n, Bsz)).max()) < _TOL:
            break
        sweep()
        done += 1

    Zt = unpack_compact(gathered(0).to(A.device), n, Bsz)[:, :n, :n]
    Vt = unpack_compact(gathered(1).to(A.device), n, Bsz)[:, :n, :n]
    vv = (Vt * Vt).sum(-1)
    w = ((Zt * Vt).sum(-1) / vv - sigma[:, None]) / scale[:, None]
    Vt = Vt * torch.rsqrt(vv).unsqueeze(-1)
    w, order = torch.sort(w, dim=-1)
    # Reorder whole rows of V^T (one row gather) rather than an elementwise
    # torch.gather on V: on the neuron device that is ~2 ms instead of
    # ~100 ms at B = 1024, n = 128.
    rows = (order + torch.arange(Bsz, device=order.device).unsqueeze(1) * n).reshape(-1)
    Vt = Vt.reshape(Bsz * n, n).index_select(0, rows).reshape(Bsz, n, n)
    return w, Vt.transpose(1, 2).contiguous(), done


_POOL = None
# (n, chunk sizes, cores) of the multi-core splits that have run once.
_WARM: set = set()


def _pool(workers: int):
    global _POOL
    if _POOL is None or _POOL._max_workers < workers:
        from concurrent.futures import ThreadPoolExecutor

        _POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="trnsolver-eigh")
    return _POOL


def split_batch(Bsz: int, n: int, parts: int) -> list[tuple[int, int]]:
    """Split ``Bsz`` matrices into at most ``parts`` chunks ``[(start, stop)]``.

    Chunks are whole tiles (``128 / ne`` matrices) and as even as possible,
    so every core runs the same launch shapes.
    """
    b = block_layout(n)[1]
    tiles = -(-Bsz // b)
    parts = max(1, min(parts, tiles))
    per = -(-tiles // parts)
    out = []
    start = 0
    while start < Bsz:
        stop = min(Bsz, start + per * b)
        out.append((start, stop))
        start = stop
    return out


def jacobi_eigh_multicore(
    A: torch.Tensor, devices: list[torch.device]
) -> tuple[torch.Tensor, torch.Tensor]:
    """`jacobi_eigh_device` with the batch split across NeuronCores.

    Chunk ``i`` is solved on ``devices[i]`` by its own host thread
    (launches, convergence checks and the final sort run per core). A chunk
    whose core is A's own stays where it is; the others go through the
    host, copied out of A's device before any kernel is queued there, and
    their results come back through the host. The matrices are
    independent, so the result equals the single-core one up to each chunk
    converging on its own sweep count.

    Measured on Trn2 (#48), the host staging is needed: with direct
    ``neuron:0 -> neuron:k`` copies the split ran no faster than one core
    (B = 1024, n = 128: 284, 290 and 302 ms on 1, 2 and 4 cores), and a
    copy out of a busy core waits behind its queued kernels.
    """
    global last_sweeps
    chunks = split_batch(A.shape[0], A.shape[-1], len(devices))
    local = [devices[i] == A.device for i in range(len(chunks))]
    staged = [None if local[i] else A[s:e].cpu() for i, (s, e) in enumerate(chunks)]

    def solve(i):
        s, e = chunks[i]
        x = A[s:e] if local[i] else staged[i].to(devices[i])
        w, V, sweeps = _solve(x)
        if not local[i]:
            w, V = w.cpu().to(A.device), V.cpu().to(A.device)
        return w, V, sweeps

    # NKI tracing is not thread-safe, and it runs on every dispatch-cache
    # miss, including for kernels behind some aten ops (Trn2: concurrent
    # first calls failed inside the tracer, or crashed). So the first call
    # for a set of chunk shapes and cores runs the chunks one after another,
    # and later calls run them concurrently.
    key = (A.shape[-1], tuple(e - s for s, e in chunks), tuple(str(d) for d in devices))
    if len(chunks) == 1 or key not in _WARM:
        results = [solve(i) for i in range(len(chunks))]
        _WARM.add(key)
    else:
        results = list(_pool(len(chunks)).map(solve, range(len(chunks))))
    last_sweeps = max(r[2] for r in results)
    if len(results) == 1:
        return results[0][0], results[0][1]
    return torch.cat([r[0] for r in results]), torch.cat([r[1] for r in results])
