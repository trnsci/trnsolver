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

Measured on Trn2 (one NeuronCore, FP32), this formulation was the fastest of
four: two-sided Jacobi with block-diagonal 128x128 tiles, two-sided compact
with Tensor Engine quadrant tiling, one-sided with per-tile update matmuls,
and this one (see #48 for the table).

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

    @nki.jit
    def jacobi_sweeps_kernel(Z, V, consts, sched):
        """Run ``sched.shape[0]`` one-sided Jacobi sweeps on G compact tiles.

        Args:
            Z, V   : (G, P, ne) FP32 compact tiles of Z^T and V^T.
            consts : from `jacobi_constants`.
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

        out = nl.ndarray((2, Z.shape[0], P, ne), dtype=Z.dtype, buffer=nl.shared_hbm)

        sigt = nl.ndarray((P, m), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sigt, src=consts[0:P, 0:m])
        qfull = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=qfull, src=consts[0:P, m : m + P])

        # Split the tiles across the SPMD programs (the physical cores of a
        # logical NeuronCore under LNC=2): program k owns tiles
        # [k * G / nprog, (k + 1) * G / nprog). A group too small to split
        # runs whole on every program (identical results, identical writes).
        nprog = int(nl.num_programs(0)) if nl.program_ndim() > 0 else 1
        if nprog > 1 and G % nprog == 0:
            G = G // nprog
            base = nl.program_id(0) * G
        else:
            base = 0

        # Z^T and V^T of tile g side by side: X[:, g] = [Z^T_g | V^T_g].
        X = nl.ndarray((P, G, 2 * ne), dtype=f32, buffer=nl.sbuf)
        for gi in range(G):
            nisa.dma_copy(dst=X[0:P, gi, 0:ne], src=Z[base + gi, 0:P, 0:ne])
            nisa.dma_copy(dst=X[0:P, gi, ne : 2 * ne], src=V[base + gi, 0:P, 0:ne])
        per_chunk = max(1, 512 // (2 * ne))
        # Per-tile instructions pay off once a tile row is long (ne >= 64);
        # below that the group has many tiles and batched instructions win.
        per_tile = ne >= 64

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
            # Each matmul writes a whole 2 KiB PSUM bank (512 FP32 columns) or
            # the bank-aligned tail, never a sub-bank column slice next to
            # another matmul's output (the accumulate=False slice hazard).
            ps_m = nl.ndarray((P, ng, 2 * ne), dtype=f32, buffer=nl.psum)
            for c0 in range(0, ng, per_chunk):
                c1 = min(ng, c0 + per_chunk)
                nisa.nc_matmul(
                    dst=ps_m[0:P, c0:c1, 0 : 2 * ne],
                    stationary=pibd,
                    moving=X[0:P, g0 + c0 : g0 + c1, 0 : 2 * ne],
                    accumulate=False,
                )
            mz = ps_m[0:P, 0:ng, 0:ne]

            # Gram entries g_pp = |z_p|^2, g_pq = z_p . z_q, g_qq = g_pp[partner].
            gpp = col()
            gpq = col()
            if per_tile:
                # Large blocks: fused multiply + free-axis reduce per tile
                # (one Vector Engine pass instead of two).
                for j in range(ng):
                    junk = nl.ndarray((P, ne), dtype=f32, buffer=nl.sbuf)
                    nisa.scalar_tensor_tensor(
                        dst=junk,
                        data=Zc[0:P, j, 0:ne],
                        op0=mul,
                        operand0=1.0,
                        op1=mul,
                        operand1=Zc[0:P, j, 0:ne],
                        reduce_op=add,
                        reduce_res=gpp[0:P, j : j + 1],
                    )
                    junk2 = nl.ndarray((P, ne), dtype=f32, buffer=nl.sbuf)
                    nisa.scalar_tensor_tensor(
                        dst=junk2,
                        data=Zc[0:P, j, 0:ne],
                        op0=mul,
                        operand0=1.0,
                        op1=mul,
                        operand1=mz[0:P, j, 0:ne],
                        reduce_op=add,
                        reduce_res=gpq[0:P, j : j + 1],
                    )
            else:
                # Small blocks: one instruction over all tiles (many tiles).
                pr = nl.ndarray((P, ng, ne), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=pr, data1=Zc, data2=Zc, op=mul)
                nisa.tensor_reduce(dst=gpp, op=add, data=pr, axis=2)
                pr2 = nl.ndarray((P, ng, ne), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=pr2, data1=mz, data2=Zc, op=mul)
                nisa.tensor_reduce(dst=gpq, op=add, data=pr2, axis=2)
            ps_q = nl.ndarray((P, ng), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps_q, stationary=pibd, moving=gpp, accumulate=False)

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
            # 1 / den = rsqrt(den)^2 (rsqrt is accurate; reciprocal is ~1e-3).
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
            t1 = nl.ndarray((P, ng, 2 * ne), dtype=f32, buffer=nl.sbuf)
            if per_tile:
                # -s o M on the Scalar Engine, (c - 1) o X + that on the Vector
                # Engine, per tile (per-partition scalars, one pass each).
                for j in range(ng):
                    nisa.activation(
                        dst=t1[0:P, j, 0 : 2 * ne],
                        op=nl.copy,
                        data=ps_m[0:P, j, 0 : 2 * ne],
                        scale=negs[0:P, j : j + 1],
                    )
                    nisa.scalar_tensor_tensor(
                        dst=t1[0:P, j, 0 : 2 * ne],
                        data=Xc[0:P, j, 0 : 2 * ne],
                        op0=mul,
                        operand0=km1[0:P, j : j + 1],
                        op1=add,
                        operand1=t1[0:P, j, 0 : 2 * ne],
                    )
            else:
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
            nisa.dma_copy(dst=out[0, base + gi, 0:P, 0:ne], src=X[0:P, gi, 0:ne])
            nisa.dma_copy(dst=out[1, base + gi, 0:P, 0:ne], src=X[0:P, gi, ne : 2 * ne])
        return out


_DEVICE_CONSTS: dict = {}


def _consts_on(n: int, chains: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(consts, sched) for one launch, cached per device (no copy per launch)."""
    ne = block_layout(n)[0]
    key = (ne, chains, str(device))
    if key not in _DEVICE_CONSTS:
        consts = jacobi_constants(n).to(device)
        sched = torch.zeros((1, ne, chains), dtype=torch.float32, device=device)
        _DEVICE_CONSTS[key] = (consts, sched)
    return _DEVICE_CONSTS[key]


def _run_kernel(Z: torch.Tensor, V: torch.Tensor, n: int, chains: int) -> torch.Tensor:
    """One launch (one sweep) on a group: simulator or device. Z, V: (G, P, ne)."""
    if _use_simulator():
        import nki as _nki

        consts, sched = _consts_on(n, chains, torch.device("cpu"))
        res = _nki.simulate(jacobi_sweeps_kernel)(
            Z.detach().cpu().contiguous().numpy(),
            V.detach().cpu().contiguous().numpy(),
            consts.numpy(),
            sched.numpy(),
        )
        return torch.from_numpy(res).to(Z.device)
    from .dispatch import _launch

    consts, sched = _consts_on(n, chains, Z.device)
    return _launch(jacobi_sweeps_kernel, "jacobi_sweeps", Z, V, consts, sched, device=Z.device)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def group_plan(tiles: int, ne: int) -> list[tuple[int, int, int]]:
    """Split ``tiles`` into launches: ``[(start, count, padded), ...]``.

    Full groups of ``_GROUP_MAX[ne]`` (``TRNSOLVER_JACOBI_GROUP`` overrides),
    then one tail rounded up to a power of two, so each ne compiles at most
    log2(group) + 1 shapes whatever the batch size.
    """
    gmax = _env_int("TRNSOLVER_JACOBI_GROUP", _GROUP_MAX[ne])
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
    """Batched symmetric eigh by the one-sided Jacobi kernel.

    Args:
        A : (B, n, n) FP32, 1 <= n <= 128, any device (the lower triangle is
            used, as in ``torch.linalg.eigh``).

    Returns:
        (w (B, n) ascending, V (B, n, n)) on A's device.

    Runs ``_SWEEPS0[ne]`` sweeps (one launch per sweep and group of tiles),
    then one more sweep at a time while any matrix has a Gram off-diagonal
    norm above 1e-6, up to 15 sweeps. Preparation (symmetrise, scale,
    shift, pack) and the final Rayleigh quotients and sort run on A's
    device, so a host input costs one copy in and one copy out.
    """
    global last_sweeps
    Bsz, n = A.shape[0], A.shape[-1]
    if n == 1:
        last_sweeps = 0
        return A.reshape(Bsz, 1).clone(), torch.ones_like(A)
    ne, b, P = block_layout(n)
    if _use_simulator():
        work = torch.device("cpu")
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
    chains_max = _env_int("TRNSOLVER_JACOBI_CHAINS", _CHAINS_MAX[ne])

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

    def sweep():
        for k, (zg, vg, ch) in enumerate(groups):
            res = _run_kernel(zg, vg, n, ch)
            groups[k] = (res[0], res[1], ch)

    def gathered(i):
        parts = [g[i][:c] for g, (_, c, _) in zip(groups, plan, strict=True)]
        return parts[0] if len(parts) == 1 else torch.cat(parts, 0)

    done = 0
    for _ in range(_SWEEPS0[ne]):
        sweep()
        done += 1
    while done < _MAX_SWEEPS:
        if float(gram_off(unpack_compact(gathered(0), n, Bsz)).max()) < _TOL:
            break
        sweep()
        done += 1
    last_sweeps = done

    Zt = unpack_compact(gathered(0).to(A.device), n, Bsz)[:, :n, :n]
    Vt = unpack_compact(gathered(1).to(A.device), n, Bsz)[:, :n, :n]
    w = ((Zt * Vt).sum(-1) - sigma[:, None]) / scale[:, None]
    w, order = torch.sort(w, dim=-1)
    V = torch.gather(Vt.transpose(1, 2), -1, order.unsqueeze(-2).expand(Bsz, n, n))
    return w, V.contiguous()
