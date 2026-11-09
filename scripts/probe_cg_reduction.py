"""cg_block reduction modes on a GP-like system: LNC invariance, accuracy, cost (#128).

The operator is a compactly supported stationary kernel on a 1-D grid plus
noise, A = K + noise·I with K[i, j] = exp(-(i - j)² / (2 ℓ²)) for |i − j| ≤ w
(ℓ = 3, w = 12 grid points). It is applied matrix-free as a fixed sum of
2w + 1 shifted copies of the block, which is elementwise only, so the
operator itself does not depend on the LNC split; whatever differs between
LNC=1 and LNC=2 comes from cg_block's reductions (and the preconditioner's).
The optional preconditioner is ``PivotedLowRankPreconditioner(apply=
"projected")`` from a rank-k pivoted Cholesky of K (host float64).

For each ``--modes`` entry the script solves A X = B (B (n, R) seeded on the
host) and reports the iteration count, the time per iteration, the relative
error against a host float64 solve, and a SHA-256 of X's bytes, and saves X.
It also times the bare reductions (_cdot, _col_norm, Qᵀ V) per mode.

    python -P scripts/probe_cg_reduction.py --n 1e5 --R 33 --out DIR [--cpu]

Run it at LNC=1 and LNC=2 and compare the SHA-256 lines (tests/test_block_neuron.py
does this when TRNSOLVER_TEST_LNC=1). Output: human-readable lines and one
``RESULT {json}`` line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch

import trnsolver
from trnsolver import block

ELL, HALF_WIDTH = 3.0, 12


def kernel_taps(dtype=torch.float64):
    d = torch.arange(-HALF_WIDTH, HALF_WIDTH + 1, dtype=torch.float64)
    return torch.exp(-0.5 * d.square() / ELL**2).to(dtype)


def banded_op(n, noise, device, dtype):
    """V -> (K + noise I) V for the banded kernel, as 2w + 1 shifted adds."""
    taps = [float(t) for t in kernel_taps()]
    w = HALF_WIDTH

    def op(V):
        Z = V.new_zeros(w, V.shape[1])
        Vp = torch.cat([Z, V, Z], 0)
        out = noise * V
        for j, t in enumerate(taps):  # fixed order: tap -w .. w
            out = out + t * Vp[j : j + n]
        return out

    return op


def kernel_rows(n):
    taps = kernel_taps()
    w = HALF_WIDTH

    def rows(idx):
        out = torch.zeros(len(idx), n, dtype=torch.float64)
        for r, i in enumerate(idx.tolist()):
            lo, hi = max(0, i - w), min(n, i + w + 1)
            out[r, lo:hi] = taps[lo - i + w : hi - i + w]
        return out

    return rows


def sync(device):
    if device.type == "neuron" and hasattr(torch, "neuron"):
        torch.neuron.synchronize()


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def timed(fn, device, reps):
    fn()
    sync(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)) if ts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=float, default=1e5)
    ap.add_argument("--R", type=int, default=33)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--rank", type=int, default=128, help="preconditioner rank (0: none)")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--modes", default="device,pairwise,host64")
    ap.add_argument("--micro-reps", type=int, default=20)
    ap.add_argument("--micro-k", type=int, default=1024, help="k for the Qᵀ V micro-timing")
    ap.add_argument("--no-ref", action="store_true", help="skip the host float64 reference")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.cpu:
        device = torch.device("cpu")
    else:
        import torch_neuronx  # noqa: F401

        device = torch.device("neuron")
    n, R = int(a.n), a.R
    res = dict(
        n=n,
        R=R,
        noise=a.noise,
        rank=a.rank,
        lnc=os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"),
        cores=os.environ.get("NEURON_RT_VISIBLE_CORES"),
        device=device.type,
        fold=block._fold_plan(n),
        solves={},
        micro={},
    )
    g = torch.Generator().manual_seed(128)
    B_host = torch.randn(n, R, generator=g, dtype=torch.float64)
    B = B_host.to(torch.float32).to(device)

    L = None
    if a.rank:
        t0 = time.perf_counter()
        K_diag = torch.ones(n, dtype=torch.float64)
        L = trnsolver.pivoted_cholesky(lambda: K_diag.clone(), kernel_rows(n), a.rank, block=64)
        res["cholesky_s"] = round(time.perf_counter() - t0, 1)
        res["rank_eff"] = L.shape[1]

    X64 = None
    if not a.no_ref:
        t0 = time.perf_counter()
        op64 = banded_op(n, a.noise, torch.device("cpu"), torch.float64)
        M64 = None
        if L is not None:
            M64 = trnsolver.PivotedLowRankPreconditioner(L, a.noise, apply="native")
        X64, i64 = trnsolver.cg_block(
            op64, B_host, M64, tol=1e-12, maxiter=4 * a.maxiter, reduction="host64"
        )
        res["ref"] = dict(iterations=i64["iterations"], s=round(time.perf_counter() - t0, 1))
        print("fp64 reference:", res["ref"], flush=True)

    op = banded_op(n, a.noise, device, torch.float32)
    Ld = None if L is None else L.to(torch.float32).to(device)
    for mode in a.modes.split(","):
        M = None
        if Ld is not None:
            M = trnsolver.PivotedLowRankPreconditioner(
                Ld, a.noise, apply="projected", reduction=mode
            )
        kw = dict(tol=a.tol, maxiter=a.maxiter, reduction=mode)
        t0 = time.perf_counter()
        X, info = trnsolver.cg_block(op, B, M, **kw)  # first call: compiles
        sync(device)
        first_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        X, info = trnsolver.cg_block(op, B, M, **kw)
        sync(device)
        solve_s = time.perf_counter() - t0
        Xh = X.detach().cpu()
        np.save(f"{a.out}/X_{mode}.npy", Xh.numpy())
        r = dict(
            iterations=info["iterations"],
            col_iters_max=int(info["col_iterations"].max()),
            converged=bool(info["converged"].all()),
            true_res_max=float(info["true_residual"].max()),
            first_s=round(first_s, 1),
            solve_s=round(solve_s, 3),
            ms_per_iter=round(1e3 * solve_s / max(1, info["iterations"] + 1), 2),
            sha_X=sha(Xh),
        )
        if M is not None:
            z = M(B).detach().cpu()
            np.save(f"{a.out}/MB_{mode}.npy", z.numpy())
            r["sha_MB"] = sha(z)
            r["apply_ms"] = round(timed(lambda M=M: M(B), device, 5), 2)
        if X64 is not None:
            r["rel_err_vs_fp64"] = float(((Xh.double() - X64).norm(dim=0) / X64.norm(dim=0)).max())
        res["solves"][mode] = r
        print(mode, json.dumps(r), flush=True)

    # Bare reductions at (n, R), and Qᵀ V at (n, k) x (n, R).
    Y = torch.randn(n, R, generator=g).to(device)
    Q = torch.randn(n, a.micro_k, generator=g).div_(math.sqrt(n)).to(device)
    Qpad = block._pad_rows(Q)
    for mode in a.modes.split(","):
        m = {}
        m["cdot_ms"] = timed(lambda mode=mode: block._cdot(B, Y, mode), device, a.micro_reps)
        m["norm_ms"] = timed(lambda mode=mode: block._col_norm(B, mode), device, a.micro_reps)
        m["qtv_ms"] = timed(
            lambda mode=mode: block._tmatmul(Q, Y, mode, Qpad=Qpad).cpu(), device, a.micro_reps
        )
        m["sha_cdot"] = sha(block._cdot(B, Y, mode))
        m["sha_norm"] = sha(block._col_norm(B, mode))
        m["sha_qtv"] = sha(block._tmatmul(Q, Y, mode, Qpad=Qpad))
        res["micro"][mode] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}
        print("micro", mode, json.dumps(res["micro"][mode]), flush=True)
    with open(f"{a.out}/res.json", "w") as f:
        json.dump(res, f, indent=1)
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
