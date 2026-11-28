"""CG reduction modes per iteration and end to end at n = 1e5, 1e6, 1e7 (trnsci/trnsci#169).

Compares ``reduction=`` pairwise, host64, ff, exact (and auto, device if
asked) for ``cg`` on one vector, and optionally ``cg_block`` on R columns.

The operator is the banded stationary kernel of scripts/probe_cg_reduction.py
(K[i, j] = exp(-(i - j)² / (2 ℓ²)) for |i − j| ≤ w, ℓ = 3, w = 12, plus
noise·I), applied matrix-free as 2w + 1 shifted adds. It is elementwise only,
so it does not depend on the LNC split, and at about 25 adds per element it
is a cheap operator: the reduction overhead shows up at close to its worst
relative size. The GP exemplar's operator (trnsci/trnsci#109) costs 0.3 s
(n = 1e5) to 7 s (n = 1e6) per iteration, so there the reductions are a far
smaller fraction.

Per n and mode it reports:

- ``micro``: median ms of one ``_inner(u, v)`` and one ``_norm(u)`` (D2H
  and host finish included), and their correct bits against the exactly
  rounded value of the fp32 data (``math.fsum`` of the exact float64
  products; skipped with ``--no-reference``): ``inner_bits`` for two
  independent vectors (cond ~ sqrt(n)), ``rr_bits`` for r.r (cond 1);
- ``fixed``: ``--iters`` CG iterations at tol = 0 (no early stop), so every
  mode does the same work: ms per iteration, the operator's share, and
  ``overhead_ms`` (per iteration, everything that is not the operator:
  3 reductions, 3 axpys and the scalar reads), plus SHA-256 of x;
- ``solve``: a converged solve at ``--tol`` (warm): iterations, end-to-end
  seconds, the first (compiling) call's seconds, the residual, and SHA-256
  of x (with ``--reference``, also the error against a host float64 solve).

Every mode is run once untimed first (compiles). ``--deadline`` stops
starting new work after that many seconds, so a run fits a window slot.

    python -P scripts/bench_cg_reduction_ff.py --n 1e5,1e6,1e7 --json OUT.json [--cpu]

Run at LNC=1 and LNC=2 and compare the ``sha`` fields: pairwise, host64, ff
and exact must match (tests/test_cg_reduction_ff_neuron.py does this with
TRNSOLVER_TEST_LNC=1). Output: human-readable lines and one ``RESULT {json}``.
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
from trnsolver import block, iterative

ELL, HALF_WIDTH = 3.0, 12


def taps():
    d = torch.arange(-HALF_WIDTH, HALF_WIDTH + 1, dtype=torch.float64)
    return [float(t) for t in torch.exp(-0.5 * d.square() / ELL**2)]


def banded_op(n, noise):
    """x -> (K + noise I) x for (n,) or (n, R), as 2w + 1 shifted adds (fixed order)."""
    tp, w = taps(), HALF_WIDTH

    def op(x):
        Z = x.new_zeros((w, *x.shape[1:]))
        xp = torch.cat([Z, x, Z], 0)
        out = noise * x
        for j, t in enumerate(tp):
            out = out + t * xp[j : j + n]
        return out

    return op


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
    return 1e3 * float(np.median(ts))


def bits(v, ref):
    if not math.isfinite(v):
        return 0.0
    if v == ref:
        return 53.0
    return round(min(53.0, max(0.0, -math.log2(abs(v - ref) / abs(ref)))), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="1e5,1e6,1e7", help="comma-separated sizes")
    ap.add_argument("--modes", default="pairwise,host64,ff,exact,auto")
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=30, help="fixed CG iterations (tol = 0)")
    ap.add_argument("--tol", type=float, default=1e-6, help="converged solve tolerance")
    ap.add_argument("--maxiter", type=int, default=2000)
    ap.add_argument("--micro-reps", type=int, default=20)
    ap.add_argument("--R", type=int, default=0, help="also time cg_block on R columns (0: off)")
    ap.add_argument("--no-reference", action="store_true", help="skip correct-bits references")
    ap.add_argument("--reference", action="store_true", help="host float64 solve per n (slow)")
    ap.add_argument("--deadline", type=float, default=1080.0, help="seconds; stop starting work")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    t_start = time.perf_counter()

    def late():
        return time.perf_counter() - t_start > a.deadline

    if a.cpu:
        device = torch.device("cpu")
    else:
        import torch_neuronx  # noqa: F401

        device = torch.device("neuron")
    modes = a.modes.split(",")
    res = dict(
        lnc=os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"),
        cores=os.environ.get("NEURON_RT_VISIBLE_CORES"),
        device=device.type,
        ff_min_n=block._ff_min_n(),
        iters=a.iters,
        tol=a.tol,
        modes=modes,
        runs={},
        skipped=[],
    )
    try:
        import trnblas

        res["trnblas"] = trnblas.__version__
    except Exception:  # pragma: no cover
        res["trnblas"] = None
    res["trnsolver"] = getattr(trnsolver, "__version__", None)

    for n in [int(float(s)) for s in a.n.split(",")]:
        g = torch.Generator().manual_seed(169)
        b_h = torch.randn(n, generator=g)
        v_h = torch.randn(n, generator=g)
        b, v = b_h.to(device), v_h.to(device)
        op = banded_op(n, a.noise)
        run = dict(n=n, auto=block._resolve_reduction("auto", n), modes={})
        res["runs"][str(n)] = run
        ref_dot = ref_nrm = None
        if not a.no_reference:
            ref_dot = math.fsum((b_h.double() * v_h.double()).tolist())
            ref_rr = math.fsum((b_h.double() ** 2).tolist())
            ref_nrm = math.sqrt(ref_rr)
        op_ms = timed(lambda op=op, b=b: op(b), device, 5)
        run["op_ms"] = round(op_ms, 3)
        x64 = None
        if a.reference:
            op64 = banded_op(n, a.noise)
            x64, it64, _ = trnsolver.cg(
                op64, b_h.double(), tol=1e-12, maxiter=4 * a.maxiter, reduction="host64"
            )
            run["ref_iterations"] = it64
        print(f"n={n} auto->{run['auto']} op {op_ms:.3f} ms", flush=True)

        for mode in modes:
            if late():
                res["skipped"].append(f"{n}:{mode}")
                continue
            m = {}
            # Micro: one reduction, end to end (D2H and host finish included).
            m["inner_ms"] = round(
                timed(
                    lambda mode=mode, b=b, v=v: iterative._inner(b, v, mode), device, a.micro_reps
                ),
                3,
            )
            m["norm_ms"] = round(
                timed(lambda mode=mode, b=b: iterative._norm(b, mode), device, a.micro_reps), 3
            )
            d, nr = iterative._inner(b, v, mode), iterative._norm(b, mode)
            m["inner_value"], m["norm_value"] = d.hex(), nr.hex()
            if ref_dot is not None:
                # inner: two independent vectors (cond ~ sqrt(n)); rr: r.r (cond 1)
                m["inner_bits"], m["norm_bits"] = bits(d, ref_dot), bits(nr, ref_nrm)
                m["rr_bits"] = bits(iterative._inner(b, b, mode), ref_rr)
            # Fixed work: --iters iterations, no early stop.
            kw = dict(tol=0.0, maxiter=a.iters, reduction=mode)
            t0 = time.perf_counter()
            trnsolver.cg(op, b, **kw)
            sync(device)
            m["fixed_first_s"] = round(time.perf_counter() - t0, 3)
            t0 = time.perf_counter()
            x, it, r = trnsolver.cg(op, b, **kw)
            sync(device)
            dt = time.perf_counter() - t0
            m["ms_per_iter"] = round(1e3 * dt / a.iters, 3)
            m["overhead_ms_per_iter"] = round(1e3 * dt / a.iters - op_ms, 3)
            m["sha_fixed"] = sha(x)
            m["residual_fixed"] = r
            # Converged solve, warm (the fixed run compiled every shape).
            kw = dict(tol=a.tol, maxiter=a.maxiter, reduction=mode)
            t0 = time.perf_counter()
            x, it, r = trnsolver.cg(op, b, **kw)
            sync(device)
            m["solve_s"] = round(time.perf_counter() - t0, 3)
            m["solve_iterations"], m["solve_residual"], m["sha_solve"] = it, r, sha(x)
            if x64 is not None:
                m["rel_err_vs_fp64"] = float((x.cpu().double() - x64).norm() / x64.norm())
            if a.R and not late():
                B = torch.randn(n, a.R, generator=torch.Generator().manual_seed(170)).to(device)
                kwb = dict(tol=0.0, maxiter=a.iters, check_every=a.iters, reduction=mode)
                trnsolver.cg_block(op, B, **kwb)
                sync(device)
                t0 = time.perf_counter()
                X, info = trnsolver.cg_block(op, B, **kwb)
                sync(device)
                m["block_ms_per_iter"] = round(1e3 * (time.perf_counter() - t0) / a.iters, 3)
                m["block_sha"] = sha(X)
                m["block_reduction"] = info["reduction"]
            run["modes"][mode] = m
            print(f"  {mode}: {json.dumps(m)}", flush=True)
        with open(a.json, "w") as f:  # after every n, so a cut-off run keeps its data
            json.dump(res, f, indent=1)
    res["wall_s"] = round(time.perf_counter() - t_start, 1)
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1)
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
