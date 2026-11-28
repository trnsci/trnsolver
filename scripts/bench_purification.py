"""GEMM-only SCF density and S^{-1/2} against host eigh (#149).

Synthetic SCF-like generalized problems F C = S C eps: a known orbital
spectrum (deep core levels, a valence band, a virtual band widening upward)
in a random orthonormal basis, and an overlap S with a chosen condition
number. Three cases set the HOMO-LUMO gap:

    gapped    0.5      (insulator-like)
    small     1e-2
    neardeg   1e-6     (near-degenerate at the Fermi level)

For each (n, case, method, arm) one row: the fixed iteration counts, the
iteration where tr(P' - P'^2) first drops below the dtype's threshold, the
GEMM count, the errors against the host fp64 eigh reference (energy tr(PF),
relative density error, idempotency ||P'^2 - P'||_F, trace |tr P' - n_occ|,
S^{-1/2} residual), the purification wall time, and the host eigh time
(``torch.linalg.eigh`` of S, then of X^T F X: the host fallback's work).

Arms:

    native        iterate in --dtype on --device, torch.matmul products, and
                  S^{-1/2} and the transforms there too (transforms="native")
    fp32prod      float64 iterate on the host, fp32 products (emulated stage)
    staged        float64 iterate on the host; the first --stage iterations
                  use fp32 products, the rest (and the transforms) fp64-class
                  ``trnblas.gemm_fp64`` products on --device
    ozaki         float64 iterate on the host, every product
                  ``trnblas.gemm_fp64`` on --device (scheme --scheme)
    fp32orth      X = S^{-1/2} and F' = X^T F X once in host fp64, then the fp32
                  purification of F' on --device (the timed part); the fp32
                  error of the native arm is mostly S^{-1/2} and the transforms
    host64        density_purification(transforms="host64"), the default for
                  fp32 on a device (the fp32orth arm inside the library):
                  fp32 F on --device, fp64 S on the host. The first call computes X (host fp64 Newton-Schulz);
                  the timed calls pass X=result.X, as an SCF iteration would,
                  so t includes the four host fp64 transforms per call

With --device neuron the native arm keeps F, S and every iterate on the
device; the staged / ozaki arms move each product's operands to the device
and back (the transfer is in the time). --compile wraps the purification loop
in torch.compile (backend "neuron" on the device).

CPU:    python scripts/bench_purification.py --n 256 512 1024
Device: python scripts/bench_purification.py --device neuron --n 512 1024 2048 3508 \\
            --arms native host64 ozaki staged --out purify_trn2.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch

CASES = {"gapped": 0.5, "small": 1e-2, "neardeg": 1e-6}

# Fixed iteration counts per (method, case). Iterations to tr(P - P^2) < 1e-10
# in fp64 on these spectra, n = 64 / 512 / 1024 / 2048:
#   canonical  gapped 27 / 30 / 32 / 33   small 42 / 57 / 63 / 69   neardeg 66 / 99 / 123 / >150
#   sp2        gapped 35 / 39 / 40 / 42   small 55 / 58 / 60 / 62   neardeg 97 / 105 / 108 / 113
# Canonical slows with n at small gaps; SP2 barely moves. The counts below
# leave margin up to n ~ 4k except canonical / neardeg beyond n ~ 1k.
ITERS = {
    ("canonical", "gapped"): 45,
    ("canonical", "small"): 85,
    ("canonical", "neardeg"): 150,
    ("sp2", "gapped"): 50,
    ("sp2", "small"): 75,
    ("sp2", "neardeg"): 125,
}


def synthetic_scf(
    n: int,
    gap: float,
    *,
    seed: int = 0,
    cond_s: float = 1e4,
    occ_frac: float = 0.25,
    dtype: torch.dtype = torch.float64,
):
    """(F, S, n_occ, eps): F = S^{1/2} Q diag(eps) Q^T S^{1/2}, so the
    generalized eigenvalues of (F, S) are exactly eps (ascending).

    eps: n_occ // 8 core levels on [-20, -5], the rest of the occupied levels
    on [-1.5, -0.3] (HOMO -0.3), and virtuals HOMO + gap + 8 t^2, t in [0, 1].
    S has eigenvalues 2 * logspace(0, -log10(cond_s)).
    """
    g = torch.Generator().manual_seed(seed)
    f64 = torch.float64
    n_occ = max(1, int(round(occ_frac * n)))
    n_core = max(1, n_occ // 8)
    core = torch.linspace(-20.0, -5.0, n_core, dtype=f64)
    val = torch.linspace(-1.5, -0.3, n_occ - n_core, dtype=f64)
    vir = -0.3 + gap + 8.0 * torch.linspace(0.0, 1.0, n - n_occ, dtype=f64) ** 2
    eps = torch.cat([core, val, vir])
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=f64))
    Qs, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=f64))
    ws = 2.0 * torch.logspace(0.0, -math.log10(cond_s), n, dtype=f64)
    S = (Qs * ws) @ Qs.T
    S_half = (Qs * ws.sqrt()) @ Qs.T
    F = S_half @ ((Q * eps) @ Q.T) @ S_half
    F, S = 0.5 * (F + F.T), 0.5 * (S + S.T)
    return F.to(dtype), S.to(dtype), n_occ, eps


def reference(F: torch.Tensor, S: torch.Tensor, n_occ: int, repeat: int = 3):
    """Host fp64 eigh reference: (P, energy, best-of-``repeat`` seconds for the two eighs)."""
    F, S = F.double().cpu(), S.double().cpu()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        w, V = torch.linalg.eigh(S)
        X = (V * w.rsqrt()) @ V.T
        e, C = torch.linalg.eigh(X @ F @ X)
        ts.append(time.perf_counter() - t0)
    t = min(ts)
    Co = X @ C[:, :n_occ]
    return Co @ Co.T, float(e[:n_occ].sum()), t


def _sync(device: str) -> None:
    if device != "cpu" and hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def device_gemm(device: str, **kw):
    """Product on ``device`` through trnblas.gemm_fp64, result back on the host."""
    import trnblas

    def g(A, B):
        return trnblas.gemm_fp64(A.to(device), B.to(device), **kw).to(A.device, A.dtype)

    return g


def device_fp32_gemm(device: str):
    def g(A, B):
        return (A.float().to(device) @ B.float().to(device)).to(A.device, A.dtype)

    return g


def precision_for(arm: str, device: str, stage: int, scheme: str):
    if arm == "native":
        return None
    if arm == "fp32prod":
        return "fp32" if device == "cpu" else device_fp32_gemm(device)
    hi = device_gemm(device, scheme=scheme) if device != "cpu" else scheme
    if arm == "ozaki":
        return hi
    if arm == "staged":
        lo = "fp32" if device == "cpu" else device_fp32_gemm(device)
        return [lo] * stage + [hi]
    raise ValueError(arm)


def first_below(hist: torch.Tensor, thr: float):
    idx = (hist.double().cpu() < thr).nonzero()
    return int(idx[0]) if len(idx) else None


def run_one(args, n, case, method, arm, prob, ref):
    import trnsolver
    from trnsolver import _precision, density_purification
    from trnsolver.matfun import _purify_fixed, ns_iteration_count

    F64, S64, n_occ, _ = prob
    P_ref, E_ref, _ = ref
    iters = args.iters or ITERS[(method, case)]
    X_host = None
    if arm == "fp32orth":
        # X = S^-1/2 once per geometry in host fp64; per SCF iteration only the
        # fp32 purification of F' = X^T F X runs on the device.
        dt, prec = torch.float32, None
        k = ns_iteration_count(args.ns_cond, 50 * torch.finfo(torch.float64).eps, "minimax")
        X_host, _, ns_res_host = trnsolver.inv_sqrt_spd_ns(
            S64, k, polynomial="minimax", cond=args.ns_cond
        )
        Fo = (X_host.T @ F64 @ X_host).to(args.device, dt)
        args_in, kw = (Fo,), dict(n_occ=n_occ, method=method, iters=iters)
    elif arm == "host64":
        dt, prec = torch.float32, None
        args_in = (F64.to(args.device, dt), S64)
        kw = dict(
            n_occ=n_occ, method=method, iters=iters, ns_cond=args.ns_cond, transforms="host64"
        )
    else:
        dt = getattr(torch, args.dtype) if arm == "native" else torch.float64
        dev = args.device if arm == "native" else "cpu"
        prec = precision_for(arm, args.device, args.stage, args.scheme)
        args_in = (F64.to(dev, dt), S64.to(dev, dt))
        kw = dict(n_occ=n_occ, method=method, iters=iters, precision=prec, ns_cond=args.ns_cond)
        if arm == "native":
            kw["transforms"] = "native"

    t0 = time.perf_counter()
    r = density_purification(*args_in, **kw)  # warm-up (compiles on the device)
    _sync(args.device)
    t_first = time.perf_counter() - t0
    ns_first = None
    if arm == "host64":
        # Later SCF iterations of the same geometry reuse the host fp64 X.
        ns_first = (r.ns_iters, r.ns_residual)
        args_in, kw = (args_in[0],), dict(kw, X=r.X)
    t_loop = t_compile = compile_error = None
    if args.compile:
        gemm_k, gemm_last = _precision.schedule(prec)
        F_in = args_in[0]
        X_r = r.X
        Fo = F_in if X_r is None else (X_r.mT @ F_in.to(X_r) @ X_r).to(F_in).contiguous()

        def loop(Fo):
            return _purify_fixed(Fo, n_occ, method, iters, gemm_k, gemm_last)

        torch._dynamo.reset()  # a fresh closure per row: no stale guards, no recompile limit
        backend = "neuron" if args.device == "neuron" else args.compile_backend
        cloop = torch.compile(loop, backend=backend, fullgraph=True, dynamic=False)
        try:
            t0 = time.perf_counter()
            out = cloop(Fo)  # first call: trace + compile + run
            out[0].sum().item()
            t_compile = time.perf_counter() - t0
            ts = []
            for _ in range(args.repeat):
                t0 = time.perf_counter()
                out = cloop(Fo)
                out[0].sum().item()
                ts.append(time.perf_counter() - t0)
            t_loop = min(ts)
        except Exception as e:  # record it; the eager row is still useful
            compile_error = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"compile failed (n={n} {case} {method} {arm}): {compile_error}", flush=True)
    ts = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        r = density_purification(*args_in, **kw)
        _sync(args.device)
        ts.append(time.perf_counter() - t0)
    if X_host is not None:
        P = X_host @ r.P.double().cpu() @ X_host.T
        r.energy = float((P * F64).sum())
        r.ns_iters, r.ns_residual = k, ns_res_host
    else:
        P = r.P.double().cpu()
    if ns_first is not None:
        r.ns_iters, r.ns_residual = ns_first
    gemms_per = 2 if method == "canonical" else 1
    thr = 1e-10 if dt == torch.float64 else 1e-3
    return {
        "n": n,
        "case": case,
        "gap": CASES[case],
        "method": method,
        "arm": arm,
        "dtype": str(dt).replace("torch.", ""),
        "device": args.device,
        "iters": iters,
        "iters_to_conv": first_below(r.history, thr),
        "conv_thr": thr,
        "ns_iters": r.ns_iters,
        # purification + NS (first step 1, then 3, + residual) + 4 transforms + final check
        "gemms": gemms_per * iters + (3 * r.ns_iters - 1 if r.ns_iters else 0) + 4 + 1,
        "E_err": abs(r.energy - E_ref),
        "E_rel": abs(r.energy - E_ref) / abs(E_ref),
        "P_rel": float((P - P_ref).norm() / P_ref.norm()),
        "idem": r.idempotency_error,
        "trace_err": r.trace_error,
        "ns_res": r.ns_residual,
        "t_s": min(ts),
        "t_first_call_s": t_first,
        "t_loop_compiled_s": t_loop,
        "t_compile_first_call_s": t_compile,
        "compile_error": compile_error,
        "t_eigh_host_s": ref[2],
    }


def _fmt(v, spec: str) -> str:
    return "-" if v is None else format(v, spec)


def table(rows) -> str:
    h = (
        "| n | case | method | arm | dtype | iters (conv) | NS | GEMMs | E err | P rel | "
        "idem | trace | NS res | t (s) | compiled loop (s) | host eigh (s) |"
    )
    out = [h, "|" + "---|" * (h.count("|") - 1)]
    for r in rows:
        out.append(
            f"| {r['n']} | {r['case']} | {r['method']} | {r['arm']} | {r['dtype']} | "
            f"{r['iters']} ({r['iters_to_conv']}) | {r['ns_iters']} | {r['gemms']} | "
            f"{r['E_err']:.1e} | {r['P_rel']:.1e} | {r['idem']:.1e} | {r['trace_err']:.1e} | "
            f"{_fmt(r['ns_res'], '.1e')} | {r['t_s']:.3f} | "
            f"{_fmt(r['t_loop_compiled_s'], '.3f')} | {r['t_eigh_host_s']:.3f} |"
        )
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--methods", nargs="+", default=["canonical", "sp2"])
    ap.add_argument(
        "--arms",
        nargs="+",
        default=["native"],
        choices=["native", "fp32orth", "host64", "fp32prod", "staged", "ozaki"],
    )
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--iters", type=int, default=None, help="override the per-case counts")
    ap.add_argument("--stage", type=int, default=20, help="fp32-product iterations (staged)")
    ap.add_argument("--scheme", default="ozaki1", choices=["ozaki1", "ozaki2"])
    ap.add_argument("--cond-s", type=float, default=1e4)
    ap.add_argument("--ns-cond", type=float, default=1e5)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile-backend", default="aot_eager")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.device != "cpu":
        import torch_neuronx  # noqa: F401

    rows = []
    for n in args.n:
        for case in args.cases:
            prob = synthetic_scf(n, CASES[case], cond_s=args.cond_s)
            ref = reference(prob[0], prob[1], prob[2], args.repeat)
            for method in args.methods:
                for arm in args.arms:
                    row = run_one(args, n, case, method, arm, prob, ref)
                    print("RESULT " + json.dumps(row), flush=True)
                    rows.append(row)
    print(table(rows))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
