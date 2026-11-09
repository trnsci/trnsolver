"""GP preconditioner benchmark (#114): build time, CG convergence, and the GP C3 run.

The problem of the trnsci GP exemplar (trnsci/trnsci#109, PR #117): RBF kernel,
d = 3, lengthscale 0.15, noise 1e-2, R = 1 + 32 probes, rank 1024, tol 1e-6.
Needs trnsparse (>= the #104 multicore kernel_matmul).

  python scripts/bench_gp_precond.py build --n 1e5 [--seq]   # preconditioner build time
  python scripts/bench_gp_precond.py conv  --n 5e4           # CG: host64 vs projected
  python scripts/bench_gp_precond.py gp    --n 1e6           # GP posterior mean (C3)

--cpu runs the same on the CPU (trnsparse pytorch backend).

Output is one JSON line per measurement (prefix RESULT) plus human-readable lines.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import trnsparse

import trnsolver

# synthetic_field and _rademacher: verbatim from trnsci/trnsci examples/gp_regression
# (PR #117), so the data are bit-identical to the #117 runs.
_RFF_BLOCK = 16384


def synthetic_field(
    n: int,
    d: int = 3,
    *,
    n_test: int = 1000,
    lengthscale: float = 0.15,
    outputscale: float = 1.0,
    noise: float = 1e-2,
    features: int = 2048,
    seed: int = 0,
):
    """(X, y, X_test, f_test): a known-hyperparameter synthetic GP field.

    f(x) = sum_k a_k cos(w_k . x + b_k), w_k ~ N(0, I / l^2), b_k ~ U(0, 2 pi),
    a_k ~ N(0, 2 s / features): random Fourier features, whose covariance
    tends to s * exp(-|x - x'|^2 / (2 l^2)) as features grows. y = f(X) + eps,
    eps ~ N(0, noise). float64 on the host.
    """
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(features, d, generator=g, dtype=torch.float64) / lengthscale
    b = torch.rand(features, generator=g, dtype=torch.float64) * (2 * math.pi)
    a = torch.randn(features, generator=g, dtype=torch.float64) * math.sqrt(
        2 * outputscale / features
    )

    def f(P):
        out = torch.empty(P.shape[0], dtype=torch.float64)
        for i in range(0, P.shape[0], _RFF_BLOCK):
            out[i : i + _RFF_BLOCK] = torch.cos(torch.addmm(b, P[i : i + _RFF_BLOCK], W.T)) @ a
        return out

    X = torch.rand(n, d, generator=g, dtype=torch.float64)
    Xt = torch.rand(n_test, d, generator=g, dtype=torch.float64)
    y = f(X) + math.sqrt(noise) * torch.randn(n, generator=g, dtype=torch.float64)
    return X, y, Xt, f(Xt)


def _rademacher(n: int, p: int, seed: int) -> torch.Tensor:
    # Random-sign probes. trnrand.rademacher (trnsci/trnrand#87) will generate
    # these on the device; a host generator keeps this example reproducible now.
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n, p), generator=g, dtype=torch.int8).to(torch.float64) * 2 - 1


LS, OS = 0.15, 1.0


def sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def emit(**kw):
    print("RESULT " + json.dumps(kw), flush=True)


class DeviceRows:
    """rows_fn / sub_fn for pivoted_cholesky: K(X, X[idx]) from trnsparse.kernel_matmul on the device."""

    def __init__(self, X, device):
        self.X32 = X.float()
        self.X = X
        self.device = device
        self.calls = self.rows = 0
        self.t_rows = 0.0
        self.shapes = set()

    def rows_fn(self, idx):
        t = time.perf_counter()
        eye = torch.eye(idx.numel(), dtype=torch.float32, device=self.device)
        out = trnsparse.kernel_matmul(self.X32, self.X32[idx], eye, "rbf", LS, OS)  # (n, j)
        out = out.cpu().T  # (j, n)
        self.t_rows += time.perf_counter() - t
        self.calls += 1
        self.rows += idx.numel()
        self.shapes.add(idx.numel())
        return out

    def sub_fn(self, idx):
        Xi = self.X[idx]
        eye = torch.eye(idx.numel(), dtype=torch.float64)
        return trnsparse.kernel_matmul_reference(Xi, Xi, eye, "rbf", LS, OS)


def prewarm(X, device, block):
    """Compile every rows_fn shape pad_rows can produce (powers of two up to block).

    The shapes don't depend on the data, so this also works in TorchNeuron
    compile-only mode, unlike a pre-warm build (trnsci/trnsci#109).
    """
    dr = DeviceRows(X, device)
    t0 = time.perf_counter()
    sizes = sorted({min(1 << b, block) for b in range(max(1, block).bit_length())})
    for j in sizes:
        dr.rows_fn(torch.arange(j))
    return time.perf_counter() - t0, sizes


def build(X, op, device, *, rank, block, relax, apply, seq=False, pad_rows=True):
    t = {}
    t0 = time.perf_counter()
    if seq:
        L = trnsolver.pivoted_cholesky(op.diag, op.rows, rank)
        info = dict(calls=rank, rows=rank, rows_s=None)
    else:
        dr = DeviceRows(X, device)
        L = trnsolver.pivoted_cholesky(
            op.diag,
            dr.rows_fn,
            rank,
            block=block,
            relax=relax,
            sub_fn=dr.sub_fn,
            pad_rows=pad_rows,
        )
        info = dict(calls=dr.calls, rows=dr.rows, shapes=len(dr.shapes), rows_s=round(dr.t_rows, 2))
    # cholesky_s includes the device rows (rows_s); host_s is the host fp64 part alone.
    t["cholesky_s"] = time.perf_counter() - t0
    if info["rows_s"] is not None:
        t["host_s"] = t["cholesky_s"] - info["rows_s"]
    t0 = time.perf_counter()
    Ld = L.to(device=device, dtype=torch.float32)
    M = trnsolver.PivotedLowRankPreconditioner(Ld, op.noise, apply=apply)
    sync()
    t["factor_s"] = time.perf_counter() - t0
    t["precond_s"] = t["cholesky_s"] + t["factor_s"]
    rem = float(op.diag().sum()) - float((L.double() ** 2).sum())
    return L, M, t, dict(info, trace_rem=rem, rank=L.shape[1])


def time_apply(M, n, device, reps=3):
    V = torch.randn(n, 33).to(device)
    M(V)
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        M(V)
    sync()
    return (time.perf_counter() - t0) / reps


def cmd_build(a):
    n = int(a.n)
    X, y, Xt, ft = synthetic_field(n, 3, noise=a.noise)
    op = trnsparse.KernelOperator(X, "rbf", LS, OS, a.noise, device=a.device)
    for apply in a.applies.split(","):
        L, M, t, info = build(
            X,
            op,
            a.device,
            rank=a.rank,
            block=a.block,
            relax=a.relax,
            apply=apply,
            seq=a.seq,
            pad_rows=a.pad_rows,
        )
        ta = time_apply(M, n, a.device)
        print(
            f"n={n} rank={info['rank']} seq={a.seq} block={a.block} relax={a.relax} apply={apply}: "
            f"pad_rows={a.pad_rows}: cholesky {t['cholesky_s']:.1f}s (rows on device {info['rows_s']}s, "
            f"{info['calls']} calls, {info.get('shapes')} shapes; host {t.get('host_s', 0):.1f}s) "
            f"factor {t['factor_s']:.1f}s total {t['precond_s']:.1f}s; apply(n,33) {ta * 1e3:.0f} ms; "
            f"trace_rem {info['trace_rem']:.4e}",
            flush=True,
        )
        emit(
            kind="build",
            n=n,
            seq=a.seq,
            block=a.block,
            relax=a.relax,
            apply=apply,
            apply_ms=ta * 1e3,
            threads=torch.get_num_threads(),
            **{k: round(v, 2) for k, v in t.items()},
            **info,
        )
        if a.seq:
            break


def solve(op, B, M, a):
    t0 = time.perf_counter()
    sol, info = trnsolver.cg_block(
        op, B, M=M, tol=a.tol, maxiter=a.maxiter, check_every=a.check_every
    )
    sync()
    return sol, info, time.perf_counter() - t0


def sampled_residual(X, y, alpha, noise, n, rows=2048):
    idx = torch.arange(n) if n <= rows else torch.linspace(0, n - 1, rows).long()
    Ka = trnsparse.kernel_matmul_reference(X[idx], X, alpha, "rbf", LS, OS)
    r = y[idx] - Ka - noise * alpha[idx]
    return float(r.norm() / y[idx].norm())


def cmd_conv(a):
    n = int(a.n)
    X, y, Xt, ft = synthetic_field(n, 3, noise=a.noise)
    op = trnsparse.KernelOperator(X, "rbf", LS, OS, a.noise, device=a.device)
    B = torch.cat([y[:, None], _rademacher(n, 32, 1)], 1).to(device=a.device, dtype=torch.float32)
    dr = DeviceRows(X, a.device)
    L = trnsolver.pivoted_cholesky(
        op.diag, dr.rows_fn, a.rank, block=a.block, relax=a.relax, sub_fn=dr.sub_fn
    )
    Ld = L.to(device=a.device, dtype=torch.float32)
    first = None
    for apply in a.applies.split(","):
        M = trnsolver.PivotedLowRankPreconditioner(Ld, a.noise, apply=apply)
        ta = time_apply(M, n, a.device)
        sol, info, ts = solve(op, B, M, a)
        alpha = sol[:, 0].detach().cpu().double()
        mean = trnsparse.kernel_matmul_reference(Xt, X, alpha, "rbf", LS, OS)
        r = dict(
            kind="conv",
            n=n,
            noise=a.noise,
            apply=apply,
            iterations=info["iterations"],
            col_iters_max=int(info["col_iterations"].max()),
            converged=bool(info["converged"].all()),
            true_res_y=float(info["true_residual"][0]),
            true_res_max=float(info["true_residual"].max()),
            sampled_res=sampled_residual(X, y, alpha, a.noise, n),
            solve_s=round(ts, 1),
            s_per_iter=round(ts / max(1, info["iterations"] + 1), 2),
            apply_ms=round(ta * 1e3, 1),
            rmse=float((mean - ft).pow(2).mean().sqrt()),
        )
        if first is None:
            first = (mean, sol.detach().cpu())
        else:
            r["mean_rel_diff_vs_first"] = float((mean - first[0]).norm() / first[0].norm())
            s = sol.detach().cpu()
            r["X_rel_diff_vs_first"] = float(
                ((s - first[1]).norm(dim=0) / first[1].norm(dim=0)).max()
            )
        print(json.dumps(r), flush=True)
        emit(**r)


def cmd_gp(a):
    """The #117 C3 run (posterior_mean) with the #114 preconditioner path."""
    n = int(a.n)
    X, y, Xt, ft = synthetic_field(n, 3, noise=a.noise)
    t = {}
    t0 = time.perf_counter()
    op = trnsparse.KernelOperator(X, "rbf", LS, OS, a.noise, device=a.device)
    B = torch.cat([y[:, None], _rademacher(n, a.probes, 1)], 1).to(device=a.device, dtype=op.dtype)
    t["setup_s"] = time.perf_counter() - t0
    L, M, tb, binfo = build(
        X,
        op,
        a.device,
        rank=min(a.rank, n),
        block=a.block,
        relax=a.relax,
        apply=a.apply,
        pad_rows=a.pad_rows,
    )
    t.update(tb)
    t["precond_rank"] = binfo["rank"]
    sol, info, ts = solve(op, B, M, a)
    t["solve_s"] = ts
    alpha = sol[:, 0].detach().cpu().double()
    t0 = time.perf_counter()
    mean = trnsparse.kernel_matmul_reference(Xt, X, alpha, "rbf", LS, OS)
    t["predict_s"] = time.perf_counter() - t0
    samp = sampled_residual(X, y, alpha, a.noise, n)
    rmse = float((mean - ft).pow(2).mean().sqrt())
    print(
        f"n={n} d=3 kernel=rbf device={a.device} R={1 + a.probes} apply={a.apply} block={a.block} relax={a.relax}"
    )
    print(f"iterations={info['iterations']} converged={bool(info['converged'][0])}")
    print(
        f"solver true residual (fp32)={float(info['true_residual'][0]):.2e}  fp64 sampled residual={samp:.2e}"
    )
    print(f"test RMSE vs latent f={rmse:.4e}  (noise sd {math.sqrt(a.noise):.3e})")
    print(
        "timings: "
        + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items())
    )
    emit(
        kind="gp",
        n=n,
        apply=a.apply,
        iterations=info["iterations"],
        converged=bool(info["converged"][0]),
        true_res=float(info["true_residual"][0]),
        sampled_res=samp,
        rmse=rmse,
        s_per_iter=ts / max(1, info["iterations"] + 1),
        **binfo,
        **{k: v for k, v in t.items()},
    )
    torch.save({"alpha": alpha, "mean": mean}, a.save) if a.save else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "conv", "gp"])
    ap.add_argument("--n", type=float, required=True)
    ap.add_argument("--noise", type=float, default=1e-2)
    ap.add_argument("--rank", type=int, default=1024)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--relax", type=float, default=0.5)
    ap.add_argument(
        "--seq", action="store_true", help="build: the one-row-per-step host build (baseline)"
    )
    ap.add_argument(
        "--pad-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pad device row calls to power-of-two shapes (trnsci/trnsci#109)",
    )
    ap.add_argument(
        "--prewarm",
        action="store_true",
        help="compile the pad_rows shapes first (timed separately as prewarm_s)",
    )
    ap.add_argument("--apply", default="projected")
    ap.add_argument("--applies", default="host64,projected")
    ap.add_argument("--probes", type=int, default=32)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--check-every", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="dry run on the CPU (pytorch backend)")
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    a.device = "cpu" if a.cpu else "neuron"
    trnsparse.set_backend("pytorch" if a.cpu else "nki")
    if a.prewarm and a.cmd != "conv":
        X = synthetic_field(int(a.n), 3, noise=a.noise)[0]
        tp, sizes = prewarm(X, a.device, a.block)
        print(f"prewarm: {len(sizes)} shapes {sizes} in {tp:.1f}s", flush=True)
        emit(kind="prewarm", n=int(a.n), prewarm_s=round(tp, 2), shapes=sizes)
    {"build": cmd_build, "conv": cmd_conv, "gp": cmd_gp}[a.cmd](a)


if __name__ == "__main__":
    main()
