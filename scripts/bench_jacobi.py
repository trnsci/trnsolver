"""Batched Jacobi eigh vs batched CPU LAPACK (#48).

Prints one ``RESULT`` line per (mode, n, B):

    RESULT mode=<m> n=<n> B=<B> sweeps=<s> ms=<total> us_per_matrix=<..>
           eig_err=<..> res=<..> orth=<..>

Modes:
    cpu        batched ``torch.linalg.eigh`` on host FP32 input (LAPACK).
    torch      the torch Jacobi of ``trnsolver/_jacobi.py`` on the ``neuron``
               device: eager aten ops, ``torch.bmm`` for the updates, no
               custom kernel.
    compiled   the same step under ``torch.compile(backend="neuron",
               dynamic=False)``: one graph per step.
    nki        ``trnsolver.eigh(A.to("neuron"), method="jacobi")``: the
               SBUF-resident one-sided Jacobi kernel (the block kernel for
               33 <= n <= 128; ``TRNSOLVER_JACOBI_BLOCK=0`` for the scalar one).
    aten       ``trnsolver.eigh(A.to("neuron"))`` with the Jacobi auto region
               off (``TRNSOLVER_EIGH_JACOBI_AUTO=off``): the default aten
               cutover, host LAPACK plus copies.
    auto       ``trnsolver.eigh(A.to("neuron"))``: the default dispatch.

Accuracy is against float64 LAPACK: ``eig_err = max|w - w64| / max|w64|``,
``res = max_b ||A V - V diag(w)||_F / ||A||_F``, ``orth = max|V^T V - I|``.
Timings are warm medians from host input to host output (copies
included), synchronised with ``torch_neuronx.synchronize()``. The ``torch``
and ``compiled`` modes first measure the sweeps to reach off(A) < tol and
then time that fixed count.

Pin the host threads for the CPU baseline, e.g.
    OMP_NUM_THREADS=2 taskset -c 10,11 python scripts/bench_jacobi.py --threads 2
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch


def _sym(B, n, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, n, n, generator=g, dtype=torch.float64)
    return (X + X.transpose(1, 2)) / 2


def _acc(A64, w, V):
    w = w.double().cpu()
    V = V.double().cpu()
    w64 = torch.linalg.eigvalsh(A64)
    eig = float((w - w64).abs().max() / w64.abs().max())
    res = float(((A64 @ V - V * w.unsqueeze(-2)).norm(dim=(1, 2)) / A64.norm(dim=(1, 2))).max())
    n = A64.shape[-1]
    orth = float((V.transpose(1, 2) @ V - torch.eye(n, dtype=torch.float64)).abs().max())
    return eig, res, orth


def _time(fn, reps, sync):
    fn()
    sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def _run_cpu(A, reps):
    out = torch.linalg.eigh(A)
    return out, _time(lambda: torch.linalg.eigh(A), reps, lambda: None), 0


def _run_torch(A, step, dev, sync, reps, tol, max_sweeps, label):
    from trnsolver import _jacobi as J

    B, n = A.shape[0], A.shape[-1]
    Ak = A.to(dev)
    Wk = torch.eye(n, device=dev).expand(B, n, n).contiguous()
    offs = []
    sweeps = max_sweeps
    for k in range(1, max_sweeps + 1):
        Ak, Wk = J.jacobi_sweeps_torch(Ak, Wk, 1, step=step)
        offs.append(float(J.off_norm(Ak).max()))
        if offs[-1] < tol:
            sweeps = k
            break
    print(f"SWEEPS mode={label} n={n} B={B} off=" + ",".join(f"{o:.1e}" for o in offs), flush=True)

    def run():
        Ad = A.to(dev)
        W = torch.eye(n, device=dev).expand(B, n, n).contiguous()
        Af, Wf = J.jacobi_sweeps_torch(Ad, W, sweeps, step=step)
        return J.finish(Af.cpu(), Wf.cpu(), n)

    out = run()
    return out, _time(run, reps, sync), sweeps


def _run_nki(A, dev, sync, reps):
    import trnsolver
    from trnsolver.nki import jacobi as JK

    def run():
        w, V = trnsolver.eigh(A.to(dev), method="jacobi")
        return w.cpu(), V.cpu()

    out = run()
    return out, _time(run, reps, sync), JK.last_sweeps


def _run_eigh_default(A, dev, sync, reps, region):
    import os

    import trnsolver

    os.environ["TRNSOLVER_EIGH_JACOBI_AUTO"] = region

    def run():
        w, V = trnsolver.eigh(A.to(dev))
        return w.cpu(), V.cpu()

    try:
        out = run()
        return out, _time(run, reps, sync), 0
    finally:
        os.environ.pop("TRNSOLVER_EIGH_JACOBI_AUTO", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="cpu,nki")
    ap.add_argument("--n", default="32,64,128")
    ap.add_argument("--B", default="16,64,256,1024")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tol", type=float, default=2e-6)
    ap.add_argument("--max-sweeps", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    modes = args.modes.split(",")

    from trnsolver import _jacobi as J

    dev = None
    sync = lambda: None  # noqa: E731
    if any(m != "cpu" for m in modes):
        import torch_neuronx

        dev = torch.device("neuron")
        sync = torch_neuronx.synchronize

    compiled_step = None
    for n in [int(x) for x in args.n.split(",")]:
        for B in [int(x) for x in args.B.split(",")]:
            A64 = _sym(B, n, seed=1000 * n + B)
            A = A64.float()
            for mode in modes:
                if mode == "cpu":
                    out, t, sweeps = _run_cpu(A, args.reps)
                elif mode == "torch":
                    out, t, sweeps = _run_torch(
                        A, J.jacobi_step, dev, sync, args.reps, args.tol, args.max_sweeps, mode
                    )
                elif mode == "compiled":
                    if compiled_step is None:
                        compiled_step = torch.compile(
                            J.jacobi_step, backend="neuron", dynamic=False
                        )
                    out, t, sweeps = _run_torch(
                        A, compiled_step, dev, sync, args.reps, args.tol, args.max_sweeps, mode
                    )
                elif mode == "nki":
                    out, t, sweeps = _run_nki(A, dev, sync, args.reps)
                elif mode == "aten":
                    out, t, sweeps = _run_eigh_default(A, dev, sync, args.reps, "off")
                elif mode == "auto":
                    out, t, sweeps = _run_eigh_default(A, dev, sync, args.reps, "")
                else:
                    raise ValueError(mode)
                eig, res, orth = _acc(A64, *out)
                print(
                    f"RESULT mode={mode} n={n} B={B} sweeps={sweeps} ms={t * 1e3:.2f} "
                    f"us_per_matrix={t * 1e6 / B:.1f} eig_err={eig:.1e} res={res:.1e} "
                    f"orth={orth:.1e}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
