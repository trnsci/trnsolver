"""Batched Jacobi eigh split across logical NeuronCores vs CPU LAPACK (#48).

Prints one ``RESULT`` line per (mode, cores, n, B):

    RESULT mode=<m> cores=<k> n=<n> B=<B> ms=<median> sweeps=<s> eig_err=<..>

Modes:
    cpu      batched ``torch.linalg.eigh`` on host FP32 input with
             ``--threads`` threads (pin them with ``taskset``).
    cpupar   the batch split into ``--threads`` chunks, one Python thread
             each with one intra-op thread: batch-parallel LAPACK, the
             stronger CPU baseline when there are more host cores.
    host     ``trnsolver.eigh(A_host, method="jacobi")`` with
             ``TRNSOLVER_EIGH_CORES=k``: host input, each chunk copied
             straight to its core, results back on the host.
    eigh     ``trnsolver.eigh(A.to("neuron:0"), method="jacobi")`` with
             ``TRNSOLVER_EIGH_CORES=k``: device input, staged through the
             host to the other cores and back; timed from host input to
             host output like ``scripts/bench_jacobi.py``.
    proc     one process per core (``NEURON_RT_VISIBLE_CORES=i``), each
             solving its chunk from host input; the wall time of a rep is
             the slowest process. A reference for the threaded modes.

Timings are warm medians of ``--reps`` runs, synchronised on every core.
Needs ``NEURON_RT_VISIBLE_CORES`` covering the cores used (e.g. ``0-3``).
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch


def _sym(B, n, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, n, n, generator=g, dtype=torch.float64)
    return (X + X.transpose(1, 2)) / 2


def _eig_err(A64, w):
    w64 = torch.linalg.eigvalsh(A64)
    return float((w.double().cpu() - w64).abs().max() / w64.abs().max())


def _time(fn, reps, sync):
    out = fn()
    sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    return out, statistics.median(ts)


_CPU_POOL = None


def _cpu_par(A, threads):
    global _CPU_POOL
    if _CPU_POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        torch.set_num_threads(1)
        _CPU_POOL = ThreadPoolExecutor(threads)
    parts = A.chunk(threads)
    res = list(_CPU_POOL.map(torch.linalg.eigh, parts))
    return torch.cat([r[0] for r in res]), torch.cat([r[1] for r in res])


def _proc_worker(core, n, B, reps, start, stop, barrier, q):
    import torch_neuronx

    from trnsolver.nki import jacobi as JK

    A = _sym(B, n, seed=1000 * n + B).float()[start:stop].contiguous()
    dev = torch.device("neuron")

    def run():
        w, V, _ = JK._solve(A.to(dev))
        return w.cpu(), V.cpu()

    run()
    torch_neuronx.synchronize()
    ts = []
    for _ in range(reps):
        barrier.wait()
        t = time.perf_counter()
        run()
        torch_neuronx.synchronize()
        ts.append(time.perf_counter() - t)
    q.put((core, ts))


def _run_proc(n, B, k, reps, cores):
    import multiprocessing as mp

    from trnsolver.nki.jacobi import split_batch

    chunks = split_batch(B, n, k)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(chunks))
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_proc_worker, args=(cores[i], n, B, reps, s, e, barrier, q))
        for i, (s, e) in enumerate(chunks)
    ]
    saved = os.environ.get("NEURON_RT_VISIBLE_CORES")
    for i, p in enumerate(procs):
        # Set before the child starts: importing torch may load torch_neuronx.
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(cores[i])
        p.start()
    if saved is None:
        os.environ.pop("NEURON_RT_VISIBLE_CORES", None)
    else:
        os.environ["NEURON_RT_VISIBLE_CORES"] = saved
    res = [q.get() for _ in procs]
    for p in procs:
        p.join()
    per_rep = [max(ts[r] for _, ts in res) for r in range(reps)]
    return statistics.median(per_rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="cpu,host,eigh")
    ap.add_argument("--cores", default="1,2,4")
    ap.add_argument("--n", default="32,64,128")
    ap.add_argument("--B", default="64,256,1024,4096")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    modes = args.modes.split(",")
    core_counts = [int(c) for c in args.cores.split(",")]

    sync = lambda: None  # noqa: E731
    devices = []
    if any(m in ("host", "eigh") for m in modes):
        import torch_neuronx

        ndev = torch_neuronx.device_count()
        devices = [torch.device(f"neuron:{i}") for i in range(ndev)]

        def sync():
            for d in devices:
                torch_neuronx.synchronize(d)

        print(f"DEVICES {ndev}", flush=True)

    import trnsolver
    from trnsolver.nki import jacobi as JK

    for n in [int(x) for x in args.n.split(",")]:
        for B in [int(x) for x in args.B.split(",")]:
            A64 = _sym(B, n, seed=1000 * n + B)
            A = A64.float()
            for mode in modes:
                ks = [0] if mode in ("cpu", "cpupar") else core_counts
                for k in ks:
                    sweeps = 0
                    err = float("nan")
                    if mode == "cpu":
                        out, t = _time(lambda A=A: torch.linalg.eigh(A), args.reps, sync)
                        err = _eig_err(A64, out[0])
                    elif mode == "cpupar":
                        out, t = _time(lambda A=A: _cpu_par(A, args.threads), args.reps, sync)
                        err = _eig_err(A64, out[0])
                    elif mode == "host":
                        os.environ["TRNSOLVER_EIGH_CORES"] = str(k)
                        out, t = _time(
                            lambda A=A: trnsolver.eigh(A, method="jacobi"), args.reps, sync
                        )
                        os.environ.pop("TRNSOLVER_EIGH_CORES", None)
                        sweeps = JK.last_sweeps
                        err = _eig_err(A64, out[0])
                    elif mode == "eigh":
                        os.environ["TRNSOLVER_EIGH_CORES"] = str(k)

                        def run(A=A):
                            w, V = trnsolver.eigh(A.to(devices[0]), method="jacobi")
                            return w.cpu(), V.cpu()

                        out, t = _time(run, args.reps, sync)
                        os.environ.pop("TRNSOLVER_EIGH_CORES", None)
                        sweeps = JK.last_sweeps
                        err = _eig_err(A64, out[0])
                    elif mode == "proc":
                        t = _run_proc(n, B, k, args.reps, list(range(k)))
                    else:
                        raise ValueError(mode)
                    print(
                        f"RESULT mode={mode} cores={k} threads={args.threads} n={n} B={B} "
                        f"ms={t * 1e3:.2f} sweeps={sweeps} eig_err={err:.1e}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
