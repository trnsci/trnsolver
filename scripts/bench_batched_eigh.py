#!/usr/bin/env python3
"""Batched small eigh on Trainium against batched CPU LAPACK (#48).

For each n in --sizes and B in --batches, prints one line:

    RESULT n=128 B=256 chunk=16 launches=64 batched_device_ms=... \
        per_matrix_device_ms=... cpu_batched_ms=... aten_neuron_ms=... \
        batched_device_ms_per_matrix=... ...

All times are warm medians over --reps runs (after --warmup runs) of the
whole batch, in milliseconds:

* batched_device_ms: one ``trnsolver.eigh(A)`` on the (B, n, n) batch with
  ``TRNSOLVER_EIGH_TRIDIAG=device`` and ``TRNSOLVER_EIGH_NKI_MIN_N=0`` (the
  batched kernels, ``4 * ceil(B / chunk)`` launches).
* per_matrix_device_ms: B calls ``trnsolver.eigh(A[b])`` on the same path.
  Above --per-matrix-max-b matrices this is measured on the first
  --per-matrix-max-b and scaled to B (``per_matrix_scaled=1``).
* cpu_batched_ms: ``torch.linalg.eigh(A)`` on the host.
* aten_neuron_ms: ``torch.linalg.eigh`` on the batch already on the
  ``neuron`` device (the default cutover path), result left on device.

Inputs are created on the host with a fixed seed. With --input host (the
default) the trnsolver calls take the host tensor, so their time includes
the host/device copies; --input device moves the batch to ``neuron`` first.
Every device timing ends with ``torch_neuronx.synchronize()``.

Pin host threads for comparable CPU numbers, for example:

    taskset -c 0-15 python scripts/bench_batched_eigh.py --threads 16

--stages adds a STAGES line per configuration with the time of each stage
of the batched device path (reduction, host tridiagonal solves, reflector
application, projection, host eigh(H), rotation).

Off hardware (no ``torch_neuronx``) the device columns are skipped unless
TRNSOLVER_USE_SIMULATOR=1, in which case the kernels run in the CPU
simulator (correctness smoke test only; the times mean nothing).
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch

os.environ.setdefault("TRNSOLVER_EIGH_TRIDIAG", "device")
os.environ.setdefault("TRNSOLVER_EIGH_NKI_MIN_N", "0")

import trnsolver  # noqa: E402
import trnsolver.eigen as eigen  # noqa: E402
from trnsolver.nki import batched  # noqa: E402

try:
    import torch_neuronx

    _HAS_NEURON = hasattr(torch_neuronx, "synchronize")
except Exception:
    torch_neuronx = None
    _HAS_NEURON = False

_SIM = os.environ.get("TRNSOLVER_USE_SIMULATOR", "").lower() in ("1", "true", "yes")


def _sync():
    if _HAS_NEURON:
        torch_neuronx.synchronize()


def _timed(fn, warmup, reps):
    """Warm median of fn() in ms, plus the first (cold, compiling) call in ms."""
    t0 = time.perf_counter()
    fn()
    _sync()
    cold = (time.perf_counter() - t0) * 1e3
    for _ in range(max(warmup - 1, 0)):
        fn()
        _sync()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times), cold


def _sym_batch(nb, n, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(nb, n, n, generator=g)
    return 0.5 * (A + A.transpose(1, 2))


def _max_err(A, w, V):
    A, w, V = A.double(), w.detach().cpu().double(), V.detach().cpu().double()
    w_ref = torch.linalg.eigvalsh(A)
    resid = (A @ V - V * w.unsqueeze(-2)).abs().max().item()
    return (w - w_ref).abs().max().item(), resid


def _stages(A, warmup, reps):
    """Per-stage warm medians (ms) of the batched device path."""
    b = batched
    out = {}
    d, e, V_refs = b.householder_tridiag_batched_device(A)
    out["tridiag"], _ = _timed(lambda: b.householder_tridiag_batched_device(A), warmup, reps)
    out["host_tridiag_solve"], _ = _timed(
        lambda: eigen._tridiag_eigh_host_batched(d, e, 1e-10), warmup, reps
    )
    _, Q_right = eigen._tridiag_eigh_host_batched(d, e, 1e-10)
    Q_right = Q_right.to(A.device)
    out["apply_reflectors"], _ = _timed(
        lambda: b.apply_reflectors_batched(V_refs, Q_right), warmup, reps
    )
    V_T = b.apply_reflectors_batched(V_refs, Q_right)
    out["rr_project"], _ = _timed(lambda: b.rr_project_batched(A, V_T), warmup, reps)
    H = b.rr_project_batched(A, V_T)
    H = 0.5 * (H + H.transpose(1, 2))
    out["host_eigh_H"], _ = _timed(lambda: torch.linalg.eigh(H), warmup, reps)
    _, Q_corr = torch.linalg.eigh(H)
    Q_corr = Q_corr.contiguous()
    out["rr_rotate"], _ = _timed(lambda: b.rr_rotate_batched(V_T, Q_corr), warmup, reps)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 256, 1024])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--per-matrix-max-b", type=int, default=64)
    p.add_argument("--input", choices=["host", "device"], default="host")
    p.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (0: leave)")
    p.add_argument("--stages", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device_ok = _HAS_NEURON or _SIM
    if args.input == "device" and not _HAS_NEURON:
        raise SystemExit("--input device needs torch_neuronx and a neuron device")

    chunk = batched.batch_chunk_max()
    print(
        f"# torch {torch.__version__} threads={torch.get_num_threads()} chunk={chunk} "
        f"neuron={_HAS_NEURON} simulator={_SIM} input={args.input} "
        f"warmup={args.warmup} reps={args.reps} "
        f"TRNSOLVER_EIGH_TRIDIAG={os.environ['TRNSOLVER_EIGH_TRIDIAG']} "
        f"TRNSOLVER_EIGH_NKI_MIN_N={os.environ['TRNSOLVER_EIGH_NKI_MIN_N']}",
        flush=True,
    )

    for n in args.sizes:
        for nb in args.batches:
            A = _sym_batch(nb, n, args.seed)
            fields = {
                "n": n,
                "B": nb,
                "chunk": chunk,
                "launches": batched.n_launches(nb) if nb > 1 else 5,
            }

            cpu_ms, _ = _timed(lambda A=A: torch.linalg.eigh(A), args.warmup, args.reps)
            fields["cpu_batched_ms"] = cpu_ms

            if device_ok:
                A_in = A.to("neuron") if args.input == "device" else A
                arg = A_in if nb > 1 else A_in[0]
                w, V = trnsolver.eigh(arg)
                werr, resid = _max_err(A if nb > 1 else A[0], w, V)
                dev_ms, cold = _timed(lambda arg=arg: trnsolver.eigh(arg), args.warmup, args.reps)
                fields["batched_device_ms"] = dev_ms
                fields["batched_device_cold_ms"] = cold

                m = min(nb, args.per_matrix_max_b)
                mats = [A_in[b] for b in range(m)]

                def per_matrix(mats=mats):
                    for M in mats:
                        trnsolver.eigh(M)

                pm_ms, _ = _timed(per_matrix, args.warmup, args.reps)
                fields["per_matrix_device_ms"] = pm_ms * nb / m
                fields["per_matrix_scaled"] = int(m < nb)
                fields["eig_err"] = werr
                fields["resid"] = resid

            if _HAS_NEURON:
                A_dev = A.to("neuron")
                _sync()
                aten_ms, _ = _timed(
                    lambda A_dev=A_dev: torch.linalg.eigh(A_dev), args.warmup, args.reps
                )
                fields["aten_neuron_ms"] = aten_ms

            for key in ("batched_device_ms", "per_matrix_device_ms", "cpu_batched_ms"):
                if key in fields:
                    fields[key + "_per_matrix"] = fields[key] / nb
            if "aten_neuron_ms" in fields:
                fields["aten_neuron_ms_per_matrix"] = fields["aten_neuron_ms"] / nb
            if "batched_device_ms" in fields:
                fields["speedup_vs_cpu"] = cpu_ms / fields["batched_device_ms"]

            print("RESULT " + " ".join(_fmt(k, v) for k, v in fields.items()), flush=True)

            if args.stages and device_ok and nb > 1 and 2 <= n <= 128:
                A_in = A.to("neuron") if args.input == "device" else A
                st = _stages(A_in, args.warmup, args.reps)
                print(
                    f"STAGES n={n} B={nb} " + " ".join(_fmt(k + "_ms", v) for k, v in st.items()),
                    flush=True,
                )


def _fmt(k, v):
    if isinstance(v, float):
        return f"{k}={v:.4g}"
    return f"{k}={v}"


if __name__ == "__main__":
    main()
