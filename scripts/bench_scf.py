"""DF-RHF SCF end to end: per-stage times, CPU vs device, energy vs fp64 (#15).

Runs ``examples/scf_df_rhf.py``'s :func:`run_scf` in several modes and prints
one ``RESULT`` line per (system, mode) plus markdown tables. Every mode
first runs ``--warmup-iters`` iterations (compiles, first-call costs;
reported as ``cold_ms``), then the timed run to convergence.

    cpu      float32 on the host, all trnsci backends ``pytorch``. Pin the
             threads (``taskset``, ``--threads``) for a fair baseline.
    cpu64    float64 on the host: the reference loop.
    neuron   float32 on the ``neuron`` device, ``auto`` backends. Stage
             times synchronise the device around each stage.
    neuron-nosync  the warm device loop without stage synchronisation:
             the application-visible time to convergence.
    neuron-jk<k>, neuron-nosync-jk<k>  the same with the J/K build split over
             logical NeuronCores 0..k-1 (aux blocks dealt round-robin, inputs
             and partial results staged through the host); needs
             ``NEURON_RT_VISIBLE_CORES`` to cover them.

Energies are compared with the fp64 reference: PySCF's DF-RHF energy on the
same integrals when the system comes from PySCF, else the ``cpu64`` loop.
``E(C) - E_ref`` re-evaluates the energy in float64 at the run's final
occupied orbitals (S-orthonormalised), separating fp32 evaluation rounding
from the orbital error.

``--eigh-only`` times one ``eigh`` of size n (the SCF's per-iteration solve):
host LAPACK, ``trnsolver.eigh`` on a ``neuron`` tensor (default dispatch),
its host-fallback parts (D2H, LAPACK, H2D), the single-matrix Jacobi kernel
(n <= 128) and ``trnsolver.eigh`` of a host tensor with the NKI backend.

``--solver subspace`` replaces the per-iteration eigh by
``trnsolver.occupied_subspace`` (warm-started Chebyshev-filtered subspace
iteration; ``--subspace-opts`` passes its keyword arguments as JSON).

Examples:
    OMP_NUM_THREADS=2 taskset -c 2,8 python scripts/bench_scf.py \\
        --systems h2-lattice:16,pyscf:benzene --modes cpu,cpu64,neuron,neuron-nosync
    OMP_NUM_THREADS=2 taskset -c 2,8 python scripts/bench_scf.py --eigh-only \\
        --sizes 128,256,512,1024
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))

from scf_df_rhf import STAGES, System, energy_at_orbitals, make_system, run_scf  # noqa: E402


def _set_backends(name: str) -> None:
    import trnblas
    import trntensor

    import trnsolver

    for lib in (trnsolver, trnblas, trntensor):
        lib.set_backend(name)


def _load_system(spec: str, cache_dir: str | None) -> System:
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, spec.replace(":", "_") + ".pt")
        if os.path.exists(path):
            return System(**torch.load(path))
    t = time.perf_counter()
    system = make_system(spec)
    print(f"# built {spec} in {time.perf_counter() - t:.1f} s", flush=True)
    if path:
        torch.save(system.__dict__, path)
    return system


def _run_mode(system: System, mode: str, args):
    """(warm-up result, timed result). The warm-up runs ``--warmup-iters``
    iterations, which touch every op shape of the loop (device compiles,
    library first-call costs); its time is reported as ``cold``."""
    kw = dict(
        solver=args.solver,
        subspace_opts=json.loads(args.subspace_opts) if args.subspace_opts else None,
        k_cache_gb=args.k_cache_gb,
        jk_square=args.jk_square,
        jk_fused={"auto": None, "fused": True, "unpack": False}[args.jk_k],
    )
    if mode in ("cpu", "cpu64"):
        _set_backends("pytorch")
        kw.update(device="cpu", dtype=torch.float64 if mode == "cpu64" else torch.float32)
    else:
        _set_backends("auto")
        import torch_neuronx  # noqa: F401

        base, _, cores = mode.partition("-jk")
        kw.update(device="neuron", dtype=torch.float32, stage_sync=base == "neuron")
        if cores:
            kw["jk_devices"] = [f"neuron:{i}" for i in range(int(cores))]
    cold = run_scf(system, max_iter=args.warmup_iters, **kw)
    return cold, run_scf(system, max_iter=args.max_iter, **kw)


def bench_scf(args) -> None:
    rows = []
    for spec in args.systems.split(","):
        system = _load_system(spec, args.cache_dir)
        print(
            f"# {system.name}: n_ao={system.nao} naux={system.naux} nocc={system.nocc}",
            flush=True,
        )
        ref = system.ref_energy
        results = {}
        for mode in args.modes.split(","):
            cold, res = _run_mode(system, mode, args)
            results[mode] = res
            if mode == "cpu64" and ref is None:
                ref = res.energy
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                tag = f"{spec.replace(':', '_')}_{mode}_{args.solver}"
                torch.save(
                    {"C_occ": res.C_occ, "E": res.energy, "iters": res.iterations},
                    os.path.join(args.save_dir, tag + ".pt"),
                )
            e_dens = energy_at_orbitals(system, res.C_occ) if args.density_energy else float("nan")
            med = res.stage_median_ms()
            per_it = res.total_ms / max(res.iterations, 1)
            row = dict(
                system=system.name,
                nao=system.nao,
                naux=system.naux,
                nocc=system.nocc,
                mode=mode,
                iters=res.iterations,
                conv=res.converged,
                E=res.energy,
                e_dens=e_dens,
                total_ms=res.total_ms,
                cold_ms=cold.total_ms + sum(cold.setup_ms.values()),
                per_it=per_it,
                setup=res.setup_ms,
                **{f"med_{s}": med[s] for s in STAGES},
            )
            rows.append(row)
            print(
                f"RESULT system={system.name} nao={system.nao} naux={system.naux} "
                f"nocc={system.nocc} mode={mode} iters={res.iterations} conv={res.converged} "
                f"E={res.energy:.10f} total_ms={res.total_ms:.1f} cold_ms={row['cold_ms']:.1f} "
                + " ".join(f"{s}={med[s]:.3f}" for s in STAGES)
                + " setup="
                + ",".join(f"{k}:{v:.1f}" for k, v in res.setup_ms.items()),
                flush=True,
            )
        for row in rows:
            if row["system"] == system.name:
                row["ref"] = ref
        del system
    _print_tables(rows)


def _print_tables(rows) -> None:
    print("\n### Time to convergence (warm, ms)\n")
    print(
        "| system | n_ao / naux / nocc | mode | iters | total | per iter | "
        "warm-up (incl. compile) | E - E_ref (fp64) | E(C) - E_ref |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ref = r.get("ref")
        de = f"{r['E'] - ref:+.2e}" if ref is not None else "n/a"
        dd = (
            f"{r['e_dens'] - ref:+.2e}" if ref is not None and r["e_dens"] == r["e_dens"] else "n/a"
        )
        conv = "" if r["conv"] else " (not conv.)"
        print(
            f"| {r['system']} | {r['nao']} / {r['naux']} / {r['nocc']} | {r['mode']} | "
            f"{r['iters']}{conv} | {r['total_ms']:.1f} | {r['per_it']:.2f} | "
            f"{r['cold_ms']:.0f} | {de} | {dd} |"
        )
    print("\n### Median per-iteration stage time (ms; stage-synchronised runs)\n")
    print("| system | mode | J/K | Fock | energy | DIIS | eigh | density | dominant |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["mode"].startswith("neuron-nosync"):
            continue
        vals = {s: r[f"med_{s}"] for s in STAGES}
        tot = sum(vals.values()) or 1.0
        dom = max(vals, key=vals.get)
        print(
            f"| {r['system']} | {r['mode']} | {vals['jk']:.2f} | {vals['fock']:.2f} | "
            f"{vals['energy']:.2f} | {vals['diis']:.2f} | {vals['eigh']:.2f} | "
            f"{vals['density']:.2f} | {dom} ({100 * vals[dom] / tot:.0f}%) |"
        )


# ---------------------------------------------------------------------------
# eigh at SCF sizes
# ---------------------------------------------------------------------------


def _sym(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, n, generator=g, dtype=torch.float64)
    return ((X + X.T) / 2).float()


def _median(fn, reps, sync):
    fn()
    sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(1e3 * (time.perf_counter() - t))
    return statistics.median(ts)


def bench_eigh(args) -> None:
    import trnsolver

    rows = []
    have_neuron = "neuron" in args.modes or "all" in args.modes
    if have_neuron:
        import torch_neuronx

        sync = torch_neuronx.synchronize
    for n in [int(x) for x in args.sizes.split(",")]:
        A = _sym(n)
        A64 = A.double()
        w64 = torch.linalg.eigvalsh(A64)
        r = {"n": n}
        r["cpu"] = _median(lambda A=A: torch.linalg.eigh(A), args.reps, lambda: None)
        if have_neuron:
            _set_backends("auto")
            Ad = A.to("neuron")
            sync()
            r["neuron"] = _median(lambda Ad=Ad: trnsolver.eigh(Ad), args.reps, sync)
            w, _ = trnsolver.eigh(Ad)
            r["neuron_err"] = float((w.cpu().double() - w64).abs().max() / w64.abs().max())
            r["d2h"] = _median(lambda Ad=Ad: Ad.cpu(), args.reps, sync)
            Ah = Ad.cpu()
            wh, Vh = torch.linalg.eigh(Ah)
            r["h2d"] = _median(
                lambda wh=wh, Vh=Vh: (wh.to("neuron"), Vh.to("neuron")), args.reps, sync
            )
            r["aten_dev"] = _median(lambda Ad=Ad: torch.linalg.eigh(Ad), args.reps, sync)
            if n <= 128:
                r["jacobi"] = _median(
                    lambda Ad=Ad: trnsolver.eigh(Ad, method="jacobi"), args.reps, sync
                )
                wj, _ = trnsolver.eigh(Ad, method="jacobi")
                r["jacobi_err"] = float((wj.cpu().double() - w64).abs().max() / w64.abs().max())
            if n <= args.host_nki_max_n:
                # Host tensor, NKI backend on a Neuron host: one timed call.
                t = time.perf_counter()
                trnsolver.eigh(A)
                r["host_nki"] = 1e3 * (time.perf_counter() - t)
            del Ad
        rows.append(r)
        print("RESULT eigh " + " ".join(f"{k}={v}" for k, v in r.items()), flush=True)

    def f(r, k):
        return f"{r[k]:.2f}" if k in r else "-"

    print("\n### One eigh of size n (warm median ms, fp32)\n")
    print(
        "| n | host LAPACK | trnsolver.eigh (neuron tensor) | aten eigh (neuron) | "
        "D2H | H2D (w, V) | Jacobi kernel, B=1 | trnsolver.eigh (host tensor, auto) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['n']} | {f(r, 'cpu')} | {f(r, 'neuron')} | {f(r, 'aten_dev')} | "
            f"{f(r, 'd2h')} | {f(r, 'h2d')} | {f(r, 'jacobi')} | {f(r, 'host_nki')} |"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--systems", default="h2-lattice:16")
    ap.add_argument("--modes", default="cpu,cpu64")
    ap.add_argument(
        "--solver", default="orth", choices=("orth", "generalized", "purify", "subspace")
    )
    ap.add_argument(
        "--subspace-opts",
        default="",
        help="JSON keyword arguments for occupied_subspace, e.g. '{\"passes\": 2}'",
    )
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--warmup-iters", type=int, default=3)
    ap.add_argument("--k-cache-gb", type=float, default=4.0)
    ap.add_argument(
        "--jk-square",
        action="store_true",
        help="keep the unpacked (n, n, naux) integrals resident in aux blocks",
    )
    ap.add_argument(
        "--jk-k",
        default="auto",
        choices=("auto", "fused", "unpack"),
        help="K build: trntensor.df_exchange from packed rows (fused), the unpack + einsum "
        "+ syrk blocks (unpack), or fused on a device when available (auto)",
    )
    ap.add_argument("--cache-dir", default=None, help="cache built systems as .pt files")
    ap.add_argument(
        "--no-density-energy",
        dest="density_energy",
        action="store_false",
        help="skip the float64 re-evaluation at each run's density",
    )
    ap.add_argument("--save-dir", default=None, help="save each run's C_occ and energy")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--eigh-only", action="store_true")
    ap.add_argument("--sizes", default="128,256,512,1024")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--host-nki-max-n", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.eigh_only:
        bench_eigh(args)
    else:
        bench_scf(args)


if __name__ == "__main__":
    main()
