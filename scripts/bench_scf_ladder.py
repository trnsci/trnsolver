"""DF-RHF + DF-MP2 size ladder: linear alkanes and water clusters, 500 to 3500 AOs (#15).

Systems are synthetic geometries (examples/scf_df_rhf.py): all-trans zigzag
alkanes C_k H_{2k+2} (``alkane<k>``) and k-water clusters on a cubic lattice
with seeded orientations (``water<k>``), in cc-pVDZ, aug-cc-pVDZ and cc-pVTZ
with the matching -jkfit fitting basis. For each basis the unit count k is
chosen to land nearest each target size (``--targets``, default 500, 1000,
1658 (C40H82 / aug-cc-pVDZ, the window-3 system), 2500, 3500 AOs).

``--dry-run`` computes no integrals. It prints, per system, the sizes and the
memory each stage needs, and what that means for a Trn2 window:

    B fp32          packed (npair, naux) DF integrals in fp32: the J/K
                    working set, and the out-of-core store on disk
    PySCF in-core   two float64 copies (``cholesky_eri`` + transpose), what
                    the old in-memory build needed on the host
    cores / chips   logical NeuronCores (LNC=2, 24 GiB each, 20 GB usable
                    for resident chunks) to hold B resident; chips = cores / 4
    plan            trn2.3xlarge (1 chip, 96 GB HBM, ~124 GB host RAM)
                    resident, or streamed: 80 GB resident, the rest uploaded
                    every iteration from host RAM (or NVMe once B exceeds
                    host RAM); else trn2.48xlarge (16 chips) resident
    B_ia fp32       the DF-MP2 three-index MO tensor (nocc x nvir x naux)

and rough times, scaled from the window-3 measurements at C40H82 /
aug-cc-pVDZ (n_ao 1658, naux 6064, nocc 161; #15): the out-of-core build
(243 CPU-s here for 8.3e9 elements of B), the J/K stage (327 ms per
iteration on 4 cores, scaled by nocc n^2 naux and the core count), the host
eigh (301 ms at m 1622, scaled by m^3) and the per-iteration upload at
14 GB/s when streaming. These are planning estimates, not measurements.

Without ``--dry-run`` each system up to ``--max-nao`` runs end to end: the
out-of-core build (time, peak RSS), the fp32 SCF (noise-aware convergence,
iterations, time, solver), the float64 energy at the SCF orbitals, DF-MP2,
and PySCF DF-RHF + DF-MP2 (same jkfit basis, matched linear-dependency
threshold) up to ``--ref-max-nao``. One ``RESULT {json}`` line per system,
then a markdown table; ``--out`` writes the rows as JSON.

Examples:
    python scripts/bench_scf_ladder.py --dry-run
    python scripts/bench_scf_ladder.py --families alkane --bases cc-pvdz --max-nao 1100 \\
        --ref-max-nao 600 --store-dir ~/scratch/df
    python scripts/bench_scf_ladder.py --device neuron \\
        --jk-devices neuron:0,neuron:1,neuron:2,neuron:3 --max-nao 3600   # Trn2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))

import df_store  # noqa: E402
import scf_df_rhf as scf  # noqa: E402

FAMILIES = ("alkane", "water")
BASES = ("cc-pvdz", "aug-cc-pvdz", "cc-pvtz")
TARGETS = (500, 1000, 1658, 2500, 3500)  # 1658: C40H82 / aug-cc-pVDZ, window 3

# Hardware (Trn2, LNC=2).
CORE_HBM_GB = 24 * 1.073741824  # 24 GiB per logical core
CORE_RESIDENT_GB = scf.DEVICE_RESIDENT_GB  # usable for resident B chunks
CORES_PER_CHIP = 4
HOST_RAM_GB = {"trn2.3xlarge": 124.0, "trn2.48xlarge": 2000.0}
CHIPS = {"trn2.3xlarge": 1, "trn2.48xlarge": 16}
H2D_GBPS = 14.0  # host link, per chip (trntensor docs)
NVME_GBPS = 2.0  # sequential read, planning figure

# Window-3 anchors (C40H82 / aug-cc-pVDZ, #15).
_A_NAO, _A_NAUX, _A_NOCC, _A_RANK = 1658, 6064, 161, 1622
_A_JK_S_4CORES = 0.327
_A_EIGH_S = 0.301
_A_BUILD_CPU_S_PER_ELEM = 243.0 / (1375311 * 6064)  # 160 s wall, 243 CPU-s (12 threads here)


def _lindep_for(basis: str) -> tuple[str, float]:
    """(orth, lindep): canonical 1e-6 for diffuse bases (fp32 needed it at
    n_ao 1658, #15), the symmetric default otherwise."""
    return ("canonical", 1e-6) if basis.startswith("aug") else ("symmetric", 1e-7)


def ladder(families, bases, targets) -> list[dict]:
    """The systems: for each (family, basis), the unit count k nearest each target n_ao."""
    out = []
    for fam in families:
        for basis in bases:
            a = scf.pyscf_sizes(f"{fam}2", basis)
            b = scf.pyscf_sizes(f"{fam}3", basis)
            per = b["nao"] - a["nao"]
            base = a["nao"] - 2 * per  # n_ao = per * k + base
            seen = set()
            for t in targets:
                k = max(2, round((t - base) / per))
                if k in seen:
                    continue
                seen.add(k)
                sz = scf.pyscf_sizes(f"{fam}{k}", basis)
                spec = f"pyscf:{fam}{k}" + ("" if basis == "cc-pvdz" else f"@{basis}")
                out.append(dict(family=fam, k=k, basis=basis, spec=spec, target=t, **sz))
    return out


def estimate(row: dict, host_threads: int = 2, iters: int = 25) -> dict:
    """Memory and time estimates for one system (no integrals)."""
    n, naux, nocc = row["nao"], row["naux"], row["nocc"]
    npair = n * (n + 1) // 2
    nvir = n - nocc
    e = dict(npair=npair, nvir=nvir)
    e["B32_GB"] = df_store.df_bytes(n, naux, 4) / 1e9
    e["pyscf_incore_GB"] = df_store.pyscf_incore_bytes(n, naux) / 1e9
    e["store_build_peak_GB"] = (3 * (1 << 30) + naux * naux * 8 + 0.6e9) / 1e9
    e["bia32_GB"] = nocc * nvir * naux * 4 / 1e9
    e["disk_GB"] = e["B32_GB"] + e["bia32_GB"]
    e["cores_resident"] = math.ceil(e["B32_GB"] / CORE_RESIDENT_GB)
    e["chips_resident"] = math.ceil(e["cores_resident"] / CORES_PER_CHIP)
    # J/K on one chip (4 cores) scales with the K work nocc n^2 naux.
    jk = _A_JK_S_4CORES * (nocc * n * n * naux) / (_A_NOCC * _A_NAO**2 * _A_NAUX)
    e["jk_s_1chip"] = jk
    e["eigh_s"] = _A_EIGH_S * (n / _A_RANK) ** 3
    e["build_s"] = _A_BUILD_CPU_S_PER_ELEM * npair * naux / host_threads
    # MP2: nocc^2 nvir^2 naux multiply-adds (pair symmetry halves 2x) at ~20 TFLOP/s per chip.
    e["mp2_s_1chip"] = nocc * nocc * nvir * nvir * naux / 20e12
    chip_resident = CORES_PER_CHIP * CORE_RESIDENT_GB
    streamed = max(0.0, e["B32_GB"] - chip_resident)
    ram = HOST_RAM_GB["trn2.3xlarge"] - 8  # leave room for the process
    e["stream_src"] = "" if streamed == 0 else ("host RAM" if e["B32_GB"] <= ram else "NVMe")
    rate = H2D_GBPS if e["stream_src"] != "NVMe" else NVME_GBPS
    e["stream_GB_per_iter"] = streamed
    # Uploads overlap the J/K compute: an iteration takes the longer of the two.
    e["iter_s_3xl"] = max(jk, streamed / rate) + e["eigh_s"]
    chips48 = min(CHIPS["trn2.48xlarge"], max(1, e["chips_resident"]))
    e["iter_s_48xl"] = jk / chips48 + e["eigh_s"]
    e["scf_min_3xl"] = iters * e["iter_s_3xl"] / 60
    if streamed == 0:
        e["plan"] = "3xl, resident"
    elif e["stream_src"] == "host RAM":
        e["plan"] = (
            f"3xl, stream {streamed:.0f} GB/it from RAM; or 48xl, {e['chips_resident']} chips"
        )
    else:
        e["plan"] = f"48xl, {e['chips_resident']} chips (3xl: {streamed:.0f} GB/it from NVMe)"
    return e


def print_dry_run(rows: list[dict]) -> None:
    print(
        "\n| system | basis | n_ao | naux | nocc | B fp32 GB | PySCF in-core GB | "
        "cores (chips) to hold B | J/K s/it, 1 chip | s/it 3xl | s/it 48xl | B_ia GB | "
        "MP2 s, 1 chip | disk GB | plan |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        e = r["est"]
        print(
            f"| {r['family']}{r['k']} | {r['basis']} | {r['nao']} | {r['naux']} | {r['nocc']} | "
            f"{e['B32_GB']:.1f} | {e['pyscf_incore_GB']:.0f} | {e['cores_resident']} "
            f"({e['chips_resident']}) | {e['jk_s_1chip']:.2f} | {e['iter_s_3xl']:.1f} | "
            f"{e['iter_s_48xl']:.1f} | {e['bia32_GB']:.1f} | {e['mp2_s_1chip']:.0f} | "
            f"{e['disk_GB']:.0f} | {e['plan']} |"
        )


def _set_backends(name: str) -> None:
    import trnblas
    import trntensor

    import trnsolver

    for lib in (trnsolver, trnblas, trntensor):
        lib.set_backend(name)


def _pyscf_reference(row: dict, lindep: float, orth: str) -> dict:
    """PySCF DF-RHF + DF-MP2 on the same jkfit basis; for diffuse bases with the
    same relative linear-dependency threshold as the SCF's orthogonalizer
    (``scf_df_mp2.pyscf_df_mp2_reference``), so both work in the same kept basis."""
    import scf_df_mp2 as mp2

    t0 = time.perf_counter()
    e_hf, e_corr = mp2.pyscf_df_mp2_reference(
        f"{row['family']}{row['k']}",
        row["basis"],
        lindep=lindep if orth == "canonical" else None,
    )
    return {"ref_e_hf": e_hf, "ref_e_corr": e_corr, "ref_s": time.perf_counter() - t0}


def _build_child(spec, store_dir, dtype_name, conn):
    try:
        system = scf.make_system(
            spec, outcore=True, store_dir=store_dir, store_dtype=getattr(torch, dtype_name)
        )
        conn.send(("ok", system.build.get("store")))
    except Exception as exc:  # reported in the parent
        conn.send(("error", repr(exc)))


def _build_in_child(spec: str, store_dir, dtype: torch.dtype) -> scf.System:
    """Build (or reuse) the out-of-core store in a child process, so its peak
    RSS is the build's own (ru_maxrss is per process), then open it here."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    p = ctx.Process(
        target=_build_child, args=(spec, store_dir, str(dtype).replace("torch.", ""), child)
    )
    p.start()
    status, val = parent.recv()
    p.join()
    if status != "ok":
        raise RuntimeError(f"building {spec}: {val}")
    return scf.load_store_system(val)


def run_one(row: dict, args) -> dict:
    import scf_df_mp2 as mp2

    orth, lindep = _lindep_for(row["basis"])
    out = dict(row, orth=orth, lindep=lindep)
    # float64 integrals for systems with a PySCF reference (the comparison
    # then sees the orbitals, not B's fp32 rounding: 3e-6 Eh at C20H42 /
    # cc-pVDZ); fp32 otherwise. The SCF casts to fp32 as it loads chunks.
    if args.store_dtype == "auto":
        store_dtype = torch.float64 if row["nao"] <= args.ref_max_nao else torch.float32
    else:
        store_dtype = getattr(torch, args.store_dtype)
    t0 = time.perf_counter()
    system = _build_in_child(row["spec"], args.store_dir, store_dtype)
    out["build_s"] = time.perf_counter() - t0
    out["store_dtype"] = str(store_dtype).replace("torch.", "")
    out["build"] = {k: v for k, v in system.build.items() if k != "store"}
    out["store"] = system.build.get("store")
    out["naux"] = system.naux
    dev = torch.device(args.device)
    _set_backends("pytorch" if dev.type == "cpu" else "auto")
    devices = args.jk_devices.split(",") if args.jk_devices else None
    solver = args.solver
    if solver == "auto":
        # Window 3: subspace (refresh=None) won at n_ao 1658, full eigh at <= 970.
        solver = "subspace" if system.nao > 1200 else "orth"
    sub_opts = {"refresh": None} if solver == "subspace" else None
    kw = dict(
        device=dev,
        dtype=torch.float32,
        solver=solver,
        orth=orth,
        lindep=lindep,
        jk_devices=devices,
        stage_sync=False,
        subspace_opts=sub_opts,
        max_iter=args.max_iter,
    )
    if dev.type != "cpu" and args.warmup_iters:
        scf.run_scf(system, **dict(kw, max_iter=args.warmup_iters))
    res = scf.run_scf(system, **kw)
    out.update(
        solver=solver,
        rank=res.info["rank"],
        iters=res.iterations,
        converged=res.converged,
        scf_s=res.total_ms / 1e3,
        setup_s=sum(res.setup_ms.values()) / 1e3,
        e_scf32=res.energy,
        stream=res.info.get("stream")
        and {k: v for k, v in res.info["stream"].items() if k != "iters"},
    )
    if res.info.get("stream") and res.info["stream"]["iters"]:
        it = res.info["stream"]["iters"][-1]
        out["stream_GB_per_iter"] = it["bytes"] / 1e9
        out["stream_wait_s_per_iter"] = it["wait_s"]
    t1 = time.perf_counter()
    m = mp2.run_mp2(
        system,
        res,
        orth=orth,
        lindep=lindep,
        device=dev,
        dtype=torch.float32 if args.mp2_fp32 else torch.float64,
        devices=devices,
    )
    out.update(
        e_hf=m.e_hf,
        e_corr=m.e_corr,
        e_total=m.e_total,
        mp2_wall_s=time.perf_counter() - t1,
        mp2_timings=m.timings_s,
        mp2_path=m.info["mp2_path"],
        bia_GB=m.info["bia_bytes"] / 1e9,
    )
    if system.nao <= args.ref_max_nao:
        try:
            ref = _pyscf_reference(row, lindep, orth)
            out.update(ref)
            out["err_hf"] = out["e_hf"] - ref["ref_e_hf"]
            out["err_corr"] = out["e_corr"] - ref["ref_e_corr"]
        except Exception as exc:  # reference is best effort
            out["ref_error"] = repr(exc)
    out["peak_rss_GB"] = df_store.peak_rss_bytes() / 1e9
    return out


def print_results(rows: list[dict]) -> None:
    print(
        "\n| system | basis | n_ao / naux / nocc | build s (peak RSS GB; from the store's "
        "meta when reused) | solver (rank) | "
        "SCF iters | SCF s | E_HF(C) | E_MP2 | ΔE_HF vs PySCF | ΔE_MP2 vs PySCF |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        dhf = f"{r['err_hf']:+.1e}" if "err_hf" in r else "–"
        dmp = f"{r['err_corr']:+.1e}" if "err_corr" in r else "–"
        conv = "" if r["converged"] else " (not conv.)"
        print(
            f"| {r['family']}{r['k']} | {r['basis']} | {r['nao']} / {r['naux']} / {r['nocc']} | "
            f"{r['build'].get('build_s', r['build_s']):.0f} "
            f"({r['build'].get('peak_rss_bytes', 0) / 1e9:.1f}) | "
            f"{r['solver']} ({r['rank']}) | {r['iters']}{conv} | {r['scf_s']:.1f} | "
            f"{r['e_hf']:.6f} | {r['e_corr']:.6f} | {dhf} | {dmp} |"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--bases", default=",".join(BASES))
    ap.add_argument("--targets", default=",".join(map(str, TARGETS)))
    ap.add_argument("--dry-run", action="store_true", help="estimates only, no integrals")
    ap.add_argument("--host-threads", type=int, default=2, help="for the build-time estimate")
    ap.add_argument("--max-nao", type=int, default=600, help="run systems up to this n_ao")
    ap.add_argument("--ref-max-nao", type=int, default=600, help="PySCF references up to this")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--jk-devices", default=None, help="e.g. neuron:0,neuron:1,neuron:2,neuron:3")
    ap.add_argument("--solver", default="auto", choices=("auto", "orth", "subspace"))
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--warmup-iters", type=int, default=3)
    ap.add_argument("--mp2-fp32", action="store_true", help="fp32 AO->MO and MP2 energy")
    ap.add_argument("--store-dir", default=None, help="df_store root (default $TRNSCI_DF_DIR)")
    ap.add_argument(
        "--store-dtype",
        default="auto",
        choices=("auto", "float32", "float64"),
        help="integral store dtype (auto: float64 up to --ref-max-nao, else float32)",
    )
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None, help="write the rows as JSON here")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device.startswith("neuron"):
        import torch_neuronx  # noqa: F401

    rows = ladder(
        args.families.split(","), args.bases.split(","), [int(t) for t in args.targets.split(",")]
    )
    for r in rows:
        r["est"] = estimate(r, host_threads=args.host_threads)
    if args.dry_run:
        for r in rows:
            print("RESULT " + json.dumps(r), flush=True)
        print_dry_run(rows)
        total = sum(r["est"]["B32_GB"] for r in rows)
        print(f"\n{len(rows)} systems; fp32 stores total {total:.0f} GB of disk.")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(rows, f, indent=1)
        return
    done = []
    for r in rows:
        if r["nao"] > args.max_nao:
            continue
        print(f"# {r['spec']}: n_ao={r['nao']} naux={r['naux']} nocc={r['nocc']}", flush=True)
        out = run_one(r, args)
        done.append(out)
        print("RESULT " + json.dumps(out, default=str), flush=True)
    print_results(done)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(done, f, indent=1, default=str)


if __name__ == "__main__":
    main()
