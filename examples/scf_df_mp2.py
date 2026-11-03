"""
DF-MP2 on top of the DF-RHF SCF of examples/scf_df_rhf.py.

After the SCF converges:

    canonical   F = H + J - K/2 at the converged occupied orbitals (one
                J/K pass, streamed over aux chunks; float64 on the host by
                default), then eigh(X^T F X) in float64 on the host: eps, C
                (n x m, m the orthogonalizer's kept rank). An fp32 Fock is
                the largest error source: 4e-6 Eh of E_MP2 at C4H10 /
                cc-pVDZ, against 4e-8 Eh for an fp32 AO->MO + energy
    AO -> MO    B_ia^P = sum_mn C_mi C_na B_mn^P     trntensor.AuxStream.ao_to_mo,
                one aux chunk at a time (trntensor.ao_to_mo_transform per chunk),
                into a host tensor or a memmap
    energy      E_MP2 = sum_ijab T (2T - T_ijba) / (e_i + e_j - e_a - e_b),
                T_ijab = sum_P B_ia^P B_jb^P        trntensor.mp2_energy when
                B_ia fits one device, else trntensor.mp2_energy_blocked
                (occupied blocks, pairs dealt over the devices)

The fitting basis is the SCF's (``<basis>-jkfit``), so the integrals are
reused; PySCF's ``mp.dfmp2.DFMP2`` on a density-fitted mean field uses the
same one, which makes it the like-for-like reference. A dedicated RI basis
(``-ri``) changes E_MP2 by its own fitting error, not by this code.

Usage:
    python examples/scf_df_mp2.py --system pyscf:water --reference
    python examples/scf_df_mp2.py --system pyscf:alkane10 --dtype float32
    python examples/scf_df_mp2.py --system pyscf:alkane40@aug-cc-pvdz --orth canonical \\
        --lindep 1e-6 --device neuron --jk-devices neuron:0,neuron:1,neuron:2,neuron:3
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from dataclasses import dataclass, field

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scf_df_rhf as scf  # noqa: E402


@dataclass
class MP2Result:
    e_corr: float
    e_hf: float  # float64 RHF energy at the canonical orbitals
    nocc: int  # correlated occupied orbitals
    nvir: int
    naux: int
    timings_s: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    @property
    def e_total(self) -> float:
        return self.e_hf + self.e_corr


def _aux_stream(system, device, dtype, devices=None, resident_bytes=0.0, prefetch=1):
    """``trntensor.AuxStream`` over system.B (store chunks, or column views)."""
    import trntensor

    if not hasattr(trntensor, "AuxStream"):
        raise RuntimeError("DF-MP2 needs trntensor with AuxStream (trntensor#102)")
    jk = scf.DFJK(
        system.B,
        system.nao,
        device=device,
        dtype=dtype,
        devices=devices,
        stream=True,
        resident_bytes=resident_bytes,
        prefetch=prefetch,
    )
    return jk


def canonical_orbitals(
    system: scf.System,
    C_occ: torch.Tensor,
    *,
    orth: str = "symmetric",
    lindep: float = 1e-7,
    device="cpu",
    dtype: torch.dtype = torch.float64,
    devices=None,
):
    """(eps, C, e_hf) of the Fock matrix at the occupied orbitals ``C_occ``.

    The orbitals are S-orthonormalised in float64 (Loewdin), J and K are
    built on ``device`` in ``dtype`` (streamed over aux chunks), and F =
    H + J - K/2 is diagonalised in float64 on the host in the orthogonalizer's
    basis (``trnsolver.orthogonalizer(S, orth, lindep)``, as in the SCF).
    ``e_hf`` is the float64 energy at ``C_occ`` (``energy_at_orbitals``
    when ``dtype`` is float64).
    """
    import trnsolver

    C = C_occ.double()
    M = C.T @ system.S @ C
    C = C @ trnsolver.inv_sqrt_spd(0.5 * (M + M.T))
    P = 2.0 * C @ C.T
    jk = _aux_stream(system, device, dtype, devices)
    J, K = jk.jk(P.to(dtype).to(jk.device), C.to(dtype).to(jk.device))
    jk.close()
    F = system.H + J.double().cpu() - 0.5 * K.double().cpu()
    e_hf = float(0.5 * (P * (system.H + F)).sum()) + system.e_nuc
    X = trnsolver.orthogonalizer(system.S, orth, lindep).X
    Fp = X.T @ F @ X
    eps, V = torch.linalg.eigh(0.5 * (Fp + Fp.T))
    return eps, X @ V, e_hf


def df_mp2(
    system: scf.System,
    eps: torch.Tensor,
    C: torch.Tensor,
    *,
    frozen: int = 0,
    device="cpu",
    dtype: torch.dtype = torch.float64,
    devices=None,
    bia_path: str | None = None,
    device_bytes: float | None = None,
    occ_block: int | None = None,
) -> tuple[float, dict]:
    """DF-MP2 correlation energy for canonical orbitals (eps, C) of ``system``.

    ``frozen`` lowest occupied orbitals are left uncorrelated. B_ia^P is
    built chunk by chunk over aux on ``device`` / ``devices`` (``dtype``)
    into host memory, or into a ``.npy`` memmap at ``bia_path`` when given.
    The energy runs on ``trntensor.mp2_energy`` with B_ia resident on one
    device when it fits ``device_bytes`` (default
    ``scf.DEVICE_RESIDENT_GB`` on a NeuronCore, unbounded on the CPU), else
    on ``trntensor.mp2_energy_blocked`` over occupied blocks.

    Returns ``(e_corr, info)``; info has the B_ia size and stage times.
    """
    import numpy as np
    import trntensor

    nocc = system.nocc
    C_o = C[:, frozen:nocc].contiguous()
    C_v = C[:, nocc:].contiguous()
    eo, ev = eps[frozen:nocc], eps[nocc:]
    no, nv = C_o.shape[1], C_v.shape[1]
    dev = torch.device(device)
    itemsize = torch.empty((), dtype=dtype).element_size()
    bia_bytes = no * nv * system.naux * itemsize
    info = {"bia_bytes": bia_bytes, "nocc_corr": no, "nvir": nv}

    t0 = time.perf_counter()
    jk = _aux_stream(system, dev, dtype, devices)
    out = None
    if bia_path is not None:
        out = np.lib.format.open_memmap(
            bia_path,
            mode="w+",
            dtype=torch.empty((), dtype=dtype).numpy().dtype,
            shape=(no, nv, system.naux),
        )
    Bia = jk.stream.ao_to_mo(C_o.to(dtype), C_v.to(dtype), out=out)
    jk.stream.close()
    info["ao2mo_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if device_bytes is None:
        device_bytes = float("inf") if dev.type == "cpu" else scf.DEVICE_RESIDENT_GB * 1e9
    many = devices is not None and len(devices) > 1
    if bia_bytes <= device_bytes and not many and _resident_mp2_ok(dev, no, nv, system.naux):
        Bt = Bia if isinstance(Bia, torch.Tensor) else torch.from_numpy(np.asarray(Bia))
        e = trntensor.mp2_energy(Bt.to(dev), eo.to(dtype).to(dev), ev.to(dtype).to(dev))
        info["mp2_path"] = "resident"
    else:
        e = trntensor.mp2_energy_blocked(
            Bia,
            eo.to(dtype),
            ev.to(dtype),
            devices=devices or [dev],
            occ_block=occ_block,
            dtype=dtype,
        )
        info["mp2_path"] = "blocked"
    e_corr = float(e.double().cpu())
    info["mp2_s"] = time.perf_counter() - t0
    return e_corr, info


def _resident_mp2_ok(dev: torch.device, nocc: int, nvir: int, naux: int) -> bool:
    """Whether ``trntensor.mp2_energy`` on a device-resident B_ia is the better route.

    On the host, it always is. On a NeuronCore it is only when trntensor's row
    kernel fits in SBUF. Otherwise ``mp2_energy`` falls back to aten einsums
    over T blocks, which sum into an fp32 scalar. At C40H82 / aug-cc-pVDZ
    (nocc 161, nvir 1461) that route took 222 s on one core, against 184 s for
    ``mp2_energy_blocked``, and came out 3.8e-6 Eh off it. At alkane43 / cc-pVTZ
    it was 531 s against 258 s (trnsci/trnsci#115). ``mp2_energy_blocked``
    sums its per-pair partials in float64.
    """
    if dev.type == "cpu":
        return True
    try:
        from trntensor.nki import dispatch as d

        dims = d._mp2_row_dims(nocc, nvir, naux)
        return dims["sbuf_bytes"] <= d._MP2_ROW_SBUF_BUDGET
    except (ImportError, AttributeError, KeyError):
        return True


def run_mp2(
    system: scf.System,
    res: scf.SCFResult,
    *,
    orth: str = "symmetric",
    lindep: float = 1e-7,
    frozen: int = 0,
    device="cpu",
    dtype: torch.dtype = torch.float64,
    devices=None,
    bia_path: str | None = None,
    fock_device="cpu",
    fock_dtype: torch.dtype = torch.float64,
) -> MP2Result:
    """Canonical orbitals at ``res``'s converged occupied orbitals, then DF-MP2.

    The canonical Fock build runs on ``fock_device`` in ``fock_dtype``
    (default float64 on the host, so ``e_hf`` is the float64 energy at the
    SCF's orbitals); the AO->MO transform and the energy on ``device`` /
    ``devices`` in ``dtype``.
    """
    t0 = time.perf_counter()
    with (
        scf._host_backends()
        if torch.device(fock_device).type == "cpu"
        else contextlib.nullcontext()
    ):
        eps, C, e_hf = canonical_orbitals(
            system,
            res.C_occ,
            orth=orth,
            lindep=lindep,
            device=fock_device,
            dtype=fock_dtype,
            devices=None if torch.device(fock_device).type == "cpu" else devices,
        )
    t_can = time.perf_counter() - t0
    with scf._host_backends() if torch.device(device).type == "cpu" else contextlib.nullcontext():
        e_corr, info = df_mp2(
            system,
            eps,
            C,
            frozen=frozen,
            device=device,
            dtype=dtype,
            devices=devices,
            bia_path=bia_path,
        )
    timings = {"canonical_s": t_can, "ao2mo_s": info.pop("ao2mo_s"), "mp2_s": info.pop("mp2_s")}
    info["rank"] = C.shape[1]
    info["fock_dtype"] = str(fock_dtype).replace("torch.", "")
    return MP2Result(
        e_corr=e_corr,
        e_hf=e_hf,
        nocc=info["nocc_corr"],
        nvir=info["nvir"],
        naux=system.naux,
        timings_s=timings,
        info=info,
    )


def pyscf_df_mp2_reference(
    name: str,
    basis: str = "cc-pvdz",
    auxbasis: str | None = None,
    lindep: float | None = None,
):
    """(E_HF, E_corr) from PySCF: DF-RHF then ``mp.dfmp2.DFMP2`` with the same jkfit basis.

    ``lindep`` (relative to S's largest eigenvalue, as for
    ``trnsolver.orthogonalizer``) drops the same near-linear dependencies
    from PySCF's basis, so the reference is in the SCF's kept basis. PySCF
    >= 2.9 does this through ``scf.hf.remove_overlap_zero_eigenvalue``;
    ``scf.addons.remove_linear_dep_`` there only acts below 1e-10 and kept
    all 510 functions of C12H26 / aug-cc-pVDZ (1.6e-4 Eh of E_HF and 1.1e-3
    of E_MP2 against the canonical 1e-6 basis).
    """
    import numpy as np
    from pyscf import mp
    from pyscf import scf as pscf
    from pyscf.scf import hf

    mol = scf.pyscf_mol(name, basis)
    old = (
        getattr(hf, "remove_overlap_zero_eigenvalue", None),
        getattr(hf, "overlap_zero_eigenvalue_threshold", None),
    )
    try:
        mf = pscf.RHF(mol).density_fit(auxbasis=auxbasis or f"{basis}-jkfit")
        mf.conv_tol = 1e-11
        if lindep is not None:
            thr = lindep * float(np.linalg.eigvalsh(mol.intor("int1e_ovlp")).max())
            if old[0] is not None:
                hf.remove_overlap_zero_eigenvalue = True
                hf.overlap_zero_eigenvalue_threshold = thr
            else:  # older PySCF
                mf = pscf.addons.remove_linear_dep_(mf, threshold=thr, lindep=thr)
        e_hf = float(mf.kernel())
        pt = mp.dfmp2.DFMP2(mf)
        # with_t2=False: the amplitudes need not fit in memory at ladder sizes.
        e_corr = float(pt.kernel(with_t2=False)[0])
    finally:
        if old[0] is not None:
            hf.remove_overlap_zero_eigenvalue, hf.overlap_zero_eigenvalue_threshold = old
    return e_hf, e_corr


def main():
    ap = argparse.ArgumentParser(description="DF-RHF + DF-MP2 on the trnsci libraries")
    ap.add_argument("--system", default="pyscf:water")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    ap.add_argument("--mp2-dtype", default=None, choices=("float32", "float64"))
    ap.add_argument("--orth", default="symmetric", choices=("symmetric", "canonical", "cholesky"))
    ap.add_argument("--lindep", type=float, default=1e-7)
    ap.add_argument("--frozen", type=int, default=0)
    ap.add_argument(
        "--fock-on-device",
        action="store_true",
        help="build the canonical Fock on --device in --dtype (default: float64 on the host)",
    )
    ap.add_argument("--jk-devices", default=None)
    ap.add_argument("--bia-path", default=None, help="memmap B_ia^P here (.npy)")
    ap.add_argument("--reference", action="store_true", help="PySCF DF-RHF + DF-MP2")
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--guess", default="core", choices=("core", "minao"))
    args = ap.parse_args()
    if args.device.startswith("neuron"):
        import torch_neuronx  # noqa: F401

    dtype = getattr(torch, args.dtype)
    mp2_dtype = getattr(torch, args.mp2_dtype) if args.mp2_dtype else dtype
    devices = args.jk_devices.split(",") if args.jk_devices else None
    system = scf.make_system(args.system)
    print(f"{system.name}: n_ao={system.nao} naux={system.naux} nocc={system.nocc}")
    res = scf.run_scf(
        system,
        guess=args.guess,
        device=args.device,
        dtype=dtype,
        orth=args.orth,
        lindep=args.lindep,
        jk_devices=devices,
        max_iter=args.max_iter,
    )
    print(f"SCF: E = {res.energy:.10f} converged={res.converged} iters={res.iterations}")
    mp = run_mp2(
        system,
        res,
        orth=args.orth,
        lindep=args.lindep,
        frozen=args.frozen,
        device=args.device,
        dtype=mp2_dtype,
        devices=devices,
        bia_path=args.bia_path,
        fock_device=args.device if args.fock_on_device else "cpu",
        fock_dtype=dtype if args.fock_on_device else torch.float64,
    )
    print(
        f"E_HF(C) = {mp.e_hf:.10f}  E_MP2 = {mp.e_corr:.10f}  E_total = {mp.e_total:.10f}  "
        f"(nocc {mp.nocc}, nvir {mp.nvir}, naux {mp.naux}, B_ia {mp.info['bia_bytes'] / 1e9:.2f} GB, "
        f"{mp.info['mp2_path']})"
    )
    print("timings: " + ", ".join(f"{k}={v:.2f}" for k, v in mp.timings_s.items()))
    if args.reference:
        kind, _, arg = args.system.partition(":")
        name, _, basis = arg.partition("@")
        lindep = args.lindep if args.orth == "canonical" else None
        e_hf, e_corr = pyscf_df_mp2_reference(name, basis or "cc-pvdz", lindep=lindep)
        print(
            f"PySCF: E_HF = {e_hf:.10f} (diff {mp.e_hf - e_hf:+.2e})  "
            f"E_MP2 = {e_corr:.10f} (diff {mp.e_corr - e_corr:+.2e})"
        )


if __name__ == "__main__":
    main()
