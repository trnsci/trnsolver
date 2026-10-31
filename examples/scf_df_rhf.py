"""
Density-fitted restricted Hartree-Fock (DF-RHF) SCF built from the trnsci libraries.

The loop, per iteration (closed shell, P = 2 C_occ C_occ^T):

    J build    d_Q = sum_{ls} B_{Q,ls} P_ls          trnblas.gemv on packed B
               J_mn = sum_Q B_{Q,mn} d_Q             trnblas.gemv on packed B
    K build    Y_{n,iQ} = sum_m C_mi B_{Q,mn}        trntensor.einsum (one GEMM per aux block)
               K_mn = 2 sum_{iQ} Y_{m,iQ} Y_{n,iQ}   trnblas.syrk
    Fock       F = H + J - K/2, E = tr P (H + F) / 2 + E_nuc
    DIIS       Pulay on e = F P S - S P F (host-side solve of the small system)
    eigh       F' = X^T F X, trnsolver.eigh(F'), C = X C'   (X = S^{-1/2}, once)
               or trnsolver.eigh_generalized(F, S) with --solver generalized
    density    P = 2 C_occ C_occ^T                    trnblas.syrk

B is the density-fitted three-centre tensor B_{Q,mn} = sum_P (L^{-1})_{QP} (P|mn),
held packed over the m >= n triangle as (npair, naux), PySCF's ``lib.pack_tril``
order (``trntensor.pack_tril``). The K build unpacks one aux block at a time on
the tensor's device (``trntensor.unpack_tril``), or once when the full square
fits ``--k-cache-gb``.

Systems:

    h2-lattice:<n_mol>   synthetic s-type Gaussian basis on a lattice of H2
                         molecules (4 s functions per atom, 8 s fitting
                         functions per atom). Closed-form integrals, no
                         dependencies. n_ao = 8 n_mol, nocc = n_mol.
    pyscf:<name>         PySCF integrals (cc-pVDZ / cc-pvdz-jkfit) for
                         water, benzene or alkane<k> (C_k H_{2k+2}, zigzag
                         geometry). Needs PySCF.

Usage:
    python examples/scf_df_rhf.py --system h2-lattice:4
    python examples/scf_df_rhf.py --system h2-lattice:16 --dtype float64
    python examples/scf_df_rhf.py --system pyscf:benzene --device neuron

``scripts/bench_scf.py`` runs the CPU / device / fp64 comparison on top of
:func:`run_scf`.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch

STAGES = ("jk", "fock", "diis", "eigh", "density", "energy")


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------


@dataclass
class System:
    """One-electron matrices and packed DF integrals, float64 on the host."""

    name: str
    H: torch.Tensor  # (n, n) core Hamiltonian
    S: torch.Tensor  # (n, n) overlap
    B: torch.Tensor  # (npair, naux) packed DF integrals, pack_tril order
    nocc: int
    e_nuc: float
    ref_energy: float | None = None  # independent fp64 reference (PySCF), if any

    @property
    def nao(self) -> int:
        return self.H.shape[0]

    @property
    def naux(self) -> int:
        return self.B.shape[1]


def _tril_index(n: int) -> torch.Tensor:
    """Flat indices m*n + k of the m >= k triangle in pack_tril (row-major) order."""
    r = torch.tril_indices(n, n)
    return r[0] * n + r[1]


def _sym_to_packed_index(n: int) -> torch.Tensor:
    """(n*n,) packed row of each (m, k): row max(m,k)(max+1)/2 + min."""
    m = torch.arange(n).unsqueeze(1).expand(n, n)
    k = torch.arange(n).unsqueeze(0).expand(n, n)
    hi = torch.maximum(m, k)
    lo = torch.minimum(m, k)
    return (hi * (hi + 1) // 2 + lo).reshape(-1)


def _boys0(t: torch.Tensor) -> torch.Tensor:
    """Boys function F0(t) = 0.5 sqrt(pi/t) erf(sqrt t), series below 1e-8."""
    small = t < 1e-8
    ts = torch.where(small, torch.ones_like(t), t)
    big = 0.5 * torch.sqrt(math.pi / ts) * torch.erf(torch.sqrt(ts))
    return torch.where(small, 1.0 - t / 3.0, big)


# Even-tempered s exponents (bohr^-2). Orbital set: 4 per atom; fitting set:
# 8 per atom, covering the pair exponents alpha + beta of the orbital set.
_H_AO_EXP = (0.12, 0.45, 1.7, 6.4)
_H_AUX_EXP = (0.2, 0.5, 1.2, 3.0, 7.5, 18.0, 45.0, 110.0)


def h2_lattice_geometry(n_mol: int, spacing: float = 4.5, bond: float = 1.4, seed: int = 0):
    """Atom centres (bohr) of n_mol H2 molecules on a cubic lattice.

    Bond axes are drawn from a seeded generator so the system has no
    symmetry; the lattice spacing keeps a large HOMO-LUMO gap.
    """
    g = torch.Generator().manual_seed(seed)
    side = math.ceil(n_mol ** (1.0 / 3.0))
    sites = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)][:n_mol]
    centres = []
    for site in sites:
        c = torch.tensor(site, dtype=torch.float64) * spacing
        axis = torch.randn(3, generator=g, dtype=torch.float64)
        axis = axis / axis.norm()
        centres.append(c - 0.5 * bond * axis)
        centres.append(c + 0.5 * bond * axis)
    return torch.stack(centres)


def _s_shells(centres: torch.Tensor, exps) -> tuple[torch.Tensor, torch.Tensor]:
    """(centre, exponent) of every primitive s function, atom-major."""
    a = torch.tensor(exps, dtype=torch.float64)
    cen = centres.repeat_interleave(len(exps), dim=0)
    return cen, a.repeat(centres.shape[0])


def _s_norm(a: torch.Tensor) -> torch.Tensor:
    return (2.0 * a / math.pi) ** 0.75


def s_gaussian_system(
    centres: torch.Tensor,
    charges: torch.Tensor,
    ao_exps=_H_AO_EXP,
    aux_exps=_H_AUX_EXP,
    *,
    name: str = "s-gaussian",
    aux_block: int = 64,
    lindep: float = 1e-10,
) -> System:
    """Closed-form integrals for normalised primitive s Gaussians.

    Overlap, kinetic, nuclear attraction, the three-centre (mn|P) and
    two-centre (P|Q) Coulomb integrals all reduce to the Boys function F0.
    The metric is inverted canonically (eigenvalues below ``lindep`` x max
    are dropped), so ``naux`` can come out below the fitting-set size.
    """
    cen, a = _s_shells(centres, ao_exps)
    n = a.shape[0]
    na = _s_norm(a)
    p = a[:, None] + a[None, :]
    mu = a[:, None] * a[None, :] / p
    r2 = ((cen[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
    kab = torch.exp(-mu * r2)
    nn = na[:, None] * na[None, :]
    S = nn * (math.pi / p) ** 1.5 * kab
    T = mu * (3.0 - 2.0 * mu * r2) * S
    Pc = (a[:, None, None] * cen[:, None, :] + a[None, :, None] * cen[None, :, :]) / p[..., None]
    V = torch.zeros_like(S)
    for c, z in zip(centres, charges, strict=True):
        pc2 = ((Pc - c) ** 2).sum(-1)
        V -= z * nn * (2.0 * math.pi / p) * kab * _boys0(p * pc2)
    H = T + V
    S = 0.5 * (S + S.T)
    H = 0.5 * (H + H.T)

    # Fitting metric (P|Q) and its canonical inverse square root.
    xc, g = _s_shells(centres, aux_exps)
    ng = _s_norm(g)
    gq = g[:, None] + g[None, :]
    rq = ((xc[:, None, :] - xc[None, :, :]) ** 2).sum(-1)
    M = (
        ng[:, None]
        * ng[None, :]
        * 2.0
        * math.pi**2.5
        / (g[:, None] * g[None, :] * torch.sqrt(gq))
        * _boys0(g[:, None] * g[None, :] / gq * rq)
    )
    M = 0.5 * (M + M.T)
    w, U = torch.linalg.eigh(M)
    keep = w > lindep * w.max()
    Minv_half = U[:, keep] / torch.sqrt(w[keep])  # (naux_raw, naux)

    # Packed three-centre integrals (mn|P), m >= n, blocked over P.
    tri = torch.tril_indices(n, n)
    pm, pn = tri[0], tri[1]
    pp = p[pm, pn]
    pre = (nn * kab)[pm, pn]
    Pp = Pc[pm, pn]  # (npair, 3)
    out = torch.zeros(pm.shape[0], int(keep.sum()), dtype=torch.float64)
    for q0 in range(0, g.shape[0], aux_block):
        gq_ = g[q0 : q0 + aux_block]
        xq = xc[q0 : q0 + aux_block]
        d2 = ((Pp[:, None, :] - xq[None, :, :]) ** 2).sum(-1)
        rho = pp[:, None] * gq_[None, :] / (pp[:, None] + gq_[None, :])
        three = (
            pre[:, None]
            * ng[q0 : q0 + aux_block][None, :]
            * 2.0
            * math.pi**2.5
            / (pp[:, None] * gq_[None, :] * torch.sqrt(pp[:, None] + gq_[None, :]))
            * _boys0(rho * d2)
        )
        out += three @ Minv_half[q0 : q0 + aux_block]

    zz = charges[:, None] * charges[None, :]
    rr = torch.cdist(centres, centres)
    iu = torch.triu_indices(len(charges), len(charges), 1)
    e_nuc = float((zz[iu[0], iu[1]] / rr[iu[0], iu[1]]).sum())
    nelec = int(round(float(charges.sum())))
    if nelec % 2:
        raise ValueError("closed-shell RHF needs an even electron count")
    return System(name, H, S, out, nelec // 2, e_nuc)


def h2_lattice(n_mol: int, seed: int = 0, **kw) -> System:
    """Synthetic H2-lattice system: n_ao = 8 n_mol, nocc = n_mol."""
    centres = h2_lattice_geometry(n_mol, seed=seed)
    charges = torch.ones(centres.shape[0], dtype=torch.float64)
    return s_gaussian_system(centres, charges, name=f"h2-lattice:{n_mol}", **kw)


def _alkane_atoms(k: int):
    """C_k H_{2k+2} all-trans zigzag geometry (Angstrom), tetrahedral angles."""
    cc, ch = 1.54, 1.09
    half = math.radians(109.47) / 2
    dx, dy = cc * math.sin(half), cc * math.cos(half) / 2
    carbons = [(i * dx, dy if i % 2 else -dy, 0.0) for i in range(k)]
    atoms = [("C", c) for c in carbons]
    ct, st = math.cos(math.radians(54.735)), math.sin(math.radians(54.735))
    for i, (x, y, _) in enumerate(carbons):
        s = 1.0 if y > 0 else -1.0
        hs = [(0.0, s * ct, st), (0.0, s * ct, -st)]
        bonds = []
        for j in (i - 1, i + 1):
            if 0 <= j < k:
                bx, by = carbons[j][0] - x, carbons[j][1] - y
                bonds.append((bx / cc, by / cc, 0.0))
        if len(bonds) == 1:  # chain end: the fourth tetrahedral direction
            d = [-(bonds[0][t] + hs[0][t] + hs[1][t]) for t in range(3)]
            hs.append(tuple(d))
        if k == 1:
            raise ValueError("alkane needs k >= 2")
        for h in hs:
            atoms.append(("H", (x + ch * h[0], y + ch * h[1], ch * h[2])))
    return atoms


_PYSCF_GEOMETRIES = {
    "water": "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
}


def _benzene_atoms():
    atoms = []
    for i in range(6):
        t = math.radians(60 * i)
        atoms.append(("C", (1.39 * math.cos(t), 1.39 * math.sin(t), 0.0)))
        atoms.append(("H", (2.48 * math.cos(t), 2.48 * math.sin(t), 0.0)))
    return atoms


def pyscf_system(
    name: str,
    basis: str = "cc-pvdz",
    auxbasis: str = "cc-pvdz-jkfit",
    reference: bool = True,
) -> System:
    """Real-molecule integrals from PySCF; ``reference`` also runs PySCF's DF-RHF."""
    from pyscf import df, gto, lib, scf

    if name == "benzene":
        atom = _benzene_atoms()
    elif name.startswith("alkane"):
        atom = _alkane_atoms(int(name[len("alkane") :]))
    else:
        atom = _PYSCF_GEOMETRIES[name]
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    H = torch.from_numpy(mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc"))
    S = torch.from_numpy(mol.intor_symmetric("int1e_ovlp"))
    cderi = df.incore.cholesky_eri(mol, auxbasis=auxbasis)  # (naux, npair)
    ref = None
    if reference:
        mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
        mf.with_df._cderi = cderi
        mf.conv_tol = 1e-11
        ref = float(mf.kernel())
    B = torch.from_numpy(lib.transpose(cderi))  # (npair, naux)
    return System(f"pyscf:{name}", H, S, B, mol.nelectron // 2, float(mol.energy_nuc()), ref)


def make_system(spec: str, **kw) -> System:
    kind, _, arg = spec.partition(":")
    if kind == "h2-lattice":
        return h2_lattice(int(arg or 4))
    if kind == "pyscf":
        return pyscf_system(arg or "water", **kw)
    raise ValueError(f"unknown system {spec!r}")


# ---------------------------------------------------------------------------
# SCF
# ---------------------------------------------------------------------------


def _sync_fn(device: torch.device):
    if device.type == "neuron":
        import torch_neuronx

        return torch_neuronx.synchronize
    return lambda: None


@dataclass
class SCFResult:
    energy: float
    converged: bool
    iterations: int
    energies: list[float]
    stage_ms: list[dict[str, float]]  # per iteration
    setup_ms: dict[str, float]
    total_ms: float
    P: torch.Tensor  # final density, host float64
    C_occ: torch.Tensor  # final occupied orbitals, host float64
    eps: torch.Tensor  # final orbital energies, host float64
    info: dict = field(default_factory=dict)

    def stage_median_ms(self) -> dict[str, float]:
        out = {}
        for s in STAGES:
            v = sorted(it.get(s, 0.0) for it in self.stage_ms[1:] or self.stage_ms)
            out[s] = v[len(v) // 2] if v else 0.0
        return out

    def stage_total_ms(self) -> dict[str, float]:
        out = defaultdict(float)
        for it in self.stage_ms:
            for s, t in it.items():
                out[s] += t
        return dict(out)


class _Timer:
    def __init__(self, sync, enabled: bool):
        self.sync = sync
        self.enabled = enabled
        self.t = {}
        self._t0 = 0.0
        self._name = None

    def start(self, name):
        if self.enabled:
            self.sync()
        self._name = name
        self._t0 = time.perf_counter()

    def stop(self):
        if self.enabled:
            self.sync()
            self.t[self._name] = self.t.get(self._name, 0.0) + 1e3 * (
                time.perf_counter() - self._t0
            )


class DFJK:
    """J and K from packed DF integrals on ``device`` (or split over ``devices``).

    The packed (npair, naux) integrals are held as contiguous aux blocks of
    ``k_block`` columns, each uploaded separately: slicing columns of one
    multi-GB device tensor is itself a large compiled op (at n_ao = 970 the
    slice copy exceeded the 24 GB HBM limit of one Trn2 logical core). K
    unpacks one block at a time (``trntensor.unpack_tril``); when the full
    square (n, n, naux) fits ``k_cache_bytes`` there is one block and its
    square is unpacked once and kept. ``square=True`` unpacks every block
    once at setup and drops the packed copy: twice the memory, but no gather
    per iteration, which at n_ao = 970 was 28 of the 44 ms per aux block of
    532 (Trn2, #15). J then comes from the squares.

    With several ``devices`` (NeuronCores) the blocks are dealt round-robin
    and each core builds partial J and K for its blocks. The inputs (packed
    P, C_occ) and the partial results are staged through the host, because
    a tensor moved between NeuronCores directly stays on its source core.
    Work is issued to every core before any result is read back, so the
    cores run concurrently. One Python thread issues it all: at n_ao = 970
    the J/K stage went 359 -> 253 ms on 1 -> 4 logical cores (Trn2, #15).
    """

    def __init__(
        self,
        B: torch.Tensor,
        n: int,
        k_cache_bytes: float = 4e9,
        k_block=None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        devices=None,
        square: bool = False,
    ):
        import trntensor

        dev = torch.device(device) if device is not None else B.device
        self.devices = [torch.device(d) for d in devices] if devices else [dev]
        self.device = self.devices[0] if devices else dev
        dtype = dtype or B.dtype
        self.n = n
        naux = B.shape[1]
        self.naux = naux
        tril = _tril_index(n)
        npair = tril.shape[0]
        w = torch.full((npair,), 2.0, dtype=dtype)
        r = torch.tril_indices(n, n)
        w[r[0] == r[1]] = 1.0
        self.tril = tril.to(self.device)
        self.pair_w = w.to(self.device)
        self.sym = _sym_to_packed_index(n).to(self.device)
        full_bytes = n * n * naux * torch.finfo(dtype).bits // 8
        if k_block is None:
            if full_bytes <= k_cache_bytes:
                k_block = naux
            else:
                k_block = max(1, int(naux * k_cache_bytes / full_bytes / 2))
            if len(self.devices) > 1:
                k_block = min(k_block, -(-naux // len(self.devices)))
        self.k_block = min(k_block, naux)
        self.blocks = []
        self.squares = None  # per-block (n, n, nq) squares, when resident
        for i, q0 in enumerate(range(0, naux, self.k_block)):
            d = self.devices[i % len(self.devices)]
            self.blocks.append(B[:, q0 : q0 + self.k_block].to(dtype).contiguous().to(d))
        self.B3 = None
        if len(self.blocks) == 1 and not square:
            self.B3 = trntensor.unpack_tril(self.blocks[0], n)  # (n, n, naux), once
        if square:
            # Unpack every block once and drop the packed copy: twice the
            # memory, no per-iteration gather, and J from the square.
            self.squares = []
            for i in range(len(self.blocks)):
                self.squares.append(trntensor.unpack_tril(self.blocks[i], n))
                self.blocks[i] = None

    def _scatter(self, x: torch.Tensor) -> dict:
        """x on every device: itself on its own, through the host elsewhere."""
        if len(self.devices) == 1:
            return {x.device: x}
        h = x.cpu()
        return {d: (x if d == x.device else h.to(d)) for d in self.devices}

    def _gather(self, parts: dict) -> torch.Tensor:
        """Sum of per-device partial results, on the primary device."""
        if len(parts) == 1:
            return next(iter(parts.values()))
        host = [t.cpu() for t in parts.values()]
        return torch.stack(host).sum(0).to(self.device)

    def j(self, P: torch.Tensor) -> torch.Tensor:
        import trnblas

        if self.squares is not None:
            ps = self._scatter(P.reshape(-1))
            parts = {}
            for B3 in self.squares:
                Bm = B3.reshape(self.n * self.n, B3.shape[2])
                d = trnblas.gemv(1.0, Bm, ps[B3.device], trans=True)
                jb = trnblas.gemv(1.0, Bm, d)
                parts[B3.device] = jb if B3.device not in parts else parts[B3.device] + jb
            return self._gather(parts).reshape(self.n, self.n)
        p = P.reshape(-1).index_select(0, self.tril) * self.pair_w
        ps = self._scatter(p)
        parts = {}
        for Bb in self.blocks:
            d = trnblas.gemv(1.0, Bb, ps[Bb.device], trans=True)  # (nq,)
            jb = trnblas.gemv(1.0, Bb, d)  # (npair,)
            parts[Bb.device] = jb if Bb.device not in parts else parts[Bb.device] + jb
        jp = self._gather(parts)
        return jp.index_select(0, self.sym).reshape(self.n, self.n)

    def k(self, C_occ: torch.Tensor) -> torch.Tensor:
        """K for P = 2 C_occ C_occ^T: K = 2 sum_{iQ} Y_{m,iQ} Y_{n,iQ}."""
        import trnblas
        import trntensor

        n, nocc = C_occ.shape
        cs = self._scatter(C_occ)
        parts = {}
        for i, Bb in enumerate(self.blocks):
            if self.squares is not None:
                B3 = self.squares[i]
            elif self.B3 is not None:
                B3 = self.B3
            else:
                B3 = trntensor.unpack_tril(Bb, n)
            C = cs[B3.device]
            nq = B3.shape[2]
            # B3 is (m, n, Q) and symmetric in m, n: one GEMM over m.
            Y = trntensor.einsum("mi,mx->ix", C, B3.reshape(n, n * nq))  # (i, n, Q)
            Y = Y.reshape(nocc, n, nq).permute(1, 0, 2).reshape(n, nocc * nq)
            Kb = trnblas.syrk(2.0, Y)
            parts[B3.device] = Kb if B3.device not in parts else parts[B3.device] + Kb
        return self._gather(parts)

    def jk(self, P: torch.Tensor, C_occ: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(J, K) for P = 2 C_occ C_occ^T.

        With resident squares, J reuses K's half-transformed Y: d_Q =
        sum_{ls} B_{Q,ls} P_ls = 2 sum_{i,s} C_si Y_{i,s,Q}, so each square
        is read once per iteration (by the half-transform GEMM) plus once by
        the J GEMV, instead of twice by J alone.
        """
        if self.squares is None:
            return self.j(P), self.k(C_occ)
        import trnblas
        import trntensor

        n, nocc = C_occ.shape
        cs = self._scatter(C_occ)
        cflat = {d: c.T.contiguous().reshape(1, nocc * n) for d, c in cs.items()}
        jparts, kparts = {}, {}
        for B3 in self.squares:
            dev, nq = B3.device, B3.shape[2]
            Y = trntensor.einsum("mi,mx->ix", cs[dev], B3.reshape(n, n * nq))  # (i, n, Q)
            d = trnblas.gemm(2.0, cflat[dev], Y.reshape(nocc * n, nq))  # (1, Q)
            jb = trnblas.gemv(1.0, B3.reshape(n * n, nq), d.reshape(nq))
            Ym = Y.reshape(nocc, n, nq).permute(1, 0, 2).reshape(n, nocc * nq)
            Kb = trnblas.syrk(2.0, Ym)
            jparts[dev] = jb if dev not in jparts else jparts[dev] + jb
            kparts[dev] = Kb if dev not in kparts else kparts[dev] + Kb
        return self._gather(jparts).reshape(n, n), self._gather(kparts)


class DIIS:
    """Pulay DIIS on e = F P S - S P F.

    The Fock and error history live in fixed device buffers, so each update
    is one GEMV for the new row of error dot products and one GEMV for the
    extrapolated Fock matrix. The (m+1)-sized linear system is solved on the
    host in float64.
    """

    def __init__(self, n: int, space: int, dtype: torch.dtype, device: torch.device):
        self.space = space
        self.F = torch.zeros(space, n * n, dtype=dtype, device=device)
        self.e = torch.zeros(space, n * n, dtype=dtype, device=device)
        self.order: list[int] = []  # buffer slots, oldest first
        self.Bm = torch.zeros(space, space, dtype=torch.float64)

    def push(self, F: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Store (F, e); return the device row <e_k, e> over the slots (slot order)."""
        slot = self.order.pop(0) if len(self.order) == self.space else len(self.order)
        self.order.append(slot)
        self.F[slot].copy_(F.reshape(-1))
        self.e[slot].copy_(e.reshape(-1))
        self._slot = slot
        return torch.mv(self.e, e.reshape(-1))

    def extrapolate(self, row: torch.Tensor) -> torch.Tensor | None:
        """Given the host copy of :meth:`push`'s row, the DIIS Fock (None: use F)."""
        slot = self._slot
        self.Bm[slot, :] = row
        self.Bm[:, slot] = row
        idx = self.order
        m = len(idx)
        if m < 2:
            return None
        Bm = self.Bm[idx][:, idx]
        A = torch.zeros(m + 1, m + 1, dtype=torch.float64)
        A[:m, :m] = Bm / Bm.diagonal().abs().max().clamp_min(1e-300)
        A[:m, m] = -1.0
        A[m, :m] = -1.0
        rhs = torch.zeros(m + 1, 1, dtype=torch.float64)
        rhs[m] = -1.0
        c = torch.linalg.lstsq(A, rhs).solution[:m, 0]
        coef = torch.zeros(self.space, dtype=torch.float64)
        coef[idx] = c
        coef = coef.to(self.F.dtype).to(self.F.device)
        n = int(round(self.F.shape[1] ** 0.5))
        return torch.mv(self.F.T, coef).reshape(n, n)


def _purify_steps(D: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """k canonical-purification steps; returns (D, tr(D - D^2) of the last step)."""
    idem = D.new_zeros(())
    for _ in range(k):
        D2 = D @ D
        D3 = D2 @ D
        tD, tD2, tD3 = D.diagonal().sum(), D2.diagonal().sum(), D3.diagonal().sum()
        idem = tD - tD2
        c = (tD2 - tD3) / idem.clamp_min(1e-30)
        hi = ((1 + c) * D2 - D3) / c.clamp_min(1e-30)
        lo = ((1 - 2 * c) * D + (1 + c) * D2 - D3) / (1 - c).clamp_min(1e-30)
        D = torch.where(c >= 0.5, hi, lo)
    return D, idem


_PURIFY_COMPILED: dict = {}


def _purify_block_fn(device: torch.device, k: int):
    """``_purify_steps(., k)``: eager on the host, one compiled graph on a NeuronCore.

    Eagerly each step is ~12 small launches; on the ``neuron`` device that
    launch cost, not the two GEMMs, dominates. ``torch.compile`` (backend
    ``neuron``) turns k steps into one graph. ``SCF_PURIFY_COMPILE=0``
    keeps it eager.
    """
    import os

    if device.type != "neuron" or os.environ.get("SCF_PURIFY_COMPILE", "1") == "0":
        return lambda D: _purify_steps(D, k)
    if k not in _PURIFY_COMPILED:
        import functools

        _PURIFY_COMPILED[k] = torch.compile(
            functools.partial(_purify_steps, k=k), backend="neuron", dynamic=False
        )
    return _PURIFY_COMPILED[k]


def purify_occupied(
    Fp: torch.Tensor,
    nocc: int,
    seed: torch.Tensor,
    *,
    tol: float | None = None,
    max_iter: int = 100,
    check_every: int = 4,
) -> tuple[torch.Tensor, int]:
    """Orthonormal basis of the lowest-``nocc`` invariant subspace of ``Fp``, GEMMs only.

    Canonical density-matrix purification (Palser and Manolopoulos, PRB 58,
    12704, 1998) drives D, a polynomial in Fp with trace ``nocc``, to the
    projector onto the occupied subspace: two n x n GEMMs per step, with the
    step coefficient computed on the device (no host round trip per step;
    convergence is read every ``check_every`` steps, which on a NeuronCore
    run as one compiled graph, :func:`_purify_block_fn`). Then the projector is
    applied to ``seed`` (the previous occupied orbitals, n x nocc) and a few
    random columns, and the rank-nocc range is extracted from the small
    Gram matrix on the host (float64), then re-orthonormalised by one
    Loewdin pass.

    It stops when the idempotency error tr(D - D^2) is below ``tol``
    (default 10 n ulp of the dtype, its rounding floor) or stops shrinking.

    Returns (Q, steps): Q is (n, nocc) with Q^T Q = I, spanning the
    occupied subspace; there are no orbital energies. Needs a gap between
    eigenvalues nocc and nocc + 1.

    Measured on Trn2 (#15), one fp32 solve with 64-84 steps: 36 / 48 / 111 ms
    compiled (238 / 167 / 320 ms eager) at n = 250 / 490 / 970, against
    7.0 / 15.2 / 74 ms for host LAPACK through aten ``eigh``. It is kept as an
    experiment: GEMM-only, but slower than the host solve at every measured
    size.
    """
    import trnblas

    n = Fp.shape[0]
    eye = torch.eye(n, dtype=Fp.dtype, device=Fp.device)
    diag = Fp.diagonal()
    radius = Fp.abs().sum(dim=1) - diag.abs()
    hmin = (diag - radius).min()
    hmax = (diag + radius).max()
    mu = diag.sum() / n
    lam = torch.minimum(nocc / (hmax - mu), (n - nocc) / (mu - hmin))
    D = (lam / n) * (mu * eye - Fp) + (nocc / n) * eye
    if tol is None:
        tol = 10 * n * torch.finfo(Fp.dtype).eps
    last = float("inf")
    run = _purify_block_fn(Fp.device, check_every)
    steps = 0
    while steps < max_iter:
        D, idem = run(D)
        steps += check_every
        err = float(idem.abs())
        if err < tol or err >= last:
            break
        last = err
    # Range of D from the previous orbitals plus a few fixed random columns:
    # an orbital that was virtual last iteration is not lost. The small Gram
    # matrix is eigendecomposed on the host to pick the rank-nocc range.
    extra = min(16, n - nocc)
    g = torch.Generator().manual_seed(n)
    R = torch.randn(n, extra, generator=g, dtype=torch.float64).to(seed.dtype).to(seed.device)
    Z = trnblas.gemm(1.0, D, torch.cat([seed, R], dim=1))
    G = trnblas.gemm(1.0, Z, Z, transA=True).double().cpu()
    w, U = torch.linalg.eigh(0.5 * (G + G.T))
    T = U[:, -nocc:] / torch.sqrt(w[-nocc:].clamp_min(1e-300))
    Z = trnblas.gemm(1.0, Z, T.to(Z.dtype).to(Z.device))
    # One Loewdin pass, Z (Z^T Z)^{-1/2}, restores orthonormality to working
    # precision (a Cholesky QR here broke down in fp32 at n_ao = 490).
    G = trnblas.gemm(1.0, Z, Z, transA=True).double().cpu()
    w, U = torch.linalg.eigh(0.5 * (G + G.T))
    T = (U / torch.sqrt(w.clamp_min(1e-300))) @ U.T
    Z = trnblas.gemm(1.0, Z, T.to(Z.dtype).to(Z.device))
    return Z, steps


def run_scf(
    system: System,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    solver: str = "orth",
    diis: bool = True,
    diis_space: int = 8,
    max_iter: int = 60,
    e_tol: float | None = None,
    e_rtol: float | None = None,
    d_tol: float | None = None,
    stage_sync: bool = True,
    k_cache_gb: float = 4.0,
    k_block: int | None = None,
    jk_devices=None,
    jk_square: bool = False,
    verbose: bool = False,
) -> SCFResult:
    """Run DF-RHF to convergence; every tensor op runs on ``device`` in ``dtype``.

    ``solver="orth"`` diagonalises X^T F X with X = S^{-1/2} (built once in
    float64 on the host by ``trnsolver.inv_sqrt_spd``); ``"generalized"`` calls
    ``trnsolver.eigh_generalized(F, S)`` every iteration; ``"purify"`` replaces
    the per-iteration eigh by :func:`purify_occupied` on X^T F X (GEMMs only;
    the guess and one final solve still use eigh, so ``eps`` is reported).
    Convergence needs
    |dE| < e_tol + e_rtol |E| and rms(FPS - SPF) < d_tol. The defaults
    depend on dtype: float32 cannot resolve |dE| below a few ulp of |E|
    (about 1e-4 Hartree at |E| = 1500), so its energy test is relative, and
    its commutator test is 1e-5 (at 1e-4, n_ao = 970 stopped with orbitals
    whose float64 energy was 1.2e-4 Eh high).
    ``stage_sync`` synchronises the device around every stage so the stage
    times add up; turn it off to time the loop as an application would.
    ``jk_devices`` splits the J/K build over several NeuronCores and
    ``jk_square`` keeps the unpacked integrals resident (:class:`DFJK`);
    everything else stays on ``device``.
    """
    import trnblas

    import trnsolver

    device = torch.device(device)
    sync = _sync_fn(device)
    fp64 = dtype == torch.float64
    e_tol = e_tol if e_tol is not None else (1e-10 if fp64 else 1e-6)
    e_rtol = e_rtol if e_rtol is not None else (0.0 if fp64 else 2e-7)
    d_tol = d_tol if d_tol is not None else (1e-7 if fp64 else 1e-5)
    n, nocc = system.nao, system.nocc
    setup = {}

    t0 = time.perf_counter()
    X64 = trnsolver.inv_sqrt_spd(system.S)  # float64, host
    setup["orthogonalizer"] = 1e3 * (time.perf_counter() - t0)

    t0 = time.perf_counter()
    H = system.H.to(dtype).to(device)
    S = system.S.to(dtype).to(device)
    X = X64.to(dtype).to(device)
    jk = DFJK(
        system.B,
        n,
        k_cache_bytes=k_cache_gb * 1e9,
        k_block=k_block,
        device=device,
        dtype=dtype,
        devices=jk_devices,
        square=jk_square,
    )
    sync()
    setup["upload"] = 1e3 * (time.perf_counter() - t0)

    def solve(F):
        if solver == "generalized":
            eps, C = trnsolver.eigh_generalized(F, S)
        else:
            Fp = trnblas.gemm(1.0, X, trnblas.gemm(1.0, F, X), transA=True)
            Fp = 0.5 * (Fp + Fp.T)
            eps, Cp = trnsolver.eigh(Fp)
            C = trnblas.gemm(1.0, X, Cp)
        return eps, C

    # Core guess.
    t0 = time.perf_counter()
    eps, C = solve(H)
    C_occ = C[:, :nocc].contiguous()
    P = trnblas.syrk(2.0, C_occ)
    Q_occ = None
    if solver == "purify":
        # Occupied orbitals in the orthonormal basis: Q = X^{-1} C_occ = X S C_occ.
        Q_occ = trnblas.gemm(1.0, X, trnblas.gemm(1.0, S, C_occ))
    purify_steps = []
    sync()
    setup["guess"] = 1e3 * (time.perf_counter() - t0)

    mixer = DIIS(n, diis_space, dtype, device) if diis else None
    energies: list[float] = []
    stage_ms: list[dict[str, float]] = []
    converged = False
    t_loop = time.perf_counter()
    it = 0
    for it in range(1, max_iter + 1):
        tm = _Timer(sync, stage_sync)
        tm.start("jk")
        J, K = jk.jk(P, C_occ)
        tm.stop()

        tm.start("fock")
        F = H + J - 0.5 * K
        tm.stop()

        tm.start("energy")
        e_dev = 0.5 * (P * (H + F)).sum()
        tm.stop()

        tm.start("diis")
        FPS = trnblas.gemm(1.0, F, trnblas.gemm(1.0, P, S))
        err = FPS - FPS.T
        if mixer is not None:
            row = mixer.push(F, err)
            norm2 = row[mixer._slot]
        else:
            row = (err * err).sum().reshape(1)
            norm2 = row[0]
        # One device-to-host copy per iteration: energy, |e|^2 and the DIIS row.
        host = torch.cat([e_dev.reshape(1), norm2.reshape(1), row]).double().cpu()
        e = float(host[0]) + system.e_nuc
        err_rms = math.sqrt(max(float(host[1]), 0.0) / (n * n))
        F_use = F
        if mixer is not None:
            F_ex = mixer.extrapolate(host[2:])
            F_use = F if F_ex is None else F_ex
        tm.stop()

        dE = abs(e - energies[-1]) if energies else float("inf")
        energies.append(e)
        if verbose:
            print(f"  it {it:3d}  E = {e:.10f}  dE = {dE:.2e}  rms(err) = {err_rms:.2e}")
        if dE < e_tol + e_rtol * abs(e) and err_rms < d_tol:
            converged = True
            stage_ms.append(tm.t)
            break

        tm.start("eigh")
        if solver == "purify":
            Fp = trnblas.gemm(1.0, X, trnblas.gemm(1.0, F_use, X), transA=True)
            Q_occ, k = purify_occupied(0.5 * (Fp + Fp.T), nocc, Q_occ)
            purify_steps.append(k)
            C_occ = trnblas.gemm(1.0, X, Q_occ)
        else:
            eps, C = solve(F_use)
            C_occ = C[:, :nocc].contiguous()
        tm.stop()

        tm.start("density")
        P = trnblas.syrk(2.0, C_occ)
        tm.stop()
        stage_ms.append(tm.t)
    sync()
    total_ms = 1e3 * (time.perf_counter() - t_loop)
    if solver == "purify":
        # Orbital energies of the final Fock matrix: one eigh after the loop.
        eps, _ = solve(F_use)

    return SCFResult(
        energy=energies[-1],
        converged=converged,
        iterations=it,
        energies=energies,
        stage_ms=stage_ms,
        setup_ms=setup,
        total_ms=total_ms,
        P=P.detach().to("cpu", torch.float64),
        C_occ=C_occ.detach().to("cpu", torch.float64),
        eps=eps.detach().to("cpu", torch.float64),
        info={
            "k_block": jk.k_block,
            "naux": system.naux,
            "nao": n,
            "nocc": nocc,
            "purify_steps": purify_steps,
        },
    )


def energy_at_orbitals(system: System, C_occ: torch.Tensor) -> float:
    """float64 DF-RHF energy of given occupied orbitals (host).

    The orbitals are first S-orthonormalised in float64 (Loewdin), so the
    density P = 2 C C^T is exactly idempotent. The RHF energy is stationary
    there, so the error of this energy is second order in the orbital error:
    it separates a low-precision run's evaluation rounding (its own E) from
    the quality of its converged orbitals.

    Runs with every trnsci backend on ``pytorch``: with ``auto`` on a Neuron
    host, trnblas / trntensor send large host GEMMs to the device, which is
    not a float64 reference.
    """
    with _host_backends():
        return _energy_at_orbitals(system, C_occ)


def _host_backends():
    """Context manager: trnsolver, trnblas and trntensor on ``pytorch``, then restored."""
    import contextlib

    import trnblas
    import trntensor

    import trnsolver

    @contextlib.contextmanager
    def cm():
        libs = (trnsolver, trnblas, trntensor)
        old = [lib.get_backend() for lib in libs]
        for lib in libs:
            lib.set_backend("pytorch")
        try:
            yield
        finally:
            for lib, b in zip(libs, old, strict=True):
                lib.set_backend(b)

    return cm()


def _energy_at_orbitals(system: System, C_occ: torch.Tensor) -> float:
    import trnsolver

    C = C_occ.double()
    M = C.T @ system.S @ C
    C = C @ trnsolver.inv_sqrt_spd(0.5 * (M + M.T))
    P = 2.0 * C @ C.T
    jk = DFJK(
        system.B, system.nao, k_cache_bytes=0, k_block=min(256, system.naux), dtype=torch.float64
    )
    F = system.H + jk.j(P) - 0.5 * jk.k(C)
    return float(0.5 * (P * (system.H + F)).sum()) + system.e_nuc


def main():
    ap = argparse.ArgumentParser(description="DF-RHF SCF on the trnsci libraries")
    ap.add_argument("--system", default="h2-lattice:4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--solver", default="orth", choices=("orth", "generalized", "purify"))
    ap.add_argument("--no-diis", action="store_true")
    ap.add_argument("--max-iter", type=int, default=60)
    args = ap.parse_args()

    if args.device.startswith("neuron"):
        import torch_neuronx  # noqa: F401

    system = make_system(args.system)
    print(
        f"{system.name}: n_ao={system.nao} naux={system.naux} nocc={system.nocc} "
        f"device={args.device} dtype={args.dtype} solver={args.solver}"
    )
    res = run_scf(
        system,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        solver=args.solver,
        diis=not args.no_diis,
        max_iter=args.max_iter,
        verbose=True,
    )
    print(f"E = {res.energy:.10f}  converged={res.converged} iterations={res.iterations}")
    if system.ref_energy is not None:
        print(
            f"PySCF reference {system.ref_energy:.10f}  diff {res.energy - system.ref_energy:.2e}"
        )
    med = res.stage_median_ms()
    print("median stage ms: " + "  ".join(f"{s}={med[s]:.2f}" for s in STAGES))
    print(
        f"loop {res.total_ms:.1f} ms; setup "
        + ", ".join(f"{k}={v:.1f}" for k, v in res.setup_ms.items())
    )


if __name__ == "__main__":
    main()
