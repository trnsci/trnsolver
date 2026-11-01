"""CPU tests for the DF-RHF SCF example (examples/scf_df_rhf.py, #15)."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import scf_df_rhf as scf  # noqa: E402

import trnsolver  # noqa: E402


@pytest.fixture(autouse=True)
def _pytorch_backends():
    """The example drives trnblas / trntensor too; keep every library on torch."""
    import trnblas
    import trntensor

    libs = (trnsolver, trnblas, trntensor)
    old = [lib.get_backend() for lib in libs]
    for lib in libs:
        lib.set_backend("pytorch")
    yield
    for lib, b in zip(libs, old, strict=True):
        lib.set_backend(b)


def _dense_eri(system):
    """(mn|ls) = sum_Q B_Q,mn B_Q,ls from the packed DF tensor."""
    import trntensor

    n = system.nao
    B3 = trntensor.unpack_tril(system.B, n)  # (m, n, Q)
    return torch.einsum("mnQ,lsQ->mnls", B3, B3)


def _reference_scf(system, iters=200):
    """Independent dense-ERI RHF in float64 with damping, no DIIS."""
    eri = _dense_eri(system)
    H, S, nocc = system.H, system.S, system.nocc
    w, U = torch.linalg.eigh(S)
    X = U @ torch.diag(w.rsqrt()) @ U.T
    _, Cp = torch.linalg.eigh(X @ H @ X)
    C = X @ Cp
    P = 2 * C[:, :nocc] @ C[:, :nocc].T
    e_old = 0.0
    for _ in range(iters):
        J = torch.einsum("mnls,ls->mn", eri, P)
        K = torch.einsum("mlns,ls->mn", eri, P)
        F = H + J - 0.5 * K
        e = 0.5 * float((P * (H + F)).sum()) + system.e_nuc
        _, Cp = torch.linalg.eigh(X @ F @ X)
        C = X @ Cp
        P = 0.5 * P + 0.5 * (2 * C[:, :nocc] @ C[:, :nocc].T)
        if abs(e - e_old) < 1e-12:
            break
        e_old = e
    return e


def test_single_s_gaussian_integrals():
    a = 0.8
    centre = torch.zeros(1, 3, dtype=torch.float64)
    sys1 = scf.s_gaussian_system(
        centre, torch.tensor([2.0], dtype=torch.float64), ao_exps=(a,), aux_exps=(2 * a,)
    )
    assert sys1.S[0, 0].item() == pytest.approx(1.0, abs=1e-14)
    # <T> = 3a/2, <V> = -2 Z sqrt(2a/pi) for one normalised s Gaussian (Z = 2).
    assert sys1.H[0, 0].item() == pytest.approx(1.5 * a - 4 * math.sqrt(2 * a / math.pi), abs=1e-13)


def test_h2_energy_is_physical():
    res = scf.run_scf(scf.h2_lattice(1), dtype=torch.float64)
    assert res.converged
    # s-only basis at R = 1.4 bohr: above the HF limit (-1.1336), close to it.
    assert -1.1336 < res.energy < -1.11


def test_matches_dense_eri_reference():
    system = scf.h2_lattice(3)
    res = scf.run_scf(system, dtype=torch.float64)
    assert res.converged
    assert res.energy == pytest.approx(_reference_scf(system), abs=1e-9)


def test_diis_and_solver_variants_agree():
    system = scf.h2_lattice(4)
    base = scf.run_scf(system, dtype=torch.float64)
    no_diis = scf.run_scf(system, dtype=torch.float64, diis=False, max_iter=200)
    gen = scf.run_scf(system, dtype=torch.float64, solver="generalized")
    assert base.converged and no_diis.converged and gen.converged
    assert no_diis.energy == pytest.approx(base.energy, abs=1e-8)
    assert gen.energy == pytest.approx(base.energy, abs=1e-9)
    assert base.iterations <= no_diis.iterations


def test_purify_solver_matches_eigh():
    system = scf.h2_lattice(6)
    base = scf.run_scf(system, dtype=torch.float64)
    pur = scf.run_scf(system, dtype=torch.float64, solver="purify")
    assert pur.converged
    assert pur.energy == pytest.approx(base.energy, abs=1e-9)
    assert all(k > 0 for k in pur.info["purify_steps"])
    torch.testing.assert_close(pur.eps, base.eps, rtol=0, atol=1e-6)


def test_purify_occupied_projector():
    g = torch.Generator().manual_seed(1)
    n, nocc = 40, 7
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    w = torch.cat([torch.linspace(-3, -1, nocc), torch.linspace(0.5, 4, n - nocc)]).double()
    A = (Q * w) @ Q.T
    seed = torch.randn(n, nocc, generator=g, dtype=torch.float64)
    Z, steps = scf.purify_occupied(A, nocc, seed)
    torch.testing.assert_close(Z.T @ Z, torch.eye(nocc, dtype=torch.float64))
    P_ref = Q[:, :nocc] @ Q[:, :nocc].T
    torch.testing.assert_close(Z @ Z.T, P_ref, rtol=0, atol=1e-10)
    assert steps < 100


def test_fp32_energy_and_orbitals():
    system = scf.h2_lattice(8)
    ref = scf.run_scf(system, dtype=torch.float64)
    res = scf.run_scf(system, dtype=torch.float32)
    assert res.converged
    assert abs(res.energy - ref.energy) < 1e-4
    # The float64 energy at the fp32 orbitals is second order in their error.
    assert abs(scf.energy_at_orbitals(system, res.C_occ) - ref.energy) < 1e-7


def test_blocked_k_matches_cached_k():
    system = scf.h2_lattice(2)
    n = system.nao
    g = torch.Generator().manual_seed(0)
    C = torch.randn(n, system.nocc, generator=g, dtype=torch.float64)
    full = scf.DFJK(system.B, n)
    blocked = scf.DFJK(system.B, n, k_cache_bytes=0, k_block=5)
    assert (
        full.B3 is not None and blocked.B3 is None and len(blocked.blocks) == -(-system.naux // 5)
    )
    torch.testing.assert_close(blocked.k(C), full.k(C), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(blocked.j(2 * C @ C.T), full.j(2 * C @ C.T), rtol=1e-12, atol=1e-12)
    P = 2 * C @ C.T
    eri = _dense_eri(system)
    torch.testing.assert_close(full.j(P), torch.einsum("mnls,ls->mn", eri, P))
    torch.testing.assert_close(full.k(C), torch.einsum("mlns,ls->mn", eri, P))


def test_alkane_geometry():
    atoms = scf._alkane_atoms(4)
    assert sum(a == "C" for a, _ in atoms) == 4
    assert sum(a == "H" for a, _ in atoms) == 10
    xyz = {k: torch.tensor([c for a, c in atoms if a == k]) for k in ("C", "H")}
    d = torch.cdist(xyz["H"], xyz["C"]).min(dim=1).values
    torch.testing.assert_close(d, torch.full_like(d, 1.09), atol=1e-3, rtol=0)
    dhh = torch.cdist(xyz["H"], xyz["H"]) + 10 * torch.eye(10)
    assert float(dhh.min()) > 1.7  # no colliding hydrogens


def test_matches_pyscf():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto
    from pyscf import scf as pscf

    centres = scf.h2_lattice_geometry(2)
    basis = {"H": [[0, [e, 1.0]] for e in scf._H_AO_EXP]}
    aux = {"H": [[0, [e, 1.0]] for e in scf._H_AUX_EXP]}
    mol = gto.M(
        atom=[("H", tuple(c.tolist())) for c in centres], unit="Bohr", basis=basis, verbose=0
    )
    mf = pscf.RHF(mol).density_fit(auxbasis=aux)
    mf.conv_tol = 1e-12
    e_ref = mf.kernel()
    res = scf.run_scf(scf.h2_lattice(2), dtype=torch.float64)
    assert res.energy == pytest.approx(e_ref, abs=1e-9)
    assert pyscf is not None


def test_square_resident_matches_packed():
    system = scf.h2_lattice(2)
    n = system.nao
    g = torch.Generator().manual_seed(3)
    C = torch.randn(n, system.nocc, generator=g, dtype=torch.float64)
    packed = scf.DFJK(system.B, n, k_cache_bytes=0, k_block=7)
    square = scf.DFJK(system.B, n, k_cache_bytes=0, k_block=7, square=True)
    assert square.squares is not None and len(square.squares) == len(packed.blocks)
    torch.testing.assert_close(square.k(C), packed.k(C))
    torch.testing.assert_close(square.j(2 * C @ C.T), packed.j(2 * C @ C.T))
    J, K = square.jk(2 * C @ C.T, C)
    torch.testing.assert_close(J, packed.j(2 * C @ C.T))
    torch.testing.assert_close(K, packed.k(C))


def test_split_over_devices_matches_single():
    system = scf.h2_lattice(2)
    n = system.nao
    g = torch.Generator().manual_seed(2)
    C = torch.randn(n, system.nocc, generator=g, dtype=torch.float64)
    one = scf.DFJK(system.B, n)
    split = scf.DFJK(system.B, n, devices=["cpu", "cpu"])
    assert len(split.blocks) == 2
    torch.testing.assert_close(split.k(C), one.k(C))
    torch.testing.assert_close(split.j(2 * C @ C.T), one.j(2 * C @ C.T))


def test_fused_k_matches_unpacked():
    """trntensor.df_exchange / df_coulomb (trntensor#60) give the same J and K
    as the unpack + einsum + syrk blocks, including when split over devices."""
    trntensor = pytest.importorskip("trntensor")
    if not hasattr(trntensor, "df_exchange"):
        pytest.skip("trntensor without df_exchange (< 0.21)")
    system = scf.h2_lattice(2)
    n = system.nao
    g = torch.Generator().manual_seed(4)
    C = torch.randn(n, system.nocc, generator=g, dtype=torch.float64)
    ref = scf.DFJK(system.B, n, k_cache_bytes=0, k_block=7, fused=False)
    for kw in ({}, {"devices": ["cpu", "cpu"]}):
        fused = scf.DFJK(system.B, n, fused=True, **kw)
        assert fused.fused and fused.B3 is None
        assert len(fused.blocks) == len(kw.get("devices", [None]))
        torch.testing.assert_close(fused.k(C), ref.k(C))
        torch.testing.assert_close(fused.j(2 * C @ C.T), ref.j(2 * C @ C.T))
    # The default keeps the unpack path off the neuron device.
    assert not scf.DFJK(system.B, n).fused
