"""CPU tests for the DF-MP2 example on top of the DF-RHF SCF (examples/scf_df_mp2.py)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import scf_df_rhf as scf  # noqa: E402

import trnsolver  # noqa: E402

trntensor = pytest.importorskip("trntensor")
if not hasattr(trntensor, "AuxStream"):
    pytest.skip("trntensor without AuxStream (trntensor#102)", allow_module_level=True)

import scf_df_mp2 as mp2  # noqa: E402


@pytest.fixture(autouse=True)
def _pytorch_backends():
    import trnblas

    libs = (trnsolver, trnblas, trntensor)
    old = [lib.get_backend() for lib in libs]
    for lib in libs:
        lib.set_backend("pytorch")
    yield
    for lib, b in zip(libs, old, strict=True):
        lib.set_backend(b)


def _dense_mp2(system, eps, C, frozen=0):
    """Reference: B_ia^P by einsum on the unpacked square, then the explicit sum."""
    n, nocc = system.nao, system.nocc
    B3 = trntensor.unpack_tril(
        system.B if isinstance(system.B, torch.Tensor) else system.B.to_tensor(), n
    )
    Co, Cv = C[:, frozen:nocc], C[:, nocc:]
    Bia = torch.einsum("mi,na,mnP->iaP", Co, Cv, B3.double())
    T = torch.einsum("iaP,jbP->ijab", Bia, Bia)
    eo, ev = eps[frozen:nocc], eps[nocc:]
    D = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev
    return float((T * (2 * T - T.transpose(2, 3)) / D).sum())


@pytest.fixture(scope="module")
def h2():
    system = scf.h2_lattice(6)
    res = scf.run_scf(system, dtype=torch.float64)
    eps, C, e_hf = mp2.canonical_orbitals(system, res.C_occ)
    return system, res, eps, C, e_hf


def test_canonical_orbitals(h2):
    system, res, eps, C, e_hf = h2
    assert e_hf == pytest.approx(res.energy, abs=1e-10)
    torch.testing.assert_close(eps, res.eps, rtol=0, atol=1e-7)
    torch.testing.assert_close(C.T @ system.S @ C, torch.eye(C.shape[1], dtype=torch.float64))


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"device_bytes": 0},  # blocked over occupied pairs
        {"device_bytes": 0, "occ_block": 2},
        {"devices": ["cpu", "cpu"]},  # aux chunks and occupied pairs over two "devices"
        {"frozen": 1},
        {"dtype": torch.float32},
    ],
)
def test_df_mp2_matches_dense(h2, kw):
    system, _res, eps, C, _ = h2
    frozen = kw.get("frozen", 0)
    ref = _dense_mp2(system, eps, C, frozen)
    e, info = mp2.df_mp2(system, eps, C, **kw)
    tol = 1e-6 * abs(ref) if kw.get("dtype") == torch.float32 else 1e-12
    assert e == pytest.approx(ref, abs=tol)
    assert info["nocc_corr"] == system.nocc - frozen
    assert info["mp2_path"] == (
        "blocked" if "device_bytes" in kw or "devices" in kw else "resident"
    )


def test_df_mp2_from_store_with_memmap(tmp_path, h2):
    system, res, eps, C, _ = h2
    disk = scf.h2_lattice(6, store=str(tmp_path / "s"), store_dtype=torch.float64)
    ref = _dense_mp2(system, eps, C)
    out = mp2.run_mp2(disk, res, bia_path=str(tmp_path / "bia.npy"))
    assert out.e_corr == pytest.approx(ref, abs=1e-12)
    assert out.e_total == pytest.approx(res.energy + ref, abs=1e-9)
    assert os.path.getsize(tmp_path / "bia.npy") >= out.info["bia_bytes"]


def test_matches_pyscf_dfmp2():
    pytest.importorskip("pyscf")
    system = scf.pyscf_system("water", reference=False)
    res = scf.run_scf(system, dtype=torch.float64)
    out = mp2.run_mp2(system, res)
    e_hf, e_corr = mp2.pyscf_df_mp2_reference("water")
    assert out.e_hf == pytest.approx(e_hf, abs=1e-9)
    assert out.e_corr == pytest.approx(e_corr, abs=1e-8)


def test_fp32_scf_and_mp2_match_pyscf():
    pytest.importorskip("pyscf")
    system = scf.pyscf_system("alkane3", reference=False)
    res = scf.run_scf(system, dtype=torch.float32)
    out = mp2.run_mp2(system, res, dtype=torch.float32)
    e_hf, e_corr = mp2.pyscf_df_mp2_reference("alkane3")
    assert abs(out.e_hf - e_hf) < 1e-7
    assert abs(out.e_corr - e_corr) < 1e-6


def test_matched_basis_reference_with_dropped_functions():
    pytest.importorskip("pyscf")
    system = scf.pyscf_system("alkane3", "aug-cc-pvdz", "aug-cc-pvdz-jkfit", reference=False)
    res = scf.run_scf(system, dtype=torch.float64, orth="canonical", lindep=1e-3)
    assert res.info["rank"] < system.nao
    out = mp2.run_mp2(system, res, orth="canonical", lindep=1e-3)
    e_hf, e_corr = mp2.pyscf_df_mp2_reference("alkane3", "aug-cc-pvdz", lindep=1e-3)
    assert out.e_hf == pytest.approx(e_hf, abs=1e-8)
    assert out.e_corr == pytest.approx(e_corr, abs=1e-7)
