"""CPU tests for scripts/bench_scf_ladder.py (the size-ladder planner, #15)."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

import bench_scf_ladder as lad  # noqa: E402


def test_estimate_reproduces_window3_anchor():
    # C40H82 / aug-cc-pVDZ: 33.4 GB fp32, two 24 GiB cores, one chip (#15).
    e = lad.estimate({"nao": 1658, "naux": 6064, "nocc": 161})
    assert e["B32_GB"] == pytest.approx(33.36, abs=0.05)
    assert e["pyscf_incore_GB"] == pytest.approx(133.4, abs=0.2)
    assert (e["cores_resident"], e["chips_resident"]) == (2, 1)
    assert e["plan"] == "3xl, resident" and e["stream_GB_per_iter"] == 0
    assert e["jk_s_1chip"] == pytest.approx(lad._A_JK_S_4CORES)


def test_estimate_plans_streaming_and_multichip():
    big = lad.estimate({"nao": 3462, "naux": 12664, "nocc": 337})
    assert big["chips_resident"] == 4 and big["stream_src"] == "NVMe"
    assert big["plan"].startswith("48xl, 4 chips")
    assert big["iter_s_48xl"] < big["iter_s_3xl"]
    mid = lad.estimate({"nao": 2478, "naux": 9064, "nocc": 241})  # 111 GB
    assert mid["stream_src"] == "host RAM" and mid["plan"].startswith("3xl, stream")
    # Resident per core when streaming: the 20 GB budget less three chunks of headroom.
    npair = 2478 * 2479 // 2
    w = lad.df_store.chunk_width(npair, 9064, 4, lad.df_store.DEFAULT_CHUNK_BYTES)
    resident = 4 * (20.0 - 3 * npair * w * 4 / 1e9)
    assert mid["stream_GB_per_iter"] == pytest.approx(mid["B32_GB"] - resident)
    assert 55.0 < resident < 65.0


def test_ladder_picks_units_near_targets():
    pytest.importorskip("pyscf")
    rows = lad.ladder(["alkane", "water"], ["cc-pvdz"], [500, 1000])
    assert [r["spec"] for r in rows] == [
        "pyscf:alkane20",
        "pyscf:alkane41",
        "pyscf:water21",
        "pyscf:water42",
    ]
    for r in rows:
        assert abs(r["nao"] - r["target"]) <= 24


def test_dry_run_cli():
    pytest.importorskip("pyscf")
    out = subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "..", "scripts", "bench_scf_ladder.py"),
            "--dry-run",
            "--families",
            "alkane",
            "--bases",
            "aug-cc-pvdz",
            "--targets",
            "1658,3500",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "| alkane40 | aug-cc-pvdz | 1658 | 6064 | 161 | 33.4 |" in out
    assert "48xl, 4 chips" in out


def test_run_one_small_system_against_pyscf(tmp_path):
    pytest.importorskip("pyscf")
    trntensor = pytest.importorskip("trntensor")
    if not hasattr(trntensor, "AuxStream"):
        pytest.skip("trntensor without AuxStream")
    row = lad.ladder(["alkane"], ["cc-pvdz"], [80])[0]  # C3H8, 82 AOs
    args = SimpleNamespace(
        store_dir=str(tmp_path),
        device="cpu",
        jk_devices=None,
        solver="auto",
        max_iter=60,
        warmup_iters=0,
        mp2_fp32=False,
        store_dtype="auto",
        ref_max_nao=600,
    )
    out = lad.run_one(row, args)
    assert out["converged"] and out["solver"] == "orth" and out["build"]["outcore"]
    assert out["store_dtype"] == "float64"
    assert abs(out["err_hf"]) < 1e-6 and abs(out["err_corr"]) < 1e-6
    lad.print_results([out])
