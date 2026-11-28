"""Hardware checks for reduction="ff" / "exact" / "auto" (trnsci/trnsci#169).

Run on Trn2 (TorchNeuron, trnblas >= 0.14):

    pytest tests/test_cg_reduction_ff_neuron.py -v -m neuron -s

CPU contracts are in test_cg_reduction_ff.py. Here:

- the device ``_inner`` / ``_norm`` / ``_cdot`` / ``_col_norm`` under ``"ff"``
  and ``"exact"`` give the CPU reference's bits (trnblas's kernels equal
  their reference on Trn2 window 7, so trnsolver's wiring must keep that);
- ``cg`` with ``"ff"`` / ``"exact"`` on the device repeats bitwise, takes
  host64's iterations and reaches the same fp64 forward error;
- LNC=1 vs LNC=2 (gated, ``TRNSOLVER_TEST_LNC=1``, whole chip): the timing
  harness scripts/bench_cg_reduction_ff.py runs once at each LNC and every
  mode but ``"device"`` must give equal SHA-256 of x and equal reduction
  values. ``TRNSOLVER_TEST_LNC1_CORES`` / ``TRNSOLVER_TEST_LNC2_CORES`` set
  ``NEURON_RT_VISIBLE_CORES`` for the two runs (default 0).
"""

import json
import math
import os
import pathlib
import subprocess
import sys

import pytest
import torch

import trnsolver
from trnsolver import block, iterative

pytestmark = pytest.mark.neuron

pytest.importorskip("trnblas.fp64")

DEV = "neuron"
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def _vec(n, seed):
    return torch.randn(n, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("n", [100003, 1_000_003])
@pytest.mark.parametrize("red", ["ff", "exact"])
def test_vector_reductions_device_equal_cpu(red, n):
    u, v = _vec(n, 1), _vec(n, 2)
    ud, vd = u.to(DEV), v.to(DEV)
    d_dev, d_cpu = iterative._inner(ud, vd, red), iterative._inner(u, v, red)
    r_dev, r_cpu = iterative._inner(ud, ud, red), iterative._inner(u, u, red)
    n_dev, n_cpu = iterative._norm(ud, red), iterative._norm(u, red)
    ref = math.fsum((u.double() * v.double()).tolist())
    print(f"{red} n={n}: device == cpu dot {d_dev == d_cpu} rr {r_dev == r_cpu} "
          f"norm {n_dev == n_cpu}; dot rel err {abs(d_dev - ref) / abs(ref):.2e}")  # fmt: skip
    assert d_dev == d_cpu and r_dev == r_cpu and n_dev == n_cpu
    if red == "exact":
        assert d_dev == ref


@pytest.mark.parametrize("red", ["ff", "exact"])
def test_block_reductions_device_equal_cpu(red):
    n, R = 100003, 33  # the GP exemplar's R (trnsci/trnsci#109)
    g = torch.Generator().manual_seed(3)
    X, Y = torch.randn(n, R, generator=g), torch.randn(n, R, generator=g)
    Xd, Yd = X.to(DEV), Y.to(DEV)
    assert torch.equal(block._cdot(Xd, Yd, red).cpu(), block._cdot(X, Y, red))
    assert torch.equal(block._col_norm(Xd, red).cpu(), block._col_norm(X, red))


def _banded(n, noise=0.05):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from bench_cg_reduction_ff import banded_op
    finally:
        sys.path.pop(0)
    return banded_op(n, noise)


@pytest.mark.parametrize("red", ["ff", "exact"])
def test_cg_device_repeat_and_convergence(red):
    n = 100003
    op = _banded(n)
    b = _vec(n, 4).to(DEV)
    x1, i1, r1 = trnsolver.cg(op, b, tol=1e-6, maxiter=500, reduction=red)
    x2, i2, r2 = trnsolver.cg(op, b, tol=1e-6, maxiter=500, reduction=red)
    _sync()
    xh, ih, rh = trnsolver.cg(op, b, tol=1e-6, maxiter=500, reduction="host64")
    print(f"{red}: iterations {i1} (host64 {ih}), residual {r1:.3e} (host64 {rh:.3e})")
    assert torch.equal(x1.cpu(), x2.cpu()) and i1 == i2 and r1 == r2
    assert r1 < 1e-6 and abs(i1 - ih) <= max(2, ih // 20)
    # Against a host float64 solve of the same banded system.
    x64, _, _ = trnsolver.cg(_banded(n), b.cpu().double(), tol=1e-12, maxiter=2000)
    err = float((x1.cpu().double() - x64).norm() / x64.norm())
    errh = float((xh.cpu().double() - x64).norm() / x64.norm())
    print(f"{red}: fwd err {err:.2e} (host64 {errh:.2e})")
    assert err < 1e-4 and err < 2 * errh + 1e-6


def test_auto_resolves_by_n():
    assert block._resolve_reduction("auto", 100_000) == "host64"
    assert block._resolve_reduction("auto", 299_999) == "host64"
    assert block._resolve_reduction("auto", 300_000) == "ff"
    assert block._resolve_reduction("auto", 1_000_000) == "ff"
    _, info = trnsolver.cg_block(_banded(1000), _vec(1000, 5).to(DEV), reduction="auto")
    assert info["reduction"] == "host64"
    # cg_block's default stays "pairwise" until trnblas#142 (window 8).
    _, info = trnsolver.cg_block(_banded(1000), _vec(1000, 5).to(DEV))
    assert info["reduction"] == "pairwise"


def _bench(lnc, cores, out):
    script = ROOT / "scripts" / "bench_cg_reduction_ff.py"
    env = dict(os.environ, NEURON_LOGICAL_NC_CONFIG=str(lnc), NEURON_RT_VISIBLE_CORES=cores)
    if "TORCH_NEURONX_NEFF_CACHE_DIR" in env:  # keep the two LNC settings' NEFFs apart
        env["TORCH_NEURONX_NEFF_CACHE_DIR"] += f"/lnc{lnc}"
    args = [sys.executable, "-P", str(script), "--n", "1e5,1e6", "--iters", "10"]
    args += ["--modes", "pairwise,host64,ff,exact,device", "--micro-reps", "2"]
    args += ["--R", "33", "--json", str(out)]
    proc = subprocess.run(args, env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(pathlib.Path(out).read_text())


@pytest.mark.skipif(
    os.environ.get("TRNSOLVER_TEST_LNC") != "1",
    reason="runs an LNC=1 and an LNC=2 process; needs the whole chip (TRNSOLVER_TEST_LNC=1)",
)
def test_ff_exact_lnc1_vs_lnc2_bitwise(tmp_path):
    r1 = _bench(1, os.environ.get("TRNSOLVER_TEST_LNC1_CORES", "0"), tmp_path / "lnc1.json")
    r2 = _bench(2, os.environ.get("TRNSOLVER_TEST_LNC2_CORES", "0"), tmp_path / "lnc2.json")
    keys = ("inner_value", "norm_value", "sha_fixed", "sha_solve", "block_sha")
    for n, run in r1["runs"].items():
        for mode, m1 in run["modes"].items():
            m2 = r2["runs"][n]["modes"][mode]
            same = {k: m1.get(k) == m2.get(k) for k in keys}
            print(f"n={n} {mode}: bitwise {same}")
            if mode != "device":  # the old, LNC-dependent path
                assert all(same.values()), (n, mode, same)
