"""Hardware gate for the GEMM-only matrix functions (#149).

Run on Trn2 (TorchNeuron):

    pytest tests/test_matfun_neuron.py -v -m neuron

The CPU contracts are in tests/test_matfun.py. This file checks what only the
device can show: that density purification and the fixed-iteration
Newton-Schulz S^{-1/2} keep fp32 accuracy through the device's products
(an accidental bf16 product would show up as a ~1e-2 energy error), that the
precision hook reaches ``trnblas.gemm_fp64`` on the device (fp64-class
products for a host fp64 iterate, and the staged fp32-then-fp64 schedule),
and that the fixed loop compiles to one graph with the neuron backend. The
device timings against host eigh are scripts/bench_purification.py.
"""

import os
import sys

import pytest
import torch

import trnsolver
from trnsolver import _precision
from trnsolver.matfun import _purify_fixed, ns_iteration_count

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

import bench_purification as bp  # noqa: E402

pytestmark = pytest.mark.neuron

DEV = "neuron"
N = 512


@pytest.fixture(scope="module")
def prob():
    out = {}
    for case in ("gapped", "small"):
        F, S, n_occ, _ = bp.synthetic_scf(N, bp.CASES[case])
        P_ref, E_ref, _ = bp.reference(F, S, n_occ, repeat=1)
        out[case] = (F, S, n_occ, P_ref, E_ref)
    return out


def _errs(r, P_ref, E_ref):
    E_rel = abs(r.energy - E_ref) / abs(E_ref)
    P_rel = float((r.P.double().cpu() - P_ref).norm() / P_ref.norm())
    return E_rel, P_rel


@pytest.mark.parametrize("case", ["gapped", "small"])
@pytest.mark.parametrize("method", ["canonical", "sp2"])
def test_purification_fp32_on_device(prob, method, case):
    F, S, n_occ, P_ref, E_ref = prob[case]
    r = trnsolver.density_purification(
        F.float().to(DEV),
        S.float().to(DEV),
        n_occ=n_occ,
        method=method,
        iters=bp.ITERS[(method, case)],
        ns_cond=1e5,
    )
    assert r.P.device.type == DEV and r.P.dtype == torch.float32
    E_rel, P_rel = _errs(r, P_ref, E_ref)
    print(
        f"\n{method} {case} n={N}: E rel {E_rel:.2e}, P rel {P_rel:.2e}, "
        f"idem {r.idempotency_error:.2e}, trace {r.trace_error:.2e}, NS res {r.ns_residual:.2e}"
    )
    assert E_rel < 2e-5
    assert r.idempotency_error < 1e-3
    assert r.trace_error < 1e-3
    if case == "gapped":
        assert P_rel < 3e-2


def test_inv_sqrt_ns_fp32_on_device(prob):
    S = prob["gapped"][1]
    k = ns_iteration_count(1e5, 1e-6, "minimax")
    X, _, res = trnsolver.inv_sqrt_spd_ns(S.float().to(DEV), k, polynomial="minimax", cond=1e5)
    ref = trnsolver.inv_sqrt_spd(S)
    assert X.device.type == DEV
    assert res < 1e-4
    assert float((X.double().cpu() - ref).norm() / ref.norm()) < 1e-4


@pytest.mark.parametrize("scheme", ["ozaki1", "ozaki2"])
def test_purification_gemm_fp64_on_device(prob, scheme):
    pytest.importorskip("trnblas")
    F, S, n_occ, P_ref, E_ref = prob["gapped"]
    g = bp.device_gemm(DEV, scheme=scheme)  # host fp64 iterate, device products
    r = trnsolver.density_purification(F, S, n_occ=n_occ, iters=40, precision=g, ns_cond=1e5)
    E_rel, P_rel = _errs(r, P_ref, E_ref)
    assert E_rel < 1e-11 and P_rel < 1e-8
    assert r.idempotency_error < 1e-10


def test_purification_staged_on_device(prob):
    pytest.importorskip("trnblas")
    F, S, n_occ, P_ref, E_ref = prob["gapped"]
    spec = [bp.device_fp32_gemm(DEV)] * 20 + [bp.device_gemm(DEV)]
    r = trnsolver.density_purification(F, S, n_occ=n_occ, iters=40, precision=spec, ns_cond=1e5)
    E_rel, P_rel = _errs(r, P_ref, E_ref)
    assert r.idempotency_error < 1e-10
    assert E_rel < 1e-9  # second order in the fp32-stage subspace rotation
    assert P_rel < 1e-4


@pytest.mark.parametrize("method", ["canonical", "sp2"])
def test_fixed_loop_compiles_on_device(prob, method):
    F, S, n_occ, _, _ = prob["gapped"]
    X = trnsolver.inv_sqrt_spd(S)
    Fo = (X @ F @ X).float().to(DEV)
    gk, gl = _precision.schedule(None)

    def loop(Fo):
        return _purify_fixed(Fo, n_occ, method, 45, gk, gl)

    torch._dynamo.reset()
    P_c, _, idem_c, tr_c = torch.compile(loop, backend="neuron", fullgraph=True, dynamic=False)(Fo)
    P_e, _, idem_e, _ = loop(Fo)
    assert float((P_c - P_e).abs().max()) < 1e-4
    assert float(idem_c) < 1e-3 and float(tr_c) < 1e-3
