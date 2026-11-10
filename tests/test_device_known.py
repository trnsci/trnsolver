"""The device-known category rules of tests/test_conformance.py (trnsci/trnsci#35).

CPU only: checks that representative conformance IDs map to their category
(and others to None) without a device, and that the conftest summary groups
xfail reasons by category.
"""

from types import SimpleNamespace

import pytest
import torch
from conftest import _expect_category, _expect_summary
from test_conformance import DEVICE_KNOWN, _device_known_category

c64, c128, f32, f64 = torch.complex64, torch.complex128, torch.float32, torch.float64


def node(name, **params):
    return SimpleNamespace(originalname=name, name=name, callspec=SimpleNamespace(params=params))


@pytest.mark.parametrize(
    "item,cat",
    [
        (node("test_dtype", op="cholesky", dt=c64), "complex-linalg"),
        (node("test_dtype", op="eigh_generalized", dt=c128), "complex-linalg"),
        (node("test_dtype", op="schur", dt=c128), None),  # real-only: raising is a skip
        (node("test_dtype", op="cholesky", dt=f64), None),
        (node("test_dtype", op="svd", dt=f32), None),
        (node("test_mixed_real_complex", op="solve"), "complex-linalg"),
        (node("test_mixed_real_complex", op="eigh_generalized"), None),
        (node("test_eigh_randomized_dtype", dt=c128), "complex-linalg"),
        (node("test_eigh_randomized_dtype", dt=f64), None),
        (node("test_cg", n=129, tol=1e-10, M="jacobi", callable_A=False, x0=False), "device-fp64"),
        (node("test_gmres_ir", n=1, kw={}), "device-fp64"),
        (node("test_noncontiguous", op="solve_vec", dt=f64, kind="step_slice"), "device-fp64"),
        (node("test_noncontiguous", op="solve_vec", dt=f32, kind="step_slice"), None),
        (node("test_noncontiguous", op="cholesky", dt=f64, kind="step_slice"), None),
        (node("test_noncontiguous_extra", case="gmres-A_T"), "device-fp64"),
        (node("test_noncontiguous_extra", case="solve-B_T"), None),
        (node("test_gmres_truncated", maxiter=2, restart=2), "device-fp64"),
        (node("test_gmres_truncated", maxiter=1, restart=3), None),
        (node("test_cg_maxiter", maxiter=0), "device-fp64"),
        (node("test_cg_maxiter", maxiter=3), None),
        (node("test_inv_sqrt_spd_ns_kwargs", max_iters=0, tol=1e-7), "device-fp64"),
        (node("test_inv_sqrt_spd_ns_kwargs", max_iters=20, tol=1e-7), None),
        (SimpleNamespace(name="test_cg_exact_x0"), "device-fp64"),
        (node("test_iterative_dtype", solver="cg", dt=c64), None),
        (SimpleNamespace(name="test_lu_convention"), None),
    ],
)
def test_category_rules(item, cat):
    got = _device_known_category(item)
    assert got == cat
    assert got is None or got in DEVICE_KNOWN


def test_summary_groups_by_category():
    def rep(nodeid, reason=None):
        return SimpleNamespace(nodeid=nodeid, when="call", wasxfail=reason)

    assert _expect_category("[device-known:device-fp64] trnsci/trnsolver#147: x") == "device-fp64"
    assert _expect_category("trnsci/trnsolver#132, fix in #1: y") == "other xfail"
    s = _expect_summary(
        {
            "passed": [rep("a"), rep("b")],
            "xfailed": [rep("c", "[device-known:complex-linalg] #98: y"), rep("d", "plain")],
            "xpassed": [rep("e", "[device-known:device-fp64] #147: x")],
            "failed": [rep("f")],
        }
    )
    assert s["passed"] == 2
    assert s["xfailed"] == {"complex-linalg": ["c"], "other xfail": ["d"]}
    assert s["xpassed"] == {"device-fp64": ["e"]}
    assert s["new_failures"] == ["f"]
