"""GEMM-only matrix functions (#149): Newton-Schulz S^{-1/2} and density purification.

References are host fp64 ``torch.linalg.eigh`` on synthetic SCF-like
generalized problems (scripts/bench_purification.py:synthetic_scf), gapped,
small-gap and near-degenerate at the Fermi level. Large sizes (n = 1024,
2048) run a subset of the grid. The device versions are in
tests/test_matfun_neuron.py.
"""

import math
import os
import sys

import pytest
import torch

import trnsolver
from trnsolver import matfun
from trnsolver.matfun import (
    _minimax_cubic,
    _ns_fixed,
    _ns_scale,
    _purify_fixed,
    density_purification,
    ns_coefficients,
    ns_iteration_count,
)

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

import bench_purification as bp  # noqa: E402

F64, F32 = torch.float64, torch.float32


@pytest.fixture(scope="module")
def problems():
    cache = {}

    def get(n, case):
        if (n, case) not in cache:
            F, S, n_occ, eps = bp.synthetic_scf(n, bp.CASES[case])
            P_ref, E_ref, _ = bp.reference(F, S, n_occ, repeat=1)
            cache[(n, case)] = (F, S, n_occ, eps, P_ref, E_ref)
        return cache[(n, case)]

    return get


def _spd(n, cond, seed=0, dtype=F64):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=F64))
    w = torch.logspace(0, -math.log10(cond), n, dtype=F64)
    return ((Q * w) @ Q.T).to(dtype), w


# ----------------------------------------------------------------------
# Newton-Schulz schedule
# ----------------------------------------------------------------------


@pytest.mark.parametrize("a", [1e-4, 1e-2, 0.3, 0.9])
def test_minimax_cubic_equioscillates(a):
    c1, c3, E = _minimax_cubic(a, 1.0)
    g = lambda s: c1 * s + c3 * s**3  # noqa: E731
    s_star = math.sqrt((a * a + a + 1.0) / 3.0)
    assert g(a) == pytest.approx(1 - E, abs=1e-14)
    assert g(1.0) == pytest.approx(1 - E, abs=1e-14)
    assert g(s_star) == pytest.approx(1 + E, abs=1e-14)
    # minimax beats Newton-Schulz (3s - s^3)/2 on the same interval
    s = torch.linspace(a, 1.0, 2001, dtype=F64)
    err_mm = (1 - (c1 * s + c3 * s**3)).abs().max()
    err_ns = (1 - 0.5 * (3 * s - s**3)).abs().max()
    assert err_mm <= err_ns + 1e-15


def test_minimax_schedule_tends_to_newton_schulz():
    coeffs = ns_coefficients(20, "minimax", 1e6)
    assert coeffs[0][0] > 2.5  # first step stretches the small eigenvalues
    assert coeffs[-1] == pytest.approx((1.5, -0.5), abs=1e-12)
    assert ns_coefficients(3, "newton") == [(1.5, -0.5)] * 3


@pytest.mark.parametrize("cond", [1e2, 1e4, 1e6, 1e8])
def test_iteration_count_minimax_fewer(cond):
    k_ns = ns_iteration_count(cond, 1e-14, "newton")
    k_mm = ns_iteration_count(cond, 1e-14, "minimax")
    assert k_mm < k_ns
    assert k_mm <= 0.65 * k_ns + 1


def test_schedule_errors():
    with pytest.raises(ValueError, match="cond"):
        ns_coefficients(3, "minimax")
    with pytest.raises(ValueError, match="polynomial"):
        ns_coefficients(3, "chebyshev")
    with pytest.raises(ValueError, match="scaling"):
        _ns_scale(torch.eye(3, dtype=F64), "spectral")
    with pytest.raises(ValueError, match="no convergence"):
        ns_iteration_count(1e6, 1e-14, "newton", max_iters=5)


# ----------------------------------------------------------------------
# inv_sqrt_spd_ns, fixed-iteration mode
# ----------------------------------------------------------------------


@pytest.mark.parametrize("polynomial", ["newton", "minimax"])
@pytest.mark.parametrize("cond", [1e2, 1e4, 1e6])
@pytest.mark.parametrize("n", [64, 256])
def test_ns_fixed_fp64_vs_eigh(n, cond, polynomial):
    A, w = _spd(n, cond)
    s = float(_ns_scale(A, "bound"))
    c = s / float(w.min())  # the bound the schedule wants: s / lambda_min
    k = ns_iteration_count(c, 1e-13, polynomial)
    X, iters, res = trnsolver.inv_sqrt_spd_ns(A, k, polynomial=polynomial, cond=c)
    ref = trnsolver.inv_sqrt_spd(A)
    assert iters == k
    assert res < 1e-10
    assert (X - ref).norm() / ref.norm() < 1e-9
    assert torch.equal(X, X.mT)


@pytest.mark.parametrize("polynomial", ["newton", "minimax"])
def test_ns_fixed_fp32(polynomial):
    A, w = _spd(256, 1e3, dtype=F32)
    k = ns_iteration_count(1e4, 1e-6, polynomial)
    X, _, res = trnsolver.inv_sqrt_spd_ns(A, k, polynomial=polynomial, cond=1e4)
    ref = trnsolver.inv_sqrt_spd(A.double())
    assert X.dtype == F32
    assert res < 1e-4
    assert (X.double() - ref).norm() / ref.norm() < 1e-4


def test_ns_cond_underestimate_converges_slower():
    A, w = _spd(128, 1e5)
    c = float(_ns_scale(A, "bound")) / float(w.min())
    k = ns_iteration_count(c, 1e-13, "minimax")
    _, _, res_low = trnsolver.inv_sqrt_spd_ns(A, k, polynomial="minimax", cond=c / 100)
    _, _, res_ok = trnsolver.inv_sqrt_spd_ns(A, k, polynomial="minimax", cond=c)
    _, _, res_more = trnsolver.inv_sqrt_spd_ns(A, k + 6, polynomial="minimax", cond=c / 100)
    assert res_ok < 1e-10 and res_more < 1e-10
    assert res_low > res_ok


def test_ns_minimax_fewer_iterations_same_accuracy():
    A, w = _spd(128, 1e6)
    c = float(_ns_scale(A, "bound")) / float(w.min())
    k_ns = ns_iteration_count(c, 1e-13, "newton")
    k_mm = ns_iteration_count(c, 1e-13, "minimax")
    X_ns, _, r_ns = trnsolver.inv_sqrt_spd_ns(A, k_ns)
    X_mm, _, r_mm = trnsolver.inv_sqrt_spd_ns(A, k_mm, polynomial="minimax", cond=c)
    assert k_mm < k_ns
    assert r_mm < 1e-10 and r_ns < 1e-10
    assert (X_mm - X_ns).norm() / X_ns.norm() < 1e-9


def test_ns_scaling_options():
    A, _ = _spd(32, 1e2)
    lam_max = float(torch.linalg.eigvalsh(A).max())
    for scaling in ("bound", "fro", "inf", 1.5 * lam_max):
        s = float(_ns_scale(A, scaling))
        assert s >= lam_max * (1 - 1e-12)
        X, _, res = trnsolver.inv_sqrt_spd_ns(A, 30, scaling=scaling)
        assert res < 1e-12
    with pytest.raises(ValueError):
        _ns_scale(A, -1.0)


def test_ns_adaptive_mode_unchanged():
    # iters=None keeps the original adaptive behaviour: Frobenius scaling,
    # max_iters=0 returns I / sqrt(||A||_F).
    A, _ = _spd(24, 1e2)
    X, iters, _ = trnsolver.inv_sqrt_spd_ns(A, max_iters=0)
    assert iters == 0
    assert torch.equal(X, torch.eye(24, dtype=F64) / torch.linalg.norm(A).sqrt())
    X, iters, res = trnsolver.inv_sqrt_spd_ns(A, tol=1e-12, max_iters=60)
    assert res < 1e-12 and 0 < iters < 60
    X2, iters2, _ = trnsolver.inv_sqrt_spd_ns(
        A, tol=1e-12, max_iters=60, polynomial="minimax", cond=1e3
    )
    assert iters2 < iters
    assert (X2 - X).norm() / X.norm() < 1e-10


# ----------------------------------------------------------------------
# precision hook
# ----------------------------------------------------------------------


def _tagged(tag, log):
    def g(A, B):
        log.append(tag)
        return A @ B

    return g


def test_precision_list_routes_per_iteration():
    A, _ = _spd(16, 10.0)
    log = []
    spec = [_tagged("lo", log)] * 2 + [_tagged("hi", log)]
    trnsolver.inv_sqrt_spd_ns(A, 4, precision=spec)
    # step 0: Y T (Z_0 = I); steps 1..3: Z Y, Y T, T Z; then the residual
    assert log == ["lo"] + ["lo"] * 3 + ["hi"] * 6 + ["hi"]


def test_precision_purification_gemm_count():
    F, S, n_occ, *_ = bp.synthetic_scf(32, 0.5)
    log = []
    r = density_purification(
        F,
        S,
        n_occ=n_occ,
        iters=5,
        ns_iters=4,
        precision=[_tagged("it", log)] * 5 + [_tagged("x", log)],
    )
    # NS (last spec): 1 + 3 * 3 + residual 1; transforms 2; canonical 2 / iter;
    # final idempotency 1; back-transform 2.
    assert log.count("it") == 10
    assert log.count("x") == 11 + 2 + 1 + 2
    assert r.ns_iters == 4 and r.iters == 5


def test_precision_specs_on_host():
    pytest.importorskip("trnblas")
    F, S, n_occ, _ = bp.synthetic_scf(64, 0.5)
    base = density_purification(F, S, n_occ=n_occ, iters=40)
    for spec in ("fp64", "ozaki1", "ozaki2", {"scheme": "ozaki1", "slices": 8}):
        r = density_purification(F, S, n_occ=n_occ, iters=40, precision=spec)
        assert abs(r.energy - base.energy) < 1e-10
        assert (r.P - base.P).norm() / base.P.norm() < 1e-10
    lo = density_purification(F, S, n_occ=n_occ, iters=40, precision="fp32")
    assert 1e-9 < abs(lo.energy - base.energy) < 1e-3
    assert lo.P.dtype == F64


def test_precision_invalid():
    with pytest.raises(ValueError, match="precision"):
        trnsolver.inv_sqrt_spd_ns(torch.eye(4, dtype=F64), 2, precision="bf16")
    with pytest.raises(ValueError, match="empty"):
        trnsolver.inv_sqrt_spd_ns(torch.eye(4, dtype=F64), 2, precision=[])


# ----------------------------------------------------------------------
# density purification against eigh
# ----------------------------------------------------------------------

TOL = {
    # (E rel, idempotency, trace, P rel)
    F64: (1e-12, 1e-11, 1e-11, 1e-8),
    F32: (2e-5, 1e-3, 1e-3, 3e-2),
}


def _check(r, prob, dtype, case):
    F, S, n_occ, eps, P_ref, E_ref = prob
    e_tol, idem_tol, tr_tol, p_tol = TOL[dtype]
    E_rel = abs(r.energy - E_ref) / abs(E_ref)
    P_rel = float((r.P.double() - P_ref).norm() / P_ref.norm())
    assert E_rel < e_tol, f"E rel {E_rel:.2e}"
    assert r.idempotency_error < idem_tol, f"idempotency {r.idempotency_error:.2e}"
    assert r.trace_error < tr_tol, f"trace {r.trace_error:.2e}"
    if dtype == F64 or case == "gapped":
        # fp32 noise (~1e-6 of the spectral width) exceeds the small gaps:
        # the density there is not determined at fp32, the energy is.
        assert P_rel < (1e-6 if case == "neardeg" else p_tol), f"P rel {P_rel:.2e}"
    # with S: P S P = P and tr(P S) = n_occ in the AO basis
    P = r.P.double()
    PSP = P @ S.double() @ P
    assert (PSP - P).norm() / P.norm() < (1e-9 if dtype == F64 else 1e-3)
    assert abs(float((P * S.double()).sum()) - n_occ) < (1e-9 if dtype == F64 else 1e-2)


@pytest.mark.parametrize("dtype", [F64, F32], ids=["fp64", "fp32"])
@pytest.mark.parametrize("case", ["gapped", "small", "neardeg"])
@pytest.mark.parametrize("method", ["canonical", "sp2"])
@pytest.mark.parametrize("n", [64, 256])
def test_purification_vs_eigh(problems, n, method, case, dtype):
    prob = problems(n, case)
    F, S, n_occ = prob[:3]
    iters = bp.ITERS[(method, case)]
    r = density_purification(F.to(dtype), S.to(dtype), n_occ=n_occ, method=method, iters=iters)
    assert r.P.dtype == dtype and r.P.shape == (n, n)
    assert r.iters == iters and r.history.shape == (iters,)
    _check(r, prob, dtype, case)
    thr = 1e-10 if dtype == F64 else 1e-3
    k = bp.first_below(r.history, thr)
    assert k is not None and k < iters, f"tr(P - P^2) never below {thr} in {iters} iterations"


@pytest.mark.parametrize(
    "n,case,dtype",
    [(1024, "gapped", F64), (1024, "small", F64), (1024, "gapped", F32), (2048, "gapped", F64)],
    ids=["1024-gapped-fp64", "1024-small-fp64", "1024-gapped-fp32", "2048-gapped-fp64"],
)
@pytest.mark.parametrize("method", ["canonical", "sp2"])
def test_purification_vs_eigh_large(problems, n, case, dtype, method):
    prob = problems(n, case)
    F, S, n_occ = prob[:3]
    r = density_purification(
        F.to(dtype), S.to(dtype), n_occ=n_occ, method=method, iters=bp.ITERS[(method, case)]
    )
    _check(r, prob, dtype, case)


def test_iteration_counts_grow_with_shrinking_gap(problems):
    for method in ("canonical", "sp2"):
        ks = []
        for case in ("gapped", "small", "neardeg"):
            F, S, n_occ = problems(256, case)[:3]
            r = density_purification(F, S, n_occ=n_occ, method=method, iters=120)
            ks.append(bp.first_below(r.history, 1e-10))
        assert ks[0] < ks[1] < ks[2], (method, ks)


def test_sp2_fewer_gemms_than_canonical(problems):
    F, S, n_occ = problems(256, "small")[:3]
    k = {}
    for method in ("canonical", "sp2"):
        r = density_purification(F, S, n_occ=n_occ, method=method, iters=120)
        k[method] = bp.first_below(r.history, 1e-10)
    assert k["sp2"] < 2 * k["canonical"]  # 1 product per SP2 step, 2 per canonical step


def test_staged_precision_energy_quadratic(problems):
    # fp32 products for the first 20 iterations, fp64 after: the late
    # iterations restore idempotency, but the subspace keeps an O(fp32)
    # rotation, so the density error is fp32-level while the energy error
    # (second order in the rotation) is far smaller.
    prob = problems(256, "gapped")
    F, S, n_occ, eps, P_ref, E_ref = prob
    r = density_purification(F, S, n_occ=n_occ, iters=40, precision=["fp32"] * 20 + [None])
    P_rel = float((r.P - P_ref).norm() / P_ref.norm())
    E_rel = abs(r.energy - E_ref) / abs(E_ref)
    assert r.idempotency_error < 1e-12
    assert 1e-8 < P_rel < 1e-4
    assert E_rel < 1e-9


def test_purification_orthonormal_basis():
    F, S, n_occ, eps = bp.synthetic_scf(64, 0.5, cond_s=1.0)  # S = 2 I
    Fo = F / 2.0  # X = I / sqrt 2
    r = density_purification(Fo, n_occ=n_occ, method="sp2", iters=45)
    assert r.X is None and r.ns_iters == 0 and r.ns_residual is None
    assert abs(r.energy - float(eps[:n_occ].sum())) < 1e-10
    assert torch.equal(r.P, r.P_orth)


def test_purification_rectangular_X():
    # A near-dependent basis: orthogonalizer drops directions (m < n).
    F, S, n_occ, eps = bp.synthetic_scf(64, 0.5, cond_s=1e4)
    # Duplicate four basis functions: S2 = T^T S T, rank 64 in 68 dims
    T = torch.cat([torch.eye(64, dtype=F64), torch.eye(64, dtype=F64)[:, :4]], 1)
    S2, F2 = T.T @ S @ T, T.T @ F @ T
    X = trnsolver.orthogonalizer(S2, method="canonical", lindep=1e-10).X
    assert X.shape == (68, 64)
    r = density_purification(F2, S2, n_occ=n_occ, X=X, iters=40)
    assert r.P_orth.shape == (64, 64) and r.P.shape == (68, 68)
    assert abs(r.energy - float(eps[:n_occ].sum())) < 1e-9
    assert abs(float((r.P * S2).sum()) - n_occ) < 1e-9


@pytest.mark.parametrize("method", ["canonical", "sp2"])
def test_purification_empty_and_full(method):
    F, _, _, _ = bp.synthetic_scf(32, 0.5, cond_s=1.0)
    r0 = density_purification(F, n_occ=0, method=method, iters=60)
    r1 = density_purification(F, n_occ=32, method=method, iters=60)
    assert r0.P.abs().max() < 1e-10
    assert (r1.P - torch.eye(32, dtype=F64)).abs().max() < 1e-10


def test_purification_bounds_argument(problems):
    F, S, n_occ, eps, P_ref, E_ref = problems(64, "gapped")
    r = density_purification(F, S, n_occ=n_occ, iters=40, bounds=(-25.0, 15.0))
    assert abs(r.energy - E_ref) / abs(E_ref) < 1e-12


def test_purification_errors():
    F = torch.eye(4, dtype=F64)
    with pytest.raises(ValueError, match="method"):
        density_purification(F, n_occ=2, method="trs4")
    with pytest.raises(ValueError, match="n_occ"):
        density_purification(F, n_occ=5)
    with pytest.raises(ValueError, match=r"\(n, n\)"):
        density_purification(torch.ones(3, 4, dtype=F64), n_occ=1)
    with pytest.raises(NotImplementedError):
        density_purification(F.to(torch.complex128), n_occ=2)
    with pytest.raises(ValueError, match="S must match"):
        density_purification(F, torch.eye(3, dtype=F64), n_occ=2)


# ----------------------------------------------------------------------
# transforms=: host fp64 S^{-1/2} and basis transforms (window 8, PR #154)
# ----------------------------------------------------------------------


def test_transforms_auto_resolution():
    r = matfun._resolve_transforms
    dev32 = torch.empty(4, 4, device="meta")  # stands in for a device tensor
    assert r("auto", dev32, True) == "host64"
    assert r("auto", dev32, False) == "native"  # no basis to transform
    assert r("auto", torch.empty(4, 4, device="meta", dtype=F64), True) == "native"
    assert r("auto", torch.empty(4, 4, device="meta", dtype=torch.bfloat16), True) == "native"
    assert r("auto", torch.empty(4, 4), True) == "native"  # fp32 on the CPU
    assert r("auto", torch.empty(4, 4, dtype=F64), True) == "native"
    for m in ("native", "host64"):
        assert r(m, dev32, True) == m and r(m, torch.empty(4, 4), False) == m
    with pytest.raises(ValueError, match="transforms"):
        density_purification(torch.eye(4), n_occ=2, transforms="fp64")


def test_transforms_default_on_host_unchanged(problems):
    F, S, n_occ, *_ = problems(64, "gapped")
    for dt in (F32, F64):
        a = density_purification(F.to(dt), S.to(dt), n_occ=n_occ, iters=40)
        b = density_purification(F.to(dt), S.to(dt), n_occ=n_occ, iters=40, transforms="native")
        assert a.X.dtype == dt and torch.equal(a.P, b.P) and a.energy == b.energy


@pytest.mark.parametrize("method", ["canonical", "sp2"])
def test_transforms_host64(problems, method):
    # The fp32orth arm: X and F' in host fp64, the loop in fp32 (F's dtype
    # and device). On Trn2 at n >= 2048 it kept ~3 more digits of energy.
    n, case = 1024, "gapped"
    prob = problems(n, case)
    F, S, n_occ, _, P_ref, E_ref = prob
    iters = bp.ITERS[(method, case)]
    kw = dict(n_occ=n_occ, method=method, iters=iters, ns_cond=1e5)
    nat = density_purification(F.float(), S.float(), transforms="native", **kw)
    r = density_purification(F.float(), S, transforms="host64", **kw)
    assert r.P.dtype == F32 and r.P_orth.dtype == F32 and r.P.shape == (n, n)
    assert r.X.dtype == F64 and r.X.device.type == "cpu"
    assert r.ns_iters == ns_iteration_count(1e5, 50 * torch.finfo(F64).eps)
    assert r.ns_residual < 1e-12
    _check(r, prob, F32, case)
    E_rel = abs(r.energy - E_ref) / abs(E_ref)
    E_nat = abs(nat.energy - E_ref) / abs(E_ref)
    assert E_rel < 5e-7 and E_rel < E_nat / 10, (E_rel, E_nat)
    # fp32 S is upcast exactly; the result stays close
    r32 = density_purification(F.float(), S.float(), transforms="host64", **kw)
    assert abs(r32.energy - E_ref) / abs(E_ref) < 2e-6
    # Reusing the host X (a later SCF iteration) skips Newton-Schulz.
    r2 = density_purification(F.float(), X=r.X, transforms="host64", **kw)
    assert r2.ns_iters == 0 and r2.ns_residual is None
    assert torch.equal(r2.P, r.P) and r2.energy == r.energy


def test_transforms_host64_rectangular_X_and_precision():
    F, S, n_occ, eps = bp.synthetic_scf(64, 0.5, cond_s=1e4)
    T = torch.cat([torch.eye(64, dtype=F64), torch.eye(64, dtype=F64)[:, :4]], 1)
    S2, F2 = T.T @ S @ T, T.T @ F @ T
    X = trnsolver.orthogonalizer(S2, method="canonical", lindep=1e-10).X
    log = []
    r = density_purification(
        F2.float(), X=X, n_occ=n_occ, iters=40, transforms="host64", precision=_tagged("p", log)
    )
    assert r.P_orth.shape == (64, 64) and r.P.shape == (68, 68) and r.P.dtype == F32
    assert abs(r.energy - float(eps[:n_occ].sum())) / abs(float(eps[:n_occ].sum())) < 1e-6
    # precision= routes the loop (2 per canonical iteration) and the final
    # idempotency check only; the transforms are host fp64 torch.matmul.
    assert len(log) == 2 * 40 + 1
    with pytest.raises(ValueError, match="n_occ"):
        density_purification(F2.float(), X=X, n_occ=65, transforms="host64")
    with pytest.raises(ValueError, match="S must match"):
        density_purification(F.float(), torch.eye(3, dtype=F64), n_occ=2, transforms="host64")
    # No S and no X: nothing to transform, the loop runs on F as given.
    o = density_purification(F.float() / 2, n_occ=n_occ, iters=40, transforms="host64")
    assert o.X is None and torch.equal(o.P, o.P_orth)


# ----------------------------------------------------------------------
# static schedule: no host synchronisation, traceable with fullgraph
# ----------------------------------------------------------------------


def test_fixed_loops_never_sync(monkeypatch):
    F, S, n_occ, _ = bp.synthetic_scf(32, 0.5)

    def boom(*a, **k):
        raise AssertionError("host synchronisation inside a fixed-iteration loop")

    monkeypatch.setattr(torch.Tensor, "item", boom)
    monkeypatch.setattr(torch.Tensor, "__bool__", boom)
    monkeypatch.setattr(torch.Tensor, "__float__", boom)
    coeffs = ns_coefficients(10, "minimax", 1e5)
    X, _ = _ns_fixed(S, coeffs, _ns_scale(S, "bound"), lambda k: torch.matmul, torch.matmul)
    Fo = X @ F @ X
    for method in ("canonical", "sp2"):
        _purify_fixed(Fo, n_occ, method, 20, lambda k: torch.matmul, torch.matmul)


@pytest.mark.parametrize("method", ["canonical", "sp2"])
def test_fixed_loop_compiles_fullgraph(method):
    F, S, n_occ, _ = bp.synthetic_scf(32, 0.5, dtype=F32)
    X = trnsolver.inv_sqrt_spd(S)
    Fo = X @ F @ X

    def loop(Fo):
        return _purify_fixed(Fo, n_occ, method, 30, lambda k: torch.matmul, torch.matmul)

    torch._dynamo.reset()
    P_c, h_c, idem_c, _ = torch.compile(loop, backend="aot_eager", fullgraph=True)(Fo)
    P_e, h_e, idem_e, _ = loop(Fo)
    assert torch.allclose(P_c, P_e, atol=1e-5)
    assert h_c.shape == (30,)


def test_public_exports():
    assert trnsolver.density_purification is matfun.density_purification
    assert trnsolver.PurificationResult is matfun.PurificationResult
