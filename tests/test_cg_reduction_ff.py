"""reduction="ff" / "exact" / "auto" in cg, gmres and cg_block (trnsci/trnsci#169).

``"ff"`` and ``"exact"`` hand the working-dtype vectors to trnblas.fp64:
the double-word ``dot`` / ``nrm2`` and the integer-limb exact ``dot``. On the
host these run trnblas's CPU reference, which issues the device kernels'
fp32 operations one for one (the kernels are bitwise equal to it on Trn2,
trnblas#134), so these tests pin the bits the device gives. ``"auto"`` picks
``"host64"`` below ``block._FF_AUTO_MIN_N`` rows and ``"ff"`` from there.

The layout-invariance checks (the CPU stand-in for LNC=1 vs LNC=2) for the
new modes are in test_cg_reduction.py, which runs every non-"device" mode.
"""

import math
import sys

import pytest
import torch

import trnsolver
from trnsolver import block, iterative

fp64 = pytest.importorskip("trnblas.fp64")

U_FF = 2.0**-48  # float-float representation error bound (relative)

# Awkward lengths around trnblas.fp64's layout: one element, a ragged
# partition (128 rows), the W_ALIGN / W_MAX tile edges, and the tile-count
# step at 128 · 4 · 512 = 262,144.
AWKWARD_N = [1, 2, 3, 127, 129, 2047, 8193, 65537, 262143, 262145, 1_000_003]


def _vec(n, seed, shift=0.0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, generator=g) + shift) * scale


def _exact_dot(u, v):
    """Exactly rounded Σ uᵢ vᵢ: fp32 products are exact in float64, fsum is exact."""
    return math.fsum((u.double() * v.double()).tolist())


def _gp(n, noise=1e-1, seed=0, ell=0.2):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 3, generator=g, dtype=torch.float64)
    K = torch.exp(-0.5 * torch.cdist(X, X).square() / ell**2)
    return K, K + noise * torch.eye(n, dtype=torch.float64)


def _nonsym(n, seed=11):
    g = torch.Generator().manual_seed(seed)
    _, A = _gp(n, noise=1.0, seed=seed)
    return A + 0.05 * torch.randn(n, n, generator=g, dtype=torch.float64) / n**0.5


# ── Correctness against fp64 at awkward n ────────────────────────────────────


class TestCorrectness:
    @pytest.mark.parametrize("n", AWKWARD_N)
    def test_dot_vs_exact(self, n):
        u, v = _vec(n, 1, shift=0.3), _vec(n, 2, shift=-0.1)
        ref = _exact_dot(u, v)
        scale = float((u.double() * v.double()).abs().sum())
        ff = iterative._inner(u, v, "ff")
        assert abs(ff - ref) <= 4 * U_FF * scale, (n, ff, ref)
        # The exact arm is exactly rounded: the same float as fsum's.
        assert iterative._inner(u, v, "exact") == ref

    @pytest.mark.parametrize("n", AWKWARD_N)
    def test_norm_vs_exact(self, n):
        u = _vec(n, 3, shift=0.2)
        ref = math.sqrt(_exact_dot(u, u))
        assert abs(iterative._norm(u, "ff") - ref) <= 4 * U_FF * ref
        assert iterative._norm(u, "exact") == ref

    def test_wiring_is_trnblas(self):
        u, v = _vec(100003, 4), _vec(100003, 5)
        assert iterative._inner(u, v, "ff") == float(fp64.dot(u, v))
        assert iterative._inner(u, v, "exact") == float(fp64.dot(u, v, accumulate="exact"))
        assert iterative._norm(u, "ff") == float(fp64.nrm2(u))

    def test_ill_conditioned_dot(self):
        # cond = Σ|uᵢvᵢ| / |u·v| ≈ 1e10: float-float error grows like 2^-48 cond,
        # the exact arm stays exactly rounded.
        n = 65537
        u = _vec(n, 6)
        v = _vec(n, 7)
        d = _exact_dot(u, v)
        # Cancel all but a tiny remainder with one extra term pair.
        u = torch.cat([u, torch.tensor([1.0])])
        v = torch.cat([v, torch.tensor([float(-d) + 1e-6], dtype=torch.float32)])
        ref = _exact_dot(u, v)
        scale = float((u.double() * v.double()).abs().sum())
        assert scale / abs(ref) > 1e6
        assert iterative._inner(u, v, "exact") == ref
        assert abs(iterative._inner(u, v, "ff") - ref) <= 4 * U_FF * scale

    def test_complex(self):
        g = torch.Generator().manual_seed(8)
        u = torch.randn(30011, generator=g, dtype=torch.complex64)
        v = torch.randn(30011, generator=g, dtype=torch.complex64)
        ref = float(torch.vdot(u.to(torch.complex128), v.to(torch.complex128)).real)
        scale = float((u.abs().double() * v.abs().double()).sum())
        for red in ("ff", "exact"):
            assert abs(iterative._inner(u, v, red) - ref) <= 4 * U_FF * scale, red
        ur, vr = torch.view_as_real(u).reshape(-1), torch.view_as_real(v).reshape(-1)
        assert iterative._inner(u, v, "ff") == float(fp64.dot(ur, vr))
        nref = float(torch.linalg.vector_norm(u.to(torch.complex128)))
        for red in ("ff", "exact"):
            assert abs(iterative._norm(u, red) - nref) <= 4 * U_FF * nref, red

    def test_float64_data(self):
        # Host float64 data: "ff" reduces in float64 (float-float would lose
        # bits), "exact" is trnblas's exactly rounded exact_dot.
        g = torch.Generator().manual_seed(9)
        u = torch.randn(4097, generator=g, dtype=torch.float64)
        v = torch.randn(4097, generator=g, dtype=torch.float64)
        assert iterative._inner(u, v, "ff") == float((u * v).sum())
        assert iterative._inner(u, v, "exact") == float(fp64.exact_dot(u, v))
        assert iterative._norm(u, "ff") == float(torch.linalg.vector_norm(u))

    def test_norm_no_overflow(self):
        # fp32 squares overflow above ~1.8e19; nrm2 scales per chunk, and the
        # exact arm's out-of-window terms take the exact host path.
        u = _vec(10007, 10) * 1e30
        ref = math.sqrt(math.fsum((u.double() ** 2).tolist()))
        assert abs(iterative._norm(u, "ff") - ref) <= 4 * U_FF * ref
        assert abs(iterative._norm(u, "exact") - ref) <= 2.0**-52 * ref

    @pytest.mark.parametrize("red", ["ff", "exact"])
    def test_nonfinite_in_nan_out(self, red):
        u = _vec(1000, 21)
        u[17] = float("inf")
        v = _vec(1000, 22)
        assert math.isnan(iterative._inner(u, v, red))
        u[17] = float("nan")
        assert math.isnan(iterative._norm(u, red))

    def test_block_columns_match_vectors(self):
        g = torch.Generator().manual_seed(11)
        X, Y = torch.randn(4099, 5, generator=g), torch.randn(4099, 5, generator=g)
        for red in ("ff", "exact"):
            d = block._cdot(X, Y, red)
            nr = block._col_norm(X, red)
            d2, nr2 = block._cdot_and_norm(X, Y, red)
            assert d.dtype == nr.dtype == torch.float32
            assert torch.equal(d, d2) and torch.equal(nr, nr2)
            for j in range(5):
                xj, yj = X[:, j].contiguous(), Y[:, j].contiguous()
                assert float(d[j]) == float(torch.tensor(iterative._inner(xj, yj, red)).float())
                assert float(nr[j]) == float(torch.tensor(iterative._norm(xj, red)).float())

    def test_missing_trnblas_fp64(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "trnblas.fp64", None)
        u = torch.ones(8)
        with pytest.raises(ImportError, match="trnblas >= 0.14"):
            iterative._inner(u, u, "ff")
        # The other modes do not touch trnblas.
        assert iterative._inner(u, u, "pairwise") == 8.0


# ── Bitwise reproducibility ──────────────────────────────────────────────────


class TestReproducible:
    @pytest.mark.parametrize("red", ["ff", "exact"])
    def test_repeat_and_copy_bitwise(self, red):
        u, v = _vec(262145, 12), _vec(262145, 13)
        a = iterative._inner(u, v, red)
        assert iterative._inner(u.clone(), v.clone(), red) == a
        # A strided view (a gmres basis column) gives the contiguous bits.
        V = torch.stack([v, u], 1)
        assert iterative._inner(u, V[:, 0], red) == a

    @pytest.mark.parametrize("red", ["ff", "exact"])
    def test_solvers_repeat_bitwise(self, red):
        _, A = _gp(700)
        A = A.float()
        b = torch.randn(700, generator=torch.Generator().manual_seed(14))
        x1, i1, r1 = trnsolver.cg(A, b, tol=1e-6, reduction=red)
        x2, i2, r2 = trnsolver.cg(A, b, tol=1e-6, reduction=red)
        assert torch.equal(x1, x2) and i1 == i2 and r1 == r2
        B = torch.randn(700, 3, generator=torch.Generator().manual_seed(15))
        X1, f1 = trnsolver.cg_block(A, B, tol=1e-6, reduction=red)
        X2, f2 = trnsolver.cg_block(A, B, tol=1e-6, reduction=red)
        assert torch.equal(X1, X2) and f1["iterations"] == f2["iterations"]


# ── Convergence equivalence ──────────────────────────────────────────────────


class TestConvergence:
    """ff / exact take host64's iterations and reach the same fp64 forward error."""

    @pytest.mark.parametrize("red", ["ff", "exact"])
    def test_cg(self, red):
        n = 900
        _, A = _gp(n)
        b = torch.randn(n, generator=torch.Generator().manual_seed(16))
        x64 = torch.linalg.solve(A, b.double())
        x, it, res = trnsolver.cg(A.float(), b, tol=1e-6, reduction=red)
        xh, ith, _ = trnsolver.cg(A.float(), b, tol=1e-6, reduction="host64")
        assert res < 1e-6 and abs(it - ith) <= 1, (it, ith)
        assert float((x.double() - x64).norm() / x64.norm()) < 1e-4
        assert torch.allclose(x, xh, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("red", ["ff", "exact"])
    def test_cg_jacobi_preconditioned(self, red):
        n = 900
        _, A = _gp(n, noise=1e-2)
        A = A + torch.diag(torch.linspace(0, 5, n, dtype=torch.float64))  # uneven diagonal
        b = torch.randn(n, generator=torch.Generator().manual_seed(17))
        M = trnsolver.jacobi_preconditioner(A.float())
        x, it, res = trnsolver.cg(A.float(), b, tol=1e-6, M=M, maxiter=3000, reduction=red)
        _, ith, _ = trnsolver.cg(A.float(), b, tol=1e-6, M=M, maxiter=3000, reduction="host64")
        assert res < 1e-6 and abs(it - ith) <= max(2, ith // 20), (it, ith)
        x64 = torch.linalg.solve(A, b.double())
        assert float((x.double() - x64).norm() / x64.norm()) < 1e-3

    @pytest.mark.parametrize("red", ["ff", "exact"])
    def test_gmres(self, red):
        n = 500
        A = _nonsym(n, seed=15)
        b = torch.randn(n, generator=torch.Generator().manual_seed(15))
        x, it, res = trnsolver.gmres(A.float(), b, tol=1e-5, restart=20, reduction=red)
        _, ith, _ = trnsolver.gmres(A.float(), b, tol=1e-5, restart=20, reduction="host64")
        assert res < 1e-5 and it == ith, (it, ith)
        x64 = torch.linalg.solve(A, b.double())
        assert float((x.double() - x64).norm() / x64.norm()) < 1e-5

    @pytest.mark.parametrize("red", ["ff", "exact"])
    def test_cg_block_projected_preconditioner(self, red):
        # The GP exemplar's shape (trnsci/trnsci#109): cg_block with the
        # projected pivoted-Cholesky preconditioner, whose Qᵀ V takes the
        # pairwise blocked product under ff / exact.
        n = 1200
        K, A = _gp(n, noise=1e-2)
        L = trnsolver.pivoted_cholesky(lambda: torch.diagonal(K).clone(), lambda i: K[i], 64)
        B = torch.randn(n, 4, generator=torch.Generator().manual_seed(18))
        M = trnsolver.PivotedLowRankPreconditioner(
            L.float(), 1e-2, apply="projected", reduction=red
        )
        assert M._Qpad is not None  # pairwise Qᵀ V
        # The same preconditioner bits for both (Qᵀ V is pairwise under ff /
        # exact), so only cg_block's own reductions differ.
        Mp = trnsolver.PivotedLowRankPreconditioner(
            L.float(), 1e-2, apply="projected", reduction="pairwise"
        )
        X, info = trnsolver.cg_block(A.float(), B, M, tol=1e-5, maxiter=500, reduction=red)
        _, ih = trnsolver.cg_block(A.float(), B, Mp, tol=1e-5, maxiter=500, reduction="host64")
        assert bool(info["converged"].all()) and info["reduction"] == red
        # Per column: fp32 CG here stalls near tol, so where the recurrence
        # residual first crosses it moves by a step or two between modes.
        dc = (info["col_iterations"] - ih["col_iterations"]).abs().max()
        assert int(dc) <= max(2, int(ih["col_iterations"].max()) // 20), dc
        X64 = torch.linalg.solve(A, B.double())
        assert float(((X.double() - X64).norm(dim=0) / X64.norm(dim=0)).max()) < 1e-3


# ── The "auto" policy ────────────────────────────────────────────────────────


class TestAuto:
    def test_threshold(self, monkeypatch):
        monkeypatch.delenv(block._FF_MIN_N_ENV, raising=False)
        assert block._FF_AUTO_MIN_N == 500_000
        r = block._resolve_reduction
        assert r("auto", 1) == "host64"
        assert r("auto", 499_999) == "host64"
        assert r("auto", 500_000) == "ff"
        assert r("auto", 10_000_000) == "ff"
        for m in ("pairwise", "host64", "device", "ff", "exact"):
            assert r(m, 1) == m and r(m, 10**7) == m

    @pytest.mark.parametrize(
        "v,m", [("1000", 1000), ("0", 0), ("5e5", 500_000), (" 2e6 ", 2 * 10**6)]
    )
    def test_env_override(self, monkeypatch, v, m):
        monkeypatch.setenv(block._FF_MIN_N_ENV, v)
        assert block._ff_min_n() == m
        assert block._resolve_reduction("auto", m) == "ff"
        if m:
            assert block._resolve_reduction("auto", m - 1) == "host64"

    @pytest.mark.parametrize("v", ["abc", "-1", "1.5", "inf", "nan"])
    def test_env_invalid(self, monkeypatch, v):
        monkeypatch.setenv(block._FF_MIN_N_ENV, v)
        with pytest.raises(ValueError, match=block._FF_MIN_N_ENV):
            block._resolve_reduction("auto", 10)
        # Explicit modes never read the threshold.
        assert block._resolve_reduction("ff", 10) == "ff"

    def test_solvers_resolve_once(self, monkeypatch):
        monkeypatch.delenv(block._FF_MIN_N_ENV, raising=False)
        monkeypatch.delenv("TRNSOLVER_CG_REDUCTION", raising=False)
        monkeypatch.delenv("TRNSOLVER_GMRES_REDUCTION", raising=False)
        A, b = torch.eye(6) * 2, torch.ones(6)
        calls = []
        orig = iterative._inner

        def spy(u, v, reduction="device"):
            calls.append(reduction)
            return orig(u, v, reduction)

        monkeypatch.setattr(iterative, "_inner", spy)
        trnsolver.cg(A, b, reduction="auto")
        assert set(calls) == {"host64"}
        calls.clear()
        trnsolver.gmres(A, b, reduction="auto")
        assert set(calls) == {"host64"}
        calls.clear()
        monkeypatch.setenv(block._FF_MIN_N_ENV, "6")
        trnsolver.cg(A, b, reduction="auto")
        assert set(calls) == {"ff"}
        calls.clear()
        monkeypatch.setenv("TRNSOLVER_GMRES_REDUCTION", "auto")
        trnsolver.gmres(A, b)
        assert set(calls) == {"ff"}

    def test_cg_block_reports_mode(self, monkeypatch):
        monkeypatch.delenv(block._FF_MIN_N_ENV, raising=False)
        A, B = torch.eye(8) * 3, torch.ones(8, 2)
        _, info = trnsolver.cg_block(A, B, reduction="auto")
        assert info["reduction"] == "host64"
        monkeypatch.setenv(block._FF_MIN_N_ENV, "8")
        X, info = trnsolver.cg_block(A, B, reduction="auto")
        assert info["reduction"] == "ff"
        assert torch.allclose(X, B / 3)
        monkeypatch.setenv("TRNSOLVER_CG_REDUCTION", "auto")
        _, info = trnsolver.cg_block(A, B)
        assert info["reduction"] == "ff"
        _, info = trnsolver.cg_block(A, B, reduction="pairwise")
        assert info["reduction"] == "pairwise"

    def test_auto_equals_resolved_mode_bitwise(self, monkeypatch):
        _, A = _gp(400)
        b = torch.randn(400, generator=torch.Generator().manual_seed(19))
        for thresh, mode in (("401", "host64"), ("400", "ff")):
            monkeypatch.setenv(block._FF_MIN_N_ENV, thresh)
            xa, ia, ra = trnsolver.cg(A.float(), b, tol=1e-6, reduction="auto")
            xm, im, rm = trnsolver.cg(A.float(), b, tol=1e-6, reduction=mode)
            assert torch.equal(xa, xm) and ia == im and ra == rm, mode

    def test_preconditioner_auto_uses_pairwise_qtv(self):
        K, _ = _gp(300)
        L = trnsolver.pivoted_cholesky(lambda: torch.diagonal(K).clone(), lambda i: K[i], 16)
        for r in ("auto", "ff", "exact"):
            P = trnsolver.PivotedLowRankPreconditioner(
                L.float(), 1e-2, apply="projected", reduction=r
            )
            Pp = trnsolver.PivotedLowRankPreconditioner(
                L.float(), 1e-2, apply="projected", reduction="pairwise"
            )
            assert P.reduction == r and P._Qpad is not None
            V = torch.randn(300, 3, generator=torch.Generator().manual_seed(20))
            assert torch.equal(P(V), Pp(V))
