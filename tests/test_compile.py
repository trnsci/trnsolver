"""torch.compile traceability on CPU (#41).

The Neuron failures in #41 (``with_effects`` in the FxImporter; no lowering
for ``linalg_cholesky_ex``, ``linalg_solve_ex`` or ``linalg_qr``) need Trn2 to
reproduce. These tests check the CPU-visible preconditions with
``backend="aot_eager", fullgraph=True``: no graph breaks, none of those ops in
the traced aten graph, and results that match eager.
"""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import trnsolver
from trnsolver.factor import (
    _cholesky_traceable,
    _cholesky_unblocked,
    _householder_qr,
    _householder_qr_unblocked,
    _solve_lu_traceable,
    _solve_lu_unblocked,
    _solve_triangular_traceable,
    _trsm_unblocked,
)

# 1 and 7 take the unblocked path; 64 and 128 are whole panels (nb = 32);
# 100 and 257 leave a ragged last panel.
_SIZES = [1, 7, 64, 100, 128, 257]


def _compile(fn):
    torch._dynamo.reset()
    return torch.compile(fn, backend="aot_eager", fullgraph=True)


def _traced_aten_ops(fn, *args):
    """Aten ops (pre-functionalization, so effectful no-output ops survive)
    in the graph Dynamo captures for fn(*args)."""
    ops = set()

    def backend(gm, example_inputs):
        # Retrace the Dynamo graph to aten with real, well-conditioned inputs.
        # make_fx without functionalization keeps ops like
        # aten._linalg_check_errors that AOTAutograd would otherwise DCE.
        real = [torch.eye(e.shape[0], e.shape[-1], dtype=e.dtype) for e in example_inputs]
        g = make_fx(gm)(*real)
        ops.update(str(n.target) for n in g.graph.nodes if n.op == "call_function")
        return gm

    torch._dynamo.reset()
    torch.compile(fn, backend=backend, fullgraph=True)(*args)
    return ops


def _graph_breaks(fn, *args):
    torch._dynamo.reset()
    return torch._dynamo.explain(fn)(*args).graph_break_count


_TOL = {torch.float32: dict(atol=1e-5, rtol=1e-5), torch.float64: dict(atol=1e-12, rtol=1e-12)}

# Ops Neuron cannot import or lower (#41): the error check that becomes
# ``with_effects``, and the LAPACK-backed linalg kernels with no StableHLO path.
_UNLOWERABLE = (
    "_linalg_check_errors",
    "linalg_cholesky_ex",
    "linalg_solve_ex",
    "linalg_solve_triangular",
    "linalg_lu",
    "linalg_qr",
)


# Every aten op the compile paths may emit: the set that lowered on Trn2
# (#41, round 2). A new op here needs a Neuron run before it is added.
_ALLOWED = {
    "<built-in function getitem>",
    *(
        f"aten.{op}"
        for op in (
            "abs.default",
            "add.Tensor",
            "alias.default",
            "arange.default",
            "argmax.default",
            "cat.default",
            "div.Tensor",
            "eq.Scalar",
            "eq.Tensor",
            "expand.default",
            "eye.default",  # inv_spd's torch.eye(n)
            "eye.m",
            "gather.default",
            "ge.Scalar",
            "gt.Scalar",
            "mm.default",
            "mul.Tensor",
            "neg.default",
            "ones.default",
            "permute.default",
            "scalar_tensor.default",
            "select.int",
            "slice.Tensor",
            "sqrt.default",
            "sub.Tensor",
            "sum.dim_IntList",
            "transpose.int",
            "tril.default",
            "triu.default",
            "unsqueeze.default",
            "where.self",
            "zeros.default",
            "zeros_like.default",
        )
    ),
}


def _assert_lowerable(ops):
    bad = sorted(op for op in ops if any(name in op for name in _UNLOWERABLE))
    assert not bad, bad
    unknown = sorted(set(ops) - _ALLOWED)
    assert not unknown, unknown


class TestCompileFactor:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("n", [8, 100, 128])
    def test_cholesky(self, spd_matrix, n, dtype):
        A = spd_matrix(n, dtype=dtype)
        assert _graph_breaks(trnsolver.cholesky, A) == 0
        L = _compile(trnsolver.cholesky)(A)
        torch.testing.assert_close(L, trnsolver.cholesky(A), **_TOL[dtype])

    def test_cholesky_upper(self, spd_matrix):
        A = spd_matrix(16)
        fn = _compile(lambda a: trnsolver.cholesky(a, upper=True))
        torch.testing.assert_close(fn(A), trnsolver.cholesky(A, upper=True), **_TOL[A.dtype])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("n", [8, 100, 128])
    def test_solve(self, spd_matrix, rng, n, dtype):
        A = spd_matrix(n, dtype=dtype)
        B = torch.randn(n, 3, generator=rng, dtype=dtype)
        assert _graph_breaks(trnsolver.solve, A, B) == 0
        X = _compile(trnsolver.solve)(A, B)
        torch.testing.assert_close(X, trnsolver.solve(A, B), **_TOL[dtype])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("n", [16, 100])
    def test_solve_requires_pivoting(self, rng, dtype, n):
        """Zero diagonal: elimination without row swaps would divide by zero."""
        # Cyclic shift (zero diagonal, well-conditioned) plus small noise.
        shift = torch.roll(torch.eye(n, dtype=dtype), 1, dims=1)
        A = 4 * shift + 0.1 * torch.randn(n, n, generator=rng, dtype=dtype)
        A = A - torch.diag(torch.diagonal(A))  # force a zero diagonal
        B = torch.randn(n, 2, generator=rng, dtype=dtype)
        X = _compile(trnsolver.solve)(A, B)
        torch.testing.assert_close(X, trnsolver.solve(A, B), **_TOL[dtype])
        torch.testing.assert_close(A @ X, B, **_TOL[dtype])

    def test_solve_vector_rhs(self, random_matrix, rng):
        A = random_matrix(12, 12) + 12 * torch.eye(12)
        b = torch.randn(12, generator=rng)
        x = _compile(trnsolver.solve)(A, b)
        assert x.shape == (12,)
        torch.testing.assert_close(x, trnsolver.solve(A, b), **_TOL[A.dtype])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_solve_spd(self, spd_matrix, rng, dtype):
        A = spd_matrix(32, dtype=dtype)
        b = torch.randn(32, generator=rng, dtype=dtype)
        assert _graph_breaks(trnsolver.solve_spd, A, b) == 0
        x = _compile(trnsolver.solve_spd)(A, b)
        torch.testing.assert_close(x, trnsolver.solve_spd(A, b), **_TOL[dtype])

    def test_inv_spd(self, spd_matrix):
        A = spd_matrix(16)
        torch.testing.assert_close(
            _compile(trnsolver.inv_spd)(A), trnsolver.inv_spd(A), **_TOL[A.dtype]
        )

    @pytest.mark.parametrize(
        "fn",
        [
            lambda A, B: trnsolver.cholesky(A),
            lambda A, B: trnsolver.solve(A, B),
            lambda A, B: trnsolver.solve_spd(A, B),
            lambda A, B: trnsolver.inv_spd(A),
        ],
        ids=["cholesky", "solve", "solve_spd", "inv_spd"],
    )
    @pytest.mark.parametrize("n", [16, 48])  # unblocked, blocked
    def test_traced_graph_is_lowerable(self, spd_matrix, rng, fn, n):
        A = spd_matrix(n)
        B = torch.randn(n, 2, generator=rng)
        _assert_lowerable(_traced_aten_ops(fn, A, B))

    def test_eager_still_raises_on_non_spd(self):
        A = -torch.eye(4)
        with pytest.raises(torch.linalg.LinAlgError):
            trnsolver.cholesky(A)
        with pytest.raises(torch.linalg.LinAlgError):
            trnsolver.solve(torch.zeros(4, 4), torch.ones(4, 1))

    @pytest.mark.parametrize(
        "shape", [(8, 8), (100, 100), (128, 128), (12, 5), (5, 12), (3, 7, 4), (2, 70, 40)]
    )
    def test_qr(self, rng, shape):
        A = torch.randn(*shape, generator=rng)
        assert _graph_breaks(trnsolver.qr, A) == 0
        Q, R = _compile(trnsolver.qr)(A)
        Qe, Re = trnsolver.qr(A)
        assert Q.shape == Qe.shape and R.shape == Re.shape
        # Same LAPACK sign convention, so Q and R agree elementwise.
        torch.testing.assert_close(Q, Qe, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(R, Re, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("n", [16, 48])  # unblocked, blocked
    def test_qr_no_linalg_qr_in_graph(self, random_matrix, n):
        _assert_lowerable(_traced_aten_ops(trnsolver.qr, random_matrix(n, n)))

    def test_qr_bf16(self, random_matrix):
        A = random_matrix(16, 16).bfloat16()
        Q, R = _compile(trnsolver.qr)(A)
        Qe, Re = trnsolver.qr(A)
        assert Q.dtype == torch.bfloat16
        torch.testing.assert_close(Q, Qe)
        torch.testing.assert_close(R, Re)


class TestHouseholderQR:
    """The compile-path QR, exercised eagerly (fp64 for tight tolerances)."""

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1),
            (6, 6),
            (10, 4),
            (4, 10),
            (2, 5, 5),
            (100, 100),
            (128, 128),
            (257, 257),
            (257, 64),
            (64, 257),
            (2, 70, 70),
        ],
    )
    def test_matches_lapack(self, rng, shape):
        A = torch.randn(*shape, generator=rng, dtype=torch.float64)
        Q, R = _householder_qr(A)
        Qe, Re = torch.linalg.qr(A)
        torch.testing.assert_close(Q, Qe)
        torch.testing.assert_close(R, Re)

    def test_zero_columns(self):
        """Zero sub-columns skip the reflector (tau = 0), as LAPACK does."""
        A = torch.randn(5, 5, dtype=torch.float64)
        A[:, 1] = 0.0
        A[2:, 0] = 0.0
        Q, R = _householder_qr(A)
        torch.testing.assert_close(Q @ R, A)
        torch.testing.assert_close(Q.T @ Q, torch.eye(5, dtype=torch.float64))
        assert torch.isfinite(Q).all() and torch.isfinite(R).all()
        assert torch.equal(R, torch.triu(R))

    def test_zero_matrix(self):
        Q, R = _householder_qr(torch.zeros(4, 3, dtype=torch.float64))
        torch.testing.assert_close(Q, torch.eye(4, 3, dtype=torch.float64))
        torch.testing.assert_close(R, torch.zeros(3, 3, dtype=torch.float64))


class TestTraceableKernels:
    """The compile-path Cholesky / LU / triangular solves, run eagerly in fp64
    against LAPACK, on both the unblocked (n <= nb) and blocked paths."""

    @pytest.mark.parametrize("batch", [(), (3,)])
    @pytest.mark.parametrize("n", _SIZES)
    def test_cholesky(self, rng, n, batch):
        G = torch.randn(*batch, n, n, generator=rng, dtype=torch.float64)
        A = G @ G.transpose(-2, -1) + n * torch.eye(n, dtype=torch.float64)
        torch.testing.assert_close(_cholesky_traceable(A), torch.linalg.cholesky(A))

    def test_cholesky_reads_lower_triangle_only(self, spd_matrix):
        A = spd_matrix(8, dtype=torch.float64)
        junk = A + torch.triu(torch.ones_like(A), diagonal=1) * 5.0
        torch.testing.assert_close(_cholesky_traceable(junk), torch.linalg.cholesky(A))

    def test_cholesky_non_spd_gives_nan(self):
        assert torch.isnan(_cholesky_traceable(-torch.eye(3, dtype=torch.float64))).any()

    @pytest.mark.parametrize("batch", [(), (4,)])
    @pytest.mark.parametrize("n", _SIZES)
    def test_lu_solve(self, rng, n, batch):
        # Plain Gaussian matrix: not diagonally dominant, so pivoting matters.
        A = torch.randn(*batch, n, n, generator=rng, dtype=torch.float64)
        B = torch.randn(*batch, n, 3, generator=rng, dtype=torch.float64)
        X, Xe = _solve_lu_traceable(A, B), torch.linalg.solve(A, B)
        assert ((X - Xe).norm() / Xe.norm()).item() < 1e-10

    def test_lu_solve_blocked_pivoting(self, rng):
        """Zero diagonal at a size that takes the blocked path."""
        n = 100
        A = 4 * torch.roll(torch.eye(n, dtype=torch.float64), 1, dims=1)
        A = A + 0.1 * torch.randn(n, n, generator=rng, dtype=torch.float64)
        A = A - torch.diag(torch.diagonal(A))
        B = torch.randn(n, 2, generator=rng, dtype=torch.float64)
        torch.testing.assert_close(_solve_lu_traceable(A, B), torch.linalg.solve(A, B))

    def test_lu_solve_pivots(self):
        """A[0, 0] = 0 and the column max is not on the diagonal at any step."""
        A = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [4.0, 5.0, 0.0]], dtype=torch.float64)
        eye = torch.eye(3, dtype=torch.float64)
        torch.testing.assert_close(_solve_lu_traceable(A, eye), torch.linalg.inv(A))

    @pytest.mark.parametrize("n", [20, 100, 257])
    @pytest.mark.parametrize("upper", [False, True])
    def test_triangular(self, rng, upper, n):
        T = torch.randn(n, n, generator=rng, dtype=torch.float64) + n * torch.eye(
            n, dtype=torch.float64
        )
        T = torch.triu(T) if upper else torch.tril(T)
        B = torch.randn(n, 4, generator=rng, dtype=torch.float64)
        torch.testing.assert_close(
            _solve_triangular_traceable(T, B, upper),
            torch.linalg.solve_triangular(T, B, upper=upper),
        )

    @pytest.mark.parametrize("batch", [(), (2,)])
    @pytest.mark.parametrize("n", [9, 20, 24])
    def test_blocked_matches_unblocked(self, rng, n, batch):
        """Small nb drives several panels (incl. a ragged last one) at small n."""
        nb = 8
        G = torch.randn(*batch, n, n, generator=rng, dtype=torch.float64)
        S = G @ G.transpose(-2, -1) + n * torch.eye(n, dtype=torch.float64)
        B = torch.randn(*batch, n, 3, generator=rng, dtype=torch.float64)
        torch.testing.assert_close(_cholesky_traceable(S, nb=nb), _cholesky_unblocked(S))
        torch.testing.assert_close(_solve_lu_traceable(G, B, nb=nb), _solve_lu_unblocked(G, B))
        L = torch.tril(S)
        for upper, T in [(False, L), (True, L.transpose(-2, -1))]:
            torch.testing.assert_close(
                _solve_triangular_traceable(T, B, upper, nb=nb), _trsm_unblocked(T, B, upper)
            )
        for A in [G, G[..., :, : n // 2], G[..., : n // 2, :]]:
            Q, R = _householder_qr(A, nb=nb)
            Qu, Ru = _householder_qr_unblocked(A)
            torch.testing.assert_close(Q, Qu)
            torch.testing.assert_close(R, Ru)


class TestEagerOnly:
    """Iterative solvers are torch.compiler.disable'd: one graph break, eager result."""

    def test_cg_runs_eagerly_under_compile(self, spd_matrix, rng):
        A = spd_matrix(16)
        b = torch.randn(16, generator=rng)
        torch._dynamo.reset()
        x, iters, res = torch.compile(trnsolver.cg, backend="aot_eager")(A, b)
        xe, iters_e, res_e = trnsolver.cg(A, b)
        assert iters == iters_e
        torch.testing.assert_close(x, xe)

    def test_cg_fullgraph_raises(self, spd_matrix, rng):
        A = spd_matrix(8)
        b = torch.randn(8, generator=rng)
        # torch 2.11 raises Unsupported; 2.14 raises UserError for the
        # data-dependent guard.
        with pytest.raises((torch._dynamo.exc.Unsupported, torch._dynamo.exc.UserError)):
            _compile(trnsolver.cg)(A, b)
