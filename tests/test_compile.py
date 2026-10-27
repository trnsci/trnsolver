"""torch.compile traceability on CPU (#41).

The Neuron failures in #41 (``with_effects`` in the FxImporter, no StableHLO
lowering for ``aten.linalg_qr``) need Trn2 to reproduce. These tests check
the CPU-visible preconditions with ``backend="aot_eager", fullgraph=True``:
no graph breaks, no side-effecting ``_linalg_check_errors`` op and no
``linalg_qr`` in the traced graph, and results that match eager.
"""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import trnsolver
from trnsolver.factor import _householder_qr


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


class TestCompileFactor:
    @pytest.mark.parametrize("n", [8, 128])
    def test_cholesky(self, spd_matrix, n):
        A = spd_matrix(n)
        assert _graph_breaks(trnsolver.cholesky, A) == 0
        torch.testing.assert_close(_compile(trnsolver.cholesky)(A), trnsolver.cholesky(A))

    def test_cholesky_upper(self, spd_matrix):
        A = spd_matrix(16)
        fn = _compile(lambda a: trnsolver.cholesky(a, upper=True))
        torch.testing.assert_close(fn(A), trnsolver.cholesky(A, upper=True))

    @pytest.mark.parametrize("n", [8, 128])
    def test_solve(self, spd_matrix, rng, n):
        A = spd_matrix(n)
        B = torch.randn(n, 3, generator=rng)
        assert _graph_breaks(trnsolver.solve, A, B) == 0
        torch.testing.assert_close(_compile(trnsolver.solve)(A, B), trnsolver.solve(A, B))

    def test_solve_spd(self, spd_matrix, rng):
        A = spd_matrix(32)
        b = torch.randn(32, generator=rng)
        assert _graph_breaks(trnsolver.solve_spd, A, b) == 0
        torch.testing.assert_close(_compile(trnsolver.solve_spd)(A, b), trnsolver.solve_spd(A, b))

    @pytest.mark.parametrize(
        "fn",
        [
            lambda A, B: trnsolver.cholesky(A),
            lambda A, B: trnsolver.solve(A, B),
            lambda A, B: trnsolver.solve_spd(A, B),
        ],
        ids=["cholesky", "solve", "solve_spd"],
    )
    def test_no_error_check_side_effect(self, spd_matrix, rng, fn):
        """`aten._linalg_check_errors` is what becomes `with_effects` on Neuron."""
        A = spd_matrix(16)
        B = torch.randn(16, 2, generator=rng)
        ops = _traced_aten_ops(fn, A, B)
        assert not any("_linalg_check_errors" in op for op in ops), ops

    def test_eager_still_raises_on_non_spd(self):
        A = -torch.eye(4)
        with pytest.raises(torch.linalg.LinAlgError):
            trnsolver.cholesky(A)
        with pytest.raises(torch.linalg.LinAlgError):
            trnsolver.solve(torch.zeros(4, 4), torch.ones(4, 1))

    @pytest.mark.parametrize("shape", [(8, 8), (128, 128), (12, 5), (5, 12), (3, 7, 4)])
    def test_qr(self, rng, shape):
        A = torch.randn(*shape, generator=rng)
        assert _graph_breaks(trnsolver.qr, A) == 0
        Q, R = _compile(trnsolver.qr)(A)
        Qe, Re = trnsolver.qr(A)
        assert Q.shape == Qe.shape and R.shape == Re.shape
        # Same LAPACK sign convention, so Q and R agree elementwise.
        torch.testing.assert_close(Q, Qe, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(R, Re, atol=1e-4, rtol=1e-4)

    def test_qr_no_linalg_qr_in_graph(self, random_matrix):
        ops = _traced_aten_ops(trnsolver.qr, random_matrix(16, 16))
        assert not any("linalg_qr" in op for op in ops), ops

    def test_qr_bf16(self, random_matrix):
        A = random_matrix(16, 16).bfloat16()
        Q, R = _compile(trnsolver.qr)(A)
        Qe, Re = trnsolver.qr(A)
        assert Q.dtype == torch.bfloat16
        torch.testing.assert_close(Q, Qe)
        torch.testing.assert_close(R, Re)


class TestHouseholderQR:
    """The compile-path QR, exercised eagerly (fp64 for tight tolerances)."""

    @pytest.mark.parametrize("shape", [(1, 1), (6, 6), (10, 4), (4, 10), (2, 5, 5)])
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
        with pytest.raises(torch._dynamo.exc.Unsupported):
            _compile(trnsolver.cg)(A, b)
