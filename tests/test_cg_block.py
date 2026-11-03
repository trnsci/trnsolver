"""Tests for cg_block, cg_block_ir, the pivoted low-rank preconditioner and SLQ (#97)."""

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse.linalg
import torch

import trnsolver


def _spd_with_cond(n, kappa, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    eig = torch.logspace(0, -np.log10(kappa), n, dtype=torch.float64)
    return ((Q * eig) @ Q.T).to(dtype)


def _rbf(n, d=3, noise=1e-2, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, d, generator=g, dtype=torch.float64) * 4
    K = torch.exp(-0.5 * torch.cdist(X, X) ** 2)
    K = 0.5 * (K + K.T)
    return K.to(dtype), (K + noise * torch.eye(n, dtype=torch.float64)).to(dtype)


class _Counted:
    def __init__(self, A):
        self.A = A
        self.calls = 0

    def __call__(self, V):
        self.calls += 1
        assert V.dim() == 2
        return self.A @ V


class TestCGBlockAccuracy:
    @pytest.mark.parametrize(
        "kappa,dtype,tol,maxiter",
        [
            (1e2, torch.float32, 1e-5, None),
            (1e2, torch.float64, 1e-10, None),
            (1e4, torch.float64, 1e-10, None),
            (1e6, torch.float64, 1e-10, 4000),
        ],
    )
    def test_matches_scipy_cg_and_cho_solve(self, kappa, dtype, tol, maxiter):
        n, R = 80, 5
        A = _spd_with_cond(n, kappa, dtype)
        g = torch.Generator().manual_seed(1)
        B = torch.randn(n, R, generator=g, dtype=torch.float64).to(dtype)
        X, info = trnsolver.cg_block(A, B, tol=tol, maxiter=maxiter)
        assert X.shape == (n, R) and X.dtype == dtype
        assert bool(info["converged"].all())
        A64, B64 = A.double().numpy(), B.double().numpy()
        X_ref = scipy.linalg.cho_solve(scipy.linalg.cho_factor(A64), B64)
        # Forward error bound ~ kappa * residual.
        fwd = 10 * kappa * tol if dtype == torch.float64 else 10 * kappa * 1e-5
        for j in range(R):
            x_sp, flag = scipy.sparse.linalg.cg(A64, B64[:, j], rtol=tol, maxiter=maxiter or 10 * n)
            assert flag == 0
            xj = X[:, j].double().numpy()
            ref = np.linalg.norm(X_ref[:, j])
            assert np.linalg.norm(xj - X_ref[:, j]) / ref < fwd
            assert np.linalg.norm(xj - x_sp) / ref < 2 * fwd
        assert float(info["true_residual"].max()) < 10 * tol

    def test_r1_matches_cg(self, spd_matrix):
        n = 40
        A = spd_matrix(n)
        b = torch.randn(n)
        x_ref, iters, _ = trnsolver.cg(A, b, tol=1e-6)
        x, info = trnsolver.cg_block(A, b, tol=1e-6)
        assert x.shape == (n,)
        assert int(info["col_iterations"][0]) == iters
        np.testing.assert_allclose(x.numpy(), x_ref.numpy(), rtol=1e-4, atol=1e-5)

    def test_block_equals_vector_solves(self):
        n = 60
        A = _spd_with_cond(n, 1e3)
        B = torch.randn(n, 4, dtype=torch.float64)
        X, _ = trnsolver.cg_block(A, B, tol=1e-12)
        for j in range(4):
            xj, _ = trnsolver.cg_block(A, B[:, j], tol=1e-12)
            np.testing.assert_allclose(X[:, j].numpy(), xj.numpy(), rtol=1e-9, atol=1e-12)

    def test_x0(self):
        n = 50
        A = _spd_with_cond(n, 1e2)
        B = torch.randn(n, 3, dtype=torch.float64)
        X_ref = torch.linalg.solve(A, B)
        X, info = trnsolver.cg_block(A, B, tol=1e-12, X0=X_ref + 1e-3)
        assert torch.allclose(X, X_ref, atol=1e-9)
        assert float(info["true_residual"].max()) < 1e-11

    def test_tensor_preconditioner(self):
        n = 60
        A = _spd_with_cond(n, 1e4)
        B = torch.randn(n, 2, dtype=torch.float64)
        Minv = torch.linalg.inv(A)
        X, info = trnsolver.cg_block(A, B, Minv, tol=1e-10)
        assert int(info["col_iterations"].max()) <= 2
        assert torch.allclose(X, torch.linalg.solve(A, B), atol=1e-8)


class TestCGBlockEdgeCases:
    def test_zero_columns(self):
        n = 30
        A = _spd_with_cond(n, 10.0)
        B = torch.randn(n, 3, dtype=torch.float64)
        B[:, 1] = 0
        X, info = trnsolver.cg_block(A, B, tol=1e-10)
        assert torch.all(X[:, 1] == 0)
        assert bool(info["converged"][1])
        assert int(info["col_iterations"][1]) == 0
        assert float(info["residual"][1]) == 0.0
        assert float(info["true_residual"][1]) == 0.0
        assert torch.isfinite(X).all()

    def test_all_zero(self):
        A = _Counted(_spd_with_cond(10, 10.0))
        X, info = trnsolver.cg_block(A, torch.zeros(10, 2, dtype=torch.float64))
        assert torch.all(X == 0) and info["iterations"] == 0
        assert bool(info["converged"].all())
        assert A.calls == 1  # the final true-residual check only

    def test_converged_column_freezes(self):
        n = 50
        A = _spd_with_cond(n, 1e4)
        evals, evecs = torch.linalg.eigh(A)
        B = torch.randn(n, 3, dtype=torch.float64)
        B[:, 0] = evecs[:, -1]  # converges in one step
        B[:, 1] = evecs[:, [-2, -15, -30]].sum(dim=1)  # three steps
        X_short, info_s = trnsolver.cg_block(A, B, tol=1e-10, maxiter=5, check_every=1)
        X_long, info_l = trnsolver.cg_block(A, B, tol=1e-10, check_every=1)
        cols = info_l["col_iterations"].tolist()
        assert cols[0] == 1 and cols[1] == 3 and cols[2] > 5
        # Columns that converged by iteration 5 are bitwise unchanged after.
        assert torch.equal(X_short[:, 0], X_long[:, 0])
        assert torch.equal(X_short[:, 1], X_long[:, 1])
        assert not torch.equal(X_short[:, 2], X_long[:, 2])
        assert info_s["converged"].tolist() == [True, True, False]

    def test_low_precision_restored(self, spd_matrix):
        A = spd_matrix(16)
        B = torch.randn(16, 2).to(torch.bfloat16)
        X, info = trnsolver.cg_block(A, B, tol=1e-3)
        assert X.dtype == torch.bfloat16

    def test_bad_shapes(self):
        with pytest.raises(ValueError):
            trnsolver.cg_block(torch.eye(3), torch.ones(3, 2, 2))
        with pytest.raises(ValueError):
            trnsolver.cg_block(torch.eye(3), torch.ones(3), check_every=0)


class TestCGBlockLaunches:
    @pytest.mark.parametrize("check_every", [1, 4, 10, 1000])
    def test_one_operator_call_per_iteration(self, check_every):
        n = 64
        A64 = _spd_with_cond(n, 1e3)
        B = torch.randn(n, 8, dtype=torch.float64)
        op = _Counted(A64)
        X, info = trnsolver.cg_block(op, B, tol=1e-8, check_every=check_every)
        # One block call per iteration, plus one for the final true residual.
        assert op.calls == info["iterations"] + 1
        assert info["iterations"] >= int(info["col_iterations"].max())
        if check_every < 1000:
            assert info["iterations"] < int(info["col_iterations"].max()) + check_every
        ref, ref_info = trnsolver.cg_block(A64, B, tol=1e-8, check_every=1)
        # check_every changes only when the loop stops, never the solution.
        assert torch.equal(X, ref)
        assert torch.equal(info["col_iterations"], ref_info["col_iterations"])

    def test_preconditioner_one_call_per_iteration(self):
        n = 64
        A64 = _spd_with_cond(n, 1e3)
        B = torch.randn(n, 4, dtype=torch.float64)
        d = torch.diagonal(A64)
        M = _Counted(torch.diag(1.0 / d))
        _, info = trnsolver.cg_block(A64, B, M, tol=1e-8)
        assert M.calls == info["iterations"] + 1  # + the initial z0 = M r0

    def test_coefficients_shape(self):
        n = 40
        A = _spd_with_cond(n, 1e2)
        B = torch.randn(n, 3, dtype=torch.float64)
        _, info = trnsolver.cg_block(A, B, tol=1e-10, return_coeffs=True)
        assert info["alpha"].shape == (info["iterations"], 3)
        assert info["beta"].shape == (info["iterations"], 3)
        for j in range(3):
            m = int(info["col_iterations"][j])
            assert torch.all(info["alpha"][:m, j] > 0)
            assert torch.all(info["alpha"][m:, j] == 0)


class TestPivotedCholesky:
    def test_exact_at_full_rank(self):
        n, r = 60, 7
        G = torch.randn(n, r, dtype=torch.float64)
        K = G @ G.T
        L = trnsolver.pivoted_cholesky(lambda: torch.diagonal(K).clone(), lambda i: K[i], n)
        assert L.shape[1] == r
        assert torch.allclose(L @ L.T, K, atol=1e-10 * float(K.abs().max()))

    def test_one_row_per_step(self):
        K, _ = _rbf(200)
        asked = []

        def rows(idx):
            asked.append(idx.numel())
            return K[idx]

        L = trnsolver.pivoted_cholesky(lambda: torch.diagonal(K).clone(), rows, 20)
        assert L.shape == (200, 20)
        assert asked == [1] * 20

    def test_tol_stops_early(self):
        K, _ = _rbf(300)
        L = trnsolver.pivoted_cholesky(
            lambda: torch.diagonal(K).clone(), lambda i: K[i], 300, tol=1e-6
        )
        assert L.shape[1] < 300
        assert float(torch.trace(K - L @ L.T)) <= 1e-6 * float(torch.trace(K)) * 1.0001

    def test_preconditioner_matches_dense(self):
        K, _ = _rbf(150)
        shift = 1e-2
        P = trnsolver.pivoted_lowrank_preconditioner(
            lambda: torch.diagonal(K).clone(), lambda i: K[i], 30, shift
        )
        assert P.L.shape == (150, 30) and P.shift == shift
        dense = P.L @ P.L.T + shift * torch.eye(150, dtype=torch.float64)
        V = torch.randn(150, 4, dtype=torch.float64)
        assert torch.allclose(P(V), torch.linalg.solve(dense, V), rtol=1e-8, atol=1e-8)
        assert torch.allclose(P(V[:, 0]), torch.linalg.solve(dense, V[:, 0]), atol=1e-8)
        assert P.logdet() == pytest.approx(torch.linalg.slogdet(dense)[1].item(), rel=1e-10)

    @pytest.mark.parametrize("noise", [1e-2, 1e-4])
    def test_preconditioned_fewer_iterations(self, noise):
        n = 2000
        K, A = _rbf(n, noise=noise, dtype=torch.float32)
        B = torch.randn(n, 4)
        P = trnsolver.pivoted_lowrank_preconditioner(
            lambda: torch.diagonal(K).clone(), lambda i: K[i], 100, noise
        )
        _, plain = trnsolver.cg_block(A, B, tol=1e-3, maxiter=3000)
        _, prec = trnsolver.cg_block(A, B, P, tol=1e-3, maxiter=3000)
        assert bool(prec["converged"].all())
        assert prec["iterations"] * 4 < plain["iterations"]

    def test_fp32_apply_is_host64_and_converges(self):
        """fp32 L: the default applies in float64 on the host (the fp32 Woodbury
        form cancels; with a near-complete low-rank factor it breaks CG)."""
        n, noise = 1500, 1e-3
        K, A = _rbf(n, noise=noise, dtype=torch.float32)
        K64 = K.double()
        L = trnsolver.pivoted_cholesky(lambda: torch.diagonal(K64).clone(), lambda i: K64[i], 400)
        P = trnsolver.PivotedLowRankPreconditioner(L.float(), noise)
        assert P.apply == "host64"
        V = torch.randn(n, 3)
        dense = L @ L.T + noise * torch.eye(n, dtype=torch.float64)
        ref = torch.linalg.solve(dense, V.double())
        out = P(V)
        assert out.dtype == torch.float32
        assert float((out.double() - ref).norm() / ref.norm()) < 1e-6
        B = torch.randn(n, 2)
        _, good = trnsolver.cg_block(A, B, P, tol=1e-5, maxiter=300)
        assert bool(good["converged"].all())
        native = trnsolver.PivotedLowRankPreconditioner(L.float(), noise, apply="native")
        assert native.apply == "native"
        with pytest.raises(ValueError):
            trnsolver.PivotedLowRankPreconditioner(L, noise, apply="fp16")


class TestCGBlockIR:
    def test_reaches_1e10_with_fp32_inner(self):
        n, noise = 1000, 1e-4
        K, A = _rbf(n, noise=noise)
        B = torch.randn(n, 3, dtype=torch.float64)
        A32 = A.float()
        K32 = K.float()
        inner_calls = _Counted(A32)
        P = trnsolver.pivoted_lowrank_preconditioner(
            lambda: torch.diagonal(K32).clone(), lambda i: K32[i], 100, noise
        )
        X, info = trnsolver.cg_block_ir(inner_calls, B, A, P, tol=1e-10, max_outer=15)
        assert X.dtype == torch.float64 and X.shape == B.shape
        assert bool(info["converged"].all())
        assert float(info["true_residual"].max()) < 1e-10
        hist = info["residual_history"]
        assert hist.shape == (info["outer_iterations"] + 1, 3)
        assert float(hist[-1].max()) < float(hist[0].max())
        X_ref = scipy.linalg.cho_solve(scipy.linalg.cho_factor(A.numpy()), B.numpy())
        err = np.linalg.norm(X.numpy() - X_ref, axis=0) / np.linalg.norm(X_ref, axis=0)
        assert err.max() < 1e-6
        # fp32 cg_block alone stalls far above this.
        X32, info32 = trnsolver.cg_block(A32, B.float(), P, tol=1e-10, maxiter=500)
        assert float(info32["true_residual"].max()) > 1e-8

    def test_vector_and_zero_columns(self):
        A = _spd_with_cond(40, 1e3)
        b = torch.randn(40, dtype=torch.float64)
        x, info = trnsolver.cg_block_ir(A, b, tol=1e-12)
        assert x.shape == (40,)
        assert float(info["true_residual"][0]) < 1e-12
        B = torch.zeros(40, 2, dtype=torch.float64)
        B[:, 0] = b
        X, info = trnsolver.cg_block_ir(A, B, tol=1e-12)
        assert torch.all(X[:, 1] == 0) and bool(info["converged"].all())

    def test_callable_default_a_hi(self):
        A = _spd_with_cond(40, 1e3)
        b = torch.randn(40, 2, dtype=torch.float64)
        A32 = A.float()

        def op(V):
            return A32.to(V.dtype) @ V

        X, info = trnsolver.cg_block_ir(op, b, tol=1e-12)
        assert float(info["true_residual"].max()) < 1e-12


class TestSLQ:
    def test_lanczos_tridiag_eigs(self):
        # Full-length CG on a small well-conditioned matrix: the Ritz values
        # of T are the eigenvalues of A.
        n = 12
        A = _spd_with_cond(n, 10.0)
        b = torch.randn(n, 1, dtype=torch.float64)
        _, info = trnsolver.cg_block(A, b, tol=1e-14, maxiter=n, return_coeffs=True)
        (T,) = trnsolver.lanczos_tridiag_from_cg(
            info["alpha"], info["beta"], info["col_iterations"]
        )
        assert T.shape == (n, n)
        np.testing.assert_allclose(
            torch.linalg.eigvalsh(T).numpy(), torch.linalg.eigvalsh(A).numpy(), rtol=1e-8
        )

    @pytest.mark.parametrize("preconditioned", [False, True])
    def test_logdet_vs_slogdet(self, preconditioned):
        n, noise = 1024, 1e-2
        K, A = _rbf(n, noise=noise)
        ref = torch.linalg.slogdet(A)[1].item()
        M = None
        if preconditioned:
            M = trnsolver.pivoted_lowrank_preconditioner(
                lambda: torch.diagonal(K).clone(), lambda i: K[i], 60, noise
            )
        g = torch.Generator().manual_seed(0)
        est, info = trnsolver.slq_logdet(
            A, n, num_probes=64, num_iters=100, generator=g, M=M, dtype=torch.float64
        )
        # Stated tolerance: 4 standard errors at 64 probes (and a 2% cap).
        assert abs(est - ref) <= 4 * info["std_error"]
        assert abs(est - ref) <= 0.02 * abs(ref)
