"""Test matrix factorizations."""

import warnings

import numpy as np
import pytest
import torch

import trnsolver


class TestCholesky:
    def test_identity(self):
        L = trnsolver.cholesky(torch.eye(4))
        np.testing.assert_allclose(L.numpy(), np.eye(4), atol=1e-6)

    def test_reconstruction(self, spd_matrix):
        n = 16
        A = spd_matrix(n)
        L = trnsolver.cholesky(A)
        reconstructed = L @ L.T
        np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-4)

    def test_lower_triangular(self, spd_matrix):
        L = trnsolver.cholesky(spd_matrix(8))
        upper = torch.triu(L, diagonal=1)
        np.testing.assert_allclose(upper.numpy(), np.zeros_like(upper.numpy()), atol=1e-7)

    def test_upper(self, spd_matrix):
        A = spd_matrix(8)
        U = trnsolver.cholesky(A, upper=True)
        reconstructed = U.T @ U
        np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-4)


class TestLU:
    def test_reconstruction(self, random_matrix):
        A = random_matrix(8, 8)
        P, L, U = trnsolver.lu(A)
        reconstructed = P @ L @ U
        np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-5)


class TestQR:
    def test_reconstruction(self, random_matrix):
        A = random_matrix(8, 6)
        Q, R = trnsolver.qr(A)
        reconstructed = Q @ R
        np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-5)

    def test_orthogonality(self, random_matrix):
        A = random_matrix(8, 6)
        Q, _ = trnsolver.qr(A)
        QtQ = Q.T @ Q
        np.testing.assert_allclose(QtQ.numpy(), np.eye(6), atol=1e-5)


class TestSVD:
    def test_reconstruction(self, random_matrix):
        A = random_matrix(8, 6)
        U, s, Vh = trnsolver.svd(A)
        recon = U @ torch.diag(s) @ Vh
        np.testing.assert_allclose(recon.numpy(), A.numpy(), atol=1e-5)

    def test_left_orthogonality(self, random_matrix):
        A = random_matrix(8, 6)
        U, _, _ = trnsolver.svd(A)
        np.testing.assert_allclose((U.T @ U).numpy(), np.eye(6), atol=1e-5)

    def test_right_orthogonality(self, random_matrix):
        A = random_matrix(8, 6)
        _, _, Vh = trnsolver.svd(A)
        np.testing.assert_allclose((Vh @ Vh.T).numpy(), np.eye(6), atol=1e-5)

    def test_singular_values_descending(self, random_matrix):
        A = random_matrix(8, 6)
        _, s, _ = trnsolver.svd(A)
        assert torch.all(s[:-1] >= s[1:])

    def test_vs_torch(self, random_matrix):
        A = random_matrix(8, 6)
        _, s, _ = trnsolver.svd(A)
        s_ref = torch.linalg.svdvals(A)
        np.testing.assert_allclose(s.numpy(), s_ref.numpy(), atol=1e-5)

    def test_full_matrices_shape(self, random_matrix):
        m, n = 8, 6
        A = random_matrix(m, n)
        U_full, s_full, Vh_full = trnsolver.svd(A, full_matrices=True)
        assert U_full.shape == (m, m)
        assert Vh_full.shape == (n, n)
        assert s_full.shape == (n,)


class TestSchur:
    def test_reconstruction(self):
        torch.manual_seed(42)
        A = torch.randn(12, 12)
        Q, T = trnsolver.schur(A)
        recon = Q @ T @ Q.T
        np.testing.assert_allclose(recon.numpy(), A.numpy(), atol=1e-4)

    def test_q_orthogonal(self):
        torch.manual_seed(43)
        A = torch.randn(12, 12)
        Q, _ = trnsolver.schur(A)
        np.testing.assert_allclose((Q.T @ Q).numpy(), np.eye(12), atol=1e-5)

    def test_quasi_upper_triangular(self):
        torch.manual_seed(44)
        A = torch.randn(12, 12)
        _, T = trnsolver.schur(A)
        # Everything two or more below the diagonal must be zero
        for i in range(12):
            for j in range(i - 1):
                assert abs(T[i, j].item()) < 1e-4, f"T[{i},{j}] = {T[i, j].item()}"

    def test_symmetric_gives_diagonal(self, spd_matrix):
        A = spd_matrix(10)
        Q, T = trnsolver.schur(A)
        # Off-diagonal of T should be small for symmetric A
        off_diag = T - torch.diag(torch.diagonal(T))
        np.testing.assert_allclose(off_diag.numpy(), np.zeros((10, 10)), atol=1e-3)

    def test_eigenvalues_match(self, spd_matrix):
        A = spd_matrix(10)
        _, T = trnsolver.schur(A)
        # Diagonal of T gives eigenvalues for symmetric A
        schur_eigs = torch.sort(torch.diagonal(T)).values
        ref_eigs = torch.sort(torch.linalg.eigvalsh(A)).values
        np.testing.assert_allclose(schur_eigs.numpy(), ref_eigs.numpy(), atol=1e-3)


class TestSolve:
    def test_identity(self):
        A = torch.eye(4)
        b = torch.tensor([1.0, 2.0, 3.0, 4.0])
        x = trnsolver.solve(A, b)
        np.testing.assert_allclose(x.numpy(), b.numpy(), atol=1e-6)

    def test_vs_torch(self, random_matrix):
        A = random_matrix(16, 16)
        b = torch.randn(16)
        x = trnsolver.solve(A, b)
        residual = A @ x - b
        np.testing.assert_allclose(residual.numpy(), np.zeros(16), atol=1e-4)


class TestSolveSPD:
    def test_vs_solve(self, spd_matrix):
        n = 16
        A = spd_matrix(n)
        b = torch.randn(n)
        x_spd = trnsolver.solve_spd(A, b)
        x_gen = trnsolver.solve(A, b)
        np.testing.assert_allclose(x_spd.numpy(), x_gen.numpy(), atol=1e-4)

    def test_residual(self, spd_matrix):
        n = 16
        A = spd_matrix(n)
        b = torch.randn(n)
        x = trnsolver.solve_spd(A, b)
        residual = A @ x - b
        np.testing.assert_allclose(residual.numpy(), np.zeros(n), atol=1e-4)

    def test_iterative_refinement_improves_residual(self):
        # SPD matrix with cond ≈ 1e6 — large enough that the plain FP32
        # Cholesky solve leaves a visible residual, but below 1/eps32 ≈ 1e7
        # so the mixed-precision FP64 residual still captures the error.
        torch.manual_seed(7)
        n = 32
        Q, _ = torch.linalg.qr(torch.randn(n, n))
        eigs = torch.logspace(0, 6, n)  # cond ≈ 1e6
        A = Q @ torch.diag(eigs) @ Q.T
        A = 0.5 * (A + A.T)
        b = torch.randn(n)

        x_plain = trnsolver.solve_spd(A, b)
        x_refined = trnsolver.solve_spd(A, b, iterative_refinement=True)

        # Compare forward error against an FP64 reference. The FP32 residual
        # ||A x - b|| is already at its rounding floor (~||A|| eps32 ||x||) for
        # both solutions, so comparing residuals is a coin flip that depends
        # on the CPU's BLAS kernels. Refinement improves x itself.
        x_ref = torch.linalg.solve(A.double(), b.double())
        err_plain = torch.linalg.norm(x_plain.double() - x_ref).item()
        err_refined = torch.linalg.norm(x_refined.double() - x_ref).item()
        assert err_refined <= 0.1 * err_plain

    def test_iterative_refinement_rhs_matrix(self, spd_matrix):
        # iterative_refinement=True works for B of shape (n, k)
        n, k = 16, 4
        A = spd_matrix(n)
        B = torch.randn(n, k)
        X = trnsolver.solve_spd(A, B, iterative_refinement=True)
        residual = A @ X - B
        assert torch.linalg.norm(residual, ord="fro").item() < 1e-3


class TestSolveSPDBatchedVectors:
    """B (*, n) with B.shape == A.shape[:-1] is a batch of vectors (#79).

    Reference: numpy.linalg.solve per matrix (LAPACK), independent of torch.
    """

    @staticmethod
    def _spd_batch(batch, n, seed=0, complex_=False):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((*batch, n, n))
        if complex_:
            X = X + 1j * rng.standard_normal((*batch, n, n))
        return X @ np.conj(np.swapaxes(X, -1, -2)) + n * np.eye(n)

    @pytest.mark.parametrize("batch", [(3,), (1,), (2, 3)])
    @pytest.mark.parametrize("complex_", [False, True])
    @pytest.mark.parametrize("ir", [False, True])
    def test_batch_of_vectors(self, batch, complex_, ir):
        n = 5
        A = self._spd_batch(batch, n, complex_=complex_)
        rng = np.random.default_rng(1)
        B = rng.standard_normal((*batch, n)) + (
            1j * rng.standard_normal((*batch, n)) if complex_ else 0
        )
        ref = np.linalg.solve(A, B[..., None])[..., 0]
        X = trnsolver.solve_spd(torch.from_numpy(A), torch.from_numpy(B), iterative_refinement=ir)
        assert X.shape == B.shape
        np.testing.assert_allclose(X.numpy(), ref, rtol=0, atol=1e-10)

    def test_issue_repro(self):
        A = torch.eye(4, dtype=torch.float64).expand(3, 4, 4)
        B = torch.ones(3, 4, dtype=torch.float64)
        np.testing.assert_allclose(trnsolver.solve_spd(A, B).numpy(), np.ones((3, 4)))


def _spd_np(batch, n, seed=0):
    X = np.random.default_rng(seed).standard_normal((*batch, n, n))
    return X @ np.swapaxes(X, -1, -2) + n * np.eye(n)


def _no_warnings(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn(*args, **kwargs)


class TestSolveSPDAmbiguousRHS:
    """A (*, n, n, n), B (*, n, n): deprecated in 0.22.1, switches in 0.23.0 (#79, #130).

    Today B reads as (*, n, k) right-hand sides broadcast over A's batch;
    0.23.0 reads it as a batch of vectors, as torch.linalg.solve does. The
    explicit forms B.unsqueeze(-1) / B.unsqueeze(-3) have B.dim() == A.dim(),
    so neither reading applies to them: same result in both releases.
    Reference: numpy.linalg.solve (LAPACK).
    """

    n = 4

    def _case(self, batch=()):
        n = self.n
        A = _spd_np((*batch, n), n, seed=2)
        B = np.random.default_rng(3).standard_normal((*batch, n, n))
        return A, B

    @pytest.mark.parametrize("ir", [False, True])
    def test_warns_and_keeps_matrix_reading(self, ir):
        A, B = self._case()
        with pytest.warns(FutureWarning, match=r"0\.23\.0") as rec:
            X = trnsolver.solve_spd(
                torch.from_numpy(A), torch.from_numpy(B), iterative_refinement=ir
            )
        assert X.shape == (self.n,) * 3
        np.testing.assert_allclose(X.numpy(), np.linalg.solve(A, B), rtol=0, atol=1e-10)
        msg = str(rec[0].message)
        assert "B.unsqueeze(-1)" in msg and "B.unsqueeze(-3)" in msg
        # Points at the caller, not at trnsolver internals.
        assert rec[0].filename == __file__

    def test_warns_with_leading_batch(self):
        # A (n, n, n, n), B (n, n, n): B broadcasts as (1, n, n, n), so today's
        # result is B.unsqueeze(0); the message says so.
        A, B = self._case((self.n,))
        with pytest.warns(FutureWarning, match=r"B\.unsqueeze\(0\) gives today's result"):
            X = trnsolver.solve_spd(torch.from_numpy(A), torch.from_numpy(B))
        Xe = _no_warnings(
            trnsolver.solve_spd, torch.from_numpy(A), torch.from_numpy(B).unsqueeze(0)
        )
        assert torch.equal(X, Xe)

    def test_explicit_batch_of_vectors(self):
        # B.unsqueeze(-1): the 0.23.0 reading of B, as torch.linalg.solve.
        A, B = self._case()
        At, Bt = torch.from_numpy(A), torch.from_numpy(B)
        X = _no_warnings(trnsolver.solve_spd, At, Bt.unsqueeze(-1))
        assert X.shape == (self.n, self.n, 1)
        ref = np.linalg.solve(A, B[..., None])[..., 0]
        np.testing.assert_allclose(X.squeeze(-1).numpy(), ref, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            X.squeeze(-1).numpy(), torch.linalg.solve(At, Bt).numpy(), rtol=0, atol=1e-10
        )

    def test_explicit_shared_matrix(self):
        # B.unsqueeze(-3): one shared (n, k) right-hand side; for a 3-D A that
        # is exactly today's result.
        A, B = self._case()
        At, Bt = torch.from_numpy(A), torch.from_numpy(B)
        X = _no_warnings(trnsolver.solve_spd, At, Bt.unsqueeze(-3))
        with pytest.warns(FutureWarning):
            X_today = trnsolver.solve_spd(At, Bt)
        assert torch.equal(X, X_today)
        np.testing.assert_allclose(X.numpy(), np.linalg.solve(A, B), rtol=0, atol=1e-10)

    @pytest.mark.parametrize(
        "a_shape,b_shape",
        [
            ((4, 4), (4,)),
            ((4, 4), (4, 4)),
            ((3, 4, 4), (3, 4)),  # batch of vectors, batch != n
            ((4, 4, 4), (4, 4, 4)),
            ((4, 4, 4), (4, 2)),
            ((2, 3, 4, 4), (2, 3, 4)),
        ],
    )
    def test_no_warning_outside_ambiguous_case(self, a_shape, b_shape):
        n = a_shape[-1]
        A = torch.from_numpy(_spd_np(a_shape[:-2], n))
        B = torch.ones(b_shape, dtype=torch.float64)
        _no_warnings(trnsolver.solve_spd, A, B)

    def test_inv_spd_batch_equals_n_does_not_warn(self):
        A = torch.from_numpy(_spd_np((self.n,), self.n))
        Ainv = _no_warnings(trnsolver.inv_spd, A)
        np.testing.assert_allclose(Ainv.numpy(), np.linalg.inv(A.numpy()), rtol=0, atol=1e-10)


class TestInvSqrtSPD:
    def test_identity(self):
        A = torch.eye(4)
        A_inv_sqrt = trnsolver.inv_sqrt_spd(A)
        np.testing.assert_allclose(A_inv_sqrt.numpy(), np.eye(4), atol=1e-6)

    def test_reconstruction(self, spd_matrix):
        """A^{-1/2} @ A @ A^{-1/2} should equal I."""
        n = 8
        A = spd_matrix(n)
        A_inv_sqrt = trnsolver.inv_sqrt_spd(A)
        product = A_inv_sqrt @ A @ A_inv_sqrt
        np.testing.assert_allclose(product.numpy(), np.eye(n), atol=1e-3)


class TestPinv:
    def test_full_rank_square(self, spd_matrix):
        n = 8
        A = spd_matrix(n)
        Ap = trnsolver.pinv(A)
        # Moore-Penrose: A @ A^+ @ A == A
        np.testing.assert_allclose((A @ Ap @ A).numpy(), A.numpy(), atol=1e-4)

    def test_rank_deficient(self):
        # Rank-1 matrix: outer product u v^T
        u = torch.tensor([1.0, 2.0, 3.0])
        v = torch.tensor([4.0, 5.0, 6.0])
        A = u.unsqueeze(1) @ v.unsqueeze(0)  # (3, 3), rank 1
        Ap = trnsolver.pinv(A)
        # Moore-Penrose condition: A @ A^+ @ A == A
        np.testing.assert_allclose((A @ Ap @ A).numpy(), A.numpy(), atol=1e-5)

    def test_rcond_threshold(self):
        # Two singular values: 1.0 and 1e-10; rcond=1e-6 zeroes the small one
        U, _, Vh = torch.linalg.svd(torch.randn(4, 4))
        s = torch.tensor([1.0, 0.5, 1e-10, 1e-12])
        A = U[:, :4] @ torch.diag(s) @ Vh[:4, :]
        Ap = trnsolver.pinv(A, rcond=1e-6)
        # Pseudoinverse singular values should be ~1/1.0, ~1/0.5, then 0, 0
        s_pinv = torch.linalg.svdvals(Ap)
        assert s_pinv[0].item() > 1.5  # ~2.0 = 1/0.5
        assert s_pinv[2].item() < 1e-3  # zeroed by rcond


class TestInvSqrtSpdNS:
    def test_identity(self):
        A = 2.0 * torch.eye(4, dtype=torch.float64)
        X, iters, res = trnsolver.inv_sqrt_spd_ns(A)
        # Expected A^{-1/2} = (1/sqrt(2)) I
        expected = (1.0 / np.sqrt(2.0)) * np.eye(4)
        np.testing.assert_allclose(X.numpy(), expected, atol=1e-6)
        assert iters >= 1
        assert res < 1e-6

    def test_matches_eig_reference(self, spd_matrix):
        n = 32
        A = spd_matrix(n, dtype=torch.float64)
        X_ref = trnsolver.inv_sqrt_spd(A)
        X_ns, iters, res = trnsolver.inv_sqrt_spd_ns(A, tol=1e-9)
        # Allow some slack; NS is quadratically convergent but not exact.
        np.testing.assert_allclose(X_ns.numpy(), X_ref.numpy(), rtol=1e-4, atol=1e-5)
        assert iters < 20

    def test_reconstruction(self, spd_matrix):
        n = 16
        A = spd_matrix(n, dtype=torch.float64)
        X, _, _ = trnsolver.inv_sqrt_spd_ns(A, tol=1e-9)
        product = X @ A @ X
        np.testing.assert_allclose(product.numpy(), np.eye(n), atol=1e-4)

    def test_returns_iteration_count(self):
        A = torch.eye(4, dtype=torch.float64) * 3.0
        _, iters, res = trnsolver.inv_sqrt_spd_ns(A)
        assert isinstance(iters, int) and iters >= 1
        assert isinstance(res, float)


class TestLUConvention:
    """lu returns A = P @ L @ U (torch.linalg.lu), as documented (trnsolver#71)."""

    def _cyclic(self):
        # Pivot order (2, 0, 1) is a 3-cycle, so P != P.T and the two
        # conventions P @ A = L @ U and A = P @ L @ U differ.
        A = torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 10.0], [1.0, 2.0, 3.0]], dtype=torch.float64)
        P, L, U = trnsolver.lu(A)
        assert not torch.equal(P, P.T)
        return A, P, L, U

    def test_a_equals_plu(self):
        A, P, L, U = self._cyclic()
        torch.testing.assert_close(P @ L @ U, A)
        torch.testing.assert_close(P.T @ A, L @ U)

    def test_docs_state_the_returned_convention(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        assert "A = P @ L @ U" in trnsolver.lu.__doc__.splitlines()[0]
        assert "`A = P L U`" in (root / "docs/api/factor.md").read_text()
        assert "PA = LU" not in (root / "docs/quickstart.md").read_text()
        assert "`PA = LU`" not in (root / "README.md").read_text()
