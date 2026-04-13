"""Benchmark suite for trnsolver.

Each operation runs against up to three baselines:
  * ``*_nki``               — trnsolver with ``set_backend("nki")`` (Trainium)
  * ``*_trnsolver_pytorch`` — trnsolver with ``set_backend("pytorch")``
  * ``*_torch``             — vanilla ``torch.linalg`` / ``scipy`` reference

NKI benchmarks are marked ``@pytest.mark.neuron`` and only run on Trainium
hardware. The PyTorch baselines run anywhere and provide the reference
numbers that ``pytest benchmarks/ -m "not neuron"`` will emit in CI.

Run all benchmarks (saving JSON):
    pytest benchmarks/ -v --benchmark-only --benchmark-json=results.json

Run only PyTorch baselines (no hardware needed):
    pytest benchmarks/ -v -m "not neuron" --benchmark-only
"""

from __future__ import annotations

import pytest
import torch

import trnsolver


def _set(backend: str):
    trnsolver.set_backend(backend)


def _warm(fn, *args, **kwargs):
    fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Symmetric eigendecomposition — the primary NKI target
# ---------------------------------------------------------------------------


class TestEigh:
    @pytest.mark.neuron
    def test_eigh_nki(self, benchmark, symmetric_matrix):
        _set("nki")
        try:
            _warm(trnsolver.eigh, symmetric_matrix)
            benchmark(trnsolver.eigh, symmetric_matrix)
        finally:
            _set("auto")

    def test_eigh_trnsolver_pytorch(self, benchmark, symmetric_matrix):
        _set("pytorch")
        try:
            benchmark(trnsolver.eigh, symmetric_matrix)
        finally:
            _set("auto")

    def test_eigh_torch(self, benchmark, symmetric_matrix):
        benchmark(torch.linalg.eigh, symmetric_matrix)


class TestEighGeneralized:
    def test_eigh_gen_trnsolver_pytorch(self, benchmark, spd_matrix, symmetric_matrix):
        _set("pytorch")
        try:
            benchmark(trnsolver.eigh_generalized, symmetric_matrix, spd_matrix)
        finally:
            _set("auto")


# ---------------------------------------------------------------------------
# Factorizations
# ---------------------------------------------------------------------------


class TestCholesky:
    def test_cholesky_trnsolver(self, benchmark, spd_matrix):
        benchmark(trnsolver.cholesky, spd_matrix)

    def test_cholesky_torch(self, benchmark, spd_matrix):
        benchmark(torch.linalg.cholesky, spd_matrix)


class TestLU:
    def test_lu_trnsolver(self, benchmark, random_matrix):
        benchmark(trnsolver.lu, random_matrix)


class TestQR:
    def test_qr_trnsolver(self, benchmark, random_matrix):
        benchmark(trnsolver.qr, random_matrix)


class TestSolveSpd:
    def test_solve_spd_trnsolver(self, benchmark, spd_matrix, random_vector):
        benchmark(trnsolver.solve_spd, spd_matrix, random_vector)

    def test_solve_torch(self, benchmark, spd_matrix, random_vector):
        benchmark(torch.linalg.solve, spd_matrix, random_vector)


class TestInvSqrtSpd:
    def test_inv_sqrt_spd_trnsolver(self, benchmark, spd_matrix):
        benchmark(trnsolver.inv_sqrt_spd, spd_matrix)

    def test_inv_sqrt_spd_ns_trnsolver(self, benchmark, spd_matrix):
        # Newton-Schulz (all-GEMM) variant — CPU baseline for the NKI target.
        benchmark(trnsolver.inv_sqrt_spd_ns, spd_matrix)


# ---------------------------------------------------------------------------
# Iterative solvers
# ---------------------------------------------------------------------------


class TestCG:
    def test_cg_no_precond(self, benchmark, spd_matrix, random_vector):
        benchmark(trnsolver.cg, spd_matrix, random_vector)

    def test_cg_jacobi_precond(self, benchmark, spd_matrix, random_vector):
        M = trnsolver.jacobi_preconditioner(spd_matrix)
        benchmark(trnsolver.cg, spd_matrix, random_vector, M=M)


class TestGMRES:
    def test_gmres(self, benchmark, random_matrix, random_vector):
        # Shift to guarantee solvability without tight conditioning assumptions.
        n = random_matrix.shape[0]
        A = random_matrix + n * torch.eye(n)
        benchmark(trnsolver.gmres, A, random_vector)
