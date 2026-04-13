"""CUDA / cuSOLVER baselines for trnsolver.

Matches the shape of benchmarks/bench_solver.py but runs each op on a CUDA
device so torch.linalg dispatches to cuSOLVER / cuBLAS. These are the
vintage-matched GPU baselines for the Trainium benchmarks — run on an
A10G-class instance (g5.xlarge) to compare against trn1, or on H100
(p5.48xlarge) to compare against trn2. See docs/benchmarks.md for the
vintage-matching rationale.

Skipped automatically when no CUDA device is available. Run manually with:

    pytest benchmarks/bench_cuda.py -v -m cuda --benchmark-only \\
        --benchmark-json=/tmp/cuda_results.json

or via the SSM wrapper:

    AWS_PROFILE=aws ./scripts/run_cuda_tests.sh g5

Each benchmarked callable includes an explicit ``torch.cuda.synchronize()``
so the timer captures kernel execution, not async launch.
"""

from __future__ import annotations

import pytest
import torch

import trnsolver

cuda_available = torch.cuda.is_available()
pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not cuda_available, reason="CUDA not available"),
]


def _sync(fn):
    """Wrap a call so pytest-benchmark times the synchronous kernel cost."""
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        return out
    return wrapper


# ---------------------------------------------------------------------------
# Symmetric eigendecomposition
# ---------------------------------------------------------------------------


class TestEighCUDA:
    def test_eigh_cuda(self, benchmark, symmetric_matrix):
        A = symmetric_matrix.cuda()
        torch.cuda.synchronize()
        benchmark(_sync(torch.linalg.eigh), A)


# ---------------------------------------------------------------------------
# Factorizations
# ---------------------------------------------------------------------------


class TestCholeskyCUDA:
    def test_cholesky_cuda(self, benchmark, spd_matrix):
        A = spd_matrix.cuda()
        torch.cuda.synchronize()
        benchmark(_sync(torch.linalg.cholesky), A)


class TestQRCUDA:
    def test_qr_cuda(self, benchmark, random_matrix):
        A = random_matrix.cuda()
        torch.cuda.synchronize()
        benchmark(_sync(torch.linalg.qr), A)


class TestSolveSpdCUDA:
    def test_solve_spd_cuda(self, benchmark, spd_matrix, random_vector):
        A = spd_matrix.cuda()
        b = random_vector.cuda()
        torch.cuda.synchronize()
        benchmark(_sync(torch.linalg.solve), A, b)


class TestInvSqrtSpdCUDA:
    def test_inv_sqrt_spd_cuda(self, benchmark, spd_matrix):
        A = spd_matrix.cuda()
        torch.cuda.synchronize()
        benchmark(_sync(trnsolver.inv_sqrt_spd), A)

    def test_inv_sqrt_spd_ns_cuda(self, benchmark, spd_matrix):
        A = spd_matrix.cuda()
        torch.cuda.synchronize()
        benchmark(_sync(trnsolver.inv_sqrt_spd_ns), A)
