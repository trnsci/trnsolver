"""Benchmark fixtures."""

import pytest
import torch


@pytest.fixture(params=[64, 128, 256, 512])
def matrix_size(request):
    return request.param


@pytest.fixture
def symmetric_matrix(matrix_size):
    torch.manual_seed(42)
    A = torch.randn(matrix_size, matrix_size)
    return 0.5 * (A + A.T)


@pytest.fixture
def spd_matrix(matrix_size):
    torch.manual_seed(42)
    A = torch.randn(matrix_size, matrix_size)
    return A @ A.T + matrix_size * torch.eye(matrix_size)


@pytest.fixture
def random_matrix(matrix_size):
    torch.manual_seed(42)
    return torch.randn(matrix_size, matrix_size)


@pytest.fixture
def random_vector(matrix_size):
    torch.manual_seed(42)
    return torch.randn(matrix_size)
