"""Test configuration and fixtures."""

import os

import pytest
import torch

# Test runs come from source checkouts / editable installs, which the NKI trace
# cache cannot version (it keys on wheel .dist-info). A stale hit would run an
# old kernel, so disable it before anything imports nki. Set it explicitly to
# override.
os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "0")


def pytest_addoption(parser):
    # Reference-conformance suite (tests/test_conformance.py): run the same
    # cases against a non-default backend and/or with device inputs on a
    # Trainium box, e.g. ``pytest tests/test_conformance.py --backend nki
    # --conformance-device neuron``.
    parser.addoption(
        "--backend",
        default="pytorch",
        choices=("pytorch", "nki", "auto"),
        help="trnsolver backend for tests/test_conformance.py (default: pytorch)",
    )
    parser.addoption(
        "--conformance-device",
        default="cpu",
        help="device for tests/test_conformance.py inputs (default: cpu; e.g. neuron)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "neuron: requires Neuron hardware")


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(42)


@pytest.fixture
def sym_matrix(rng):
    """Random symmetric matrix."""

    def _make(n, dtype=torch.float32):
        A = torch.randn(n, n, generator=rng, dtype=dtype)
        return 0.5 * (A + A.T)

    return _make


@pytest.fixture
def spd_matrix(rng):
    """Symmetric positive definite matrix."""

    def _make(n, dtype=torch.float32):
        A = torch.randn(n, n, generator=rng, dtype=dtype)
        return A @ A.T + n * torch.eye(n, dtype=dtype)

    return _make


@pytest.fixture
def random_matrix(rng):
    def _make(m, n, dtype=torch.float32):
        return torch.randn(m, n, generator=rng, dtype=dtype)

    return _make
