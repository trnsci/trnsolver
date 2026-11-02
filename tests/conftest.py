"""Test configuration and fixtures."""

import atexit
import os
import shutil
import tempfile

import pytest
import torch

# NKI trace cache (aws-neuron/aws-neuron-sdk#1454, trnsci/trnsci#35).
# `import torch_neuronx` turns the cache on (NKI_ENABLE_TRACE_CACHE=1), so
# that is how users run these kernels, and the neuron tests run the same way.
# The cache lives on disk and is versioned by wheel .dist-info, which an
# editable install or source checkout does not change, so a shared cache could
# serve a trace of an older kernel. Each session therefore gets a fresh,
# empty cache directory (NKI_TRACE_CACHE_URL), removed at exit.
# tests/test_trace_cache_safety.py checks that no kernel depends on a value
# the cache key misses (closure or global). Overrides: NKI_ENABLE_TRACE_CACHE=0
# runs with the cache off; an explicit NKI_TRACE_CACHE_URL is kept as is.
os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "1")
if os.environ["NKI_ENABLE_TRACE_CACHE"] != "0" and "NKI_TRACE_CACHE_URL" not in os.environ:
    _trace_cache_dir = tempfile.mkdtemp(prefix="trnsolver-nki-trace-cache-")
    os.environ["NKI_TRACE_CACHE_URL"] = _trace_cache_dir
    atexit.register(shutil.rmtree, _trace_cache_dir, ignore_errors=True)


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
