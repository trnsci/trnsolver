"""NKI kernel correctness via the CPU simulator (Neuron SDK 2.29+).

Runs the Jacobi rotation kernels on CPU using `nki.simulate_kernel`, so we
can iterate on kernel design without a trn1 instance. Tests here take
`numpy.ndarray` inputs (the simulator contract), in contrast to the
`@pytest.mark.neuron` tests which take torch tensors through the XLA path.

Run:
    pytest tests/test_jacobi_simulator.py -v -m simulator

On hosts without the Neuron SDK installed (e.g. macOS dev boxes), the
entire module is skipped at collection time. The CI `test-simulator` job
installs `neuronx-cc` from the Neuron pip index and runs these tests
on `ubuntu-latest`.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module if neuronxcc isn't installed. The CI simulator job
# pulls it from https://pip.repos.neuron.amazonaws.com.
nki = pytest.importorskip(
    "neuronxcc.nki",
    reason="neuronxcc not installed; install via pip install neuronx-cc "
    "--extra-index-url https://pip.repos.neuron.amazonaws.com",
)

from trnsolver.nki.dispatch import rotate_pairs_kernel  # noqa: E402

pytestmark = pytest.mark.simulator


def _reference_rotate(even: np.ndarray, odd: np.ndarray, c: np.ndarray, s: np.ndarray):
    """Analytic reference for rotate_pairs_kernel.

    Applies, for each row i:
        new_even[i, :] = c[i] * even[i, :] - s[i] * odd[i, :]
        new_odd[i, :]  = s[i] * even[i, :] + c[i] * odd[i, :]
    """
    new_even = c * even - s * odd
    new_odd = s * even + c * odd
    return new_even, new_odd


@pytest.mark.parametrize("half", [2, 4, 8, 16])
def test_rotate_pairs_identity(half):
    """c=1, s=0 rotation is a no-op."""
    n = 2 * half
    rng = np.random.default_rng(42)
    even = rng.standard_normal((half, n)).astype(np.float32)
    odd = rng.standard_normal((half, n)).astype(np.float32)
    c = np.ones((half, 1), dtype=np.float32)
    s = np.zeros((half, 1), dtype=np.float32)

    new_even, new_odd = nki.simulate_kernel(rotate_pairs_kernel, even, odd, c, s)

    np.testing.assert_allclose(new_even, even, atol=1e-6)
    np.testing.assert_allclose(new_odd, odd, atol=1e-6)


@pytest.mark.parametrize("half", [2, 4, 8])
def test_rotate_pairs_vs_reference(half):
    """Simulator result matches the analytic formula."""
    n = 2 * half
    rng = np.random.default_rng(0)
    even = rng.standard_normal((half, n)).astype(np.float32)
    odd = rng.standard_normal((half, n)).astype(np.float32)
    theta = rng.uniform(-np.pi, np.pi, size=(half, 1)).astype(np.float32)
    c = np.cos(theta)
    s = np.sin(theta)

    ref_even, ref_odd = _reference_rotate(even, odd, c, s)
    new_even, new_odd = nki.simulate_kernel(rotate_pairs_kernel, even, odd, c, s)

    np.testing.assert_allclose(new_even, ref_even, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(new_odd, ref_odd, atol=1e-5, rtol=1e-5)


def test_rotate_pairs_orthogonality():
    """Orthogonal rotation preserves pair-wise squared norm.

    For each pair i: new_even[i]² + new_odd[i]² == even[i]² + odd[i]²
    (element-wise).
    """
    half, n = 4, 16
    rng = np.random.default_rng(7)
    even = rng.standard_normal((half, n)).astype(np.float32)
    odd = rng.standard_normal((half, n)).astype(np.float32)
    theta = rng.uniform(-np.pi, np.pi, size=(half, 1)).astype(np.float32)
    c = np.cos(theta)
    s = np.sin(theta)

    new_even, new_odd = nki.simulate_kernel(rotate_pairs_kernel, even, odd, c, s)

    lhs = new_even**2 + new_odd**2
    rhs = even**2 + odd**2
    np.testing.assert_allclose(lhs, rhs, atol=1e-4, rtol=1e-4)
