"""``TRNSOLVER_JACOBI_ONEZERO`` reaches the Jacobi kernels as an argument.

The Jacobi kernels used to read the ``_ONEZERO`` module global from their
body. The NKI trace cache, which ``import torch_neuronx`` turns on and which
persists on disk, keys on a kernel's arguments but not on the globals or
closure values its body reads (aws-neuron/aws-neuron-sdk#1454), so a process
running with ``TRNSOLVER_JACOBI_ONEZERO=0`` could get a trace made with the
flag on, or the reverse. The flag is now the kernels' ``onezero`` argument
(a static ``nki_op`` argument on the native path).

The CPU test models the trace cache: a launch is keyed on what the real
cache keys on (op, tensor shapes and dtypes, static scalars), and a hit
returns the stored trace's result. Each kernel stand-in returns its
``onezero`` value, so a stale trace would show up as the wrong value.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from trnsolver.nki import dispatch
from trnsolver.nki import jacobi as JK
from trnsolver.nki.dispatch import HAS_NKI


class _TraceCacheModel:
    """Keys like the NKI trace cache: arguments yes, globals and closures no."""

    def __init__(self):
        self.entries: dict = {}

    def op(self, name, trace):
        def run(*args):
            key = (name,) + tuple(
                (tuple(a.shape), a.dtype) if isinstance(a, torch.Tensor) else a for a in args
            )
            if key not in self.entries:
                self.entries[key] = trace(*args)
            return self.entries[key]

        return run


def _onezero_trace(Z, V, consts, sched, onezero):
    # Stand-in for the traced kernel: its output records the flag it saw.
    return torch.full((2,) + tuple(Z.shape), float(onezero))


@pytest.fixture
def cache(monkeypatch):
    model = _TraceCacheModel()
    ops = {
        "jacobi_sweeps": model.op("jacobi_sweeps", _onezero_trace),
        "jacobi_block": model.op("jacobi_block", _onezero_trace),
    }
    monkeypatch.setattr(dispatch, "_get_dispatch_path", lambda: "native")
    monkeypatch.setattr(dispatch, "_get_native_ops", lambda: ops)
    monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
    monkeypatch.setattr(JK, "_use_simulator", lambda: False)
    monkeypatch.setattr(JK, "_DEVICE_CONSTS", {})
    # Without nki the kernels are not defined; the native path never calls them.
    for name in ("jacobi_sweeps_kernel", "jacobi_block_kernel"):
        monkeypatch.setattr(JK, name, getattr(JK, name, None), raising=False)
    return model


@pytest.mark.parametrize("block", [False, True])
def test_each_onezero_setting_gets_its_own_trace(cache, monkeypatch, block):
    n, chains = 64, 1
    ne, b, P = JK.block_layout(n)
    Z = torch.zeros((1, P, ne))
    run = (
        (lambda: JK._run_block_kernel(Z, Z, n, chains))
        if block
        else (lambda: JK._run_kernel(Z, Z, n, chains))
    )
    got = {}
    for flag in (True, False, True):
        monkeypatch.setattr(JK, "_ONEZERO", flag)
        got[flag] = run()
        assert torch.all(got[flag] == float(flag)), f"onezero={flag} ran a stale trace"
    assert len(cache.entries) == 2  # one trace per setting, reused on the repeat


def test_cache_model_reproduces_1454(cache):
    """Sanity: the model does serve a stale trace for a global-specialised kernel."""
    setting = {"onezero": 1}
    run = cache.op("global_kernel", lambda Z: torch.full(Z.shape, float(setting["onezero"])))
    Z = torch.zeros(2, 2)
    assert torch.all(run(Z) == 1.0)
    setting["onezero"] = 0
    assert torch.all(run(Z) == 1.0)  # stale: the global is not in the key


@pytest.mark.nki_simulator
def test_simulator_onezero_settings_match_emulation(monkeypatch, tmp_path):
    """Both settings, one after the other with the trace cache on, give the emulated sweep."""
    if not (
        HAS_NKI and os.environ.get("TRNSOLVER_USE_SIMULATOR", "").lower() in ("1", "true", "yes")
    ):
        pytest.skip("TRNSOLVER_USE_SIMULATOR=1 and nki required")
    import nki
    from test_jacobi import _emulate_launch, _sym_batch

    monkeypatch.setenv("NKI_ENABLE_TRACE_CACHE", "1")
    monkeypatch.setenv("NKI_TRACE_CACHE_URL", str(tmp_path))
    n, G, chains = 8, 2, 2
    ne, b, P = JK.block_layout(n)
    A = _sym_batch(G * b, n)
    sigma = A.abs().sum(-1).amax(-1)
    Z = JK.pack_compact(A + sigma[:, None, None] * torch.eye(n), n)
    V = JK.pack_compact(torch.eye(n).expand(G * b, n, n), n)
    ref = _emulate_launch(Z, V, n, chains)
    for onezero in (1, 0):
        got = nki.simulate(JK.jacobi_sweeps_kernel)(
            Z.numpy(),
            V.numpy(),
            JK.jacobi_constants(n).numpy(),
            np.zeros((1, ne, chains), dtype=np.float32),
            onezero,
        )
        torch.testing.assert_close(torch.from_numpy(got), ref, rtol=1e-4, atol=1e-4)
