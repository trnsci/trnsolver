"""
Dispatch-path selection (native TorchNeuron vs XLA vs CPU) and lazy
registration of the `trnsolver::` custom ops.

CPU-only: `torch_neuronx` and `torch_xla` are replaced with stand-ins via
`sys.modules` / `find_spec` monkeypatching, so these run in the default
`pytest -m "not neuron"` job on any host.
"""

import importlib.util
import subprocess
import sys
import types
import warnings

import pytest
import torch

import trnsolver
import trnsolver.eigen as eigen
import trnsolver.nki
from trnsolver.nki import dispatch

_REAL_FIND_SPEC = importlib.util.find_spec

EXPECTED_OPS = {
    "matvec",
    "rank2_update",
    "matmul",
    "matmul_fp32out",
    "apply_reflectors",
    "jacobi_sweeps",
    "jacobi_block",
    "subspace_filter",
}


@pytest.fixture(autouse=True)
def reset_dispatch_state(monkeypatch):
    """Each test sees an undetected path and an unregistered op table."""
    monkeypatch.setattr(dispatch, "_dispatch_path", None)
    monkeypatch.setattr(dispatch, "_native_ops", None)
    monkeypatch.setattr(dispatch, "_xla_deprecation_warned", False)
    monkeypatch.delenv("TRNSCI_NKI_DISPATCH", raising=False)
    monkeypatch.delenv("TRNSCI_XLA_DEPRECATION", raising=False)
    # Drop any `_native` left by an earlier test so registration re-runs.
    monkeypatch.delitem(sys.modules, "trnsolver.nki._native", raising=False)
    monkeypatch.delattr(trnsolver.nki, "_native", raising=False)
    yield


def _fake_native_torch_neuronx(device_count=1, recorded=None):
    """Stand-in for the TorchNeuron `torch_neuronx` (exports `nki_op`)."""
    mod = types.ModuleType("torch_neuronx")

    def nki_op(name, fn=None, mutates_args={}, fake_fn=None):  # noqa: B006
        def dec(f):
            if recorded is not None:
                recorded[name] = (f, mutates_args, fake_fn)
            return f

        return dec if fn is None else dec(fn)

    mod.nki_op = nki_op
    mod.device_count = lambda: device_count
    return mod


def _fake_xla_torch_neuronx():
    """Stand-in for the XLA-lineage `torch_neuronx` (no `nki_op`)."""
    mod = types.ModuleType("torch_neuronx")
    mod.trace = lambda *a, **k: None
    return mod


def _set_torch_xla_present(monkeypatch, present: bool):
    def find_spec(name, *args, **kwargs):
        if name == "torch_xla":
            return object() if present else None
        return _REAL_FIND_SPEC(name, *args, **kwargs)

    monkeypatch.setattr(dispatch.importlib.util, "find_spec", find_spec)


def _accept_neuron_device(monkeypatch):
    # Off-hardware, "neuron" is not a registered torch device type.
    real_device = torch.device

    def device(spec, *args, **kwargs):
        if spec == dispatch._NEURON_DEVICE:
            return real_device("cpu")
        return real_device(spec, *args, **kwargs)

    monkeypatch.setattr(dispatch.torch, "device", device)


class TestDispatchPathSelection:
    def test_native_when_nki_op_and_device(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "native"

    def test_native_probe_does_not_touch_torch_xla(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        # Any import of torch_xla would raise ImportError; any spec lookup
        # would fail the test.
        monkeypatch.setitem(sys.modules, "torch_xla", None)

        def find_spec(name, *args, **kwargs):
            assert name != "torch_xla", "native probe must not look up torch_xla"
            return _REAL_FIND_SPEC(name, *args, **kwargs)

        monkeypatch.setattr(dispatch.importlib.util, "find_spec", find_spec)
        assert dispatch._get_dispatch_path() == "native"

    def test_xla_when_torch_neuronx_has_no_nki_op(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_xla_torch_neuronx())
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "xla"

    def test_xla_when_native_has_no_device(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(device_count=0)
        )
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "xla"

    def test_not_native_when_neuron_device_type_unregistered(self, monkeypatch):
        # torch_neuronx imported but "neuron" is not a torch device type
        # (e.g. another backend owns PrivateUse1).
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_not_native_when_device_count_raises(self, monkeypatch):
        mod = _fake_native_torch_neuronx()

        def boom():
            raise RuntimeError("no runtime")

        mod.device_count = boom
        monkeypatch.setitem(sys.modules, "torch_neuronx", mod)
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_cpu_when_nothing_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", None)  # import raises
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_detection_is_cached(self, monkeypatch):
        calls = []

        def detect():
            calls.append(1)
            return "xla"

        monkeypatch.setattr(dispatch, "_detect_dispatch_path", detect)
        assert dispatch._get_dispatch_path() == "xla"
        assert dispatch._get_dispatch_path() == "xla"
        assert len(calls) == 1


class TestDispatchOverride:
    """TRNSCI_NKI_DISPATCH=auto|native|xla|cpu (suite-wide) pins the path."""

    def _both_lineages(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)

    def test_auto_prefers_native(self, monkeypatch):
        self._both_lineages(monkeypatch)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "auto")
        assert dispatch._get_dispatch_path() == "native"

    def test_force_xla_when_both_available(self, monkeypatch):
        self._both_lineages(monkeypatch)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "XLA")
        assert dispatch._get_dispatch_path() == "xla"

    def test_force_cpu_when_both_available(self, monkeypatch):
        self._both_lineages(monkeypatch)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "cpu")
        assert dispatch._get_dispatch_path() == "cpu"

    def test_force_native_without_backend_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_xla_torch_neuronx())
        _set_torch_xla_present(monkeypatch, True)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "native")
        with pytest.raises(RuntimeError, match="TRNSCI_NKI_DISPATCH=native"):
            dispatch._get_dispatch_path()

    def test_force_xla_without_torch_xla_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, False)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "xla")
        with pytest.raises(RuntimeError, match="TRNSCI_NKI_DISPATCH=xla"):
            dispatch._get_dispatch_path()

    def test_unknown_value_raises(self, monkeypatch):
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "cuda")
        with pytest.raises(ValueError, match="auto, native, xla, or cpu"):
            dispatch._get_dispatch_path()


class TestXlaDeprecationHook:
    def _xla(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_xla_torch_neuronx())
        _set_torch_xla_present(monkeypatch, True)

    def test_off_by_default(self, monkeypatch):
        self._xla(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert dispatch._get_dispatch_path() == "xla"
            assert dispatch._get_dispatch_path() == "xla"

    def test_fires_once_when_enabled(self, monkeypatch):
        self._xla(monkeypatch)
        monkeypatch.setenv("TRNSCI_XLA_DEPRECATION", "1")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                assert dispatch._get_dispatch_path() == "xla"
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1
        assert str(dep[0].message) == (
            "the PyTorch/XLA path is deprecated on Trn2/Trn3; use TorchNeuron (native). "
            "Trn1/Inf2 remain supported in maintenance mode."
        )

    def test_not_on_native_path(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        monkeypatch.setenv("TRNSCI_XLA_DEPRECATION", "1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert dispatch._get_dispatch_path() == "native"


class TestLaunchRouting:
    def test_native_launch_uses_custom_op_not_xla(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
        monkeypatch.setitem(sys.modules, "torch_xla", None)
        seen = {}

        def op(a, b):
            seen["args"] = (a, b)
            return a + b

        monkeypatch.setattr(dispatch, "_native_ops", {"fake": op})

        def no_kernel(*t):
            raise AssertionError("raw kernel must not be called on the native path")

        out = dispatch._launch(no_kernel, "fake", torch.ones(2, 1), torch.ones(2, 1))
        assert torch.equal(out, torch.full((2, 1), 2.0))
        assert out.device.type == "cpu"
        assert "args" in seen

    def test_native_work_device_and_no_sync(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
        monkeypatch.setitem(sys.modules, "torch_xla", None)  # any import raises
        assert dispatch._work_device() == torch.device("cpu")
        dispatch._step_sync()  # no torch_xla import on the native path

    def test_xla_launch_calls_kernel_and_syncs(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        syncs = []
        fake_xla = types.ModuleType("torch_xla")
        fake_xla.device = lambda: torch.device("cpu")
        fake_xla.sync = lambda: syncs.append(1)
        monkeypatch.setitem(sys.modules, "torch_xla", fake_xla)
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_xla_torch_neuronx())

        def no_ops():
            raise AssertionError("native ops must not load on the XLA path")

        monkeypatch.setattr(dispatch, "_get_native_ops", no_ops)
        out = dispatch._launch(lambda x: x * 3, "unused", torch.ones(4, 1))
        assert torch.equal(out, torch.full((4, 1), 3.0))
        assert syncs == [1]

        out = dispatch._launch(lambda x: x * 2, "unused", torch.ones(4, 1), sync=False)
        assert torch.equal(out, torch.full((4, 1), 2.0))
        assert syncs == [1]

        dispatch._step_sync()
        assert syncs == [1, 1]

    def test_cpu_launch_keeps_pre_native_behaviour(self, monkeypatch):
        # No device path: the launch imports torch_xla, which raises when it
        # is absent — the same failure as before the native path existed.
        monkeypatch.setattr(dispatch, "_dispatch_path", "cpu")
        monkeypatch.setitem(sys.modules, "torch_xla", None)
        with pytest.raises(ImportError):
            dispatch._launch(lambda x: x, "unused", torch.ones(1, 1))
        with pytest.raises(ImportError):
            dispatch._work_device()


def _reference_ops(calls):
    """Torch implementations of the five ops, recording each call."""

    def rec(name, fn):
        def op(*args):
            calls.append(name)
            return fn(*args)

        return op

    def apply_reflectors(V_refs, Q):
        Q = Q.clone()
        for k in range(V_refs.shape[1] - 1, -1, -1):
            v = V_refs[:, k]
            Q = Q - 2.0 * torch.outer(v, v @ Q)
        return Q.T.contiguous()

    return {
        "matvec": rec("matvec", lambda A, v: A @ v),
        "rank2_update": rec("rank2_update", lambda A, u, v: A - u @ v.T - v @ u.T),
        "matmul": rec("matmul", lambda A, V: A @ V),
        "matmul_fp32out": rec("matmul_fp32out", lambda A, V: A.float() @ V.float()),
        "apply_reflectors": rec("apply_reflectors", apply_reflectors),
    }


class TestNativeHostDrivers:
    """The eigh / eigh_randomized drivers launch only trnsolver:: ops natively.

    The `neuron` device is aliased to CPU and the ops are torch references,
    so this checks the routing and the host-side scalar plumbing, not the
    kernels.
    """

    @pytest.fixture
    def native(self, monkeypatch):
        calls = []
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
        monkeypatch.setattr(dispatch, "_native_ops", _reference_ops(calls))
        monkeypatch.setitem(sys.modules, "torch_xla", None)  # any import raises
        for name in (
            "matvec_kernel",
            "rank2_update_kernel",
            "matmul_kernel",
            "matmul_kernel_fp32out",
            "apply_reflectors_kernel",
        ):
            monkeypatch.setattr(dispatch, name, object(), raising=False)
        monkeypatch.setattr(eigen, "HAS_NKI", True)
        monkeypatch.setattr(eigen, "_use_nki", lambda: True)
        monkeypatch.setattr(eigen, "_use_simulator", lambda: False)
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", True)
        return calls

    @pytest.mark.parametrize("n", [3, 8, 17, 32])
    def test_eigh_kernel_path(self, native, sym_matrix, n):
        A = sym_matrix(n)
        w, V = trnsolver.eigh(A)
        w_ref = torch.linalg.eigh(A).eigenvalues
        torch.testing.assert_close(w, w_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(V @ torch.diag(w) @ V.T, A, rtol=1e-3, atol=1e-3)
        assert native.count("matvec") == native.count("rank2_update") == n - 2
        assert native.count("apply_reflectors") == 1
        assert native.count("matmul") == 3
        assert "torch_xla" not in sys.modules or sys.modules["torch_xla"] is None

    def test_tridiag_matches_host_householder(self, native, sym_matrix):
        n = 16
        A = sym_matrix(n)
        diag, subdiag, V_refs = eigen._householder_tridiag(A)
        T = torch.diag(diag) + torch.diag(subdiag, 1) + torch.diag(subdiag, -1)
        torch.testing.assert_close(
            torch.linalg.eigvalsh(T), torch.linalg.eigvalsh(A), rtol=1e-4, atol=1e-4
        )
        assert V_refs.shape == (n, n - 1)

    def test_eigh_randomized_bf16_routes_fp32out(self, native, sym_matrix):
        A = sym_matrix(64).to(torch.bfloat16)
        w, V = trnsolver.eigh_randomized(A, 4, oversampling=6, n_iter=1, seed=0)
        assert w.shape == (4,) and V.shape == (64, 4)
        assert native.count("matmul_fp32out") == 3


class TestLazyRegistration:
    def test_import_does_not_register_ops(self):
        # Fresh interpreter: importing trnsolver (and the dispatch module)
        # must not load `_native` or import torch_neuronx / torch_xla.
        code = (
            "import sys, trnsolver, trnsolver.nki.dispatch, trnsolver.eigen; "
            "bad = [m for m in ('trnsolver.nki._native', 'torch_neuronx', 'torch_xla') "
            "if m in sys.modules]; "
            "assert not bad, bad"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_ops_registered_on_first_native_use(self, monkeypatch):
        recorded = {}
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(recorded=recorded)
        )
        assert "trnsolver.nki._native" not in sys.modules
        assert recorded == {}

        ops = dispatch._get_native_ops()

        assert set(ops) == EXPECTED_OPS
        assert set(recorded) == {f"trnsolver::{name}" for name in EXPECTED_OPS}
        assert all(mutates == {} for _, mutates, _ in recorded.values())
        assert all(fake is not None for _, _, fake in recorded.values())

        # Second call reuses the table; no re-registration.
        recorded.clear()
        assert dispatch._get_native_ops() is ops
        assert recorded == {}

    def test_op_signatures_build_a_schema(self, monkeypatch):
        # nki_op derives the schema with torch.library.infer_schema; check
        # every wrapper is fully annotated (Tensor args, Tensor return).
        recorded = {}
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(recorded=recorded)
        )
        dispatch._get_native_ops()
        assert len(recorded) == len(EXPECTED_OPS)
        for name, (fn, mutates, _) in recorded.items():
            schema = torch.library.infer_schema(fn, mutates_args=mutates)
            assert schema.endswith("-> Tensor"), (name, schema)
            assert "Tensor" in schema.split("->")[0], (name, schema)

    def test_fake_fns_give_kernel_output_shapes(self, monkeypatch):
        recorded = {}
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(recorded=recorded)
        )
        dispatch._get_native_ops()

        def meta(*shape, dtype=torch.float32):
            return torch.empty(shape, dtype=dtype, device="meta")

        def fake(name):
            return recorded[f"trnsolver::{name}"][2]

        n, k = 32, 12
        out = fake("matvec")(meta(n, n), meta(n, 1))
        assert out.shape == (n, 1) and out.dtype == torch.float32
        assert fake("rank2_update")(meta(n, n), meta(n, 1), meta(n, 1)).shape == (n, n)
        assert fake("matmul")(meta(n, n), meta(n, k)).shape == (n, k)
        bf16 = fake("matmul_fp32out")(
            meta(n, n, dtype=torch.bfloat16), meta(n, k, dtype=torch.bfloat16)
        )
        assert bf16.shape == (n, k) and bf16.dtype == torch.float32
        out = fake("apply_reflectors")(meta(n, n - 2), meta(n, n))
        assert out.shape == (n, n) and out.dtype == torch.float32
        out = fake("subspace_filter")(meta(n, n), meta(n, k), meta(128, 3 * 8 + k), meta(8, 1))
        assert out.shape == (n + 6 * k + 1, k) and out.dtype == torch.float32
