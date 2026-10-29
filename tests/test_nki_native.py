"""
Native TorchNeuron dispatch on hardware (Trn2/Trn3, torch-neuronx 2.14+).

Skipped unless the dispatch path resolves to "native": these do not run on
the XLA lineage (Trn1) or off-hardware. `test_eigh_neuron.py` also runs the
end-to-end eigh tests on a TorchNeuron host; this file covers what is
specific to the native path — custom-op registration, no torch_xla, each op
against a reference in eager mode and inside torch.compile, device-resident
inputs, and the solvers whose scalar loops now run on host copies.

Run on the instance with:
    NKI_ENABLE_TRACE_CACHE=0 TRNSOLVER_REQUIRE_NKI=1 pytest tests/test_nki_native.py -v -m neuron
"""

import subprocess
import sys

import pytest
import torch

import trnsolver
from trnsolver.nki import HAS_NKI
from trnsolver.nki.dispatch import _get_dispatch_path, _use_simulator


def _native() -> bool:
    return HAS_NKI and not _use_simulator() and _get_dispatch_path() == "native"


pytestmark = [
    pytest.mark.neuron,
    pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path"),
]

DEV = "neuron"


def _sym(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    return 0.5 * (A + A.T)


def _householder_inputs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(n, n - 2, generator=g)
    V = V / V.norm(dim=0, keepdim=True)
    Q = torch.linalg.qr(torch.randn(n, n, generator=g)).Q
    return V, Q


def _apply_reflectors_ref(V, Q):
    for k in range(V.shape[1] - 1, -1, -1):
        v = V[:, k]
        Q = Q - 2.0 * torch.outer(v, v @ Q)
    return Q.T


def _cases():
    n, k = 64, 32
    g = torch.Generator().manual_seed(1)
    A = _sym(n)
    v = torch.randn(n, 1, generator=g)
    u = torch.randn(n, 1, generator=g)
    Vk = torch.randn(n, k, generator=g)
    Vr, Q = _householder_inputs(n)
    Ab, Vb = A.to(torch.bfloat16), Vk.to(torch.bfloat16)
    return {
        "matvec": ((A, v), A @ v, 1e-4),
        "rank2_update": ((A, u, v), A - u @ v.T - v @ u.T, 1e-4),
        "matmul": ((A, Vk), A @ Vk, 1e-3),
        "matmul_fp32out": ((Ab, Vb), Ab.float() @ Vb.float(), 1e-2),
        "apply_reflectors": ((Vr, Q), _apply_reflectors_ref(Vr, Q), 1e-3),
    }


OPS = sorted(_cases())


class TestNativeDispatch:
    def test_native_path_never_imports_torch_xla(self):
        # The subprocess stops after path detection and op registration and
        # never launches a kernel: a launch would call nrt_init on a
        # NeuronCore the parent pytest process may already hold.
        code = (
            "import sys, trnsolver; "
            "from trnsolver.nki.dispatch import _get_dispatch_path, _get_native_ops; "
            "assert _get_dispatch_path() == 'native'; "
            "ops = _get_native_ops(); "
            "import torch; assert all(hasattr(torch.ops.trnsolver, n) for n in ops); "
            "assert 'torch_xla' not in sys.modules, 'torch_xla imported'"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_launch_in_process_never_imports_torch_xla(self):
        # In-process complement: a real launch through the host wrapper.
        from trnsolver.eigen import _call_matmul

        _call_matmul(torch.randn(16, 16), torch.randn(16, 8))
        assert "torch_xla" not in sys.modules

    def test_ops_registered_in_torch_ops(self):
        from trnsolver.nki.dispatch import _get_native_ops

        _get_native_ops()
        for name in OPS:
            assert hasattr(torch.ops.trnsolver, name), name

    @pytest.mark.parametrize("name", OPS)
    def test_op_eager_matches_reference(self, name):
        args, expected, tol = _cases()[name]
        out = getattr(torch.ops.trnsolver, name)(*(a.to(DEV) for a in args))
        assert out.device.type == DEV
        torch.testing.assert_close(out.cpu().float(), expected.float(), rtol=tol, atol=tol)

    @pytest.mark.parametrize("name", OPS)
    def test_op_inside_torch_compile(self, name):
        from trnsolver.nki.dispatch import _get_native_ops

        op = _get_native_ops()[name]

        # dynamic=False: NkiTensor rejects symbolic dims, and Dynamo would
        # otherwise mark the shapes dynamic on a recompile.
        @torch.compile(backend="neuron", fullgraph=True, dynamic=False)
        def scaled(*args):
            return op(*args) * 2.0

        args, expected, tol = _cases()[name]
        out = scaled(*(a.to(DEV) for a in args)).cpu().float()
        torch.testing.assert_close(out, expected.float() * 2.0, rtol=tol, atol=2 * tol)


class TestNativeDeviceTridiag:
    """trnsolver::householder_tridiag (TRNSOLVER_EIGH_TRIDIAG=device, #42)."""

    def test_op_registered(self):
        from trnsolver.nki.dispatch import _get_native_ops

        _get_native_ops()
        assert hasattr(torch.ops.trnsolver, "householder_tridiag")

    @pytest.mark.parametrize("n", [8, 64, 128])
    def test_op_eager_tridiagonal_spectrum(self, n):
        from trnsolver.nki.tridiag import n_out_cols, tridiag_constants

        A = _sym(n)
        out = torch.ops.trnsolver.householder_tridiag(A.to(DEV), tridiag_constants(n).to(DEV))
        assert out.device.type == DEV and out.shape == (n, n_out_cols(n))
        out = out.cpu()
        d, e = out[:, 0], out[: n - 1, 1]
        T = torch.diag(d) + torch.diag(e, 1) + torch.diag(e, -1)
        torch.testing.assert_close(
            torch.linalg.eigvalsh(T), torch.linalg.eigvalsh(A), rtol=1e-3, atol=1e-3
        )

    @pytest.mark.parametrize("n", [8, 32, 128])
    def test_eigh_device_tridiag_no_torch_xla(self, n, monkeypatch):
        monkeypatch.setenv("TRNSOLVER_EIGH_TRIDIAG", "device")
        monkeypatch.setenv("TRNSOLVER_EIGH_NKI_MIN_N", "0")
        A = _sym(n)
        w, V = trnsolver.eigh(A.to(DEV))
        w, V = w.cpu(), V.cpu()
        torch.testing.assert_close(w, torch.linalg.eigh(A).eigenvalues, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(V @ torch.diag(w) @ V.T, A, rtol=1e-2, atol=1e-2)
        assert "torch_xla" not in sys.modules


class TestNativeDrivers:
    @pytest.mark.parametrize("n", [8, 32, 128])
    def test_eigh_kernel_path_device_input(self, n, monkeypatch):
        # Force the kernels for a device-resident input (bypass the aten
        # cutover) and check the result stays on the device.
        monkeypatch.setenv("TRNSOLVER_EIGH_NKI_MIN_N", "0")
        A = _sym(n)
        w, V = trnsolver.eigh(A.to(DEV))
        assert w.device.type == DEV and V.device.type == DEV
        w, V = w.cpu(), V.cpu()
        torch.testing.assert_close(w, torch.linalg.eigh(A).eigenvalues, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(V @ torch.diag(w) @ V.T, A, rtol=1e-2, atol=1e-2)

    def test_eigh_default_cutover_uses_aten(self):
        A = _sym(32)
        w, _ = trnsolver.eigh(A.to(DEV))
        assert w.device.type == DEV

    def test_eigh_randomized_bf16(self):
        A = _sym(96).to(torch.bfloat16)
        w, V = trnsolver.eigh_randomized(A.to(DEV), 4, oversampling=8, seed=0)
        assert w.shape == (4,) and V.shape == (96, 4)

    def test_schur_device_input_returns_on_device(self):
        A = torch.randn(24, 24, generator=torch.Generator().manual_seed(3))
        Q, T = trnsolver.schur(A.to(DEV))
        assert Q.device.type == DEV and T.device.type == DEV
        torch.testing.assert_close(Q.cpu() @ T.cpu() @ Q.cpu().T, A, rtol=1e-3, atol=1e-3)

    def test_gmres_device_rhs(self):
        n = 48
        g = torch.Generator().manual_seed(4)
        A = torch.randn(n, n, generator=g) + n * torch.eye(n)
        b = torch.randn(n, generator=g)
        x, _, res = trnsolver.gmres(A.to(DEV), b.to(DEV), tol=1e-5)
        assert x.device.type == DEV
        assert res < 1e-4
