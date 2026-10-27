"""eigh routes to aten on the native neuron device below TRNSOLVER_EIGH_NKI_MIN_N (#42).

CPU-only: the device check is exercised by pointing the module's native
device type at "cpu" and by stub tensors that report device type "neuron".
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import trnsolver
import trnsolver.eigen as eigen


def _stub(n, device_type):
    """Object with the two attributes the cutover predicate reads."""
    return SimpleNamespace(device=SimpleNamespace(type=device_type), shape=(n, n))


def _fix_signs(V):
    """Flip each column so its largest-magnitude entry is positive."""
    idx = V.abs().argmax(dim=0)
    signs = torch.sign(V[idx, torch.arange(V.shape[1])])
    signs[signs == 0] = 1.0
    return V * signs


@pytest.fixture
def no_require(monkeypatch):
    monkeypatch.setattr(eigen, "_REQUIRE_NKI", False)
    monkeypatch.delenv("TRNSOLVER_EIGH_NKI_MIN_N", raising=False)


class TestCutoverPredicate:
    @pytest.mark.parametrize("n", [1, 16, 128, 511])
    def test_neuron_below_default_threshold(self, no_require, n):
        assert eigen._eigh_aten_cutover(_stub(n, "neuron"))

    @pytest.mark.parametrize("n", [512, 1024])
    def test_neuron_at_or_above_default_threshold(self, no_require, n):
        assert not eigen._eigh_aten_cutover(_stub(n, "neuron"))

    @pytest.mark.parametrize("device_type", ["cpu", "xla", "cuda", "meta"])
    def test_other_devices_never_cut_over(self, no_require, device_type):
        assert not eigen._eigh_aten_cutover(_stub(16, device_type))

    def test_env_override(self, no_require, monkeypatch):
        monkeypatch.setenv("TRNSOLVER_EIGH_NKI_MIN_N", "32")
        assert eigen._eigh_aten_cutover(_stub(16, "neuron"))
        assert not eigen._eigh_aten_cutover(_stub(32, "neuron"))
        assert not eigen._eigh_aten_cutover(_stub(64, "neuron"))

    def test_env_zero_disables(self, no_require, monkeypatch):
        monkeypatch.setenv("TRNSOLVER_EIGH_NKI_MIN_N", "0")
        assert not eigen._eigh_aten_cutover(_stub(2, "neuron"))

    def test_env_garbage_uses_default(self, no_require, monkeypatch):
        monkeypatch.setenv("TRNSOLVER_EIGH_NKI_MIN_N", "lots")
        assert eigen._eigh_nki_min_n() == eigen._EIGH_NKI_MIN_N_DEFAULT

    def test_require_nki_bypasses(self, monkeypatch):
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", True)
        monkeypatch.delenv("TRNSOLVER_EIGH_NKI_MIN_N", raising=False)
        assert not eigen._eigh_aten_cutover(_stub(16, "neuron"))


class TestCutoverRouting:
    """End to end through trnsolver.eigh with the NKI backend selected.

    _NATIVE_DEVICE_TYPE is pointed at "cpu" so CPU tensors take the
    native-device branch; _householder_qr_eigh is replaced by a spy.
    """

    @pytest.fixture
    def routed(self, monkeypatch, no_require):
        calls = []

        def fake_nki(A, tol=1e-10):
            calls.append(A.shape[0])
            return torch.linalg.eigh(A)

        monkeypatch.setattr(eigen, "_use_nki", lambda: True)
        monkeypatch.setattr(eigen, "_householder_qr_eigh", fake_nki)
        monkeypatch.setattr(eigen, "_NATIVE_DEVICE_TYPE", "cpu")
        return calls

    def test_small_n_uses_aten(self, routed, sym_matrix):
        trnsolver.eigh(sym_matrix(16))
        assert routed == []

    def test_env_threshold_sends_large_n_to_nki(self, routed, sym_matrix, monkeypatch):
        monkeypatch.setenv("TRNSOLVER_EIGH_NKI_MIN_N", "32")
        trnsolver.eigh(sym_matrix(16))
        trnsolver.eigh(sym_matrix(64))
        assert routed == [64]

    def test_require_nki_forces_kernel_path(self, routed, sym_matrix, monkeypatch):
        monkeypatch.setattr(eigen, "_REQUIRE_NKI", True)
        trnsolver.eigh(sym_matrix(16))
        assert routed == [16]

    def test_other_device_uses_kernel_path(self, routed, sym_matrix, monkeypatch):
        monkeypatch.setattr(eigen, "_NATIVE_DEVICE_TYPE", "neuron")
        trnsolver.eigh(sym_matrix(16))
        assert routed == [16]

    @pytest.mark.parametrize("n", [1, 2, 8, 16, 64, 128])
    def test_matches_torch(self, routed, sym_matrix, n):
        A = sym_matrix(n)
        w, V = trnsolver.eigh(A)
        w_ref, V_ref = torch.linalg.eigh(A)
        assert routed == []
        assert w.device == A.device and V.device == A.device
        np.testing.assert_allclose(w.numpy(), w_ref.numpy(), atol=1e-5)
        np.testing.assert_allclose(_fix_signs(V).numpy(), _fix_signs(V_ref).numpy(), atol=1e-4)

    def test_bf16_restored(self, routed, sym_matrix):
        A = sym_matrix(16).to(torch.bfloat16)
        w, V = trnsolver.eigh(A)
        assert routed == []
        assert w.dtype == torch.bfloat16 and V.dtype == torch.bfloat16


class TestAtenOnDevice:
    def test_result_on_input_device(self, sym_matrix):
        A = sym_matrix(8)
        w, V = eigen._aten_eigh_on_device(A)
        assert w.device == A.device and V.device == A.device
