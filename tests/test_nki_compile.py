"""CPU compile check: every NKI kernel lowers to MLIR that passes verification.

`nki.simulate` does not run the MLIR verifier, so a kernel can pass every
simulator test and still fail to compile for hardware. #40 was such a case:
`apply_reflectors_kernel` sliced a single SBUF row `V_T[k:k+1, :]` at an
arbitrary partition offset k, which NKI 0.7.0 rejects ("'nisa.tensor_copy' op
Expected 'src' partition start in {0, 32, 64, 96} for a 1-partition access").
This module runs the NKI frontend (trace -> MLIR -> verify) on CPU for every
kernel and target. It needs `nki` but no Neuron device and no `neuronx-cc`;
it does not run the backend compiler, so it cannot catch errors raised after
MLIR generation.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.nki_simulator

from trnsolver.nki.dispatch import HAS_NKI  # noqa: E402

try:
    from nki.compiler.frontend import FrontendKind, source_to_mlir
except ImportError:
    source_to_mlir = None


def _f32(*shape):
    return np.zeros(shape, dtype=np.float32)


def _bf16(*shape):
    import ml_dtypes

    return np.zeros(shape, dtype=ml_dtypes.bfloat16)


# kernel name -> n -> example inputs (shapes/dtypes match the host wrappers)
_KERNEL_INPUTS = {
    "matvec_kernel": lambda n: (_f32(n, n), _f32(n, 1)),
    "rank2_update_kernel": lambda n: (_f32(n, n), _f32(n, 1), _f32(n, 1)),
    "matmul_kernel": lambda n: (_f32(n, n), _f32(n, n)),
    "matmul_kernel_sim": lambda n: (_f32(n, n), _f32(n, n)),
    "matmul_kernel_fp32out": lambda n: (_bf16(n, n), _bf16(n, n)),
    "matmul_kernel_fp32out_sim": lambda n: (_f32(n, n), _f32(n, n)),
    "apply_reflectors_kernel": lambda n: (_f32(n, n - 2), _f32(n, n)),
}

# Sizes from tests/test_eigh_neuron.py; 8 and 16 exercise the
# non-multiple-of-32 partition offsets that #40 tripped on.
_SIZES = [8, 16, 32, 128]


@pytest.fixture(autouse=True)
def _frontend_available():
    if not HAS_NKI:
        pytest.skip("nki not importable on this host")
    if source_to_mlir is None:
        pytest.skip("nki.compiler.frontend.source_to_mlir not available in this nki")


@pytest.mark.parametrize("target", ["trn1", "trn2"])
@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize("kernel_name", sorted(_KERNEL_INPUTS))
def test_kernel_mlir_verifies(kernel_name, n, target):
    from trnsolver.nki import dispatch

    if kernel_name == "matmul_kernel_fp32out":
        pytest.importorskip("ml_dtypes")
    kernel = getattr(dispatch, kernel_name)
    result = source_to_mlir(
        kernel, *_KERNEL_INPUTS[kernel_name](n), target=target, frontend=FrontendKind.TRACER
    )
    result.verify()
