"""
Native TorchNeuron registration of the trnsolver NKI kernels.

On the TorchNeuron lineage (Neuron SDK 2.33+, torch-neuronx 2.14,
PyTorch 2.14, Trn2/Trn3), Trainium is a native PyTorch `neuron` device
and an `@nki.jit` kernel is exposed to PyTorch as a custom operator via
`torch_neuronx.nki_op`. Each wrapper below calls an existing kernel from
`dispatch.py`, `tridiag.py` or `batched.py` unchanged; the op schema is inferred from the
annotations.

Importing this module registers the ops, so `dispatch.py` imports it only
after selecting the native path. CPU-only installs and the Trn1 / XLA
lineage (whose `torch_neuronx` has no `nki_op`) never load it.

This module deliberately does not use `from __future__ import annotations`:
`nki_op` builds the op schema from the live annotation objects.

The kernels are looked up on `dispatch` / `tridiag` / `batched` at call time rather
than imported by name, so registration itself does not trace or compile
anything; the first call per input shape does. The simulator-only variants
(`*_sim`) are not registered: they never run on a device.
"""

import torch
from torch_neuronx import nki_op

from . import batched as _b
from . import dispatch as _d
from . import tridiag as _t

# Fake (meta) implementations: output shape/dtype only, no kernel trace.
# Used by torch.compile tracing and for meta-device inputs.


@nki_op(
    "trnsolver::matvec",
    mutates_args={},
    fake_fn=lambda A, v: A.new_empty((A.shape[0], 1)),
)
def matvec(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _d.matvec_kernel(A, v)


@nki_op(
    "trnsolver::rank2_update",
    mutates_args={},
    fake_fn=lambda A, u, v: A.new_empty(A.shape),
)
def rank2_update(A: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _d.rank2_update_kernel(A, u, v)


@nki_op(
    "trnsolver::matmul",
    mutates_args={},
    fake_fn=lambda A, V: A.new_empty((A.shape[0], V.shape[1])),
)
def matmul(A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return _d.matmul_kernel(A, V)


@nki_op(
    "trnsolver::matmul_fp32out",
    mutates_args={},
    fake_fn=lambda A, V: A.new_empty((A.shape[0], V.shape[1]), dtype=torch.float32),
)
def matmul_fp32out(A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return _d.matmul_kernel_fp32out(A, V)


@nki_op(
    "trnsolver::apply_reflectors",
    mutates_args={},
    fake_fn=lambda V_refs, Q_in: Q_in.new_empty(Q_in.shape),
)
def apply_reflectors(V_refs: torch.Tensor, Q_in: torch.Tensor) -> torch.Tensor:
    return _d.apply_reflectors_kernel(V_refs, Q_in)


@nki_op(
    "trnsolver::householder_tridiag",
    mutates_args={},
    fake_fn=lambda A, consts: A.new_empty((A.shape[0], _t.n_out_cols(A.shape[0]))),
)
def householder_tridiag(A: torch.Tensor, consts: torch.Tensor) -> torch.Tensor:
    # Packed (n, 2 + max(n - 1, 1)) output: diag, subdiag, reflectors.
    return _t.householder_tridiag_kernel(A, consts)


# Batched kernels (#48): leading chunk dimension C, one launch per chunk.


@nki_op(
    "trnsolver::householder_tridiag_batched",
    mutates_args={},
    fake_fn=lambda A, consts: A.new_empty((A.shape[0], A.shape[1], _t.n_out_cols(A.shape[1]))),
)
def householder_tridiag_batched(A: torch.Tensor, consts: torch.Tensor) -> torch.Tensor:
    # Packed (C, n, 2 + max(n - 1, 1)) output, per matrix as householder_tridiag.
    return _b.householder_tridiag_batched_kernel(A, consts)


@nki_op(
    "trnsolver::apply_reflectors_batched",
    mutates_args={},
    fake_fn=lambda V_refs, Q_in: Q_in.new_empty(Q_in.shape),
)
def apply_reflectors_batched(V_refs: torch.Tensor, Q_in: torch.Tensor) -> torch.Tensor:
    return _b.apply_reflectors_batched_kernel(V_refs, Q_in)


@nki_op(
    "trnsolver::rr_project_batched",
    mutates_args={},
    fake_fn=lambda A, V_T: A.new_empty(A.shape),
)
def rr_project_batched(A: torch.Tensor, V_T: torch.Tensor) -> torch.Tensor:
    return _b.rr_project_batched_kernel(A, V_T)


@nki_op(
    "trnsolver::rr_rotate_batched",
    mutates_args={},
    fake_fn=lambda V_T, Q: Q.new_empty(Q.shape),
)
def rr_rotate_batched(V_T: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    return _b.rr_rotate_batched_kernel(V_T, Q)


# Keyed by the short op name used in `dispatch._launch`.
OPS = {
    "matvec": matvec,
    "rank2_update": rank2_update,
    "matmul": matmul,
    "matmul_fp32out": matmul_fp32out,
    "apply_reflectors": apply_reflectors,
    "householder_tridiag": householder_tridiag,
    "householder_tridiag_batched": householder_tridiag_batched,
    "apply_reflectors_batched": apply_reflectors_batched,
    "rr_project_batched": rr_project_batched,
    "rr_rotate_batched": rr_rotate_batched,
}
