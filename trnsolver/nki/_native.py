"""
Native TorchNeuron registration of the trnsolver NKI kernels.

On the TorchNeuron lineage (Neuron SDK 2.33+, torch-neuronx 2.14,
PyTorch 2.14, Trn2/Trn3), Trainium is a native PyTorch `neuron` device
and an `@nki.jit` kernel is exposed to PyTorch as a custom operator via
`torch_neuronx.nki_op`. Each wrapper below calls an existing kernel from
`dispatch.py` unchanged; the op schema is inferred from the annotations.

Importing this module registers the ops, so `dispatch.py` imports it only
after selecting the native path. CPU-only installs and the Trn1 / XLA
lineage (whose `torch_neuronx` has no `nki_op`) never load it.

This module deliberately does not use `from __future__ import annotations`:
`nki_op` builds the op schema from the live annotation objects.

The kernels are looked up on `dispatch` at call time rather than imported
by name, so registration itself does not trace or compile anything; the
first call per input shape does. The simulator-only variants
(`*_sim`) are not registered: they never run on a device.
"""

import torch
from torch_neuronx import nki_op

from . import dispatch as _d

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
    "trnsolver::jacobi_sweeps",
    mutates_args={},
    fake_fn=lambda Z, V, consts, sched, onezero: Z.new_empty((2,) + tuple(Z.shape)),
)
def jacobi_sweeps(
    Z: torch.Tensor, V: torch.Tensor, consts: torch.Tensor, sched: torch.Tensor, onezero: int
) -> torch.Tensor:
    # The Jacobi kernel lives in its own module (nki/jacobi.py), imported
    # here at call time like the dispatch kernels above.
    # `onezero` is a static op argument, so the compiled kernel and the NKI
    # trace cache key on it (aws-neuron/aws-neuron-sdk#1454).
    from . import jacobi as _j

    return _j.jacobi_sweeps_kernel(Z, V, consts, sched, int(onezero))


@nki_op(
    "trnsolver::jacobi_block",
    mutates_args={},
    fake_fn=lambda Z, V, consts, sched, onezero: Z.new_empty((2,) + tuple(Z.shape)),
)
def jacobi_block(
    Z: torch.Tensor, V: torch.Tensor, consts: torch.Tensor, sched: torch.Tensor, onezero: int
) -> torch.Tensor:
    from . import jacobi as _j

    return _j.jacobi_block_kernel(Z, V, consts, sched, int(onezero))


@nki_op(
    "trnsolver::subspace_filter",
    mutates_args={},
    fake_fn=lambda F, X, table, sched: F.new_empty((F.shape[0] + 6 * X.shape[1] + 1, X.shape[1])),
)
def subspace_filter(
    F: torch.Tensor, X: torch.Tensor, table: torch.Tensor, sched: torch.Tensor
) -> torch.Tensor:
    from . import subspace as _s

    return _s.subspace_filter_kernel(F, X, table, sched)


# Keyed by the short op name used in `dispatch._launch`.
OPS = {
    "matvec": matvec,
    "rank2_update": rank2_update,
    "matmul": matmul,
    "matmul_fp32out": matmul_fp32out,
    "apply_reflectors": apply_reflectors,
    "jacobi_sweeps": jacobi_sweeps,
    "jacobi_block": jacobi_block,
    "subspace_filter": subspace_filter,
}
