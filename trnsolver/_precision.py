"""GEMM precision routing for the GEMM-only matrix functions (#149).

``inv_sqrt_spd_ns`` and ``density_purification`` are chains of matrix
products with O(n^2) elementwise work in between. The ``precision=``
argument picks how each product is computed, per iteration, so a caller can
stage precision: cheap products early, fp64-class products late
(trnsci/trnsci#172). The iterate itself stays in the input dtype; low
precision only enters as the split pieces inside ``trnblas.gemm_fp64``,
never as the iterate's precision (trnsci/trnsci#176, rule 7).

A spec is one of:

- ``None`` or ``"native"``: ``torch.matmul`` in the operands' dtype (fp32 PSUM
  accumulation on the Tensor Engine for fp32 operands).
- ``"fp32"``: operands cast to fp32, ``torch.matmul``, result cast back. On a
  float64 iterate this emulates an fp32-product stage on the host.
- ``"fp64"`` or ``"ozaki1"``: ``trnblas.gemm_fp64(A, B)`` (Ozaki scheme I, the
  default 8 slices), result cast back to the iterate dtype.
- ``"ozaki2"``: ``trnblas.gemm_fp64(A, B, scheme="ozaki2")``.
- a dict: keyword arguments for ``trnblas.gemm_fp64``, e.g.
  ``{"scheme": "ozaki1", "slices": 6}`` or ``{"scheme": "ozaki2", "moduli": 14}``.
- a callable ``(A, B) -> C``: full control, e.g. a host fp64 iterate whose
  products run on the device.

``precision`` itself is a spec, or a list / tuple of specs: entry ``k`` is used
for iteration ``k``, and iterations past the end use the last entry (so
``["fp32"] * 9 + ["ozaki1"]`` is nine fp32-product iterations, then
fp64-class). Products outside the iteration loop (basis transforms, the final
residual) use the last entry.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

Gemm = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_FP64_SCHEMES = {"fp64": "ozaki1", "ozaki1": "ozaki1", "ozaki2": "ozaki2"}


def _trnblas():
    try:
        import trnblas
    except ImportError as e:  # pragma: no cover - trnblas is a dev dependency
        raise ImportError("precision routes through trnblas.gemm_fp64; install trnblas>=0.9") from e
    return trnblas


def _gemm_fp64(**kw) -> Gemm:
    tb = _trnblas()

    def g(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return tb.gemm_fp64(A, B, **kw).to(A.dtype)

    return g


def _fp32(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A.float(), B.float()).to(A.dtype)


def _trnblas_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return _trnblas().gemm(1.0, A, B)


def resolve(spec) -> Gemm:
    """The GEMM ``(A, B) -> A @ B`` for one precision spec (module docstring)."""
    if spec is None or spec == "native":
        return torch.matmul
    if callable(spec):
        return spec
    if isinstance(spec, dict):
        return _gemm_fp64(**spec)
    if spec == "fp32":
        return _fp32
    if spec == "trnblas":
        return _trnblas_gemm
    if spec in _FP64_SCHEMES:
        return _gemm_fp64(scheme=_FP64_SCHEMES[spec])
    raise ValueError(
        "precision must be None, 'native', 'fp32', 'fp64', 'ozaki1', 'ozaki2', a dict of "
        f"trnblas.gemm_fp64 keywords, a callable (A, B) -> C, or a list of these; got {spec!r}"
    )


def schedule(precision) -> tuple[Callable[[int], Gemm], Gemm]:
    """``(per_iteration, last)``: ``per_iteration(k)`` is iteration ``k``'s GEMM,
    ``last`` the GEMM for products outside the loop."""
    if isinstance(precision, (list, tuple)):
        if not precision:
            raise ValueError("precision list must not be empty")
        gemms = [resolve(s) for s in precision]
        return (lambda k: gemms[min(k, len(gemms) - 1)]), gemms[-1]
    g = resolve(precision)
    return (lambda k: g), g
