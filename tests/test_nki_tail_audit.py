"""Tail-tile audit (trnsci/trnfft#100): every trnsolver NKI kernel at awkward sizes.

trnfft found kernels that silently dropped a partial M, K or N tile. This
file runs each trnsolver kernel (and the drivers that pad for it) at sizes
the regular suites avoid: n = 1, 2, 3, odd n, 127, 128 and the first size
past the single-tile limit (129, which must fall back, not truncate), and
batches that leave a partial tile or launch group (127, 129, 200, 1000,
4097 matrices). Every result is checked against an FP64 reference.

Each case is a plain function returning ``(error, tolerance)`` so that
the standalone probe can run the same list and tell silent wrong output
(error above tolerance) from loud failures (an exception).

Run on the instance with:
    NKI_ENABLE_TRACE_CACHE=0 pytest tests/test_nki_tail_audit.py -v -m neuron
"""

from __future__ import annotations

import pytest
import torch

import trnsolver
import trnsolver.eigen as eigen
from trnsolver.nki import HAS_NKI


def _native() -> bool:
    if not HAS_NKI:
        return False
    from trnsolver.nki.dispatch import _get_dispatch_path, _use_simulator

    return not _use_simulator() and _get_dispatch_path() == "native"


# ----------------------------------------------------------------------
# Inputs (built on the host in FP64, then rounded) and error measures
# ----------------------------------------------------------------------


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _sym(n, seed=0):
    A = torch.randn(n, n, generator=_gen(seed), dtype=torch.float64)
    return (0.5 * (A + A.T)).float()


def _sym_batch(nb, n, seed=0):
    A = torch.randn(nb, n, n, generator=_gen(seed), dtype=torch.float64)
    return (0.5 * (A + A.transpose(1, 2))).float()


def _rel(x, ref):
    """max |x - ref| / max |ref|, in FP64."""
    x = x.detach().cpu().double()
    ref = ref.detach().cpu().double()
    return float((x - ref).abs().max() / ref.abs().max().clamp(min=1e-30))


def _eig_err(A, w, V):
    """Worst of eigenvalue error, residual and orthogonality (FP64 reference)."""
    A64 = A.double().reshape(-1, A.shape[-1], A.shape[-1])
    n = A64.shape[-1]
    w64 = w.detach().cpu().double().reshape(-1, n)
    V64 = V.detach().cpu().double().reshape(-1, n, n)
    wr = torch.linalg.eigvalsh(A64)
    scale = wr.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    e_w = float(((w64 - wr).abs() / scale).max())
    res = (A64 @ V64 - V64 * w64.unsqueeze(-2)).abs().amax(dim=(1, 2)) / scale.squeeze(-1)
    orth = (V64.transpose(1, 2) @ V64 - torch.eye(n, dtype=torch.float64)).abs().max()
    return max(e_w, float(res.max()), float(orth))


# ----------------------------------------------------------------------
# Cases: kernel name, shape label, callable -> (error, tolerance)
# ----------------------------------------------------------------------


def _matvec(n):
    def run():
        from trnsolver.eigen import _call_matvec

        A = _sym(n, 1)
        v = torch.randn(n, 1, generator=_gen(2))
        return _rel(_call_matvec(A, v), A.double() @ v.double()), 1e-5

    return run


def _rank2(n):
    def run():
        from trnsolver.eigen import _call_rank2_update

        A = _sym(n, 3)
        u = torch.randn(n, 1, generator=_gen(4))
        v = torch.randn(n, 1, generator=_gen(5))
        ref = A.double() - u.double() @ v.double().T - v.double() @ u.double().T
        return _rel(_call_rank2_update(A, u, v), ref), 1e-5

    return run


def _matmul(n, k):
    def run():
        from trnsolver.eigen import _call_matmul

        A = torch.randn(n, n, generator=_gen(6))
        V = torch.randn(n, k, generator=_gen(7))
        return _rel(_call_matmul(A, V), A.double() @ V.double()), 1e-3

    return run


def _matmul_fp32out(n, k):
    def run():
        from trnsolver.eigen import _call_matmul_fp32out

        A = torch.randn(n, n, generator=_gen(8)).to(torch.bfloat16)
        V = torch.randn(n, k, generator=_gen(9)).to(torch.bfloat16)
        C = _call_matmul_fp32out(A, V)
        assert C.dtype == torch.float32
        return _rel(C, A.double() @ V.double()), 1e-3

    return run


def _apply_reflectors(n):
    def run():
        from trnsolver.eigen import _call_apply_reflectors

        g = _gen(10)
        Vr = torch.randn(n, n - 1, generator=g, dtype=torch.float64)
        Vr[:, -1] = 0.0  # the driver's last column is always zero
        Vr = Vr / Vr.norm(dim=0, keepdim=True).clamp(min=1e-30)
        Q = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64)).Q
        ref = Q.clone()
        for k in range(Vr.shape[1] - 1, -1, -1):
            v = Vr[:, k]
            ref = ref - 2.0 * torch.outer(v, v @ ref)
        out = _call_apply_reflectors(Vr.float(), Q.float())
        return _rel(out, ref), 1e-4

    return run


def _householder_eigh(n):
    def run():
        A = _sym(n, 11)
        w, V = trnsolver.eigh(A)  # REQUIRE_NKI on: no aten cutover, no fallback
        return _eig_err(A, w, V), 1e-3

    return run


def _householder_fallback(n):
    """n > 128: the single-tile driver must fall back to aten, not truncate."""

    def run():
        A = _sym(n, 12)
        eigen._REQUIRE_NKI = False
        w, V = trnsolver.eigh(A)
        return _eig_err(A, w, V), 1e-4

    return run


def _randomized(n, k, p):
    def run():
        # Dominant rank-k part with well separated eigenvalues, small tail.
        g = _gen(13)
        Q = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64)).Q
        lam = torch.full((n,), 1e-3, dtype=torch.float64)
        lam[:k] = 10.0 + 3.0 * torch.arange(k, dtype=torch.float64)
        A = ((Q * lam) @ Q.T).float()
        w, V = trnsolver.eigh_randomized(A, k, oversampling=p, n_iter=1, seed=0)
        ref = torch.sort(lam[:k]).values
        e_w = _rel(w, ref)
        R = A.double() @ V.double() - V.double() * w.double()
        res = float(R.abs().max() / ref.abs().max())
        return max(e_w, res), 1e-3

    return run


def _jacobi(n, nb, block=True, cores=None):
    def run():
        import os

        old = {k: os.environ.get(k) for k in ("TRNSOLVER_JACOBI_BLOCK", "TRNSOLVER_EIGH_CORES")}
        try:
            os.environ["TRNSOLVER_JACOBI_BLOCK"] = "1" if block else "0"
            if cores is None:
                os.environ.pop("TRNSOLVER_EIGH_CORES", None)
            else:
                import torch_neuronx

                if torch_neuronx.device_count() < cores:
                    raise _Skip(f"needs {cores} visible NeuronCores")
                os.environ["TRNSOLVER_EIGH_CORES"] = str(cores)
            A = _sym_batch(nb, n, seed=n * 7 + nb)
            w, V = trnsolver.eigh(A, method="jacobi")
            return _eig_err(A, w, V), 2e-4
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    return run


class _Skip(Exception):
    pass


def cases():
    out = []
    for n in (1, 2, 3, 5, 31, 127, 128):
        out.append(("matvec", f"n={n}", _matvec(n)))
        out.append(("rank2_update", f"n={n}", _rank2(n)))
    for n, k in ((2, 2), (3, 2), (5, 3), (31, 17), (100, 33), (127, 5), (127, 127), (5, 127)):
        out.append(("matmul", f"n={n},k={k}", _matmul(n, k)))
        out.append(("matmul_fp32out", f"n={n},k={k}", _matmul_fp32out(n, k)))
    for n in (3, 4, 5, 31, 127):
        out.append(("apply_reflectors", f"n={n}", _apply_reflectors(n)))
    for n in (2, 3, 5, 33, 127):
        out.append(("householder_eigh", f"n={n}", _householder_eigh(n)))
    out.append(("householder_eigh(fallback)", "n=129", _householder_fallback(129)))
    out.append(("householder_eigh(fallback)", "n=200", _householder_fallback(200)))
    for n, k, p in ((127, 4, 7), (100, 3, 1), (128, 5, 0)):
        out.append(("eigh_randomized", f"n={n},k={k},p={p}", _randomized(n, k, p)))
    # Scalar kernel (ne <= 32), every padded size class, partial tiles and groups.
    for n, nb in (
        (2, 1),
        (2, 4097),
        (3, 129),
        (3, 4097),
        (5, 200),
        (7, 1000),
        (17, 127),
        (31, 129),
        (33, 200),
        (63, 129),
    ):
        out.append(("jacobi_sweeps", f"n={n},B={nb}", _jacobi(n, nb, block=True)))
    # Block kernel (ne = 64, 128) and the scalar kernel at the same sizes.
    for n, nb in ((65, 3), (100, 129), (127, 1), (127, 129), (128, 200), (127, 1000)):
        out.append(("jacobi_block", f"n={n},B={nb}", _jacobi(n, nb, block=True)))
    for n, nb in ((65, 3), (127, 129)):
        out.append(("jacobi_sweeps(no block)", f"n={n},B={nb}", _jacobi(n, nb, block=False)))
    # Multi-core thread split with a ragged batch.
    for n, nb in ((5, 1000), (127, 257)):
        out.append(("jacobi_multicore", f"n={n},B={nb},cores=2", _jacobi(n, nb, cores=2)))
    return out


_CASES = cases()


@pytest.mark.neuron
@pytest.mark.parametrize("kernel,shape,run", _CASES, ids=[f"{k}[{s}]" for k, s, _ in _CASES])
def test_tail(kernel, shape, run, monkeypatch):
    if not _native():
        pytest.skip("requires TorchNeuron (native) dispatch path")
    monkeypatch.setattr(eigen, "_REQUIRE_NKI", True)
    for var in ("TRNSOLVER_EIGH_METHOD", "TRNSOLVER_JACOBI_GROUP", "TRNSOLVER_EIGH_POOL"):
        monkeypatch.delenv(var, raising=False)
    old = trnsolver.get_backend()
    trnsolver.set_backend("nki")
    try:
        err, tol = run()
    except _Skip as e:
        pytest.skip(str(e))
    finally:
        trnsolver.set_backend(old)
    assert err < tol, f"{kernel} {shape}: error {err:.3e} > {tol:.0e}"
