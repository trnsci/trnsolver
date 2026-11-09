"""cg_block reductions over n that do not depend on the LNC split (#128).

On Trn2, device ``sum`` / ``vector_norm`` / matmul reductions over n add in
an order that depends on how the work is split across logical cores, so
``cg_block`` gave a different solution at LNC=1 and LNC=2. The default
``reduction="pairwise"`` uses only elementwise adds of contiguous halves on
the device and finishes in float64 on the host.

The CPU stand-in here, ``_Dev``, is a tensor subclass for "device" tensors
whose reductions over dim 0 (``sum``, ``vector_norm``, a matmul contracting
over n) add in an order set by a layout: layout 1 reduces all n rows in one
go, layout 2 reduces the two halves separately and adds them, as two cores
would. Elementwise ops are unchanged, and a copy to the host returns a plain
tensor. Results that are bitwise equal across the two layouts do not depend
on the reduction order of the device. The order inside one 128-row bmm block
(the projected apply's Qᵀ V) is not perturbed: that a single block
contraction is LNC-invariant is a property of the hardware, which
tests/test_block_neuron.py checks.
"""

import contextlib

import numpy as np
import pytest
import torch

import trnsolver
from trnsolver import block


class _Dev(torch.Tensor):
    layout = 1

    @staticmethod
    def _wrap(out, strip):
        if isinstance(out, torch.Tensor):
            return out if strip else out.as_subclass(_Dev)
        if isinstance(out, (tuple, list)):
            return type(out)(_Dev._wrap(o, strip) for o in out)
        return out

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", "")
        with torch._C.DisableTorchFunctionSubclass():
            out = None
            if cls.layout == 2:
                out = _split_reduce(name, args, kwargs)
            if out is None:
                out = func(*args, **kwargs)
        strip = name in ("to", "cpu") and _to_host(name, args, kwargs)
        return _Dev._wrap(out, strip)


def _to_host(name, args, kwargs):
    if name == "cpu":
        return True
    dev = kwargs.get("device")
    for a in args[1:]:
        if isinstance(a, (torch.device, str)):
            dev = a
    return dev is not None and torch.device(dev).type == "cpu"


def _dim0(args, kwargs, pos=1):
    dim = kwargs.get("dim", args[pos] if len(args) > pos else None)
    return dim in (0, (0,), [0])


def _split_reduce(name, args, kwargs):
    """Layout 2: the same reduction over dim 0 as two halves, then one add."""
    if name == "sum" and _dim0(args, kwargs) and args[0].dim() >= 1 and args[0].shape[0] > 1:
        t = args[0]
        h = t.shape[0] // 2
        return t[:h].sum(dim=0) + t[h:].sum(dim=0)
    if (
        name in ("vector_norm", "linalg_vector_norm")
        and _dim0(args, kwargs, pos=2)
        and args[0].shape[0] > 1
    ):
        t = args[0]
        h = t.shape[0] // 2
        return torch.sqrt((t[:h] * t[:h]).sum(dim=0) + (t[h:] * t[h:]).sum(dim=0))
    if name in ("matmul", "__matmul__", "mm") and len(args) == 2:
        A, B = args
        if A.dim() == 2 and B.dim() == 2 and A.shape[1] > 1:
            h = A.shape[1] // 2
            return A[:, :h] @ B[:h] + A[:, h:] @ B[h:]
    return None


@contextlib.contextmanager
def _layout(k):
    old = _Dev.layout
    _Dev.layout = k
    try:
        yield
    finally:
        _Dev.layout = old


def _dev(t):
    return t.clone().as_subclass(_Dev)


def _plain(t):
    return t.as_subclass(torch.Tensor) if isinstance(t, torch.Tensor) else t


def _kernel_op(A):
    """A fixed-order operator (an LNC-invariant kernel, as on the device)."""

    def op(V):
        with torch._C.DisableTorchFunctionSubclass():
            return (A @ V).as_subclass(_Dev)

    return op


def _gp(n, noise=1e-2, seed=0, ell=0.2):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 3, generator=g, dtype=torch.float64)
    K = torch.exp(-0.5 * torch.cdist(X, X).square() / ell**2)
    return K, K + noise * torch.eye(n, dtype=torch.float64)


def _np_tree(p):
    """Reference for the fixed tree: pad, halve, then float64 sum (numpy)."""
    n = p.shape[0]
    lev = 0
    while -(-n // (1 << lev)) > block._FOLD_LEAF:
        lev += 1
    m = -(-n // (1 << lev))
    t = np.zeros((m << lev,) + p.shape[1:], dtype=p.dtype)
    t[:n] = p
    for _ in range(lev):
        h = t.shape[0] // 2
        t = t[:h] + t[h:]
    return t.astype(np.float64).sum(0)


class TestFoldPlan:
    @pytest.mark.parametrize(
        "n,lev,m",
        [(1, 0, 1), (64, 0, 64), (65, 1, 33), (128, 1, 64), (1000, 4, 63), (100000, 11, 49)],
    )
    def test_plan(self, n, lev, m):
        assert block._fold_plan(n) == (lev, m)
        assert m <= block._FOLD_LEAF and (m << lev) >= n > ((m - 1) << lev)

    @pytest.mark.parametrize("n", [1, 7, 64, 65, 1000, 4097, 30011])
    def test_matches_numpy_tree_bitwise(self, n):
        g = torch.Generator().manual_seed(n)
        X = torch.randn(n, 5, generator=g)
        Y = torch.randn(n, 5, generator=g)
        got = block._cdot(X, Y, "pairwise")
        ref = _np_tree((X * Y).numpy()).astype(np.float32)
        assert np.array_equal(got.numpy(), ref)


class TestReductionAccuracy:
    @pytest.mark.parametrize("n", [3, 100, 4097, 100003])
    def test_cdot_and_norm_vs_fp64(self, n):
        g = torch.Generator().manual_seed(1)
        X = torch.randn(n, 4, generator=g) + 0.5
        Y = torch.randn(n, 4, generator=g) + 0.5
        X64, Y64 = X.double(), Y.double()
        ref_dot = (X64 * Y64).sum(0)
        ref_norm = X64.norm(dim=0)
        scale = (X64 * Y64).abs().sum(0)
        u = 2.0**-24
        lev = block._fold_plan(n)[0]
        for red, bound in (("pairwise", (lev + 2) * u), ("host64", 2 * u), ("device", None)):
            d = block._cdot(X, Y, red)
            nr = block._col_norm(X, red)
            assert d.dtype == torch.float32 and nr.dtype == torch.float32
            err_dot = float(((d.double() - ref_dot).abs() / scale).max())
            err_norm = float(((nr.double() - ref_norm).abs() / ref_norm).max())
            if bound is not None:
                assert err_dot <= bound, (red, err_dot)
                assert err_norm <= bound, (red, err_norm)

    @pytest.mark.parametrize("red", ["pairwise", "host64", "device"])
    def test_fused_dot_and_norm_equal_separate(self, red):
        g = torch.Generator().manual_seed(4)
        X, Y = torch.randn(4097, 7, generator=g), torch.randn(4097, 7, generator=g)
        d, nr = block._cdot_and_norm(X, Y, red)
        assert torch.equal(d, block._cdot(X, Y, red))
        assert torch.equal(nr, block._col_norm(X, red))

    def test_complex(self):
        g = torch.Generator().manual_seed(2)
        X = torch.randn(999, 3, generator=g, dtype=torch.complex64)
        Y = torch.randn(999, 3, generator=g, dtype=torch.complex64)
        ref = (X.conj().to(torch.complex128) * Y.to(torch.complex128)).sum(0).real
        nref = torch.linalg.vector_norm(X.to(torch.complex128), dim=0)
        for red in block._REDUCTIONS:
            d = block._cdot(X, Y, red)
            nr = block._col_norm(X, red)
            assert d.dtype == torch.float32 and nr.dtype == torch.float32
            assert torch.allclose(d.double(), ref, rtol=1e-5, atol=1e-4)
            assert torch.allclose(nr.double(), nref, rtol=1e-6)

    @pytest.mark.parametrize("n,k", [(1, 3), (127, 5), (128, 5), (1000, 17), (5003, 40)])
    def test_tmatmul_vs_fp64(self, n, k):
        g = torch.Generator().manual_seed(3)
        Q = torch.randn(n, k, generator=g)
        V = torch.randn(n, 6, generator=g)
        ref = Q.double().T @ V.double()
        scale = Q.double().abs().T @ V.double().abs()
        for red in block._REDUCTIONS:
            c = block._tmatmul(Q, V, red)
            assert c.shape == (k, 6) and c.dtype == torch.float32
            assert float(((c.double() - ref).abs() / scale).max()) < 1e-5


class TestLayoutInvariance:
    """Bitwise equal results under two simulated LNC layouts."""

    def _both(self, fn):
        with _layout(1):
            a = fn()
        with _layout(2):
            b = fn()
        return a, b

    def test_standin_has_teeth(self):
        g = torch.Generator().manual_seed(5)
        X, Y = _dev(torch.randn(30011, 33, generator=g)), _dev(torch.randn(30011, 33, generator=g))
        a, b = self._both(lambda: _plain(block._cdot(X, Y, "device")))
        assert not torch.equal(a, b)
        a, b = self._both(lambda: _plain(block._col_norm(X, "device")))
        assert not torch.equal(a, b)
        Q = _dev(torch.randn(30011, 20, generator=g))
        a, b = self._both(lambda: _plain(block._tmatmul(Q, Y, "device")))
        assert not torch.equal(a, b)

    @pytest.mark.parametrize("red", ["pairwise", "host64"])
    @pytest.mark.parametrize("n", [1000, 30011])
    def test_reductions_bitwise(self, red, n):
        g = torch.Generator().manual_seed(6)
        X, Y = _dev(torch.randn(n, 33, generator=g)), _dev(torch.randn(n, 33, generator=g))
        Q = _dev(torch.randn(n, 20, generator=g))
        for fn in (
            lambda: block._cdot(X, Y, red),
            lambda: block._col_norm(X, red),
            lambda: block._tmatmul(Q, Y, red),
        ):
            a, b = self._both(lambda fn=fn: _plain(fn()))
            assert torch.equal(a, b)

    def _gp_solve(self, red, M=None, n=1500, R=33):
        K, A = _gp(n, noise=1e-1)
        op = _kernel_op(A.float())
        B = _dev(torch.randn(n, R, generator=torch.Generator().manual_seed(7)))
        X, info = trnsolver.cg_block(op, B, M, tol=1e-5, maxiter=2000, reduction=red)
        return _plain(X), info, A

    @pytest.mark.parametrize("red", ["pairwise", "host64"])
    def test_cg_block_bitwise(self, red):
        (Xa, ia, A), (Xb, ib, _) = self._both(lambda: self._gp_solve(red))
        assert torch.equal(Xa, Xb)
        assert ia["iterations"] == ib["iterations"]
        assert torch.equal(_plain(ia["col_iterations"]), _plain(ib["col_iterations"]))
        assert bool(ia["converged"].all())
        # Error against the float64 solution.
        B = torch.randn(Xa.shape, generator=torch.Generator().manual_seed(7))
        X64 = torch.linalg.solve(A, B.double())
        fwd = float(((Xa.double() - X64).norm(dim=0) / X64.norm(dim=0)).max())
        assert fwd < 1e-2

    def test_cg_block_device_mode_differs(self):
        (Xa, _, _), (Xb, _, _) = self._both(lambda: self._gp_solve("device"))
        assert not torch.equal(Xa, Xb)

    def test_cg_block_projected_preconditioner_bitwise(self):
        n = 1500
        K, A = _gp(n)
        L = trnsolver.pivoted_cholesky(lambda: torch.diagonal(K).clone(), lambda i: K[i], 64)

        def run():
            P = trnsolver.PivotedLowRankPreconditioner(_dev(L.float()), 1e-2, apply="projected")
            assert P.reduction == "pairwise"
            B = _dev(torch.randn(n, 33, generator=torch.Generator().manual_seed(8)))
            z = _plain(P(B))
            X, info = trnsolver.cg_block(_kernel_op(A.float()), B, P, tol=1e-5, maxiter=300)
            return z, _plain(X), info["iterations"]

        (za, Xa, ka), (zb, Xb, kb) = self._both(run)
        assert torch.equal(za, zb)
        assert torch.equal(Xa, Xb) and ka == kb

    def test_projected_apply_device_mode_differs(self):
        n = 3000
        K, _ = _gp(n)
        L = trnsolver.pivoted_cholesky(lambda: torch.diagonal(K).clone(), lambda i: K[i], 64)
        B = _dev(torch.randn(n, 33, generator=torch.Generator().manual_seed(9)))

        def run():
            P = trnsolver.PivotedLowRankPreconditioner(
                _dev(L.float()), 1e-2, apply="projected", reduction="device"
            )
            return _plain(P(B))

        a, b = self._both(run)
        assert not torch.equal(a, b)


class TestModes:
    def test_env_default_and_validation(self, monkeypatch):
        monkeypatch.delenv("TRNSOLVER_CG_REDUCTION", raising=False)
        assert block._reduction_mode(None) == "pairwise"
        monkeypatch.setenv("TRNSOLVER_CG_REDUCTION", "Device")
        assert block._reduction_mode(None) == "device"
        assert block._reduction_mode("host64") == "host64"
        monkeypatch.setenv("TRNSOLVER_CG_REDUCTION", "fast")
        with pytest.raises(ValueError, match="reduction"):
            block._reduction_mode(None)
        with pytest.raises(ValueError, match="reduction"):
            trnsolver.cg_block(torch.eye(3), torch.ones(3), reduction="exact")

    def test_modes_agree_and_iterations_match(self):
        n = 800
        _, A = _gp(n, noise=1e-1)
        B = torch.randn(n, 8, generator=torch.Generator().manual_seed(10))
        out = {
            r: trnsolver.cg_block(A.float(), B, tol=1e-5, reduction=r) for r in block._REDUCTIONS
        }
        its = {r: info["iterations"] for r, (_, info) in out.items()}
        assert max(its.values()) - min(its.values()) <= 1, its
        for r, (X, info) in out.items():
            assert bool(info["converged"].all()), r
            assert torch.allclose(X, out["host64"][0], rtol=1e-3, atol=1e-4), r

    def test_projected_modes_agree(self):
        K, _ = _gp(700)
        L = trnsolver.pivoted_cholesky(lambda: torch.diagonal(K).clone(), lambda i: K[i], 40)
        V = torch.randn(700, 5, dtype=torch.float64)
        ref = trnsolver.PivotedLowRankPreconditioner(L, 1e-2, apply="host64")(V)
        for r in block._REDUCTIONS:
            P = trnsolver.PivotedLowRankPreconditioner(L, 1e-2, apply="projected", reduction=r)
            assert P.reduction == r
            assert torch.allclose(P(V), ref, rtol=1e-8, atol=1e-10), r
            assert torch.allclose(P(V[:, 0]), ref[:, 0], rtol=1e-8, atol=1e-10), r
