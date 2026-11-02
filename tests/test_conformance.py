"""Reference-conformance suite (trnsci/trnsci#105).

Every public trnsolver function is checked against an *independent*
reference: ``torch.linalg`` / ``scipy.linalg`` in float64 or complex128, or
the defining identity (``A = L Lᴴ``, ``A V = V Λ``, ``Q R = A``, the true
residual of a solve). A differential nki-vs-pytorch test cannot see a bug
that every backend shares; this suite can. The cases cover awkward sizes
(1, 2, 127, 129, 300 and 1000 for the iterative solvers), dtypes (bf16,
fp16, fp32, fp64, complex64, complex128), every non-default keyword value,
strided / non-contiguous inputs, batched and empty inputs, input mutation and
output aliasing, determinism, and fp64 ``gradcheck`` where differentiable.

Known behaviours that wait on a decision are ``xfail(strict=True)`` with the
issue number, so the test flips (XPASS -> failure) once they are decided.
Bugs with an open fix PR are ``xfail(strict=False)`` so CI stays green when
the fix merges; drop the marker in that PR or right after it.

CPU CI runs this file with the default ``--backend pytorch`` on CPU inputs.
The same cases run on a Trainium box (manual path, no hardware CI)::

    pytest tests/test_conformance.py --backend nki
    pytest tests/test_conformance.py --backend auto --conformance-device neuron
    TRNSOLVER_REQUIRE_NKI=1 pytest tests/test_conformance.py --backend nki \\
        --conformance-device neuron

Off the pytorch backend the tolerances relax to the FP32 kernel level and
the fp64 gradchecks are skipped on a non-CPU device.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest
import scipy.linalg as sla
import torch

import trnsolver as ts

# Set per module by the ``_conformance_env`` fixture from the CLI options.
BACKEND = "pytorch"
DEV = torch.device("cpu")

F64, C128 = torch.float64, torch.complex128
DTYPES = (
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)
TOL = {
    torch.bfloat16: 6e-2,
    torch.float16: 1e-2,
    torch.float32: 2e-4,
    torch.float64: 1e-9,
    torch.complex64: 2e-4,
    torch.complex128: 1e-9,
}
N = 12  # default size for the dtype / layout / alias axes


@pytest.fixture(scope="module", autouse=True)
def _conformance_env(request):
    global BACKEND, DEV
    BACKEND = request.config.getoption("--backend")
    DEV = torch.device(request.config.getoption("--conformance-device"))
    if BACKEND == "nki" and not ts.HAS_NKI:
        pytest.skip("--backend nki needs the nki package (a Neuron host)")
    prev = ts.get_backend()
    ts.set_backend(BACKEND)
    yield
    ts.set_backend(prev)


# ---------------------------------------------------------------- helpers


def _dn(dt):
    return str(dt).removeprefix("torch.")


def tol_for(dt, scale=1.0):
    t = TOL[dt] * scale
    if BACKEND != "pytorch":
        t = max(t, 2e-3)  # FP32 kernels (Jacobi sweeps, Householder) on device
    return t


def g(seed):
    return torch.Generator().manual_seed(seed)


def rand(*shape, dtype=F64, seed=0):
    if dtype.is_complex:
        re = torch.randn(*shape, generator=g(seed), dtype=F64)
        im = torch.randn(*shape, generator=g(seed + 1000), dtype=F64)
        return torch.complex(re, im).to(dtype)
    return torch.randn(*shape, generator=g(seed), dtype=F64).to(dtype)


def ref_dtype(dt):
    return C128 if dt.is_complex else F64


def real_of(dt):
    return {torch.complex64: torch.float32, torch.complex128: torch.float64}.get(dt, dt)


def spd(n, dtype=F64, batch=(), seed=0):
    """Well-conditioned SPD / HPD matrix (cond ~ 10)."""
    X = rand(*batch, n, n, dtype=ref_dtype(dtype), seed=seed)
    A = X @ X.mH / n + 2.0 * torch.eye(n, dtype=ref_dtype(dtype))
    return A.to(dtype)


def herm(n, dtype=F64, batch=(), seed=0):
    """Indefinite Hermitian matrix."""
    return (spd(n, ref_dtype(dtype), batch, seed) - 2.5 * torch.eye(n, dtype=ref_dtype(dtype))).to(
        dtype
    )


def gen(n, dtype=F64, batch=(), seed=0, m=None):
    """General well-conditioned (m, n) matrix."""
    m = n if m is None else m
    X = rand(*batch, m, n, dtype=ref_dtype(dtype), seed=seed) / max(m, n, 1) ** 0.5
    if m == n:
        X = X + 3.0 * torch.eye(n, dtype=ref_dtype(dtype))
    return X.to(dtype)


def dev(x):
    return x.to(DEV) if isinstance(x, torch.Tensor) else x


def host(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, (tuple, list)):
        return type(x)(host(t) for t in x)
    return x


def up(x):
    """float64 / complex128 host copy, the reference precision."""
    return x.detach().cpu().to(ref_dtype(x.dtype))


def relerr(x, ref):
    if ref.numel() == 0:
        return 0.0
    x = host(x).to(ref.dtype)
    den = max(ref.abs().max().item(), 1e-30)
    return (x - ref).abs().max().item() / den


def assert_close(x, ref, tol, what):
    x = host(x)
    assert tuple(x.shape) == tuple(ref.shape), (
        f"{what}: shape {tuple(x.shape)} != {tuple(ref.shape)}"
    )
    if x.numel():
        assert torch.isfinite(x.to(ref.dtype)).all(), f"{what}: non-finite output"
    e = relerr(x, ref)
    assert e <= tol, f"{what}: rel err {e:.3g} > tol {tol:.1g}"


def assert_dtype(x, dt, what):
    assert x.dtype == dt, f"{what}: dtype {x.dtype} != {dt}"


def guarded(fn, *inputs):
    """Call fn(*inputs) and fail if a tensor input is mutated or aliased by the output."""
    tin = [t for t in inputs if isinstance(t, torch.Tensor)]
    snaps = [t.detach().clone() for t in tin]
    out = fn(*inputs)
    for t, s in zip(tin, snaps, strict=True):
        assert torch.equal(host(t), host(s)), "input mutated in place"
    for o in out if isinstance(out, (tuple, list)) else (out,):
        if isinstance(o, torch.Tensor) and o.numel():
            for t in tin:
                if t.numel():
                    assert o.untyped_storage().data_ptr() != t.untyped_storage().data_ptr(), (
                        "output shares storage with an input"
                    )
    return out


def run(fn, inputs, check):
    out = guarded(fn, *[dev(x) for x in inputs])
    check(out)
    return out


def quiet(fn):
    def wrapped(*a, **k):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*a, **k)

    return wrapped


def P(id_, *values, marks=()):
    """pytest.param with the KNOWN marker for ``id_`` added."""
    m = list(marks if isinstance(marks, (list, tuple)) else [marks])
    if id_ in KNOWN:
        m.append(KNOWN[id_])
    return pytest.param(*values, id=id_, marks=m)


# ---------------------------------------------------------------- known issues


def _pending(issue, pr, what):
    return pytest.mark.xfail(strict=False, reason=f"trnsci/trnsolver#{issue}, fix in #{pr}: {what}")


def _decide(what):
    return pytest.mark.xfail(strict=True, reason=f"trnsci/trnsolver#76 needs decision: {what}")


KNOWN: dict[str, pytest.MarkDecorator] = {}  # filled in next to each case table


# ---------------------------------------------------------------- reference checkers


def chk_cholesky(A, upper, tol, dt):
    def f(out):
        assert_dtype(out, dt, "factor")
        assert_close(
            out, torch.linalg.cholesky(up(A), upper=upper), tol, "factor vs torch.linalg.cholesky"
        )

    return f


def chk_lu(A, tol, dt):
    def f(out):
        for t, nm in zip(out, "PLU", strict=True):
            assert_dtype(t, dt, nm)
        Pm, L, U = (host(t).to(ref_dtype(dt)) for t in out)
        A64 = up(A)
        assert_close(Pm @ L @ U, A64, tol, "P @ L @ U vs A")
        assert not (torch.tril(U, -1) != 0).any(), "U not upper triangular"
        assert not (torch.triu(L, 1) != 0).any(), "L not lower triangular"
        if A64.dim() == 2 and A64.numel():
            # Partial pivoting gives |L| <= 1 and the same P as LAPACK getrf.
            assert L.abs().max().item() <= 1 + 1e-6, "|L| > 1: not partial pivoting"
            Ps, Ls, Us = sla.lu(A64.numpy())
            assert_close(U, torch.from_numpy(Us), tol, "U vs scipy.linalg.lu")

    return f


def chk_qr(A, tol, dt):
    def f(out):
        Q, R = out
        assert_dtype(Q, dt, "Q")
        assert_dtype(R, dt, "R")
        A64 = up(A)
        Q, R = host(Q).to(A64.dtype), host(R).to(A64.dtype)
        Qr, Rr = torch.linalg.qr(A64)
        assert Q.shape == Qr.shape and R.shape == Rr.shape, (
            f"Q{tuple(Q.shape)} R{tuple(R.shape)} (reduced)"
        )
        assert_close(Q @ R, A64, tol, "Q @ R vs A")
        k = Q.shape[-1]
        assert_close(
            Q.mH @ Q, torch.eye(k, dtype=A64.dtype).expand(*Q.shape[:-2], k, k), tol, "Qᴴ Q vs I"
        )
        if R.numel():
            assert not (torch.tril(R, -1).abs() > tol * max(R.abs().max().item(), 1)).any(), (
                "R not upper triangular"
            )

    return f


def chk_svd(A, tol, dt, full=False):
    def f(out):
        U, s, Vh = out
        assert_dtype(U, dt, "U")
        assert_dtype(s, real_of(dt), "s")
        assert_dtype(Vh, dt, "Vh")
        A64 = up(A)
        U, s, Vh = host(U).to(A64.dtype), host(s).to(real_of(A64.dtype)), host(Vh).to(A64.dtype)
        Ur, sr, Vhr = torch.linalg.svd(A64, full_matrices=full)
        assert U.shape == Ur.shape and Vh.shape == Vhr.shape, (
            f"U{tuple(U.shape)} Vh{tuple(Vh.shape)}"
        )
        assert_close(s, sr, tol, "s vs svdvals")
        k = s.shape[-1]
        Us, Vhs = (U[..., :, :k], Vh[..., :k, :]) if full else (U, Vh)
        assert_close((Us * s.unsqueeze(-2).to(A64.dtype)) @ Vhs, A64, tol, "U diag(s) Vh vs A")
        m, n = U.shape[-1], Vh.shape[-2]
        assert_close(
            U.mH @ U, torch.eye(m, dtype=A64.dtype).expand(*U.shape[:-2], m, m), tol, "Uᴴ U"
        )
        assert_close(
            Vh @ Vh.mH, torch.eye(n, dtype=A64.dtype).expand(*Vh.shape[:-2], n, n), tol, "Vh Vhᴴ"
        )

    return f


def chk_eigh(A, tol, dt):
    def f(out):
        w, V = out
        assert_dtype(w, real_of(dt), "w")
        assert_dtype(V, dt, "V")
        A64 = up(A)
        w, V = host(w).to(real_of(A64.dtype)), host(V).to(A64.dtype)
        assert_close(w, torch.linalg.eigvalsh(A64), tol, "w vs eigvalsh (ascending)")
        n = A64.shape[-1]
        assert_close(
            V @ torch.diag_embed(w.to(A64.dtype)) @ V.mH, A64, tol * 5, "V diag(w) Vᴴ vs A"
        )
        assert_close(V.mH @ V, torch.eye(n, dtype=A64.dtype).expand_as(A64), tol * 5, "Vᴴ V vs I")

    return f


def chk_eigh_generalized(A, B, tol, dt):
    def f(out):
        w, V = out
        assert_dtype(w, real_of(dt), "w")
        assert_dtype(V, dt, "V")
        A64, B64 = up(A), up(B)
        w, V = host(w).to(real_of(A64.dtype)), host(V).to(A64.dtype)
        if A64.dim() == 2:
            wr = torch.from_numpy(sla.eigh(A64.numpy(), B64.numpy(), eigvals_only=True))
        else:
            Li = torch.linalg.inv(torch.linalg.cholesky(B64))
            wr = torch.linalg.eigvalsh(Li @ A64 @ Li.mH)
        assert_close(w, wr.to(w.dtype), tol * 5, "w vs scipy.linalg.eigh(A, B)")
        assert_close(
            A64 @ V, B64 @ V * w.to(A64.dtype).unsqueeze(-2), tol * 20, "A V vs B V diag(w)"
        )
        n = A64.shape[-1]
        assert_close(
            V.mH @ B64 @ V, torch.eye(n, dtype=A64.dtype).expand_as(A64), tol * 20, "Vᴴ B V vs I"
        )

    return f


def chk_solve(A, B, tol, dt):
    def f(out):
        assert_dtype(out, dt, "X")
        assert_close(out, torch.linalg.solve(up(A), up(B)), tol, "X vs torch.linalg.solve")

    return f


def chk_mat(ref, tol, dt, what):
    def f(out):
        assert_dtype(out, dt, what)
        assert_close(out, ref, tol, what)

    return f


def inv_sqrt_ref(A64):
    w, V = torch.linalg.eigh(A64)
    return V @ torch.diag_embed((1.0 / w.sqrt()).to(A64.dtype)) @ V.mH


def chk_schur(A, tol, dt):
    def f(out):
        Q, T = out
        assert_dtype(Q, dt, "Q")
        assert_dtype(T, dt, "T")
        A64 = up(A)
        Q, T = host(Q).to(A64.dtype), host(T).to(A64.dtype)
        n = A64.shape[-1]
        assert_close(Q @ T @ Q.mH, A64, tol * 10, "Q T Qᴴ vs A")
        assert_close(Q.mH @ Q, torch.eye(n, dtype=A64.dtype), tol * 10, "Qᴴ Q vs I")
        if n > 2:
            assert not (torch.tril(T, -2).abs() > 1e-6 * T.abs().max()).any(), (
                "T not quasi-upper-triangular"
            )
            # No two consecutive nonzero subdiagonals (2x2 blocks only).
            sub = (torch.diagonal(T, -1).abs() > 1e-6 * T.abs().max()).int()
            assert not (sub[1:] * sub[:-1]).any(), "T has a 3x3 diagonal block"
        Ts = sla.schur(A64.numpy(), output="real")[0]
        ev = np.sort_complex(np.linalg.eigvals(T.numpy()))
        er = np.sort_complex(np.linalg.eigvals(Ts))
        assert_close(
            torch.from_numpy(ev), torch.from_numpy(er), tol * 10, "eig(T) vs scipy.linalg.schur"
        )

    return f


def ns(A, **kw):
    kw.setdefault("max_iters", 60)
    kw.setdefault("tol", 1e-12)
    return ts.inv_sqrt_spd_ns(A, **kw)[0]


# name -> (fn, build(n, dt, batch) -> inputs, checker(inputs, tol, dt))
OPS = {
    "cholesky": (
        ts.cholesky,
        lambda n, dt, b=(): (spd(n, dt, b),),
        lambda inp, t, dt: chk_cholesky(inp[0], False, t, dt),
    ),
    "cholesky_upper": (
        lambda A: ts.cholesky(A, upper=True),
        lambda n, dt, b=(): (spd(n, dt, b),),
        lambda inp, t, dt: chk_cholesky(inp[0], True, t, dt),
    ),
    "lu": (ts.lu, lambda n, dt, b=(): (gen(n, dt, b),), lambda inp, t, dt: chk_lu(inp[0], t, dt)),
    "qr": (
        ts.qr,
        lambda n, dt, b=(): (gen(n, dt, b, m=n + 4),),
        lambda inp, t, dt: chk_qr(inp[0], t, dt),
    ),
    "qr_wide": (
        ts.qr,
        lambda n, dt, b=(): (gen(n + 4, dt, b, m=n),),
        lambda inp, t, dt: chk_qr(inp[0], t, dt),
    ),
    "svd": (
        ts.svd,
        lambda n, dt, b=(): (gen(n, dt, b, m=n + 3),),
        lambda inp, t, dt: chk_svd(inp[0], t, dt),
    ),
    "svd_full": (
        lambda A: ts.svd(A, full_matrices=True),
        lambda n, dt, b=(): (gen(n, dt, b, m=n + 3),),
        lambda inp, t, dt: chk_svd(inp[0], t, dt, full=True),
    ),
    "solve_vec": (
        ts.solve,
        lambda n, dt, b=(): (gen(n, dt, b), rand(*b, n, dtype=dt, seed=5)),
        lambda inp, t, dt: chk_solve(inp[0], inp[1], t, dt),
    ),
    "solve_mat": (
        ts.solve,
        lambda n, dt, b=(): (gen(n, dt, b), rand(*b, n, 3, dtype=dt, seed=5)),
        lambda inp, t, dt: chk_solve(inp[0], inp[1], t, dt),
    ),
    "solve_spd_vec": (
        ts.solve_spd,
        lambda n, dt, b=(): (spd(n, dt, b), rand(*b, n, dtype=dt, seed=5)),
        lambda inp, t, dt: chk_solve(inp[0], inp[1], t, dt),
    ),
    "solve_spd_mat": (
        ts.solve_spd,
        lambda n, dt, b=(): (spd(n, dt, b), rand(*b, n, 3, dtype=dt, seed=5)),
        lambda inp, t, dt: chk_solve(inp[0], inp[1], t, dt),
    ),
    "solve_spd_ir": (
        lambda A, B: ts.solve_spd(A, B, iterative_refinement=True),
        lambda n, dt, b=(): (spd(n, dt, b), rand(*b, n, 3, dtype=dt, seed=5)),
        lambda inp, t, dt: chk_solve(inp[0], inp[1], t, dt),
    ),
    "inv_spd": (
        ts.inv_spd,
        lambda n, dt, b=(): (spd(n, dt, b),),
        lambda inp, t, dt: chk_mat(
            torch.linalg.inv(up(inp[0])), t, dt, "inv_spd vs torch.linalg.inv"
        ),
    ),
    "pinv": (
        ts.pinv,
        lambda n, dt, b=(): (gen(n, dt, b, m=n + 3),),
        lambda inp, t, dt: chk_mat(
            torch.linalg.pinv(up(inp[0])), t, dt, "pinv vs torch.linalg.pinv"
        ),
    ),
    "inv_sqrt_spd": (
        ts.inv_sqrt_spd,
        lambda n, dt, b=(): (spd(n, dt, b),),
        lambda inp, t, dt: chk_mat(inv_sqrt_ref(up(inp[0])), t, dt, "A^-1/2 vs eigh formula"),
    ),
    "inv_sqrt_spd_ns": (
        ns,
        lambda n, dt, b=(): (spd(n, dt, b),),
        lambda inp, t, dt: chk_mat(
            inv_sqrt_ref(up(inp[0])), max(t, 1e-6) * 10, dt, "Newton-Schulz A^-1/2"
        ),
    ),
    "eigh": (
        ts.eigh,
        lambda n, dt, b=(): (herm(n, dt, b),),
        lambda inp, t, dt: chk_eigh(inp[0], t, dt),
    ),
    "eigh_auto": (
        lambda A: ts.eigh(A, method="auto"),
        lambda n, dt, b=(): (herm(n, dt, b),),
        lambda inp, t, dt: chk_eigh(inp[0], t, dt),
    ),
    "eigh_jacobi": (
        lambda A: ts.eigh(A, method="jacobi"),
        lambda n, dt, b=(): (herm(n, dt, b),),
        lambda inp, t, dt: chk_eigh(inp[0], max(t, 1e-6), dt),
    ),
    "eigh_generalized": (
        ts.eigh_generalized,
        lambda n, dt, b=(): (herm(n, dt, b, seed=1), spd(n, dt, b, seed=2)),
        lambda inp, t, dt: chk_eigh_generalized(inp[0], inp[1], t, dt),
    ),
    "schur": (
        ts.schur,
        lambda n, dt, b=(): (gen(n, dt, b),),
        lambda inp, t, dt: chk_schur(inp[0], t, dt),
    ),
}

# Documented real-only and unbatched: raising on complex or batched input is fine.
REAL_ONLY = {"schur", "inv_sqrt_spd_ns"}
UNBATCHED = {"schur", "inv_sqrt_spd_ns"}
# Pure-Python sweeps: size cap keeps CI fast.
SLOW = {"schur": 129, "eigh_jacobi": 129}


def _expect_raise_or(fn):
    """Raising is acceptable (documented unsupported input); a quiet wrong answer is not."""

    def wrapped():
        try:
            fn()
        except AssertionError:
            raise
        except Exception:  # noqa: BLE001 -- any loud rejection is fine here
            pytest.skip("rejected loudly (documented unsupported input)")

    return wrapped


# ---------------------------------------------------------------- axis 1: dtype x op

for _dt in ("complex64", "complex128"):
    KNOWN[f"eigh_jacobi-{_dt}"] = pytest.mark.xfail(
        strict=False,
        raises=NotImplementedError,
        reason="trnsci/trnsolver#79 (loud): eigh(method='jacobi') rejects complex input",
    )


@pytest.mark.parametrize(
    "op,dt",
    [P(f"{op}-{_dn(dt)}", op, dt) for op in OPS for dt in DTYPES],
)
def test_dtype(op, dt):
    fn, build, chk = OPS[op]
    inputs = build(N, dt)
    body = lambda: run(quiet(fn), inputs, chk(inputs, tol_for(dt), dt))  # noqa: E731
    (_expect_raise_or(body) if dt.is_complex and op in REAL_ONLY else body)()


# ---------------------------------------------------------------- axis 2: awkward sizes


SIZES = (1, 2, 127, 129, 300)


@pytest.mark.parametrize(
    "op,n,dt",
    [
        P(f"{op}-n{n}-{_dn(dt)}", op, n, dt)
        for op in OPS
        for n in SIZES
        for dt in ((F64, torch.float32) if n == 129 else (F64,))
        if n <= SLOW.get(op, 10**9)
    ],
)
def test_sizes(op, n, dt):
    fn, build, chk = OPS[op]
    inputs = build(n, dt)
    run(
        quiet(fn),
        inputs,
        chk(inputs, tol_for(dt, 10 if dt == torch.float32 and n > 100 else 1), dt),
    )


# ---------------------------------------------------------------- axis 3: non-contiguous


def _view(X, kind):
    """A view with X's values and different strides, built on the input device."""
    X = dev(X)
    if X.dim() < 2:
        return torch.stack([X, X], -1)[..., 0]
    if kind == "step_slice":
        big = torch.zeros(
            *X.shape[:-2], 2 * X.shape[-2], 2 * X.shape[-1], dtype=X.dtype, device=X.device
        )
        big[..., ::2, ::2] = X
        return big[..., ::2, ::2]
    if kind == "transposed_storage":
        return X.mT.contiguous().mT
    if kind == "as_strided_offset":
        st = torch.empty(X.numel() + 7, dtype=X.dtype, device=X.device)
        st[7:] = X.reshape(-1)
        return st.as_strided(X.shape, X.stride(), 7)
    if kind == "column_window":
        big = torch.zeros(*X.shape[:-1], X.shape[-1] + 5, dtype=X.dtype, device=X.device)
        big[..., 2 : 2 + X.shape[-1]] = X
        return big[..., 2 : 2 + X.shape[-1]]
    raise ValueError(kind)


VIEWS = ("step_slice", "transposed_storage", "as_strided_offset", "column_window")


@pytest.mark.parametrize(
    "op,dt,kind",
    [
        P(f"{op}-{_dn(dt)}-{k}", op, dt, k)
        for op in OPS
        for dt in (torch.float32, F64)
        for k in VIEWS
    ],
)
def test_noncontiguous(op, dt, kind):
    fn, build, chk = OPS[op]
    inputs = build(N, dt)
    views = tuple(_view(x, kind) for x in inputs)
    for a, b in zip(inputs, views, strict=True):
        assert torch.equal(a, host(b))
    assert any(not v.is_contiguous() or v.storage_offset() for v in views)
    out = guarded(quiet(fn), *views)
    chk(inputs, tol_for(dt), dt)(out)


@pytest.mark.parametrize(
    "case",
    [
        "solve-B_T",
        "solve_spd-B_expanded",
        "cg-b_step",
        "gmres-A_T",
        "gmres_ir-A_T",
        "eigh-batch_permuted",
    ],
)
def test_noncontiguous_extra(case):
    if case == "solve-B_T":
        A, B = gen(N), rand(3, N, seed=4).T
        run(ts.solve, (A, B), chk_solve(A, B.contiguous(), tol_for(F64), F64))
    elif case == "solve_spd-B_expanded":
        A, B = spd(N), rand(N, 1, seed=4).expand(N, 3)
        run(ts.solve_spd, (A, B), chk_solve(A, B.contiguous(), tol_for(F64), F64))
    elif case == "cg-b_step":
        A, b = spd(N), rand(2 * N, seed=4)[::2]
        run(lambda A, b: ts.cg(A, b, tol=1e-12)[0], (A, b), chk_solve(A, b.contiguous(), 1e-8, F64))
    elif case == "gmres-A_T":
        A, b = gen(N).T, rand(N, seed=4)
        run(
            lambda A, b: ts.gmres(A, b, tol=1e-12, restart=N)[0],
            (A, b),
            chk_solve(A.contiguous(), b, 1e-8, F64),
        )
    elif case == "gmres_ir-A_T":
        A, b = gen(N).T, rand(N, seed=4)
        run(lambda A, b: ts.gmres_ir(A, b)[0], (A, b), chk_solve(A.contiguous(), b, 1e-8, F64))
    else:
        A = spd(N, batch=(2, 3))
        Av = A.permute(1, 0, 2, 3).contiguous().permute(1, 0, 2, 3)
        run(ts.eigh, (Av,), chk_eigh(A, tol_for(F64), F64))


# ---------------------------------------------------------------- axis 4: edge shapes

EMPTY = torch.zeros(0, 0, dtype=F64)
EDGE = []
for _op in ("cholesky", "cholesky_upper", "inv_spd", "inv_sqrt_spd", "eigh", "eigh_jacobi"):
    EDGE.append((f"{_op}-0x0", _op, (EMPTY,)))
for _op in ("lu", "qr", "svd", "svd_full", "pinv"):
    for _m, _n in ((1, 1), (64, 3), (3, 64), (200, 1), (1, 200), (0, 0), (0, 3), (3, 0)):
        _A = gen(_n, m=_m) if _m and _n else torch.zeros(_m, _n, dtype=F64)
        EDGE.append((f"{_op}-{_m}x{_n}", _op, (_A,)))
EDGE += [
    ("solve_mat-Bnx0", "solve_mat", (gen(4), torch.zeros(4, 0, dtype=F64))),
    ("solve_spd_mat-Bnx0", "solve_spd_mat", (spd(4), torch.zeros(4, 0, dtype=F64))),
    ("solve_mat-Bnx1", "solve_mat", (gen(5), rand(5, 1))),
    ("solve_spd_mat-Bnx1", "solve_spd_mat", (spd(5), rand(5, 1))),
    ("eigh-n128", "eigh", (herm(128),)),
]
KNOWN["eigh_jacobi-0x0"] = pytest.mark.xfail(
    strict=False,
    raises=RuntimeError,
    reason="trnsci/trnsolver#79 (loud): eigh(method='jacobi') cannot reshape a 0x0 input",
)


def _eigh_n129_mark():
    if BACKEND != "pytorch" and os.environ.get("TRNSOLVER_REQUIRE_NKI") == "1":
        return "raise"  # Householder tridiag is single-tile n<=128 by design
    return None


@pytest.mark.parametrize("op,inputs", [P(i, op, inp) for i, op, inp in EDGE])
def test_edge_shapes(op, inputs):
    fn, _, chk = OPS[op]
    run(quiet(fn), inputs, chk(inputs, tol_for(F64), F64))


def test_eigh_n129_require_nki():
    A = herm(129)
    if _eigh_n129_mark() == "raise":
        with pytest.raises(Exception):  # noqa: B017 -- documented REQUIRE_NKI refusal
            ts.eigh(dev(A))
        return
    run(ts.eigh, (A,), chk_eigh(A, tol_for(F64), F64))


def test_schur_small():
    for n in (1, 2):
        A = gen(n)
        run(ts.schur, (A,), chk_schur(A, tol_for(F64), F64))


def test_schur_complex_eigenvalues():
    # Rotation-dominated matrix: 2x2 blocks in the real Schur form.
    th = torch.linspace(0.3, 2.5, 4, dtype=F64)
    R = torch.zeros(8, 8, dtype=F64)
    for i, t in enumerate(th):
        c, s = torch.cos(t), torch.sin(t)
        R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = torch.tensor([[c, -s], [s, c]]) * (1 + i)
    Q, _ = torch.linalg.qr(rand(8, 8, seed=3))
    A = Q @ R @ Q.T
    run(ts.schur, (A,), chk_schur(A, tol_for(F64), F64))


def test_eigh_generalized_n1():
    A, B = spd(1, seed=1), spd(1, seed=2)
    run(ts.eigh_generalized, (A, B), chk_eigh_generalized(A, B, tol_for(F64), F64))


def test_eigh_repeated_eigenvalues():
    # Degenerate spectrum: V must still be orthonormal and reconstruct A.
    Q, _ = torch.linalg.qr(rand(10, 10, seed=4))
    A = (Q * torch.tensor([1.0] * 5 + [3.0] * 5, dtype=F64)) @ Q.T
    A = 0.5 * (A + A.T)
    for method in ("auto", "jacobi"):
        run(
            lambda A, m=method: ts.eigh(A, method=m),
            (A,),
            chk_eigh(A, max(tol_for(F64), 1e-6), F64),
        )


# ---------------------------------------------------------------- axis 5: keyword value tables


@pytest.mark.parametrize("method", [None, "auto", "jacobi"])
@pytest.mark.parametrize("tol", [1e-6, 1e-10, 1e-14])
def test_eigh_kwargs(method, tol):
    A = herm(17, seed=3)
    run(lambda A: ts.eigh(A, tol, method=method), (A,), chk_eigh(A, max(tol_for(F64), 1e-6), F64))


def test_eigh_bad_method_raises():
    with pytest.raises(ValueError):
        ts.eigh(dev(herm(4)), method="householder-typo")


@pytest.mark.parametrize("tol", [1e-6, 1e-12])
def test_schur_tol(tol):
    A = gen(20, seed=2)
    run(lambda A: ts.schur(A, tol), (A,), chk_schur(A, max(tol_for(F64), 1e-5), F64))


@pytest.mark.parametrize("tol", [1e-6, 1e-12])
def test_eigh_generalized_tol(tol):
    A, B = herm(16, seed=1), spd(16, seed=2)
    run(
        lambda A, B: ts.eigh_generalized(A, B, tol),
        (A, B),
        chk_eigh_generalized(A, B, max(tol_for(F64), 1e-6), F64),
    )


@pytest.mark.parametrize("rcond", [None, 1e-12, 1e-3, 0.5])
@pytest.mark.parametrize("shape", [(9, 6), (6, 9)])
def test_pinv_rcond_rank_deficient(rcond, shape):
    m, n = shape
    # rank 3 with singular values (10, 1, 0.1) plus an exact null space.
    U, _ = torch.linalg.qr(rand(m, 3, seed=1))
    V, _ = torch.linalg.qr(rand(n, 3, seed=2))
    A = (U * torch.tensor([10.0, 1.0, 0.1], dtype=F64)) @ V.T
    ref = torch.linalg.pinv(A) if rcond is None else torch.linalg.pinv(A, rtol=rcond)
    run(
        lambda A: ts.pinv(A, rcond),
        (A,),
        chk_mat(ref, tol_for(F64, 100), F64, "pinv vs torch.linalg.pinv(rtol)"),
    )


@pytest.mark.parametrize("max_iters,tol", [(5, 1e-7), (20, 1e-7), (60, 1e-12), (0, 1e-7)])
def test_inv_sqrt_spd_ns_kwargs(max_iters, tol):
    A = spd(24, seed=3)
    X, iters, res = guarded(lambda A: ts.inv_sqrt_spd_ns(A, max_iters=max_iters, tol=tol), dev(A))
    assert 0 <= iters <= max_iters
    if max_iters == 0:
        assert torch.equal(host(X), torch.eye(24, dtype=F64) / torch.linalg.norm(A).sqrt())
        return
    assert np.isfinite(res)
    if res < tol:  # reported converged: the result must be A^-1/2
        Xh = host(X)
        true_res = (torch.linalg.norm(Xh @ A @ Xh - torch.eye(24, dtype=F64)) / 24**0.5).item()
        assert true_res <= 10 * tol + 1e-12, f"reported {res:.3g}, true ||XAX - I|| {true_res:.3g}"
        assert_close(X, inv_sqrt_ref(A), max(1e-6, 10 * tol), "Newton-Schulz A^-1/2")
    else:
        assert iters == max_iters


@pytest.mark.parametrize("n", [129, 300])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("oversampling,n_iter", [(0, 0), (2, 1), (10, 0), (10, 2)])
def test_eigh_randomized_kwargs(n, k, oversampling, n_iter):
    if k + oversampling < 4:
        pytest.skip("sketch smaller than the rank-4 test matrix: no exactness to check")
    A, lam = _lowrank(n, tail=0.0)
    w, V = guarded(
        lambda A: ts.eigh_randomized(A, k, oversampling=oversampling, n_iter=n_iter, seed=7), dev(A)
    )
    top = lam[torch.argsort(lam.abs(), descending=True)[:k]].sort().values
    assert_dtype(w, F64, "w")
    assert_close(w, top, tol_for(F64, 1e3), "top-|λ| vs dense")
    Vh = host(V)
    assert_close(Vh.T @ Vh, torch.eye(k, dtype=F64), tol_for(F64, 1e3), "Vᵀ V")
    assert_close(A @ Vh, Vh * host(w), tol_for(F64, 1e4), "A V = V diag(w)")


def test_eigh_randomized_power_iterations_help():
    A, lam = _lowrank(200, tail=0.3)
    top = lam[torch.argsort(lam.abs(), descending=True)[:4]].sort().values
    e = [
        relerr(ts.eigh_randomized(dev(A), 4, oversampling=4, n_iter=q, seed=1)[0], top)
        for q in (0, 3)
    ]
    assert e[1] < 1e-6, f"n_iter=3 error {e[1]:.3g}"
    assert e[1] <= e[0]


def _lowrank(n, tail=1e-3, dt=F64, seed=0):
    Q, _ = torch.linalg.qr(rand(n, n, seed=seed))
    lam = torch.cat(
        [
            torch.tensor([50.0, -40.0, 30.0, -20.0]),
            tail * torch.rand(n - 4, generator=g(seed + 1), dtype=F64),
        ]
    )
    A = (Q * lam) @ Q.T
    return (0.5 * (A + A.T)).to(dt), lam


# Preconditioners: the value of M(r) against the dense formula.


@pytest.mark.parametrize("block_size", [1, 5, 16, N, N + 3])
def test_block_jacobi_value(block_size):
    A, r = spd(N), rand(N, seed=6)
    Mb = torch.zeros_like(A)
    for s in range(0, N, block_size):
        e = min(s + block_size, N)
        Mb[s:e, s:e] = A[s:e, s:e]
    z = guarded(ts.block_jacobi_preconditioner(dev(A), block_size), dev(r))
    assert_close(z, torch.linalg.solve(Mb, r), tol_for(F64, 10), "z vs blockdiag(A)^-1 r")


@pytest.mark.parametrize("omega", [0.5, 1.0, 1.3, 1.9])
@pytest.mark.parametrize("dt", [torch.float32, F64])
def test_ssor_value(omega, dt):
    A, r = spd(N, dt), rand(N, dtype=dt, seed=6)
    A64, r64 = up(A), up(r)
    D = torch.diag(torch.diagonal(A64))
    L = torch.tril(A64, -1)
    M = (D + omega * L) @ torch.linalg.inv(D) @ (D + omega * L.T) / (omega * (2 - omega))
    z = guarded(ts.ssor_preconditioner(dev(A), omega), dev(r))
    assert_dtype(z, dt, "z")
    assert_close(z, torch.linalg.solve(M, r64), tol_for(dt, 10), "z vs SSOR formula")


@pytest.mark.parametrize("dt", [torch.float32, F64])
def test_jacobi_value(dt):
    A, r = spd(N, dt), rand(N, dtype=dt, seed=6)
    z = guarded(ts.jacobi_preconditioner(dev(A)), dev(r))
    assert_dtype(z, dt, "z")
    assert_close(z, up(r) / torch.diagonal(up(A)), tol_for(dt), "z vs r / diag(A)")


# Iterative solvers: the documented contract is the relative residual. Check
# the *true* residual of the returned x, that the reported residual is the
# true one, and x against a direct solve (cond ~ 10 for these matrices).


def _check_iterative(A, b, out, tol, maxiter):
    x, iters, res = out
    A64, b64 = up(A), up(b)
    xh = host(x).to(A64.dtype)
    bn = torch.linalg.norm(b64).item()
    true_res = (torch.linalg.norm(b64 - A64 @ xh) / bn).item() if bn else 0.0
    assert isinstance(iters, int) and 0 <= iters <= maxiter
    assert abs(res - true_res) <= 1e-6 * max(true_res, 1e-10) + 1e-12, (
        f"reported {res:.3g} != true {true_res:.3g}"
    )
    if res <= tol:
        x_ref = torch.linalg.solve(A64, b64)
        assert_close(xh, x_ref, max(20 * tol, 1e-12), "x vs torch.linalg.solve")
    return true_res


def _precond(kind, A):
    if kind is None:
        return None
    if kind == "jacobi":
        return ts.jacobi_preconditioner(A)
    if kind == "block5":
        return ts.block_jacobi_preconditioner(A, 5)
    if kind == "block16":
        return ts.block_jacobi_preconditioner(A)
    if kind == "ssor0.8":
        return ts.ssor_preconditioner(A, 0.8)
    if kind == "ssor1.5":
        return ts.ssor_preconditioner(A, 1.5)
    if kind == "tensor":
        return torch.diag(1.0 / torch.diagonal(A))
    raise ValueError(kind)


CG_CASES = [
    P(
        f"n{n}-tol{tol:g}-M{M}-{'op' if callable_A else 'dense'}{'-x0' if x0 else ''}",
        n,
        tol,
        M,
        callable_A,
        x0,
    )
    for n, tol, M, callable_A, x0 in [
        (1, 1e-6, None, False, False),
        (2, 1e-10, None, False, False),
        (127, 1e-6, None, False, False),
        (129, 1e-10, None, False, False),
        (1000, 1e-8, None, False, False),
        (129, 1e-10, "jacobi", False, False),
        (129, 1e-10, "block5", False, False),
        (129, 1e-10, "block16", False, False),
        (129, 1e-10, "ssor0.8", False, False),
        (129, 1e-10, "ssor1.5", False, False),
        (129, 1e-10, "tensor", False, False),
        (129, 1e-10, None, True, False),
        (129, 1e-10, "jacobi", True, True),
        (129, 1e-4, None, False, True),
        (1000, 1e-10, "jacobi", False, True),
    ]
]


@pytest.mark.parametrize("n,tol,M,callable_A,x0", CG_CASES)
def test_cg(n, tol, M, callable_A, x0):
    A, b = dev(spd(n, seed=n)), dev(rand(n, seed=n + 1))
    x0t = dev(rand(n, seed=n + 2)) if x0 else None
    Aop = (lambda v: A @ v) if callable_A else A
    out = guarded(
        lambda A_, b_: ts.cg(Aop, b_, x0=x0t, tol=tol, maxiter=1000, M=_precond(M, A)), A, b
    )
    assert _check_iterative(A, b, out, tol, 1000) <= tol * (1 + 1e-6)


@pytest.mark.parametrize("maxiter", [0, 1, 3])
def test_cg_maxiter(maxiter):
    A, b, x0 = spd(40), rand(40, seed=1), rand(40, seed=2)
    out = ts.cg(dev(A), dev(b), x0=dev(x0), tol=1e-14, maxiter=maxiter)
    _check_iterative(A, b, out, 1e-14, maxiter)
    if maxiter == 0:
        assert torch.equal(host(out[0]), x0)


GMRES_CASES = [
    P(
        f"n{n}-tol{tol:g}-restart{rs}-maxiter{mi}-{'op' if callable_A else 'dense'}{'-x0' if x0 else ''}",
        *c,
    )
    for c in [
        (1, 1e-6, 30, 100, False, False),
        (2, 1e-10, 30, 100, False, False),
        (127, 1e-6, 30, 100, False, False),
        (129, 1e-10, 30, 100, False, False),
        (1000, 1e-8, 30, 100, False, False),
        (129, 1e-10, 5, 200, False, False),
        (129, 1e-10, 1, 400, False, False),
        (129, 1e-12, 139, 2, False, False),
        (129, 1e-10, 30, 100, True, False),
        (129, 1e-10, 30, 100, False, True),
        (129, 1e-3, 10, 100, False, True),
    ]
    for n, tol, rs, mi, callable_A, x0 in [c]
]


@pytest.mark.parametrize("n,tol,restart,maxiter,callable_A,x0", GMRES_CASES)
def test_gmres(n, tol, restart, maxiter, callable_A, x0):
    A, b = dev(gen(n, seed=n)), dev(rand(n, seed=n + 1))
    x0t = dev(rand(n, seed=n + 2)) if x0 else None
    Aop = (lambda v: A @ v) if callable_A else A
    out = guarded(
        lambda A_, b_: ts.gmres(Aop, b_, x0=x0t, tol=tol, maxiter=maxiter, restart=restart), A, b
    )
    assert _check_iterative(A, b, out, tol, maxiter * max(restart, 1)) <= tol * (1 + 1e-6)


@pytest.mark.parametrize("maxiter,restart", [(1, 3), (2, 2)])
def test_gmres_truncated(maxiter, restart):
    A, b = gen(20), rand(20)
    out = ts.gmres(dev(A), dev(b), tol=1e-14, maxiter=maxiter, restart=restart)
    _check_iterative(A, b, out, 1e-14, maxiter * restart)


GMRES_IR_CASES = [
    dict(),
    dict(tol=1e-6),
    dict(tol=1e-13, max_outer=10),
    dict(max_outer=1),
    dict(inner_tol=1e-3, inner_maxiter=10, inner_restart=5),
    dict(inner_restart=1, inner_maxiter=200),
]


@pytest.mark.parametrize("n", [1, 129, 1000])
@pytest.mark.parametrize(
    "kw", GMRES_IR_CASES, ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()) or "default"
)
def test_gmres_ir(n, kw):
    if n == 1000 and kw:
        pytest.skip("1000 only with default kwargs (runtime)")
    A, b = gen(n, seed=n), rand(n, seed=n + 1)
    x, outer, res = guarded(lambda A, b: ts.gmres_ir(A, b, **kw), dev(A), dev(b))
    max_outer = kw.get("max_outer", 5)
    assert 0 <= outer <= max_outer
    xh = host(x)
    true_res = (torch.linalg.norm(b - A @ xh) / torch.linalg.norm(b)).item()
    assert abs(res - true_res) <= 1e-3 * max(true_res, 1e-15) + 1e-15, (
        f"reported {res:.3g} != true {true_res:.3g}"
    )
    tol = kw.get("tol", 1e-10)
    if outer < max_outer:
        assert true_res <= tol * 1.01
        assert_close(xh, torch.linalg.solve(A, b), 20 * tol, "x vs torch.linalg.solve")


ITER_SOLVERS = ("cg", "cg_jacobiM", "cg_blockjacobiM", "cg_ssorM", "gmres", "gmres_ir")


@pytest.mark.parametrize(
    "solver,dt", [P(f"{s}-{_dn(dt)}", s, dt) for s in ITER_SOLVERS for dt in DTYPES]
)
def test_iterative_dtype(solver, dt):
    mat = spd if solver.startswith("cg") else gen
    A, b = mat(N, dt), rand(N, dtype=dt, seed=9)
    fns = {
        "cg": lambda A, b: ts.cg(A, b, tol=1e-10, maxiter=400)[0],
        "cg_jacobiM": lambda A, b: ts.cg(
            A, b, tol=1e-10, maxiter=400, M=ts.jacobi_preconditioner(A)
        )[0],
        "cg_blockjacobiM": lambda A, b: ts.cg(
            A, b, tol=1e-10, maxiter=400, M=ts.block_jacobi_preconditioner(A, 5)
        )[0],
        "cg_ssorM": lambda A, b: ts.cg(
            A, b, tol=1e-10, maxiter=400, M=ts.ssor_preconditioner(A, 1.2)
        )[0],
        "gmres": lambda A, b: ts.gmres(A, b, tol=1e-12, maxiter=50, restart=N)[0],
        "gmres_ir": lambda A, b: ts.gmres_ir(A, b)[0],
    }
    tol = {torch.bfloat16: 6e-2, torch.float16: 1e-2}.get(dt, 1e-5)
    body = lambda: run(quiet(fns[solver]), (A, b), chk_solve(A, b, tol, dt))  # noqa: E731
    (_expect_raise_or(body) if dt.is_complex else body)()


# ---------------------------------------------------------------- axis 6: mixed dtype


MIXED = [
    P("solve-Af32-Bc64", "solve"),
    P("solve_spd-Af32-Bc64", "solve_spd"),
    P("eigh_generalized-Af32-Bc64", "eigh_generalized"),
]


@pytest.mark.parametrize("op", MIXED)
def test_mixed_real_complex(op):
    if op == "eigh_generalized":
        A, B = spd(N, torch.float32, seed=1), spd(N, torch.complex64, seed=2)

        def body():
            w, V = guarded(quiet(ts.eigh_generalized), dev(A), dev(B))
            Li = torch.linalg.inv(torch.linalg.cholesky(up(B)))
            assert_close(
                w,
                torch.linalg.eigvalsh(Li @ up(A).to(C128) @ Li.mH).to(real_of(w.dtype)),
                1e-3,
                "w",
            )

        _expect_raise_or(body)()
        return
    A = (gen if op == "solve" else spd)(N, torch.float32)
    B = rand(N, 2, dtype=torch.complex64, seed=3)
    X = guarded(quiet(getattr(ts, op)), dev(A), dev(B))
    assert X.is_complex(), f"imaginary part of B dropped: X dtype {X.dtype}"
    assert_close(X, torch.linalg.solve(up(A).to(C128), up(B)), 1e-4, "X")


def test_cg_mixed_real_complex():
    A, b = spd(N, torch.float32), rand(N, dtype=torch.complex64, seed=3)

    def body():
        x = guarded(quiet(lambda A, b: ts.cg(A, b, tol=1e-10)[0]), dev(A), dev(b))
        assert x.is_complex(), f"imaginary part of b dropped: x dtype {x.dtype}"
        assert_close(x, torch.linalg.solve(up(A).to(C128), up(b)), 1e-4, "x")

    _expect_raise_or(body)()


@pytest.mark.parametrize(
    "dt",
    [P("bfloat16", torch.bfloat16), P("float64", F64), P("complex128", C128)],
)
def test_eigh_randomized_dtype(dt):
    n = 64
    Q, _ = torch.linalg.qr(rand(n, n, dtype=ref_dtype(dt), seed=7))
    lam = torch.cat(
        [
            torch.tensor([50.0, -40.0, 30.0, -20.0]),
            1e-3 * torch.rand(n - 4, generator=g(8), dtype=F64),
        ]
    )
    A = ((Q * lam.to(Q.dtype)) @ Q.mH).to(dt)
    w, V = guarded(lambda A: ts.eigh_randomized(A, 4, seed=0), dev(A))
    assert_dtype(w, real_of(dt), "w")
    assert_dtype(V, dt, "V")
    assert_close(
        w, torch.sort(lam[:4]).values, {torch.bfloat16: 6e-2}.get(dt, 1e-5), "top-|λ| vs dense"
    )


# ---------------------------------------------------------------- axis 7: autograd (fp64 gradcheck)


def _sym(f):
    """Symmetrise the first input (gradcheck perturbs entries independently)."""
    return lambda A, *r: f(0.5 * (A + A.mT), *r)


_n = 5
_A_s, _A_g, _B_r, _b_r = spd(_n), gen(_n), rand(_n, 2, seed=3), rand(_n, seed=4)
_A_t = gen(3, m=6)
GRAD = [
    ("cholesky", _sym(ts.cholesky), (_A_s,), _sym(torch.linalg.cholesky)),
    (
        "cholesky_upper",
        _sym(lambda A: ts.cholesky(A, upper=True)),
        (_A_s,),
        _sym(lambda A: torch.linalg.cholesky(A, upper=True)),
    ),
    (
        "lu-LU",
        lambda A: (lambda P_, L, U: L @ U)(*ts.lu(A)),
        (_A_g,),
        lambda A: (lambda P_, L, U: L @ U)(*torch.linalg.lu(A)),
    ),
    ("qr-R", lambda A: ts.qr(A)[1], (_A_t,), lambda A: torch.linalg.qr(A)[1]),
    ("svd-s", lambda A: ts.svd(A)[1], (_A_t,), torch.linalg.svdvals),
    ("svd-USVh", lambda A: (lambda U, s, Vh: (U * s) @ Vh)(*ts.svd(A)), (_A_t,), lambda A: A * 1.0),
    ("solve", ts.solve, (_A_g, _B_r), torch.linalg.solve),
    ("solve_spd", _sym(ts.solve_spd), (_A_s, _B_r), _sym(torch.linalg.solve)),
    (
        "solve_spd_ir",
        _sym(lambda A, B: ts.solve_spd(A, B, iterative_refinement=True)),
        (_A_s, _B_r),
        _sym(torch.linalg.solve),
    ),
    ("inv_spd", _sym(ts.inv_spd), (_A_s,), _sym(torch.linalg.inv)),
    ("pinv", ts.pinv, (_A_t,), torch.linalg.pinv),
    ("inv_sqrt_spd", _sym(ts.inv_sqrt_spd), (_A_s,), _sym(inv_sqrt_ref)),
    ("inv_sqrt_spd_ns", _sym(lambda A: ns(A, tol=1e-14)), (_A_s,), _sym(inv_sqrt_ref)),
    ("eigh-w", _sym(lambda A: ts.eigh(A)[0]), (_A_s,), _sym(torch.linalg.eigvalsh)),
    (
        "eigh-VwVt",
        _sym(lambda A: (lambda w, V: (V * w) @ V.mT)(*ts.eigh(A))),
        (_A_s,),
        _sym(lambda A: A * 1.0),
    ),
    (
        "eigh_jacobi-w",
        _sym(lambda A: ts.eigh(A, method="jacobi")[0]),
        (_A_s,),
        _sym(torch.linalg.eigvalsh),
    ),
    (
        "eigh_generalized-w",
        lambda A, B: ts.eigh_generalized(0.5 * (A + A.mT), 0.5 * (B + B.mT))[0],
        (herm(_n), spd(_n, seed=8)),
        lambda A, B: torch.linalg.eigvalsh(
            (lambda Li: Li @ (0.5 * (A + A.mT)) @ Li.mT)(
                torch.linalg.inv(torch.linalg.cholesky(0.5 * (B + B.mT)))
            )
        ),
    ),
    ("schur-T", lambda A: ts.schur(A)[1], (spd(_n),), None),
    (
        "cg-b",
        lambda b: ts.cg(_A_s, b, tol=1e-14)[0],
        (_b_r,),
        lambda b: torch.linalg.solve(_A_s, b),
    ),
    ("cg-A", _sym(lambda A, b: ts.cg(A, b, tol=1e-14)[0]), (_A_s, _b_r), _sym(torch.linalg.solve)),
    (
        "gmres-Ab",
        lambda A, b: ts.gmres(A, b, tol=1e-14, restart=_n)[0],
        (_A_g, _b_r),
        torch.linalg.solve,
    ),
    ("gmres_ir-Ab", lambda A, b: ts.gmres_ir(A, b, tol=1e-14)[0], (_A_g, _b_r), torch.linalg.solve),
]
KNOWN["schur-T"] = _decide("schur silently detaches (no gradient)")
KNOWN["cg-A"] = _decide("cg gradient w.r.t. A is wrong")
KNOWN["cg-A-gradgrad"] = _decide("cg gradient w.r.t. A is wrong")
KNOWN["gmres-Ab"] = _decide("gmres backward raises (in-place op on a saved tensor)")
KNOWN["gmres-Ab-gradgrad"] = _decide("gmres backward raises (in-place op on a saved tensor)")


def _skip_grad_off_cpu():
    if DEV.type != "cpu":
        pytest.skip("fp64 gradcheck runs on CPU inputs only")


@pytest.mark.parametrize("name,f,inputs,ref", [P(c[0], *c) for c in GRAD])
def test_gradcheck(name, f, inputs, ref):
    _skip_grad_off_cpu()
    ins = [dev(x.clone()).requires_grad_(True) for x in inputs]
    out = f(*ins)
    assert out.requires_grad, "output detached from the inputs"
    assert torch.autograd.gradcheck(f, ins, eps=1e-6, atol=1e-5, rtol=1e-4)
    if ref is not None:
        ins_r = [x.clone().requires_grad_(True) for x in inputs]
        out_r = ref(*ins_r)
        assert_close(out, out_r.detach(), 1e-8, "forward vs torch")
        gout = rand(*out_r.shape, seed=77)
        gr = torch.autograd.grad(out, ins, dev(gout), allow_unused=True)
        grr = torch.autograd.grad(out_r, ins_r, gout, allow_unused=True)
        for a, b in zip(gr, grr, strict=True):
            assert a is not None, "gradient is None (input disconnected)"
            assert_close(a, b, 1e-6, "grad vs torch")


@pytest.mark.parametrize(
    "name,f,inputs", [P(f"{c[0]}-gradgrad", *c[:3]) for c in GRAD if c[3] is not None]
)
def test_gradgradcheck(name, f, inputs):
    _skip_grad_off_cpu()
    ins = [dev(x.clone()).requires_grad_(True) for x in inputs]
    assert torch.autograd.gradgradcheck(f, ins, eps=1e-6, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------- axis 8: aliasing / mutation


@pytest.mark.parametrize("op", list(OPS))
def test_no_mutation_no_alias(op):
    fn, build, _ = OPS[op]
    guarded(quiet(fn), *[dev(x) for x in build(N, F64)])


def test_eigh_n1_output_does_not_alias():
    A = dev(torch.tensor([[3.0]], dtype=F64))
    w, V = guarded(ts.eigh, A)
    w.add_(1.0)
    V.add_(1.0)
    assert host(A).item() == 3.0


@pytest.mark.parametrize("solver", ["cg", "gmres"])
def test_x0_not_mutated(solver):
    A = spd(N) if solver == "cg" else gen(N)
    b, x0 = rand(N), rand(N, seed=2)
    guarded(lambda A, b, x0: getattr(ts, solver)(A, b, x0=x0), dev(A), dev(b), dev(x0))


@pytest.mark.parametrize("solver", ["cg", "gmres"])
def test_zero_rhs_returns_zero(solver):
    A = spd(N) if solver == "cg" else gen(N)
    b, x0 = torch.zeros(N, dtype=F64), rand(N, seed=2)
    x, _, res = guarded(lambda A, b, x0: getattr(ts, solver)(A, b, x0=x0), dev(A), dev(b), dev(x0))
    assert_close(x, torch.zeros(N, dtype=F64), 1e-12, "x for b = 0")
    assert res == 0.0


@pytest.mark.parametrize("solver", ["cg", "gmres"])
@pytest.mark.parametrize("scale", [1e-16, 1e16])
def test_scaled_system(solver, scale):
    A = scale * (spd(8) if solver == "cg" else gen(8))
    b = scale * rand(8)
    fn = (
        (lambda A, b: ts.cg(A, b, tol=1e-10))
        if solver == "cg"
        else (lambda A, b: ts.gmres(A, b, tol=1e-12, restart=8))
    )
    out = guarded(fn, dev(A), dev(b))
    _check_iterative(A, b, out, 1e-10, 1000)
    assert_close(out[0], torch.linalg.solve(A, b), 1e-8, "x vs torch.linalg.solve")


def test_cg_exact_x0():
    A, b = spd(6), rand(6)
    x0 = torch.linalg.solve(A, b)
    out = ts.cg(dev(A), dev(b), x0=dev(x0))
    assert_close(out[0], x0, 1e-9, "x")


@pytest.mark.parametrize("pname", ["jacobi", "block_jacobi", "ssor"])
def test_preconditioner_captures_copy(pname):
    build = {
        "jacobi": ts.jacobi_preconditioner,
        "block_jacobi": lambda A: ts.block_jacobi_preconditioner(A, 4),
        "ssor": lambda A: ts.ssor_preconditioner(A, 1.3),
    }[pname]
    A, r = dev(spd(N)), dev(rand(N, seed=6))
    M = guarded(build, A)
    z0 = host(M(r))
    A.mul_(7.0)  # mutating A after building must not change M
    assert torch.equal(host(guarded(M, r)), z0)


# ---------------------------------------------------------------- axis 9: batch / broadcast

BATCH_IDS = []
for _op in OPS:
    BATCH_IDS.append((f"{_op}-batch2x3", _op, (2, 3)))
    BATCH_IDS.append((f"{_op}-batch1", _op, (1,)))
    BATCH_IDS.append(
        (f"{_op}-batch{N}", _op, (N,))
    )  # batch == n: a .T over all dims keeps the shape
KNOWN["solve_spd_vec-batch2x3"] = pytest.mark.xfail(
    strict=False, reason="trnsci/trnsolver#79 (loud): solve_spd rejects a batch of vectors B (*, n)"
)
KNOWN["solve_spd_vec-batch1"] = KNOWN["solve_spd_vec-batch2x3"]
KNOWN[f"solve_spd_vec-batch{N}"] = KNOWN["solve_spd_vec-batch2x3"]


@pytest.mark.parametrize("op,batch", [P(i, op, b) for i, op, b in BATCH_IDS])
def test_batch(op, batch):
    fn, build, chk = OPS[op]
    inputs = build(N, F64, batch)
    body = lambda: run(quiet(fn), inputs, chk(inputs, tol_for(F64), F64))  # noqa: E731
    (_expect_raise_or(body) if op in UNBATCHED else body)()


_Ab, _As, _Asb = gen(N, batch=(3,)), gen(N), spd(N, batch=(3,))
BCAST = [
    ("solve-A2d-B3d", ts.solve, torch.linalg.solve, (_As, rand(3, N, 2))),
    ("solve-A3d-B2d", ts.solve, torch.linalg.solve, (_Ab, rand(N, 2))),
    ("solve-A3d-Bbatchvec", ts.solve, torch.linalg.solve, (_Ab, rand(3, N))),
    ("solve-A3d-B1d", ts.solve, torch.linalg.solve, (_Ab, rand(N))),
    ("solve_spd-A2d-B3d", ts.solve_spd, torch.linalg.solve, (spd(N), rand(3, N, 2))),
    ("solve_spd-A3d-B2d", ts.solve_spd, torch.linalg.solve, (_Asb, rand(N, 2))),
    ("solve_spd-A3d-Bbatchvec", ts.solve_spd, torch.linalg.solve, (_Asb, rand(3, N))),
    ("solve_spd-A3d-B1d", ts.solve_spd, torch.linalg.solve, (_Asb, rand(N))),
    (
        "solve_spd_ir-A3d-B3d",
        lambda A, B: ts.solve_spd(A, B, iterative_refinement=True),
        torch.linalg.solve,
        (_Asb, rand(3, N, 2)),
    ),
    (
        "eigh-empty_batch",
        lambda A: ts.eigh(A)[0],
        torch.linalg.eigvalsh,
        (torch.zeros(0, N, N, dtype=F64),),
    ),
    (
        "cholesky-empty_batch",
        ts.cholesky,
        torch.linalg.cholesky,
        (torch.zeros(0, N, N, dtype=F64),),
    ),
]
KNOWN["solve_spd-A3d-Bbatchvec"] = KNOWN["solve_spd_vec-batch2x3"]


@pytest.mark.parametrize("f,ref,inputs", [P(c[0], *c[1:]) for c in BCAST])
def test_broadcast(f, ref, inputs):
    out = guarded(quiet(f), *(dev(x) for x in inputs))
    assert_close(out, ref(*[up(x) for x in inputs]), tol_for(F64), "vs torch.linalg")


@pytest.mark.parametrize("kind", ["empty_batch", "batch1"])
def test_eigh_jacobi_batch_edges(kind):
    A = torch.zeros(0, N, N, dtype=F64) if kind == "empty_batch" else herm(N, batch=(1,))
    w = guarded(lambda A: ts.eigh(A, method="jacobi")[0], dev(A))
    assert_close(w, torch.linalg.eigvalsh(A), max(tol_for(F64), 1e-6), "w")


def test_lu_convention():
    # A == P @ L @ U (torch convention); a 3-cycle pivot tells P from Pᵀ.
    A = torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 10.0], [1.0, 2.0, 3.0]], dtype=F64)
    Pm, L, U = host(ts.lu(dev(A)))
    assert not torch.equal(Pm, Pm.T)
    torch.testing.assert_close(Pm @ L @ U, A)
    Pt, Lt, Ut = torch.linalg.lu(A)
    torch.testing.assert_close(Pm, Pt)


# ---------------------------------------------------------------- axis 10: determinism / seeding


def test_eigh_randomized_same_seed():
    A, _ = _lowrank(48)
    a = host(ts.eigh_randomized(dev(A), 3, seed=11, n_iter=1))
    b = host(ts.eigh_randomized(dev(A), 3, seed=11, n_iter=1))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_eigh_randomized_seeds_differ_answer_same():
    A, _ = _lowrank(48)
    a = host(ts.eigh_randomized(dev(A), 3, seed=1))
    b = host(ts.eigh_randomized(dev(A), 3, seed=2))
    assert_close(a[0], b[0], 1e-8, "eigenvalues across seeds")
    assert not torch.equal(a[1], b[1]), "different seeds gave bit-identical vectors (seed ignored?)"


def test_eigh_randomized_does_not_touch_global_rng():
    A, _ = _lowrank(48)
    torch.manual_seed(123)
    s0 = torch.get_rng_state()
    ts.eigh_randomized(dev(A), 3, seed=5)
    assert torch.equal(s0, torch.get_rng_state())


def test_eigh_randomized_f32_vs_f64():
    A, _ = _lowrank(48)
    a = host(ts.eigh_randomized(dev(A), 3, seed=11))
    b = host(ts.eigh_randomized(dev(A.float()), 3, seed=11))
    assert_dtype(b[0], torch.float32, "w")
    assert_close(b[0], a[0], 1e-4, "f32 vs f64 eigenvalues")


def test_eigh_jacobi_batch_order_independent():
    A = spd(N, batch=(4,))
    w1, _ = host(ts.eigh(dev(A), method="jacobi"))
    w2, _ = host(ts.eigh(dev(A.flip(0)), method="jacobi"))
    assert_close(w2.flip(0), w1, 1e-12, "w across batch order")
    w3, _ = host(ts.eigh(dev(A[2:3]), method="jacobi"))
    assert_close(w3[0], w1[2], 1e-12, "w alone vs in batch")


@pytest.mark.parametrize("method", ["auto", "jacobi"])
def test_eigh_repeatable(method):
    A = herm(N)
    a = host(ts.eigh(dev(A), method=method))
    b = host(ts.eigh(dev(A), method=method))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
