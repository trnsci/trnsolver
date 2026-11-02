"""No NKI kernel is reached with an operand dtype it does not take.

Every trnsolver kernel computes in FP32 (``matmul_fp32out``: BF16 in, FP32
out). float64, integer, bool and complex operands used to reach them as is
and came back as garbage, non-finite, float32 for a float64 caller, or a
compile error on Trn2 (#83, #84; trnsci/trnsci#35).

CPU-only. A stand-in ``torch_neuronx`` is installed whose ``nki_op`` returns
a stub per ``trnsolver::`` op: it records every launch and the operand
dtypes, and computes a cheap torch reference so the host drivers terminate.
``_native`` is imported fresh under it. The XLA path and the simulator path
get stand-in kernels with the same check. Every public API that can reach a
kernel is then driven with each dtype, backend ``nki`` and ``auto``, with
and without ``TRNSOLVER_REQUIRE_NKI``, and must:

- never hand a stub an operand outside the kernel's allowed set (and never
  trip the launch-boundary TypeError, which is the backstop);
- return the dtypes and shapes the pytorch backend returns for the same
  input, or raise where the pytorch backend raises (under REQUIRE_NKI a
  NotImplementedError is also allowed).

The float64 policy (trnsolver#89) is tested on every kernel entry: under
backend ``auto`` float64 computes in torch at float64 accuracy with no
launch; under backend ``nki`` the FP32 kernels run with FP32 operands, the
result is float64 and one UserWarning is emitted per op; under
``TRNSOLVER_REQUIRE_NKI=1`` it raises NotImplementedError.

float64 already on the ``neuron`` device (trnsolver#89 amendment): under
backend ``auto`` every public op copies it to the host, computes there at
float64 accuracy and returns the result on the original device (one
UserWarning per op); ``TRNSOLVER_FP64_ON_DEVICE=1`` keeps the device path.
Here the stand-in device is CPU, so the host and return transfers are spied
on rather than observed as device changes.
"""

import sys
import types
import warnings

import numpy as np
import pytest
import torch

import trnsolver
import trnsolver.eigen as eigen
import trnsolver.nki
import trnsolver.subspace as tsub
from trnsolver.nki import dispatch
from trnsolver.nki import jacobi as nj
from trnsolver.nki import subspace as ns

# The stand-in neuron device is CPU, so float64 calls under backend "auto"
# take the host path and warn; only the tests below that check it look.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*float64 on the neuron device is computed on the host:UserWarning"
)

F32 = (torch.float32,)
# The test's own copy of the contract (the library's table must equal it).
ALLOWED = {
    "matvec": F32,
    "rank2_update": F32,
    "matmul": F32,
    "matmul_fp32out": (torch.bfloat16,),
    "apply_reflectors": F32,
    "jacobi_sweeps": F32,
    "jacobi_block": F32,
    "subspace_filter": F32,
    "subspace_step": F32,
}
SIM_ALLOWED = {**ALLOWED, "matmul_fp32out": F32}

DTYPES = [
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
]


# ── torch references for the kernels (cheap; enough for the drivers to run) ──


def _apply_reflectors(V_refs, Q):
    Q = Q.clone()
    for k in range(V_refs.shape[1] - 1, -1, -1):
        v = V_refs[:, k]
        Q = Q - 2.0 * torch.outer(v, v @ Q)
    return Q.T.contiguous()


def _filter(F, X, row, emit_x):
    out = tsub.filter_reference(F, X, row)
    return torch.cat([X, out], 0) if emit_x else out


def _step(F, prev, qtab):
    kb = prev.shape[1]
    return tsub.step_reference(F, prev, qtab[: 2 * kb, :kb], qtab[2 * kb])


REF = {
    "matvec": lambda A, v: A @ v,
    "rank2_update": lambda A, u, v: A - u @ v.T - v @ u.T,
    "matmul": lambda A, V: A @ V,
    "matmul_fp32out": lambda A, V: A.float() @ V.float(),
    "apply_reflectors": _apply_reflectors,
    # No rotation: the Jacobi driver runs its sweep cap and returns.
    "jacobi_sweeps": lambda Z, V, consts, sched, onezero: torch.stack([Z, V]),
    "jacobi_block": lambda Z, V, consts, sched, onezero: torch.stack([Z, V]),
    "subspace_filter": lambda F, X, tab, sched: _filter(F, X, tab[0], sched.shape[1] == 2),
    "subspace_step": lambda F, prev, qtab, sched: _step(F, prev, qtab),
}


class Recorder:
    def __init__(self):
        self.launches = []  # (path, op, dtypes)
        self.violations = []  # (path, op, dtypes)
        self.rejected = []  # TypeErrors raised by the launch-boundary check

    def check(self, path, op, args, table):
        dts = tuple(a.dtype for a in args if isinstance(a, torch.Tensor))
        self.launches.append((path, op, dts))
        if any(d not in table[op] for d in dts):
            self.violations.append((path, op, dts))


def _stub(rec, op):
    def stub(*args):
        rec.check("native", op, args, ALLOWED)
        return REF[op](*args)

    stub.__name__ = op
    return stub


def _fake_native_torch_neuronx(rec, registered):
    mod = types.ModuleType("torch_neuronx")

    def nki_op(name, fn=None, mutates_args={}, fake_fn=None):  # noqa: B006
        short = name.split("::", 1)[1]

        def dec(f):
            registered[short] = f
            return _stub(rec, short)

        return dec if fn is None else dec(fn)

    mod.nki_op = nki_op
    mod.device_count = lambda: 1
    return mod


class FakeKernel:
    """Stand-in `@nki.jit` kernel for the XLA path and `nki.simulate`."""

    def __init__(self, rec, op):
        self.rec, self.op = rec, op

    def __call__(self, *args):  # XLA path: torch tensors
        self.rec.check("xla", self.op, args, ALLOWED)
        return REF[self.op](*args)

    def simulate(self, *args):  # simulator: numpy arrays
        targs = [torch.from_numpy(a) if isinstance(a, np.ndarray) else a for a in args]
        self.rec.check("sim", self.op, targs, SIM_ALLOWED)
        return REF[self.op](*targs).numpy()


KERNEL_ATTRS = [
    (dispatch, "matvec_kernel", "matvec"),
    (dispatch, "matvec_kernel_sim", "matvec"),
    (dispatch, "rank2_update_kernel", "rank2_update"),
    (dispatch, "matmul_kernel", "matmul"),
    (dispatch, "matmul_kernel_sim", "matmul"),
    (dispatch, "matmul_kernel_fp32out", "matmul_fp32out"),
    (dispatch, "matmul_kernel_fp32out_sim", "matmul_fp32out"),
    (dispatch, "apply_reflectors_kernel", "apply_reflectors"),
    (dispatch, "apply_reflectors_kernel_sim", "apply_reflectors"),
    (nj, "jacobi_sweeps_kernel", "jacobi_sweeps"),
    (nj, "jacobi_block_kernel", "jacobi_block"),
    (ns, "subspace_filter_kernel", "subspace_filter"),
    (ns, "subspace_step_kernel", "subspace_step"),
]


def _install(monkeypatch, path, backend, require):
    """Force kernel dispatch on CPU along ``path`` (native / xla / sim)."""
    rec = Recorder()
    registered = {}
    for mod in (dispatch, eigen, tsub, ns):
        monkeypatch.setattr(mod, "HAS_NKI", True)
    for mod in (dispatch, eigen, tsub, trnsolver.nki):
        monkeypatch.setattr(mod, "_REQUIRE_NKI", require)
    monkeypatch.setattr(dispatch, "_backend", backend)
    monkeypatch.setattr(dispatch, "_WARNED_FP32", set())
    monkeypatch.setattr(dispatch, "_fp64_host_warned", set())
    monkeypatch.setattr(dispatch, "_FP64_ON_DEVICE", False)
    monkeypatch.setattr(dispatch, "_native_ops", None)
    monkeypatch.setattr(dispatch, "_USE_SIMULATOR", path == "sim")
    # The "neuron" device is CPU here, so host inputs stand in for device ones.
    monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
    monkeypatch.setattr(eigen, "_NATIVE_DEVICE_TYPE", "cpu")
    for mod, attr, op in KERNEL_ATTRS:
        monkeypatch.setattr(mod, attr, FakeKernel(rec, op), raising=False)
    monkeypatch.delitem(sys.modules, "trnsolver.nki._native", raising=False)
    monkeypatch.delattr(trnsolver.nki, "_native", raising=False)
    monkeypatch.setattr(nj, "_DEVICE_CONSTS", {})
    monkeypatch.setattr(ns, "_sched", lambda m, device, cols=1: torch.zeros((m, cols)))

    if path == "native":
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(rec, registered)
        )
        monkeypatch.setitem(sys.modules, "torch_xla", None)  # any import raises
    elif path == "xla":
        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        xla = types.ModuleType("torch_xla")
        xla.device = lambda: torch.device("cpu")
        xla.sync = lambda: None
        monkeypatch.setitem(sys.modules, "torch_xla", xla)
        monkeypatch.setitem(sys.modules, "torch_neuronx", types.ModuleType("torch_neuronx"))
    else:
        monkeypatch.setattr(dispatch, "_dispatch_path", "cpu")
        fake_nki = types.ModuleType("nki")
        fake_nki.simulate = lambda kernel: kernel.simulate
        monkeypatch.setitem(sys.modules, "nki", fake_nki)

    real_check = getattr(dispatch, "_check_kernel_dtypes", None)
    if real_check is not None:  # (absent before the guard: the test still runs)

        def check(op_name, tensors, table=None):
            try:
                real_check(op_name, tensors, table)
            except TypeError as e:
                rec.rejected.append(str(e))
                raise

        monkeypatch.setattr(dispatch, "_check_kernel_dtypes", check)
    return rec, registered


# ── inputs and public APIs ───────────────────────────────────────────────────


def _sym(n, dtype, batch=(), seed=0):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(*batch, n, n, generator=g, dtype=torch.float64)
    A = M + M.transpose(-1, -2)
    return _as(A, dtype)


def _spd(n, dtype, seed=1):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(n, n, generator=g, dtype=torch.float64)
    return _as(M @ M.T + n * torch.eye(n, dtype=torch.float64), dtype)


def _as(A, dtype):
    if dtype == torch.bool:
        return A > 0
    if not dtype.is_floating_point and not dtype.is_complex:
        return torch.round(3 * A).to(dtype)
    if dtype.is_complex:
        H = A + 1j * (A - A.transpose(-1, -2)).to(torch.complex128)
        return H.to(dtype)
    return A.to(dtype)


def _occupied(F):
    state = trnsolver.occupied_subspace(F, 3, buffer=3).state
    # Two passes with no early stop: the second runs the fold (step) kernel.
    return trnsolver.occupied_subspace(
        F, 3, guess=state, buffer=3, passes=2, tol=None, rtol=None, refresh=None
    )


def _env(**env):
    """APIs run with extra environment variables (set in `_call`)."""
    return env


# (name, env, fn(dtype) -> result). The env variants force the kernel paths
# that the defaults cut over to aten on a native host.
APIS = [
    ("eigh", {}, lambda dt: trnsolver.eigh(_sym(6, dt))),
    ("eigh[min_n=0]", _env(TRNSOLVER_EIGH_NKI_MIN_N="0"), lambda dt: trnsolver.eigh(_sym(6, dt))),
    ("eigh[n=1]", _env(TRNSOLVER_EIGH_NKI_MIN_N="0"), lambda dt: trnsolver.eigh(_sym(1, dt))),
    ("eigh[jacobi]", {}, lambda dt: trnsolver.eigh(_sym(6, dt), method="jacobi")),
    (
        "eigh[jacobi batch]",
        {},
        lambda dt: trnsolver.eigh(_sym(6, dt, batch=(3,)), method="jacobi"),
    ),
    (
        "eigh[jacobi block]",
        {},
        lambda dt: trnsolver.eigh(_sym(40, dt, batch=(2,)), method="jacobi"),
    ),
    (
        "eigh[auto jacobi region]",
        _env(TRNSOLVER_EIGH_JACOBI_AUTO="1-128:1"),
        lambda dt: trnsolver.eigh(_sym(6, dt, batch=(2,))),
    ),
    (
        "eigh[env method=jacobi]",
        _env(TRNSOLVER_EIGH_METHOD="jacobi"),
        lambda dt: trnsolver.eigh(_sym(6, dt)),
    ),
    (
        "eigh_generalized",
        _env(TRNSOLVER_EIGH_NKI_MIN_N="0"),
        lambda dt: trnsolver.eigh_generalized(_sym(6, dt), _spd(6, dt)),
    ),
    (
        "eigh_randomized",
        _env(TRNSOLVER_EIGH_NKI_MIN_N="0"),
        lambda dt: trnsolver.eigh_randomized(_sym(16, dt), 2, oversampling=3, seed=0),
    ),
    (
        "eigh_randomized[n_iter=1]",
        {},
        lambda dt: trnsolver.eigh_randomized(_sym(16, dt), 2, oversampling=3, n_iter=1, seed=0),
    ),
    ("occupied_subspace", {}, lambda dt: _occupied(_sym(12, dt))),
    (
        "eigh_generalized[B float64]",
        _env(TRNSOLVER_EIGH_NKI_MIN_N="0"),
        lambda dt: trnsolver.eigh_generalized(_sym(6, dt), _spd(6, torch.float64)),
    ),
    (
        "eigh_generalized[B float32]",
        _env(TRNSOLVER_EIGH_NKI_MIN_N="0"),
        lambda dt: trnsolver.eigh_generalized(_sym(6, dt), _spd(6, torch.float32)),
    ),
    ("cholesky", {}, lambda dt: trnsolver.cholesky(_spd(5, dt))),
    ("lu", {}, lambda dt: trnsolver.lu(_spd(5, dt))),
    ("qr", {}, lambda dt: trnsolver.qr(_spd(5, dt))),
    ("solve", {}, lambda dt: trnsolver.solve(_spd(5, dt), _as(torch.ones(5, 1), dt))),
    ("solve_spd", {}, lambda dt: trnsolver.solve_spd(_spd(5, dt), _as(torch.ones(5, 1), dt))),
    ("inv_sqrt_spd", {}, lambda dt: trnsolver.inv_sqrt_spd(_spd(5, dt))),
    ("cg", {}, lambda dt: trnsolver.cg(_spd(5, dt), _as(torch.ones(5), dt), maxiter=20)),
    ("gmres", {}, lambda dt: trnsolver.gmres(_spd(5, dt), _as(torch.ones(5), dt), maxiter=20)),
    ("gmres_ir", {}, lambda dt: trnsolver.gmres_ir(_spd(5, dt), _as(torch.ones(5), dt))),
]


def _signature(out):
    """Dtypes and shapes of every tensor in a (nested) result."""
    if isinstance(out, torch.Tensor):
        return [(out.dtype, tuple(out.shape))]
    if isinstance(out, (tuple, list)):
        return [s for o in out for s in _signature(o)]
    return []


def _call(monkeypatch, env, fn, dt):
    with monkeypatch.context() as m:
        for k, v in env.items():
            m.setenv(k, v)
        try:
            return _signature(fn(dt)), None
        except Exception as e:  # noqa: BLE001 - classified by the caller
            return None, e


def _pytorch_reference(monkeypatch, env, fn, dt):
    with monkeypatch.context() as m:
        m.setattr(dispatch, "_backend", "pytorch")
        for mod in (dispatch, eigen, tsub):
            m.setattr(mod, "_REQUIRE_NKI", False)
        return _call(m, env, fn, dt)


# ── the contract ─────────────────────────────────────────────────────────────


def test_table_matches_contract():
    assert dispatch.KERNEL_DTYPES == ALLOWED
    assert dispatch.SIM_KERNEL_DTYPES == SIM_ALLOWED


def test_every_registered_op_has_a_table_entry(monkeypatch):
    _, registered = _install(monkeypatch, "native", "nki", False)
    ops = dispatch._get_native_ops()
    assert set(registered) == set(ops) == set(dispatch.KERNEL_DTYPES)


@pytest.mark.parametrize("path", ["native", "xla"])
def test_launch_refuses_unlisted_dtypes(monkeypatch, path):
    _install(monkeypatch, path, "nki", False)
    for op, allowed in ALLOWED.items():
        bad = torch.float64 if allowed == F32 else torch.float32
        with pytest.raises(TypeError, match=op):
            dispatch._launch(dispatch.matvec_kernel, op, torch.ones(2, 2, dtype=bad))
    with pytest.raises(TypeError, match="no entry"):
        dispatch._launch(dispatch.matvec_kernel, "not_an_op", torch.ones(2, 2))


def test_simulate_refuses_unlisted_dtypes(monkeypatch):
    _install(monkeypatch, "sim", "nki", False)
    for dt in (torch.float64, torch.int32, torch.bool, torch.complex64, torch.bfloat16):
        with pytest.raises(TypeError, match="matvec"):
            dispatch._simulate(dispatch.matvec_kernel, "matvec", torch.ones(2, 2, dtype=dt))


def test_every_kernel_is_reached(monkeypatch):
    """The drivers below do reach each op, so the dtype checks are not vacuous."""
    rec, _ = _install(monkeypatch, "native", "nki", True)
    for _name, env, fn in APIS:
        _call(monkeypatch, env, fn, torch.float32)
    _call(monkeypatch, {}, APIS[9][2], torch.bfloat16)  # eigh_randomized, BF16 sketch
    assert APIS[9][0] == "eigh_randomized"
    assert {op for _, op, _ in rec.launches} == set(ALLOWED)
    assert not rec.violations and not rec.rejected


@pytest.mark.parametrize("require", [False, True], ids=["default", "require_nki"])
@pytest.mark.parametrize("backend", ["nki", "auto"])
@pytest.mark.parametrize("path", ["native", "xla", "sim"])
@pytest.mark.parametrize("dt", DTYPES, ids=[str(d).removeprefix("torch.") for d in DTYPES])
def test_no_unsupported_dtype_reaches_a_kernel(monkeypatch, path, backend, require, dt):
    refs = [_pytorch_reference(monkeypatch, env, fn, dt) for _, env, fn in APIS]
    rec, _ = _install(monkeypatch, path, backend, require)
    problems = []
    for (name, env, fn), (ref, ref_err) in zip(APIS, refs, strict=True):
        got, err = _call(monkeypatch, env, fn, dt)
        if ref_err is not None:
            continue  # the pytorch backend raises too; any error is fine
        if err is not None:
            if require and isinstance(err, NotImplementedError):
                continue
            problems.append(f"{name}: raised {type(err).__name__}: {err}")
        elif got != ref:
            problems.append(f"{name}: result {got} != pytorch backend {ref}")
    assert not rec.violations, f"unsupported operand dtypes reached kernels: {rec.violations}"
    assert not rec.rejected, f"a wrapper relied on the launch-boundary check: {rec.rejected}"
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("dt", [torch.int32, torch.bool])
def test_require_nki_refuses_non_float_eigh(monkeypatch, dt):
    _install(monkeypatch, "native", "nki", True)
    with pytest.raises(NotImplementedError, match="TRNSOLVER_REQUIRE_NKI"):
        trnsolver.eigh(_sym(6, dt))


# ── the float64 policy (trnsolver#89, trnsci/trnsci#35) ──────────────────────
#
# Where an FP32 input would run the kernels:
#   backend "auto"          -> torch at float64 accuracy, no kernel launch;
#   backend "nki"           -> FP32 kernels, float64 result, one UserWarning per op;
#   TRNSOLVER_REQUIRE_NKI=1 -> NotImplementedError (over either backend).

# (name, env, fn(dtype) -> (w, V) or SubspaceResult, kernel ops it reaches,
#  warning op name, dispatch paths with a kernel)
FP64_KERNEL_APIS = [
    (
        "eigh[householder]",
        _env(TRNSOLVER_EIGH_NKI_MIN_N="0"),
        lambda dt: trnsolver.eigh(_sym(6, dt)),
        {"matvec", "rank2_update"},
        "eigh",
        ("native", "xla", "sim"),
    ),
    (
        "eigh_generalized",
        _env(TRNSOLVER_EIGH_NKI_MIN_N="0"),
        lambda dt: trnsolver.eigh_generalized(_sym(6, dt), _spd(6, dt)),
        {"matvec", "rank2_update"},
        "eigh",
        ("native", "xla", "sim"),
    ),
    (
        "eigh_randomized",
        {},
        lambda dt: trnsolver.eigh_randomized(_sym(16, dt), 2, oversampling=3, n_iter=1, seed=0),
        {"matmul"},
        "eigh_randomized",
        ("native", "xla", "sim"),
    ),
    (
        "eigh[jacobi]",
        {},
        lambda dt: trnsolver.eigh(_sym(6, dt, batch=(3,)), method="jacobi"),
        {"jacobi_sweeps"},
        "eigh(method='jacobi')",
        ("native", "sim"),
    ),
    (
        "eigh[jacobi block]",
        {},
        lambda dt: trnsolver.eigh(_sym(40, dt, batch=(2,)), method="jacobi"),
        {"jacobi_block"},
        "eigh(method='jacobi')",
        ("native", "sim"),
    ),
    (
        "eigh[auto jacobi region]",
        _env(TRNSOLVER_EIGH_JACOBI_AUTO="1-128:1"),
        lambda dt: trnsolver.eigh(_sym(6, dt, batch=(2,))),
        {"jacobi_sweeps"},
        "eigh(method='jacobi')",
        ("native",),  # the region is for the native neuron device only
    ),
    (
        "occupied_subspace",
        {},
        lambda dt: _occupied(_sym(12, dt)),
        {"subspace_filter", "subspace_step"},
        "occupied_subspace",
        ("native", "sim"),
    ),
]
FP64_CASES = [
    pytest.param(path, name, env, fn, ops, what, id=f"{name}-{path}")
    for name, env, fn, ops, what, paths in FP64_KERNEL_APIS
    for path in paths
]


def _eigvals(out):
    return out.eigenvalues if hasattr(out, "eigenvalues") else out[0]


def _run(monkeypatch, env, fn, dt):
    with monkeypatch.context() as m:
        for k, v in env.items():
            m.setenv(k, v)
        return fn(dt)


def _pytorch_values(monkeypatch, env, fn, dt):
    with monkeypatch.context() as m:
        m.setattr(dispatch, "_backend", "pytorch")
        return _run(m, env, fn, dt)


def _fp32_policy_warnings(caught):
    return [
        w
        for w in caught
        if issubclass(w.category, UserWarning) and "reduced to fp32" in str(w.message)
    ]


@pytest.mark.parametrize("path,name,env,fn,ops,what", FP64_CASES)
def test_float64_auto_computes_in_torch_at_float64(monkeypatch, path, name, env, fn, ops, what):
    """backend "auto": float64 takes torch, at float64 accuracy, and no kernel runs."""
    ref = _pytorch_values(monkeypatch, env, fn, torch.float64)
    rec, _ = _install(monkeypatch, path, "auto", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _run(monkeypatch, env, fn, torch.float64)
    assert not rec.launches, f"{name}: float64 under auto launched {rec.launches}"
    assert not _fp32_policy_warnings(caught)
    assert _signature(out) == _signature(ref)
    assert _eigvals(out).dtype == torch.float64
    torch.testing.assert_close(_eigvals(out), _eigvals(ref), atol=1e-10, rtol=1e-10)
    # Sanity: the fp32 input on the same setup does run the kernels.
    _run(monkeypatch, env, fn, torch.float32)
    assert {op for _, op, _ in rec.launches} & ops


@pytest.mark.parametrize("path,name,env,fn,ops,what", FP64_CASES)
def test_float64_nki_runs_fp32_kernels_and_warns_once(monkeypatch, path, name, env, fn, ops, what):
    """backend "nki": the FP32 kernels run, the result is float64, one warning per op."""
    ref = _pytorch_values(monkeypatch, env, fn, torch.float64)
    rec, _ = _install(monkeypatch, path, "nki", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outs = [_run(monkeypatch, env, fn, torch.float64) for _ in range(3)]
    reached = {op for _, op, _ in rec.launches}
    assert reached & ops, f"{name}: expected {ops}, launched {reached}"
    assert all(dts and all(d == torch.float32 for d in dts) for _, _, dts in rec.launches), (
        rec.launches
    )
    assert not rec.violations and not rec.rejected
    policy = _fp32_policy_warnings(caught)
    assert len(policy) == 1, [str(w.message) for w in policy]
    assert what in str(policy[0].message)
    stub_rotates = not ops & {"jacobi_sweeps", "jacobi_block"}  # the Jacobi stubs do not
    for out in outs:
        assert _signature(out) == _signature(ref)
        if stub_rotates:  # fp32 precision, not float64 (the kernels compute in FP32)
            torch.testing.assert_close(_eigvals(out), _eigvals(ref), atol=1e-3, rtol=1e-3)


def test_float64_nki_warns_once_per_op(monkeypatch):
    """Each op warns once; a second op warns too (the set is per op name)."""
    _install(monkeypatch, "sim", "nki", False)
    monkeypatch.setenv("TRNSOLVER_EIGH_NKI_MIN_N", "0")
    A = _sym(16, torch.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            trnsolver.eigh(A)
            trnsolver.eigh_randomized(A, 2, oversampling=3, seed=0)
            trnsolver.eigh(A[:6, :6].reshape(1, 6, 6).expand(2, 6, 6), method="jacobi")
    msgs = sorted(str(w.message).split(":")[0] for w in _fp32_policy_warnings(caught))
    assert msgs == [
        "trnsolver eigh",
        "trnsolver eigh(method='jacobi')",
        "trnsolver eigh_randomized",
    ]


@pytest.mark.parametrize("backend", ["nki", "auto"])
@pytest.mark.parametrize("path,name,env,fn,ops,what", FP64_CASES)
def test_float64_require_nki_raises(monkeypatch, backend, path, name, env, fn, ops, what):
    """TRNSOLVER_REQUIRE_NKI=1: float64 raises on every kernel path, over either backend."""
    rec, _ = _install(monkeypatch, path, backend, True)
    with pytest.raises(NotImplementedError, match="TRNSOLVER_REQUIRE_NKI"):
        _run(monkeypatch, env, fn, torch.float64)
    assert not rec.launches


# ── float64 on the neuron device: the host path (trnsolver#89 amendment) ─────
#
# The stand-in neuron device is CPU (`_install`), so every input below is
# "device-resident". `_fp64_to_host` / `_fp64_to_device` are spied on: the
# host path must copy each neuron-resident operand to the host and bring
# each result back to the operands' device.

FP64_DTYPES = [torch.float64, torch.complex128]
HOST_MSG = "float64 on the neuron device is computed on the host for float64 accuracy"


class Transfers:
    def __init__(self, monkeypatch):
        self.to_host, self.to_device, self.active = [], [], []
        real_h, real_d = dispatch._fp64_to_host, dispatch._fp64_to_device

        def to_host(t):
            self.to_host.append(t)
            return real_h(t)

        def to_device(t, device):
            self.to_device.append((t, device))
            return real_d(t, device)

        monkeypatch.setattr(dispatch, "_fp64_to_host", to_host)
        monkeypatch.setattr(dispatch, "_fp64_to_device", to_device)
        # Device aten on the host path would be a bug: every torch.linalg
        # call made inside a public op must run with the host flag set.
        real_eigh = torch.linalg.eigh

        def eigh(*a, **k):
            self.active.append(getattr(dispatch._fp64_host_active, "on", False))
            return real_eigh(*a, **k)

        monkeypatch.setattr(torch.linalg, "eigh", eigh)


def _host_warnings(caught):
    return [w for w in caught if HOST_MSG in str(w.message)]


def _tensors_of(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [t for o in out for t in _tensors_of(o)]
    return []


def _assert_matches(out, ref, name):
    got, want = _tensors_of(out), _tensors_of(ref)
    assert len(got) == len(want), name
    for g, w in zip(got, want, strict=True):
        assert g.dtype == w.dtype and g.shape == w.shape, name
        if g.is_floating_point() or g.is_complex():
            torch.testing.assert_close(g, w, rtol=1e-12, atol=1e-12, msg=name)
        else:
            assert torch.equal(g, w), name


def _api_ids():
    return [name for name, _, _ in APIS]


@pytest.mark.parametrize("dt", FP64_DTYPES, ids=["float64", "complex128"])
@pytest.mark.parametrize("name,env,fn", APIS, ids=_api_ids())
def test_fp64_on_device_auto_computes_on_host(monkeypatch, name, env, fn, dt):
    """auto + float64 / complex128 on the neuron device: host path, result on
    the device, float64 dtype, matching a CPU float64 reference."""
    ref, ref_err = _pytorch_reference(monkeypatch, env, fn, dt)
    if ref_err is not None:
        pytest.skip(f"the pytorch backend raises for {name} {dt}: {ref_err}")
    if name == "occupied_subspace" and dt.is_complex:
        pytest.skip("occupied_subspace is real: it casts other F to float32 (no float64 left)")
    with monkeypatch.context() as m:
        m.setattr(dispatch, "_backend", "pytorch")
        for k, v in env.items():
            m.setenv(k, v)
        ref = fn(dt)
    rec, _ = _install(monkeypatch, "native", "auto", False)
    tr = Transfers(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _run(monkeypatch, env, fn, dt)
    assert not rec.launches, f"{name}: device-resident {dt} launched {rec.launches}"
    assert tr.to_host, f"{name}: no operand was copied to the host"
    assert all(a for a in tr.active), f"{name}: torch.linalg.eigh ran outside the host path"
    moved = {id(t) for t, _ in tr.to_device}
    for t in _tensors_of(out):
        assert t.device.type == dispatch._NEURON_DEVICE
    assert {d for _, d in tr.to_device} <= {torch.device("cpu")}
    assert moved, f"{name}: no result was returned to the device"
    assert len(_host_warnings(caught)) >= 1
    _assert_matches(out, ref, name)


def test_fp64_on_device_returns_results_to_the_operand_device(monkeypatch):
    """Each result tensor goes back to the operands' device; the subspace
    state's host fields (theta, residual) stay on the host."""
    _install(monkeypatch, "native", "auto", False)
    tr = Transfers(monkeypatch)
    F = _sym(12, torch.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _occupied(F)
    moved = [t for t, _ in tr.to_device]
    assert any(t is res.state.X or t.data_ptr() == res.state.X.data_ptr() for t in moved)
    assert all(t is not res.state.theta for t in moved)
    assert res.state.theta.dtype == torch.float64
    assert res.eigenvalues.dtype == res.eigenvectors.dtype == torch.float64


def test_fp64_on_device_warns_once_per_op(monkeypatch):
    _install(monkeypatch, "native", "auto", False)
    A, b = _spd(6, torch.float64), torch.ones(6, dtype=torch.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            trnsolver.eigh(A)
            trnsolver.cholesky(A)
            trnsolver.cg(A, b, maxiter=20)
            trnsolver.eigh_generalized(A, A)  # nests eigh / cholesky: no extra warning
    msgs = sorted(str(w.message).split(":")[0] for w in _host_warnings(caught))
    assert msgs == [
        "trnsolver cg",
        "trnsolver cholesky",
        "trnsolver eigh",
        "trnsolver eigh_generalized",
    ]


def test_fp64_on_device_callable_operands(monkeypatch):
    """A matvec / preconditioner callable keeps getting device tensors."""
    _install(monkeypatch, "native", "auto", False)
    A, b = _spd(6, torch.float64), torch.ones(6, dtype=torch.float64)
    seen = []

    def matvec(x):
        seen.append(x.device.type)
        return A @ x

    M = trnsolver.jacobi_preconditioner(A)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, _, _ = trnsolver.cg(matvec, b, tol=1e-14, maxiter=100, M=M)
    assert seen and set(seen) == {dispatch._NEURON_DEVICE}
    torch.testing.assert_close(x, torch.linalg.solve(A, b), rtol=1e-10, atol=1e-10)


def test_fp64_on_device_override_keeps_device_path(monkeypatch):
    """TRNSOLVER_FP64_ON_DEVICE=1: no host copy, no host-path warning."""
    _install(monkeypatch, "native", "auto", False)
    monkeypatch.setattr(dispatch, "_FP64_ON_DEVICE", True)
    tr = Transfers(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _, env, fn in APIS:
            _call(monkeypatch, env, fn, torch.float64)
    assert not tr.to_host and not tr.to_device
    assert not _host_warnings(caught)


def test_fp64_on_device_override_reads_env(monkeypatch):
    import importlib

    monkeypatch.setenv("TRNSOLVER_FP64_ON_DEVICE", "1")
    fresh = importlib.util.module_from_spec(importlib.util.find_spec("trnsolver.nki.dispatch"))
    fresh.__spec__.loader.exec_module(fresh)
    assert fresh._FP64_ON_DEVICE is True


@pytest.mark.parametrize("backend", ["nki", "pytorch"])
def test_fp64_on_device_other_backends_unchanged(monkeypatch, backend):
    """Only backend auto takes the host path (nki: fp32 cast; pytorch: torch)."""
    _install(monkeypatch, "native", backend, False)
    tr = Transfers(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _, env, fn in APIS:
            _call(monkeypatch, env, fn, torch.float64)
    assert not tr.to_host and not _host_warnings(caught)


@pytest.mark.parametrize("path,name,env,fn,ops,what", FP64_CASES)
def test_fp64_on_device_require_nki_still_raises(monkeypatch, path, name, env, fn, ops, what):
    _install(monkeypatch, path, "auto", True)
    tr = Transfers(monkeypatch)
    with pytest.raises(NotImplementedError, match="TRNSOLVER_REQUIRE_NKI"):
        _run(monkeypatch, env, fn, torch.float64)
    assert not tr.to_host


def test_fp64_host_resident_unchanged(monkeypatch):
    """float64 already on the host: no transfer and no warning (host torch)."""
    _install(monkeypatch, "native", "auto", False)
    monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "neuron")  # CPU is the host again
    tr = Transfers(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name, env, fn in APIS:
            got, err = _call(monkeypatch, env, fn, torch.float64)
            ref, ref_err = _pytorch_reference(monkeypatch, env, fn, torch.float64)
            assert (err is None) == (ref_err is None), name
            assert got == ref, name
    assert not tr.to_host and not tr.to_device
    assert not _host_warnings(caught)
