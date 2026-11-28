"""Static lint: every PSUM accumulation group is opened by a matmul write (trnsci/trnsci#185).

``nisa.nc_matmul`` documents that on NeuronCore-v3 (Trn2) a matmul that
accumulates onto PSUM initialised by a non-matmul instruction is undefined
behaviour, and that the first matmul into a PSUM location must use
``accumulate=False``. ``nl.zeros(..., buffer=nl.psum)`` is such an
instruction (a memset), so ``nl.zeros`` followed by
``nc_matmul(..., accumulate=True)`` on the same tile relies on undefined
behaviour even when the device happens to return the right numbers.

The kernels open each group with a matmul write instead: the first K step
is peeled with ``accumulate=False`` (``nl.affine_range`` indices may be
symbolic), or a static loop passes ``accumulate=(k > 0)``.

The scan parses every module under ``<package>/nki`` and, per top-level
function (nested helpers share their enclosing kernel's tile names), flags:

* ``zeroed-psum``: a PSUM tile bound by ``nl.zeros(..., buffer=nl.psum)``
  (directly, in a list, or appended to one) that is the ``dst`` of an
  ``nc_matmul`` whose ``accumulate`` is anything but the literal ``False``;
* ``never-opened``: a PSUM tile that some ``nc_matmul`` accumulates into with
  the literal ``accumulate=True`` but that no ``nc_matmul`` writes with any
  other ``accumulate`` value, so nothing opens its group.

A PSUM tile zeroed for a non-matmul reason (say, a later Vector Engine
accumulate) may keep its memset; it just must not be a matmul accumulation
target.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

PACKAGE = "trnsolver"
NKI_DIR = Path(__file__).resolve().parents[1] / PACKAGE / "nki"

# Fewest nc_matmul calls the scan must see, so a broken detector cannot pass vacuously.
MIN_MATMULS = 10


def _is_psum_alloc(node: ast.AST, fn: str) -> bool:
    """``nl.<fn>(..., buffer=nl.psum)``."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    if node.func.attr != fn:
        return False
    return any(
        kw.arg == "buffer" and isinstance(kw.value, ast.Attribute) and kw.value.attr == "psum"
        for kw in node.keywords
    )


def _root(node: ast.AST) -> str | None:
    """``accs[j][ug]`` / ``ps[0:L, 0:RB]`` / ``ps`` -> the bound name."""
    while isinstance(node, (ast.Subscript, ast.Attribute, ast.Starred)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _contains(node: ast.AST, fn: str) -> bool:
    return any(_is_psum_alloc(n, fn) for n in ast.walk(node))


def _bound_tiles(func: ast.AST, fn: str) -> set[str]:
    """Names bound to PSUM tiles made by ``nl.<fn>`` (also lists of them)."""
    names: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            if _contains(node.value, fn):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            names.add(n.id)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("append", "extend", "insert")
            and any(_contains(a, fn) for a in node.args)
        ):
            r = _root(node.func.value)
            if r:
                names.add(r)
    return names


def _matmuls(func: ast.AST):
    """``(lineno, dst root name, accumulate node or None)`` per ``nc_matmul``."""
    for node in ast.walk(func):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "nc_matmul":
            continue
        kws = {kw.arg: kw.value for kw in node.keywords}
        dst = kws.get("dst", node.args[0] if node.args else None)
        acc = kws.get("accumulate")
        if acc is None and len(node.args) >= 9:  # positional accumulate (signature order)
            acc = node.args[8]
        yield node.lineno, _root(dst) if dst is not None else None, acc


def _is_const(node: ast.AST | None, value: bool) -> bool:
    return isinstance(node, ast.Constant) and node.value is value


def lint_source(src: str, label: str = "<src>") -> tuple[list[str], int]:
    """(findings, number of nc_matmul calls seen) for one module's source."""
    tree = ast.parse(src)
    findings: list[str] = []
    seen = 0
    tops = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    # Top-level functions only: a def nested in another is scanned with its parent.
    nested = {id(c) for t in tops for c in ast.walk(t) if c is not t and c in tops}
    for func in (t for t in tops if id(t) not in nested):
        zeroed = _bound_tiles(func, "zeros")
        psum = zeroed | _bound_tiles(func, "ndarray")
        mms = list(_matmuls(func))
        seen += len(mms)
        opened = {dst for _, dst, acc in mms if not _is_const(acc, True)}
        for line, dst, acc in mms:
            if dst in zeroed and not _is_const(acc, False):
                findings.append(
                    f"{label}:{line} {func.name}: zeroed-psum: nc_matmul into nl.zeros PSUM "
                    f"tile {dst!r} with accumulate={ast.unparse(acc) if acc else None}"
                )
            if dst in psum and _is_const(acc, True) and dst not in opened:
                findings.append(
                    f"{label}:{line} {func.name}: never-opened: nc_matmul accumulates into "
                    f"PSUM tile {dst!r} that no matmul opens with accumulate=False"
                )
    return findings, seen


def test_psum_groups_open_with_a_matmul_write():
    findings, seen = [], 0
    for path in sorted(NKI_DIR.glob("*.py")):
        f, n = lint_source(path.read_text(), str(path.relative_to(NKI_DIR.parents[1])))
        findings += f
        seen += n
    assert seen >= MIN_MATMULS, f"scan saw only {seen} nc_matmul calls; detector broken?"
    assert not findings, "PSUM accumulation opened by a memset (trnsci/trnsci#185):\n" + (
        "\n".join(findings)
    )


_BAD = """
def k(a, b, K):
    ps = nl.zeros((128, 512), dtype=nl.float32, buffer=nl.psum)
    for k in nl.affine_range(K):
        nisa.nc_matmul(dst=ps, stationary=a, moving=b, accumulate=True)
    accs = [nl.zeros((128, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(2)]
    def inner():
        nisa.nc_matmul(dst=accs[1][0:128], stationary=a, moving=b, accumulate=True)
    pp = []
    pp.append(nl.zeros((128, 512), dtype=nl.float32, buffer=nl.psum))
    nisa.nc_matmul(dst=pp[0], stationary=a, moving=b)
    q = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=q, stationary=a, moving=b, accumulate=True)
"""

_GOOD = """
def k(a, b, K):
    ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ps, stationary=a, moving=b, accumulate=False)
    for k in nl.affine_range(1, K):
        nisa.nc_matmul(dst=ps, stationary=a, moving=b, accumulate=True)
    qs = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    for k in range(K):
        nisa.nc_matmul(dst=qs, stationary=a, moving=b, accumulate=(k > 0))
    z = nl.zeros((128, 512), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=z, stationary=a, moving=b, accumulate=False)
    t = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=t, stationary=a, moving=b)
"""


def test_lint_flags_the_bad_pattern():
    findings, seen = lint_source(textwrap.dedent(_BAD))
    assert seen == 4
    kinds = sorted(f.split(": ")[1] for f in findings)
    assert kinds == ["never-opened"] * 3 + ["zeroed-psum"] * 3, findings


def test_lint_accepts_the_fixed_forms():
    findings, seen = lint_source(textwrap.dedent(_GOOD))
    assert seen == 5 and not findings, findings
