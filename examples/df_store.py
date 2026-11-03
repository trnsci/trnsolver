"""Out-of-core density-fitted integrals for examples/scf_df_rhf.py.

The packed three-centre tensor B (npair, naux) of a DF-SCF grows as
n_ao^2 naux: 33.4 GB in fp32 at C40H82 / aug-cc-pVDZ (n_ao 1658, naux 6064),
about 270 GB at n_ao ~ 3300. PySCF's in-core build (``df.incore.cholesky_eri``
plus ``lib.transpose``) needs two float64 copies, 133 GB at n_ao 1658, so it
did not fit a 124 GB host (#15, window 3). This module builds B straight to
disk, a block of AO rows at a time, with no full in-memory copy:

    for each block of AO shells [i0, i1):
        X = (mn|P) for m in the block, n <= m   (PySCF int3c2e, aosym s2ij)
        B_rows = X L^{-T}                       (L: Cholesky of (P|Q))
        write B_rows into every chunk file      (contiguous packed rows)

The store is a directory:

    meta.json         shapes, dtype, chunk widths, build stats
    system.pt         H, S, nocc, e_nuc (and a reference energy if any)
    B_00000.npy ...   C-contiguous (npair, nq) aux chunks (numpy .npy)

Each chunk is one contiguous file, so streaming it to a device is one
sequential read (served from the page cache when host RAM holds B), and
``np.load(..., mmap_mode="c")`` maps it without a copy. The chunk width is
a multiple of 128 under ``chunk_bytes`` (default 2 GiB), below the 4 GiB
transposed-LHS limit of the device matmul (aws-neuron-sdk#1457).

:class:`DFStore` behaves enough like the in-memory (npair, naux) tensor for
``DFJK`` (``shape``, ``dtype``, ``B[:, q0:q1]``) and hands its chunks to
``trntensor.AuxStream`` for streaming.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time

import numpy as np
import torch

STORE_VERSION = 1
DEFAULT_CHUNK_BYTES = 1 << 31
# Peak host bytes of one row block of the builder (float64 int3c2e + solve).
DEFAULT_BLOCK_BYTES = 1 << 30


def default_store_root() -> str:
    """``$TRNSCI_DF_DIR``, else ``~/.cache/trnsci/df``."""
    return os.environ.get("TRNSCI_DF_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "trnsci", "df"
    )


def chunk_width(npair: int, naux: int, itemsize: int, chunk_bytes: int) -> int:
    """Widest 128-multiple aux chunk of ``npair`` rows under ``chunk_bytes``."""
    w = (chunk_bytes // (npair * itemsize)) // 128 * 128
    if w == 0:
        w = max(1, chunk_bytes // (npair * itemsize))
    return int(min(naux, w))


def peak_rss_bytes() -> int:
    """Peak resident set size of this process (bytes)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


class DFStore:
    """A packed (npair, naux) DF tensor stored as aux-chunk ``.npy`` files."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        self.nbasis = self.meta["nbasis"]
        self.npair = self.meta["npair"]
        self.naux = self.meta["naux"]
        self.widths = list(self.meta["widths"])
        self.offsets = [0]
        for w in self.widths:
            self.offsets.append(self.offsets[-1] + w)
        self.dtype = getattr(torch, self.meta["dtype"])
        self.device = torch.device("cpu")

    # -- tensor-like surface -----------------------------------------------

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.npair, self.naux))

    @property
    def nbytes(self) -> int:
        return self.npair * self.naux * torch.empty((), dtype=self.dtype).element_size()

    def chunk_path(self, i: int) -> str:
        return os.path.join(self.path, f"B_{i:05d}.npy")

    def chunk(self, i: int, mmap_mode: str | None = "c"):
        """Chunk ``i`` as a numpy array (a copy-on-write memmap by default)."""
        return np.load(self.chunk_path(i), mmap_mode=mmap_mode)

    def loaders(self) -> list:
        """Zero-argument loaders, one per chunk, for ``trntensor.AuxStream``."""
        return [lambda i=i: self.chunk(i) for i in range(len(self.widths))]

    def __getitem__(self, key) -> torch.Tensor:
        """``B[:, q0:q1]`` as a host tensor (the only indexing ``DFJK`` uses)."""
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] == slice(None)):
            raise IndexError("DFStore supports B[:, q0:q1] only")
        q0, q1, step = key[1].indices(self.naux)
        if step != 1:
            raise IndexError("DFStore column slices must be contiguous")
        parts = []
        for i in range(len(self.widths)):
            a, b = self.offsets[i], self.offsets[i + 1]
            lo, hi = max(a, q0), min(b, q1)
            if lo < hi:
                parts.append(
                    torch.from_numpy(np.ascontiguousarray(self.chunk(i)[:, lo - a : hi - a]))
                )
        if not parts:
            return torch.empty(self.npair, 0, dtype=self.dtype)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    def to_tensor(self) -> torch.Tensor:
        """The whole tensor in memory (small systems and tests only)."""
        return self[:, 0 : self.naux]

    def __repr__(self) -> str:
        return (
            f"DFStore({self.path!r}, npair={self.npair}, naux={self.naux}, "
            f"dtype={self.meta['dtype']}, chunks={len(self.widths)}, {self.nbytes / 1e9:.2f} GB)"
        )

    def __reduce__(self):
        return (DFStore, (self.path,))


class StoreWriter:
    """Writes packed rows of B into a new store's chunk files."""

    def __init__(
        self,
        path: str,
        nbasis: int,
        naux: int,
        dtype: torch.dtype = torch.float32,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.nbasis = nbasis
        self.npair = nbasis * (nbasis + 1) // 2
        self.naux = naux
        self.dtype = dtype
        self.np_dtype = torch.empty((), dtype=dtype).numpy().dtype
        w = chunk_width(self.npair, naux, self.np_dtype.itemsize, chunk_bytes)
        self.widths = [min(w, naux - q0) for q0 in range(0, naux, w)]
        # Each chunk file gets its .npy header now; rows are then written
        # with positioned writes (not a writable memmap), so dirty pages
        # belong to the page cache, not to this process's footprint.
        self.files = []
        q0 = 0
        for i, wi in enumerate(self.widths):
            fn = os.path.join(path, f"B_{i:05d}.npy")
            mm = np.lib.format.open_memmap(
                fn, mode="w+", dtype=self.np_dtype, shape=(self.npair, wi)
            )
            offset = mm.offset
            del mm
            self.files.append((q0, wi, offset, open(fn, "r+b")))  # noqa: SIM115 (closed in close())
            q0 += wi
        self.rows_written = 0

    def write_rows(self, p0: int, rows) -> None:
        """Packed rows ``[p0, p0 + len(rows))`` of B, shape (nrow, naux)."""
        rows = np.asarray(rows)
        isz = self.np_dtype.itemsize
        for q0, wi, offset, fh in self.files:
            blk = np.ascontiguousarray(rows[:, q0 : q0 + wi], dtype=self.np_dtype)
            os.pwrite(fh.fileno(), blk.data, offset + p0 * wi * isz)
        self.rows_written += rows.shape[0]

    def close(self, system_fields: dict, extra_meta: dict | None = None) -> DFStore:
        if self.rows_written != self.npair:
            raise RuntimeError(f"wrote {self.rows_written} of {self.npair} packed rows")
        for *_, fh in self.files:
            fh.close()
        self.files = []
        torch.save(system_fields, os.path.join(self.path, "system.pt"))
        meta = {
            "version": STORE_VERSION,
            "nbasis": self.nbasis,
            "npair": self.npair,
            "naux": self.naux,
            "dtype": str(self.dtype).replace("torch.", ""),
            "widths": self.widths,
        }
        meta.update(extra_meta or {})
        with open(os.path.join(self.path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        return DFStore(self.path)


def store_complete(path: str, expect: dict | None = None) -> bool:
    """True when ``path`` holds a finished store whose meta matches ``expect``."""
    try:
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return False
    if meta.get("version") != STORE_VERSION:
        return False
    return all(meta.get(k) == v for k, v in (expect or {}).items())


def load_system_fields(path: str) -> dict:
    return torch.load(os.path.join(path, "system.pt"))


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _shell_row_blocks(ao_loc, nbas: int, naux: int, block_bytes: int):
    """Shell ranges [i0, i1) whose packed rows times naux float64 fit ``block_bytes``
    (at least one shell per block)."""
    blocks = []
    i0 = 0
    while i0 < nbas:
        i1 = i0 + 1
        while i1 < nbas:
            m0, m2 = ao_loc[i0], ao_loc[i1 + 1]
            rows = m2 * (m2 + 1) // 2 - m0 * (m0 + 1) // 2
            # int3c2e block + its transpose / solve copies: ~3 float64 buffers.
            if 3 * rows * naux * 8 > block_bytes:
                break
            i1 += 1
        blocks.append((i0, i1))
        i0 = i1
    return blocks


def build_pyscf_store(
    mol,
    auxbasis: str,
    path: str,
    *,
    dtype: torch.dtype = torch.float32,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    block_bytes: int = DEFAULT_BLOCK_BYTES,
    system_fields: dict | None = None,
    extra_meta: dict | None = None,
    verbose: bool = False,
) -> DFStore:
    """PySCF DF integrals B = (mn|P) L^{-T} straight to a store, row block by row block.

    The same tensor as PySCF's ``df.incore.cholesky_eri`` (transposed to
    (npair, naux)): Cholesky of the metric, or PySCF's eigen-decomposition
    fallback when it is not positive definite. Peak host memory is one row
    block (``block_bytes``) plus the metric, not B.
    """
    import scipy.linalg
    from pyscf import df
    from pyscf.df import incore

    t0 = time.perf_counter()
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    j2c = auxmol.intor("int2c2e", hermi=1)
    try:
        low = scipy.linalg.cholesky(j2c, lower=True)
        decomp = "cd"
    except scipy.linalg.LinAlgError:
        low = incore._eig_decompose(mol, j2c, incore.LINEAR_DEP_THR)
        decomp = "eig"
    del j2c
    naux = low.shape[0]
    nao = mol.nao
    writer = StoreWriter(path, nao, naux, dtype, chunk_bytes)
    ao_loc = mol.ao_loc_nr()
    blocks = _shell_row_blocks(ao_loc, mol.nbas, auxmol.nao, block_bytes)
    t_int = t_solve = t_write = 0.0
    for k, (i0, i1) in enumerate(blocks):
        m0, m1 = int(ao_loc[i0]), int(ao_loc[i1])
        ta = time.perf_counter()
        ints = incore.aux_e2(
            mol, auxmol, intor="int3c2e", aosym="s2ij", shls_slice=(i0, i1, 0, i1, 0, auxmol.nbas)
        )  # (rows, naoaux), F-ordered
        tb = time.perf_counter()
        if decomp == "cd":
            rows = scipy.linalg.solve_triangular(
                low, ints.T, lower=True, overwrite_b=True, check_finite=False
            ).T
        else:
            rows = (low @ ints.T).T
        del ints
        tc = time.perf_counter()
        writer.write_rows(m0 * (m0 + 1) // 2, rows.astype(writer.np_dtype, copy=False))
        del rows
        td = time.perf_counter()
        t_int += tb - ta
        t_solve += tc - tb
        t_write += td - tc
        if verbose:
            print(
                f"#   block {k + 1}/{len(blocks)}: AO [{m0}, {m1}) "
                f"{td - ta:.1f} s, peak RSS {peak_rss_bytes() / 1e9:.1f} GB",
                flush=True,
            )
    stats = {
        "build_s": time.perf_counter() - t0,
        "int3c_s": t_int,
        "solve_s": t_solve,
        "write_s": t_write,
        "row_blocks": len(blocks),
        "peak_rss_bytes": peak_rss_bytes(),
        "metric": decomp,
        "auxbasis": auxbasis,
    }
    meta = dict(extra_meta or {})
    meta["build"] = stats
    return writer.close(system_fields or {}, meta)


def build_rows_store(
    nbasis: int,
    naux: int,
    rows_fn,
    path: str,
    *,
    dtype: torch.dtype = torch.float32,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    row_block: int = 65536,
    system_fields: dict | None = None,
    extra_meta: dict | None = None,
) -> DFStore:
    """A store from ``rows_fn(p0, p1) -> (p1 - p0, naux)`` packed rows, block by block."""
    t0 = time.perf_counter()
    writer = StoreWriter(path, nbasis, naux, dtype, chunk_bytes)
    for p0 in range(0, writer.npair, row_block):
        p1 = min(writer.npair, p0 + row_block)
        r = rows_fn(p0, p1)
        if isinstance(r, torch.Tensor):
            r = r.numpy()
        writer.write_rows(p0, r.astype(writer.np_dtype, copy=False))
    meta = dict(extra_meta or {})
    meta["build"] = {"build_s": time.perf_counter() - t0, "peak_rss_bytes": peak_rss_bytes()}
    return writer.close(system_fields or {}, meta)


def store_from_tensor(B: torch.Tensor, nbasis: int, path: str, **kw) -> DFStore:
    """An in-memory (npair, naux) tensor written as a store (tests, conversions)."""
    return build_rows_store(nbasis, B.shape[1], lambda p0, p1: B[p0:p1], path, **kw)


# ---------------------------------------------------------------------------
# Size estimates (no integrals computed)
# ---------------------------------------------------------------------------


def df_bytes(nbasis: int, naux: int, itemsize: int = 4) -> int:
    """Bytes of the packed (npair, naux) tensor."""
    return nbasis * (nbasis + 1) // 2 * naux * itemsize


def pyscf_incore_bytes(nbasis: int, naux: int) -> int:
    """Peak host bytes of PySCF's in-core build as the example used it:
    the float64 ``_cderi`` plus its ``lib.transpose`` copy."""
    return 2 * df_bytes(nbasis, naux, 8)


def gb(x: float) -> str:
    return f"{x / 1e9:.1f}" if x < 1e11 else f"{x / 1e9:.0f}"


__all__ = [
    "DFStore",
    "StoreWriter",
    "build_pyscf_store",
    "build_rows_store",
    "store_from_tensor",
    "store_complete",
    "load_system_fields",
    "default_store_root",
    "df_bytes",
    "pyscf_incore_bytes",
    "chunk_width",
    "peak_rss_bytes",
]
