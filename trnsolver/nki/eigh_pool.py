"""
Persistent process-per-core pool for batched Jacobi ``eigh`` (#29).

``TRNSOLVER_EIGH_CORES=<k>|auto`` with ``TRNSOLVER_EIGH_POOL=proc`` solves
a host batch with one worker process per logical NeuronCore instead of one
host thread per core (`jacobi_eigh_multicore`). The thread split is
host-bound: one Python process issues every core's launches, aten ops and
copies, and on Trn2 it reached 2.3x on 4 cores where separate processes
reached 3.6x (#54).

Each worker is a spawned process with ``NEURON_RT_VISIBLE_CORES`` set to
one core id, so it owns exactly that core and runs the single-core solver
(`jacobi._solve`) on it. Workers start on the first call and stay up, so
the import, runtime start and kernel compile (seconds) are paid once.

Data moves through shared memory. Inputs: one POSIX segment per worker,
kept between calls and grown on demand; a call copies chunk i into worker
i's segment and sends a small message at once, so the copy of chunk i + 1
overlaps chunk i's solve. Outputs: one torch shared-memory buffer per call
that every worker writes its slice of straight from the device, and that
the returned ``w`` and ``V`` are views of (no copy out). The buffer is
reused once the caller has dropped those results, so steady-state calls
neither allocate nor page-fault it.

Constraint: the Neuron runtime gives a core to one process. The pool only
works while the calling process has not used those cores itself, so it
takes host input only (a tensor already on ``neuron:k`` means the caller
holds the cores, and the thread split runs instead). The cores come from
``TRNSOLVER_EIGH_POOL_CORES`` (e.g. ``0-3`` or ``0,2``), else from
``NEURON_RT_VISIBLE_CORES``.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import threading
import time
import traceback

import torch

_LOCK = threading.Lock()
_POOL: _Pool | None = None
# Set when the pool has failed in this process: the fallback then opens a
# core here, which a restarted pool's workers could not take, so later
# calls skip the pool (until `shutdown`).
_FAILED = False


def parse_cores(raw: str) -> list[int]:
    """``"0-3"`` / ``"0,2"`` / ``"1"`` -> sorted core ids (empty if unset)."""
    out: set[int] = set()
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def pool_cores() -> list[int]:
    """Core ids the pool may use: ``TRNSOLVER_EIGH_POOL_CORES``, else
    ``NEURON_RT_VISIBLE_CORES``, else every logical core."""
    raw = os.environ.get("TRNSOLVER_EIGH_POOL_CORES", "").strip()
    if not raw:
        raw = os.environ.get("NEURON_RT_VISIBLE_CORES", "").strip()
    if not raw:
        # No restriction: every logical core (device_count does not open one).
        try:
            import torch_neuronx

            return list(range(torch_neuronx.device_count()))
        except Exception:
            return []
    try:
        return parse_cores(raw)
    except ValueError:
        return []


def pool_requested() -> bool:
    return os.environ.get("TRNSOLVER_EIGH_POOL", "").strip().lower() in ("proc", "process")


def _worker_threads() -> int:
    raw = os.environ.get("TRNSOLVER_EIGH_POOL_THREADS", "").strip()
    try:
        return max(1, int(raw)) if raw else 1
    except ValueError:
        return 1


def _view(buf, offset: int, shape: tuple[int, ...]) -> torch.Tensor:
    count = 1
    for s in shape:
        count *= s
    return torch.frombuffer(buf, dtype=torch.float32, count=count, offset=offset).view(shape)


def _worker(core: int, conn, threads: int, simulate: bool) -> None:
    """Worker loop: own one NeuronCore, solve chunks from shared memory."""
    from multiprocessing import shared_memory

    torch.set_num_threads(threads)
    segs: dict[str, shared_memory.SharedMemory] = {}
    bufs: dict[int, torch.Tensor] = {}  # output buffers by id, shared with the parent

    def seg(name):
        if name not in segs:
            segs[name] = shared_memory.SharedMemory(name=name)
        return segs[name]

    try:
        from . import jacobi as JK

        if simulate:
            # CPU tests: the torch Jacobi stands in for the device kernel.
            from .._jacobi import jacobi_eigh_torch as solve

            dev = torch.device("cpu")
        else:
            solve = JK._solve
            import torch_neuronx  # noqa: F401 — registers the neuron device

            dev = torch.device("neuron")
            torch.ones(1, device=dev).cpu()  # take the core now, not on the first call
        conn.send(("ready", core))
    except Exception:
        conn.send(("error", traceback.format_exc()))
        return
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg[0] == "stop":
            break
        if msg[0] == "detach":
            s = segs.pop(msg[1], None)
            if s is not None:
                s.close()
            continue
        if msg[0] == "buf":
            bufs[msg[1]] = msg[2]
            continue
        if msg[0] == "drop":
            bufs.pop(msg[1], None)
            continue
        _, in_name, buf_id, w_off, v_off, count, n = msg
        try:
            x = _view(seg(in_name).buf, 0, (count, n, n))
            out = bufs[buf_id]
            t0 = time.perf_counter()
            w, V, sweeps = solve(x.to(dev))
            out[w_off : w_off + count * n].view(count, n).copy_(w)
            out[v_off : v_off + count * n * n].view(count, n, n).copy_(V)
            del x, w, V, out
            conn.send(("done", sweeps, time.perf_counter() - t0))
        except Exception:
            conn.send(("error", traceback.format_exc()))
    for s in segs.values():
        with contextlib.suppress(BufferError):
            s.close()


class _Seg:
    """A parent-owned shared-memory segment, replaced when too small."""

    def __init__(self):
        self.shm = None

    def ensure(self, nbytes: int, conn) -> str:
        from multiprocessing import shared_memory

        if self.shm is None or self.shm.size < nbytes:
            if self.shm is not None:
                conn.send(("detach", self.shm.name))
                self.close()
            self.shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 4096))
        return self.shm.name

    def close(self):
        if self.shm is not None:
            with contextlib.suppress(BufferError):
                self.shm.close()
            with contextlib.suppress(FileNotFoundError):
                self.shm.unlink()
            self.shm = None


def _use_count(t: torch.Tensor) -> int:
    return torch._C._storage_Use_Count(t.untyped_storage()._cdata)


class _OutBuf:
    """A torch shared-memory output buffer that results are views of.

    The caller gets ``w`` and ``V`` as views of it, so no copy-out is
    needed; the buffer is reused once the caller has dropped every view
    (its storage use count is back to this object's own references).
    """

    def __init__(self, ident: int, numel: int):
        storage = torch.UntypedStorage._new_shared(numel * 4)
        self.id = ident
        self.t = torch.empty(0, dtype=torch.float32).set_(storage, 0, (numel,))
        del storage
        self.base = _use_count(self.t)

    def free(self) -> bool:
        return _use_count(self.t) == self.base


# Free output buffers kept for reuse beyond the one a call takes.
_KEEP_FREE = 1


class _Pool:
    """One spawned worker per core id, started together and kept alive."""

    def __init__(self, cores: list[int], threads: int, simulate: bool = False):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.cores = list(cores)
        self.last_times: list[float] = []
        self.procs, self.conns = [], []
        self.ins = [_Seg() for _ in cores]
        self.bufs: list[_OutBuf] = []
        self._next_id = 0
        saved = {k: os.environ.get(k) for k in ("NEURON_RT_VISIBLE_CORES", "TRNSOLVER_EIGH_POOL")}
        try:
            # A worker never starts a pool of its own (spawn re-imports the
            # caller's main module; without a __main__ guard that would
            # call eigh again inside the worker).
            os.environ["TRNSOLVER_EIGH_POOL"] = "0"
            for c in cores:
                parent, child = ctx.Pipe()
                # The child reads its core from the environment at import.
                os.environ["NEURON_RT_VISIBLE_CORES"] = str(c)
                p = ctx.Process(target=_worker, args=(c, child, threads, simulate), daemon=True)
                p.start()
                child.close()
                self.procs.append(p)
                self.conns.append(parent)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        for conn in self.conns:
            msg = conn.recv()
            if msg[0] != "ready":
                self.close()
                raise RuntimeError(f"eigh pool worker failed to start:\n{msg[1]}")

    def _out_buf(self, numel: int) -> _OutBuf:
        """A free output buffer of at least ``numel`` floats, sent to every worker."""
        free = [b for b in self.bufs if b.free()]
        fits = [b for b in free if b.t.numel() >= numel]
        pick = min(fits, key=lambda b: b.t.numel()) if fits else None
        # Drop free buffers that are too small, and spares beyond _KEEP_FREE.
        spare = [b for b in free if b is not pick]
        drop = [b for b in spare if b.t.numel() < numel] + [
            b for b in spare if b.t.numel() >= numel
        ][_KEEP_FREE:]
        for b in drop:
            for conn in self.conns:
                conn.send(("drop", b.id))
            self.bufs.remove(b)
        if pick is None:
            import torch.multiprocessing  # noqa: F401 — shared-tensor pickling

            pick = _OutBuf(self._next_id, numel)
            self._next_id += 1
            for conn in self.conns:
                conn.send(("buf", pick.id, pick.t))
            self.bufs.append(pick)
        return pick

    def solve(self, A: torch.Tensor, chunks: list[tuple[int, int]]):
        """Solve chunk i of host FP32 ``A`` on worker i: (w, V, max sweeps).

        ``w`` and ``V`` are views of one shared output buffer that the
        workers write in place.
        """
        n = A.shape[-1]
        Bsz = A.shape[0]
        out = self._out_buf(Bsz * (n + n * n))
        w = out.t[: Bsz * n].view(Bsz, n)
        V = out.t[Bsz * n : Bsz * (n + n * n)].view(Bsz, n, n)
        for i, (s, e) in enumerate(chunks):
            c = e - s
            conn = self.conns[i]
            iname = self.ins[i].ensure(c * n * n * 4, conn)
            _view(self.ins[i].shm.buf, 0, (c, n, n)).copy_(A[s:e])
            conn.send(("solve", iname, out.id, s * n, Bsz * n + s * n * n, c, n))
        sweeps, errors = 0, []
        self.last_times = []  # per-worker seconds from input to written output
        for i in range(len(chunks)):
            try:
                msg = self.conns[i].recv()
            except EOFError:
                errors.append(f"worker for core {self.cores[i]} exited")
                continue
            if msg[0] != "done":
                errors.append(msg[1])
                continue
            sweeps = max(sweeps, msg[1])
            self.last_times.append(msg[2])
        if errors:
            raise RuntimeError("eigh pool worker failed:\n" + "\n".join(errors))
        return w, V, sweeps

    def alive(self) -> bool:
        return all(p.is_alive() for p in self.procs)

    def close(self):
        for conn in self.conns:
            with contextlib.suppress(OSError, BrokenPipeError):
                conn.send(("stop",))
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)
        for conn in self.conns:
            conn.close()
        for s in self.ins:
            s.close()
        self.bufs = []
        self.procs, self.conns = [], []


def get_pool(cores: list[int], simulate: bool = False) -> _Pool:
    """The process pool on exactly ``cores``, (re)started when they change."""
    global _POOL
    if _POOL is not None and (_POOL.cores != list(cores) or not _POOL.alive()):
        _POOL.close()
        _POOL = None
    if _POOL is None:
        _POOL = _Pool(cores, _worker_threads(), simulate)
    return _POOL


def shutdown() -> None:
    """Stop the workers and free their cores and shared memory."""
    global _POOL, _FAILED
    with _LOCK:
        if _POOL is not None:
            _POOL.close()
            _POOL = None
        _FAILED = False


atexit.register(shutdown)


def jacobi_eigh_pool(
    A: torch.Tensor, cores: list[int], parts: int | None = None, *, simulate: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """`jacobi_eigh_device` on host ``A`` (B, n, n), batch split over ``cores``.

    The pool runs one worker per entry of ``cores``; the batch goes to the
    first ``parts`` of them (default all), so a smaller split reuses the
    same workers instead of restarting them.

    ``simulate`` runs the torch Jacobi in the workers (CPU tests).

    Chunks are those of `split_batch`, so each core runs the same launch
    shapes as in the thread split. On a worker failure the pool is shut
    down (freeing the cores) and the error propagates, so the caller can
    fall back to one core in this process; later calls then raise at once
    instead of restarting workers that could no longer open the cores.
    """
    global _POOL, _FAILED
    from . import jacobi as JK

    if A.device.type != "cpu":
        raise ValueError("the eigh process pool takes host input")
    if _FAILED:
        raise RuntimeError("the eigh process pool failed earlier in this process")
    A = A.to(torch.float32).contiguous()
    parts = len(cores) if parts is None else max(1, min(parts, len(cores)))
    chunks = JK.split_batch(A.shape[0], A.shape[-1], parts)
    with _LOCK:
        # Small batches use the first workers only; the pool keeps every core.
        try:
            pool = get_pool(cores, simulate)
        except Exception:
            _FAILED = True
            raise
        try:
            w, V, sweeps = pool.solve(A, chunks)
        except Exception:
            pool.close()
            _POOL = None
            _FAILED = True
            raise
    JK.last_sweeps = sweeps
    return w, V
