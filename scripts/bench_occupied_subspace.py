"""SCF-sized symmetric eigensolve: occupied_subspace vs host LAPACK and aten.

For each n (with k = nocc, default n // 6) the script builds a Fock-like
symmetric matrix (deep core levels, a valence band, a HOMO-LUMO gap, a
virtual band) and an SCF-like sequence F_t = F_0 + d_t D_t with
d_t = d_0 * shrink^t, then times, warm, median of ``--reps``:

    host       torch.linalg.eigh of the host FP32 matrix (pin the threads:
               OMP_NUM_THREADS=2 taskset -c ...), the SCF baseline
    aten       torch.linalg.eigh of the neuron tensor (host LAPACK behind
               aten, #42), synchronised
    sub<p>     trnsolver.occupied_subspace(F_t, k, guess=state, passes=p) on
               the neuron tensor, warm-started from the previous F_t, until
               the result is on the host-visible side (one sync)
    kernel     one subspace_filter_kernel launch alone (degree --degree)

and the breakdown of one pass: kernel, D2H of the Gram blocks, the float64
host Rayleigh-Ritz, and the block update on device. Accuracy columns are
against float64 eigh of F_t: ``proj`` = |V V^T - U_k U_k^T|_2 (occupied
projector), ``eig`` = max |w - w_ref| / |w_ref|_max, ``orth`` =
|V^T V - I|_max.

Every line also prints as ``RESULT {json}`` for the issue comment.

Examples (Trn2, one logical NeuronCore, host threads pinned):
    NEURON_RT_VISIBLE_CORES=0 OMP_NUM_THREADS=2 taskset -c 2,8 \\
        python scripts/bench_occupied_subspace.py --sizes 128,250,490,970
    ... --sizes 970 --passes 1,2,3 --degree 8 --reps 20
    ... --device cpu  # the whole thing on the host (reference filter)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch


def fock_like(n: int, k: int, seed: int = 0) -> torch.Tensor:
    """Q diag(w) Q^T, float64: core near -11, valence [-1.5, -0.35], virtuals 0.15..30."""
    g = torch.Generator().manual_seed(seed)
    ncore = max(1, k // 5)
    core = -11.0 - 0.3 * torch.rand(ncore, generator=g, dtype=torch.float64)
    val = torch.linspace(-1.5, -0.35, k - ncore, dtype=torch.float64)
    virt = 0.15 + torch.sort(torch.rand(n - k, generator=g, dtype=torch.float64) ** 2 * 30).values
    w = torch.cat([core, val, virt])
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    A = (Q * w) @ Q.T
    return 0.5 * (A + A.T)


def sequence(n, k, steps, d0, shrink, seed=0):
    """SCF-like sequence of float64 matrices with shrinking symmetric changes."""
    F = fock_like(n, k, seed)
    g = torch.Generator().manual_seed(seed + 1)
    out = [F]
    for t in range(steps):
        D = torch.randn(n, n, generator=g, dtype=torch.float64)
        F = F + d0 * shrink**t * 0.5 * (D + D.T) / n**0.5
        out.append(F)
    return out


def accuracy(F64, w, V, k):
    wr, U = torch.linalg.eigh(F64)
    V = V.detach().to("cpu", torch.float64)
    proj = float(torch.linalg.matrix_norm(V @ V.T - U[:, :k] @ U[:, :k].T, 2))
    eig = float((w.detach().cpu().double() - wr[:k]).abs().max() / wr.abs().max())
    orth = float((V.T @ V - torch.eye(k, dtype=torch.float64)).abs().max())
    return proj, eig, orth


def median_ms(fn, reps, sync):
    fn()
    sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(1e3 * (time.perf_counter() - t))
    return statistics.median(ts)


def bench_one(n, k, args, passes, dev, sync, SK):
    """One size: the RESULT row."""
    import trnsolver
    from trnsolver import subspace as SS

    seq = sequence(n, k, args.steps, args.d0, args.shrink)
    F32 = [F.float() for F in seq]
    Fd = [F.to(dev) for F in F32]
    kb = k + SS.default_buffer(k)
    r = {"n": n, "k": k, "kb": kb, "degree": args.degree}
    r["host_ms"] = median_ms(lambda: torch.linalg.eigh(F32[-1]), args.reps, lambda: None)
    if dev.type == "neuron":
        r["aten_ms"] = median_ms(lambda: torch.linalg.eigh(Fd[-1]), args.reps, sync)

    for p in passes:
        # Walk the sequence once (warm states), then time calls on the
        # last matrices from the state one step back.
        st = trnsolver.occupied_subspace(Fd[0], k).state
        states = [st]
        for t in range(1, len(Fd)):
            res = trnsolver.occupied_subspace(
                Fd[t], k, guess=st, passes=p, degree=args.degree, refresh=None
            )
            st = res.state
            states.append(st)
        proj, eig, orth = accuracy(seq[-1], res.eigenvalues, res.eigenvectors, k)
        prev = states[-2]

        def call(prev=prev, p=p):
            out = trnsolver.occupied_subspace(
                Fd[-1], k, guess=prev, passes=p, degree=args.degree, refresh=None
            )
            return out.eigenvectors

        r[f"sub{p}_ms"] = median_ms(call, args.reps, sync)
        r[f"sub{p}_proj"], r[f"sub{p}_eig"], r[f"sub{p}_orth"] = proj, eig, orth

    if SK is not None and SK.kernel_fits(n, kb):
        X = st.X.contiguous()
        theta = st.theta
        b = SS._gershgorin_upper(Fd[-1])
        coef = SS.chebyshev_coefficients(args.degree, float(theta[0]), float(theta[-1]), b)
        table = SS.filter_table(coef, theta)
        out = SK.run_filter_kernel(Fd[-1], X, table)
        r["kernel_ms"] = median_ms(lambda: SK.run_filter_kernel(Fd[-1], X, table), args.reps, sync)
        r["d2h_ms"] = median_ms(lambda: out[n:].to("cpu", torch.float64), args.reps, sync)
        small = out[n:].to("cpu", torch.float64)
        G, H, _ = SS.unpack_grams(small, kb)
        r["rr_ms"] = median_ms(lambda: SS.rayleigh_ritz(G, H, kb=kb), args.reps, lambda: None)
        _, Q = SS.rayleigh_ritz(G, H, kb=kb)
        Qd = Q[:, :kb].to(dev, torch.float32)
        r["update_ms"] = median_ms(lambda: X @ Qd[:kb] + out[:n] @ Qd[kb:], args.reps, sync)
        r["gershgorin_ms"] = median_ms(lambda: SS._gershgorin_upper(Fd[-1]), args.reps, sync)
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", default="128,250,490,970")
    ap.add_argument("--nocc", default="", help="comma list matching --sizes (default n // 6)")
    ap.add_argument("--passes", default="1,2")
    ap.add_argument("--degree", type=int, default=8)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--steps", type=int, default=12, help="SCF-like matrices per size")
    ap.add_argument("--d0", type=float, default=0.05)
    ap.add_argument("--shrink", type=float, default=0.6)
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    import trnsolver

    dev = torch.device(args.device)
    if dev.type == "neuron":
        import torch_neuronx

        sync = torch_neuronx.synchronize
        from trnsolver.nki import subspace as SK
    else:
        trnsolver.set_backend("pytorch")
        SK = None

        def sync():
            return None

    sizes = [int(s) for s in args.sizes.split(",")]
    noccs = [int(s) for s in args.nocc.split(",")] if args.nocc else [n // 6 for n in sizes]
    passes = [int(p) for p in args.passes.split(",")]
    rows = []
    for n, k in zip(sizes, noccs, strict=True):
        r = bench_one(n, k, args, passes, dev, sync, SK)
        rows.append(r)
        print("RESULT " + json.dumps(r), flush=True)

    cols = ["n", "k", "kb", "host_ms", "aten_ms"]
    for p in passes:
        cols += [f"sub{p}_ms", f"sub{p}_proj"]
    cols += ["kernel_ms", "d2h_ms", "rr_ms", "update_ms", "gershgorin_ms"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.3g}" if isinstance(v, float) else str(v))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
