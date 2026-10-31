"""Can one large matrix's Jacobi sweeps pay off across NeuronCores? (#29)

Block one-sided Jacobi on an (n, n) matrix with column blocks of width b:
a step pairs the n / b blocks into n / (2 b) disjoint pairs and, for each
pair, forms the (2b, 2b) Gram matrix of its columns, diagonalises it with
the batched device Jacobi (`jacobi._solve`, 2 b <= 128), and rotates the
pair's Z and V columns by the eigenvectors. Pairs are independent, so a
step splits over cores; between steps, blocks move to new partners, which
across cores means device -> host -> device.

This probe times the pieces of one step on the devices, without running a
full solver:

    step     one block step for P pairs on one core (Gram bmm, batched
             Jacobi on the Grams, two update bmms), P = all pairs or P / k.
    xfer     staging one step's outgoing blocks of Z and V through the host
             for k cores at once (each core sends half its pairs' columns,
             the circle-method exchange), d2h then h2d.

Prints ``RESULT`` lines; the projected k-core step is
``step(P / k) + xfer(k)`` against ``step(P)`` on one core. ``step`` runs on
the first visible core for every k (``--probes step`` on one core); ``xfer``
needs k visible cores.
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import torch


def _med(fn, reps, sync):
    fn()
    sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="1024,2048,4096")
    ap.add_argument("--b", type=int, default=64)
    ap.add_argument("--cores", default="1,2,4")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--probes", default="step,xfer")
    args = ap.parse_args()

    import torch_neuronx

    from trnsolver.nki import jacobi as JK

    ndev = torch_neuronx.device_count()
    devs = [torch.device(f"neuron:{i}") for i in range(ndev)]

    def sync():
        for d in devs:
            torch_neuronx.synchronize(d)

    b = args.b
    for n in [int(x) for x in args.n.split(",")]:
        pairs = n // (2 * b)
        g = torch.Generator().manual_seed(n)
        for k in [int(x) for x in args.cores.split(",")]:
            P = pairs // k
            d0 = devs[0]
            # Z and V columns of P pairs as (P, n, 2b) on one core.
            Zh = torch.randn(P, n, 2 * b, generator=g)
            Z = Zh.to(d0)
            Vt = torch.randn(P, n, 2 * b, generator=g).to(d0)

            def step(Z=Z, Vt=Vt):
                G = torch.bmm(Z.transpose(1, 2), Z)
                _, U, _ = JK._solve(G)
                return torch.bmm(Z, U), torch.bmm(Vt, U)

            if "step" in args.probes:
                t_step = _med(step, args.reps, sync)
                print(f"RESULT probe=step n={n} b={b} pairs={P} ms={t_step:.2f}", flush=True)
            del Z, Vt
            if k == 1 or k > ndev or "xfer" not in args.probes:
                continue
            # Exchange: each core ships one block per pair of Z and V (half
            # its columns), so per core (P, n, b) twice, out and back in.
            blocks = [torch.randn(2, P, n, b, generator=g).to(devs[i]) for i in range(k)]
            pool = ThreadPoolExecutor(k)

            def xfer(blocks=blocks, k=k, pool=pool):
                def one(j):
                    h = blocks[j].cpu()
                    return h.to(devs[(j + 1) % k])

                return list(pool.map(one, range(k)))

            t_x = _med(xfer, args.reps, sync)
            mb = 2 * P * n * b * 4 / 1e6
            print(
                f"RESULT probe=xfer n={n} b={b} cores={k} mb_per_core={mb:.1f} ms={t_x:.2f}",
                flush=True,
            )
            del blocks
            pool.shutdown()


if __name__ == "__main__":
    main()
