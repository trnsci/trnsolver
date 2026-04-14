# Developing NKI kernels

trnsolver currently ships one NKI kernel (`rotate_pairs_kernel`) in
`trnsolver/nki/dispatch.py`. It's the per-round primitive for the
parallel Jacobi sweep in `_jacobi_eigh_nki`. This page is the short
trnsolver-specific version of the suite-wide guide at
[`trnsci/docs/developing_kernels.md`](https://github.com/trnsci/trnsci/blob/main/docs/developing_kernels.md).

## Three dispatch modes

| Mode | Trigger | When to use |
|------|---------|-------------|
| **PyTorch fallback** | `HAS_NKI = False` (non-Neuron host), or an exception in the NKI path gets caught | Laptops, GPUs, CI's `ubuntu-latest` test matrix — the default for anyone without Neuron installed |
| **NKI hardware** | `HAS_NKI = True`, default env. Kernel runs through `torch_xla` → NEFF compile → NeuronCore | Real perf numbers, final validation |
| **NKI simulator** | `TRNSOLVER_USE_SIMULATOR=1` + `HAS_NKI = True`. Kernel runs through `nki.simulate(kernel)(numpy_args)` on CPU | Fast correctness iteration during kernel design |

All three modes share the same kernel source — the `@nki.jit` definition
inside `if HAS_NKI:` in `trnsolver/nki/dispatch.py`. The host driver
(`_jacobi_eigh_nki` in `trnsolver/eigen.py`) picks the path based on
`_use_nki()` and `_use_simulator()`.

## Simulator workflow

NKI 0.3.0 Stable (Neuron SDK 2.29, April 2026) ships a CPU simulator
that runs kernels without Trainium hardware. It collapses the iteration
loop from ~8–12 min per attempt (instance start + SSM + NEFF compile)
to seconds per iteration.

Three equivalent ways to run:

```bash
# (1) GH Actions — automatic on every push, zero AWS cost
#     See .github/workflows/ci.yml::nki-simulator

# (2) Local Linux x86_64 with nki installed
TRNSOLVER_USE_SIMULATOR=1 pytest tests/ -m nki_simulator -v

# (3) SSM against the provisioned trn1 CI instance (uses the bundled Neuron venv)
AWS_PROFILE=aws ./scripts/run_simulator_tests.sh
```

## CI coverage

| Gate | Runner | Catches | Misses |
|------|--------|---------|--------|
| `test` matrix (py 3.10/3.11/3.12) | `ubuntu-latest` | Pure-Python correctness against `torch.linalg.eigh`. ~1 s. | Anything NKI-kernel-specific. |
| `nki-simulator` | `ubuntu-latest` | Python trace-level kernel errors: wrong `nc_matmul` kwargs, dropped ops (`nl.divide`), shape mismatches, tile-size violations. Seconds per kernel. | MLIR verifier errors — simulator explicitly skips compile. Perf. |
| `neuron` (SSM, manual) | `trn1.2xlarge` | Full NEFF compile + on-hardware execution. MLIR verification. Real perf. | Nothing (this is the ground truth). |

The `nki-simulator` gate catches the majority of the iteration pain
(Python-trace breakage) without AWS round-trips. Hardware runs are
reserved for MLIR verification and perf numbers.

## Architecture-first (reminder)

Every NKI kernel in trnsolver should pull at least one documented
architectural lever from [#36](https://github.com/trnsci/trnsolver/issues/36)
— stationary-operand reuse, FP32-PSUM-free mixed precision, 4-engine
concurrency, SBUF residency. Kernel changes that are "port scipy faster"
get re-scoped or rejected.

The rotation-kernel post-mortem on [#9](https://github.com/trnsci/trnsolver/issues/9)
is the concrete lesson.
