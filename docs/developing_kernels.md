# Developing NKI kernels

trnsolver currently ships one NKI kernel (`rotate_pairs_kernel`) in
`trnsolver/nki/dispatch.py`. It's the per-round primitive for the
parallel Jacobi sweep in `_jacobi_eigh_nki`. This page is the short
trnsolver-specific version of the suite-wide guide at
[`trnsci/docs/developing_kernels.md`](https://github.com/trnsci/trnsci/blob/main/docs/developing_kernels.md).

## Dispatch modes

| Mode | Trigger | When to use |
|------|---------|-------------|
| **PyTorch fallback** | `HAS_NKI = False` (non-Neuron host), or an exception in the NKI path gets caught | Laptops, GPUs, CI's `ubuntu-latest` test matrix — the default for anyone without Neuron installed |
| **NKI hardware, native** | `HAS_NKI = True` + TorchNeuron `torch_neuronx` (has `nki_op`) + a `neuron` device. Kernel runs as a `trnsolver::` custom op on `device="neuron"` | Trn2/Trn3 on Neuron SDK 2.33+. Set `NKI_ENABLE_TRACE_CACHE=0` with an editable install. |
| **NKI hardware, XLA** | `HAS_NKI = True` + `torch_xla` installed (and the native path unavailable). Kernel runs through `torch_xla` → NEFF compile → NeuronCore | Trn1; real perf numbers, final validation |
| **NKI simulator** | `TRNSOLVER_USE_SIMULATOR=1` + `HAS_NKI = True`. Kernel runs through `nki.simulate(kernel)(numpy_args)` on CPU | Fast correctness iteration during kernel design |

All modes share the same kernel source — the `@nki.jit` definition
inside `if HAS_NKI:` in `trnsolver/nki/dispatch.py`. The host driver
(`_jacobi_eigh_nki` in `trnsolver/eigen.py`) picks the path based on
`_use_nki()` and `_use_simulator()`.

Each host wrapper in `trnsolver/eigen.py` hands its operands to
`_launch(kernel, op_name, *tensors)` in `dispatch.py`, which picks the
native custom op or the torch_xla path. A new hardware kernel needs a
matching annotated wrapper in `trnsolver/nki/_native.py` and an entry in its
`OPS` table. On the native path, keep scalar control flow (`.item()`,
element writes) on host copies: each one is a device round trip.

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
