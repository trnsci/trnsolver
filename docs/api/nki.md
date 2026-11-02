# NKI Backend

Backend selection mirrors the sister `trnblas` and `trnfft` packages.

## `set_backend(backend)`

```python
trnsolver.set_backend("auto")  # NKI if available, else PyTorch (default)
trnsolver.set_backend("pytorch")  # force PyTorch fallback
trnsolver.set_backend("nki")  # require NKI; raises if unavailable
```

## `get_backend()`

Returns the current backend string.

## `HAS_NKI`

Module-level boolean — `True` iff the `nki` package (NKI 0.3.0 Stable, Neuron SDK 2.29+) imported successfully.

## Environment variables

| Variable | Effect |
|----------|--------|
| `TRNSOLVER_REQUIRE_NKI=1` | Kernel-path failures re-raise instead of silently falling back to PyTorch. Used by the validation suite to catch silent kernel breakage. |
| `TRNSOLVER_EIGH_NKI_MIN_N=<n>` | On the native `neuron` device (and for host tensors on a TorchNeuron host), `eigh` uses `torch.linalg.eigh` for matrices with fewer than `n` rows and the NKI path otherwise. The NKI path is single-tile (n ≤ 128) and falls back to `torch.linalg.eigh` above that. Default 512; `0` always uses NKI. Ignored when `TRNSOLVER_REQUIRE_NKI=1`. (#42) |
| `TRNSOLVER_EIGH_METHOD=auto\|jacobi` | `eigh` method when `method=` is not passed. `jacobi` selects the batched Jacobi solver (#48). Default `auto`. |
| `TRNSOLVER_JACOBI_GROUP=<g>` | Jacobi kernel: tiles per launch (rounded down to a power of two). Default 128 (n ≤ 32: scalar kernel; 33 ≤ n ≤ 128: block kernel), split across the two physical cores of a logical NeuronCore. |
| `TRNSOLVER_JACOBI_CHAINS=<c>` | Jacobi kernel: independent dependency chains per step. Default 8 for n ≤ 32, 2 for the block kernel. |
| `TRNSOLVER_EIGH_CORES=<k>\|auto` | Split a Jacobi `eigh` batch over logical NeuronCores, one host thread per core (#54). Default (unset or 1): one core. |
| `TRNSOLVER_EIGH_POOL=proc` | With `TRNSOLVER_EIGH_CORES`, solve a host batch on persistent per-core worker processes instead of threads (#57). Cores from `TRNSOLVER_EIGH_POOL_CORES`, else `NEURON_RT_VISIBLE_CORES`. `trnsolver.nki.eigh_pool.shutdown()` frees them. |
| `TRNSOLVER_JACOBI_SWEEPS_PER_LAUNCH=<k>` | Scalar Jacobi kernel (padded n ≤ 32): run the first sweeps k per launch. Default 1. |
| `TRNSOLVER_JACOBI_BLOCK=0` | Run the scalar Jacobi kernel for 33 ≤ n ≤ 128 instead of the block kernel (#48). |
| `TRNSOLVER_EIGH_JACOBI_AUTO=off\|<rows>` | Region where `method="auto"` runs the Jacobi kernel (FP32 batches on `neuron`). `off`, or rows `n_min-n_max:B_min`, e.g. `1-32:512,33-64:256`. Default: the measured table in `eigen._JACOBI_AUTO_REGION`. |
| `TRNSCI_NKI_DISPATCH=auto\|native\|xla\|cpu` | Suite-wide. Forces the native or XLA dispatch path; `auto` (default) detects. Forcing an unavailable path raises. |
| `TRNSCI_XLA_DEPRECATION=1` | The XLA path emits one `DeprecationWarning`. Off by default. |
| `NKI_ENABLE_TRACE_CACHE=0` | TorchNeuron (not trnsolver) setting. Turns off the cross-process NKI kernel cache, which cannot version editable installs. |
| `TRNSOLVER_USE_SIMULATOR=1` | Route kernel dispatch through `nki.simulate(kernel)(numpy_args)` on CPU instead of torch_xla. No hardware needed. See [Developing kernels](../developing_kernels.md). |

## Dispatch path

On Neuron hardware, kernels run on one of two `torch-neuronx` lineages,
detected on the first kernel launch:

- **Native (TorchNeuron, Trn2/Trn3)**: each kernel is a PyTorch custom op
  on `device="neuron"`, also callable directly and inside
  `torch.compile(backend="neuron")`: `torch.ops.trnsolver.matvec`,
  `rank2_update`, `matmul`, `matmul_fp32out`, `apply_reflectors`,
  `jacobi_sweeps`, `jacobi_block`. The ops
  are registered only once this path is selected.
- **XLA (Trn1)**: kernels run on the `torch_xla` device, as before.

`set_backend` is unaffected: it still chooses NKI vs PyTorch. See
[Installation](../installation.md#two-neuron-pytorch-lineages).

## Jacobi kernels

`trnsolver/nki/jacobi.py` holds the batched one-sided (Hestenes) Jacobi kernels behind `eigh(method="jacobi")` (#48): the scalar kernel (`trnsolver::jacobi_sweeps`, padded n ≤ 32) and the block kernel (`trnsolver::jacobi_block`, 33 ≤ n ≤ 128). Up to 128 tiles of 128 / n matrices stay in SBUF for a whole sweep; the Tensor Engine applies the step's partner permutation (and, in the block kernel, the Gram and block-rotation matmuls) and the Vector Engine the rotations. See [Eigensolvers](eigen.md).

The earlier `rotate_pairs_kernel` was removed; see [#9](https://github.com/trnsci/trnsolver/issues/9) for that post-mortem.
