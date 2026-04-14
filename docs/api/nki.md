# NKI Backend

Backend selection mirrors the sister `trnblas` and `trnfft` packages.

## `set_backend(backend)`

```python
trnsolver.set_backend("auto")     # NKI if available, else PyTorch (default)
trnsolver.set_backend("pytorch")  # force PyTorch fallback
trnsolver.set_backend("nki")      # require NKI; raises if unavailable
```

## `get_backend()`

Returns the current backend string.

## `HAS_NKI`

Module-level boolean — `True` iff the `nki` package (NKI 0.3.0 Stable, Neuron SDK 2.29+) imported successfully.

## Environment variables

| Variable | Effect |
|----------|--------|
| `TRNSOLVER_REQUIRE_NKI=1` | Kernel-path failures re-raise instead of silently falling back to PyTorch. Used by the validation suite to catch silent kernel breakage. |
| `TRNSOLVER_USE_SIMULATOR=1` | Route kernel dispatch through `nki.simulate(kernel)(numpy_args)` on CPU instead of torch_xla. No hardware needed. See [Developing kernels](../developing_kernels.md). |

## Jacobi rotation kernel

`trnsolver.nki.dispatch.rotate_pairs_kernel` is the primary NKI acceleration target. Each sweep round:

- Loads the `n/2` pairs of rows (even / odd at strided positions 2i, 2i+1) and rotates them by per-row `(c, -s; s, c)`
- The host driver calls the kernel three times per round (D rows, D cols, V cols) under a Brent-Luk permutation
- Compile graph is stable per `(half, n, dtype)` — NKI caches after the first invocation

See [Architecture](../architecture.md#nki-jacobi-strategy) and [#9](https://github.com/trnsci/trnsolver/issues/9) for the design rationale.
