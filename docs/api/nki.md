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

Module-level boolean — `True` iff `neuronxcc.nki` imported successfully.

## Environment variables

| Variable | Effect |
|----------|--------|
| `TRNSOLVER_REQUIRE_NKI=1` | Kernel-path failures re-raise instead of silently falling back to PyTorch. Used by the validation suite to catch silent kernel breakage. |

## Jacobi rotation kernel

`trnsolver.nki.dispatch.jacobi_rotation_kernel` is the primary NKI acceleration target. Each Givens rotation:

- Loads rows `p` and `q` of `D` (the working symmetric matrix) and rotates them via `(c, -s; s, c)`
- Mirrors the rotation on columns `p` and `q` (D is symmetric)
- Zeros the `(p, q)` and `(q, p)` off-diagonal entries
- Accumulates the rotation into `V` (the eigenvector matrix)

Currently scaffolded — falls back to `torch.linalg.eigh` until on-hardware validation completes. See [Architecture](../architecture.md#nki-jacobi-strategy) for the mapping rationale.
