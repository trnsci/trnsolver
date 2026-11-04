# Factorizations & Direct Solvers

## `cholesky(A, upper=False)`

`A = L Lᵀ` (or `Uᵀ U` with `upper=True`) for SPD `A`. Returns `L` (or `U`).

```python
L = trnsolver.cholesky(A_spd)
```

## `lu(A)`

Pivoted LU decomposition. Returns `(P, L, U)` with `A = P L U`, the
`torch.linalg.lu` convention (equivalently `Pᵀ A = L U`).

## `qr(A)`

`A = Q R` with `Q` orthogonal and `R` upper-triangular.

## `solve(A, B)`

Solves `A X = B` via LU.

## `solve_spd(A, B)`

Solves `A X = B` via Cholesky — faster than `solve` when `A` is SPD.

`B` is `(n,)`, `(*, n, k)`, or a batch of vectors `(*, n)` with
`B.shape == A.shape[:-1]`, as `torch.linalg.solve` reads it.

!!! warning "Deprecated in 0.22.1: an ambiguous `B` shape (#79)"
    When A's last batch dim equals `n` (A `(*, n, n, n)`, B `(*, n, n)`), `B`
    can be read either way. `solve_spd` still reads it as `(*, n, k)`
    right-hand-side matrices broadcast over A's batch (result
    `(*, n, n, n)`), and now emits a `FutureWarning`. From 0.23.0 it reads
    it as a batch of vectors, as `torch.linalg.solve` does (result
    `(*, n, n)`; [#130](https://github.com/trnsci/trnsolver/issues/130)).
    Be explicit; both forms mean the same in every release:

    ```python
    X = trnsolver.solve_spd(A, B.unsqueeze(-1))  # batch of vectors -> (*, n, n, 1)
    X = trnsolver.solve_spd(A, B.unsqueeze(-3))  # one shared (n, k) right-hand side
    ```

    For a 3-D `A`, `B.unsqueeze(-3)` is the current result; with more batch
    dims, `B.unsqueeze(0)` is.

## `inv_spd(A)`

Returns `A⁻¹` for SPD `A` via Cholesky.

## `inv_sqrt_spd(A)`

Returns `A⁻¹ᐟ²` for SPD `A` via eigendecomposition: `A⁻¹ᐟ² = V Λ⁻¹ᐟ² Vᵀ`.
Robust default for general SPD, including ill-conditioned metrics.

```python
M = trnsolver.inv_sqrt_spd(metric)
```

## `inv_sqrt_spd_ns(A, *, max_iters=20, tol=1e-7)`

Returns `(X, iters, residual)` where `X ≈ A⁻¹ᐟ²`, computed via the coupled
Newton-Schulz iteration `T = ½(3I − ZY); Y ← YT; Z ← TZ` starting from a
Frobenius-norm-scaled `Y₀`. All operations are GEMMs, which maps cleanly to
the Trainium Tensor Engine via `trnblas.gemm` (when trnblas is installed; on the
native `neuron` device its `auto` backend routes dense GEMMs to aten, trnblas 0.9.0).

Choose this over `inv_sqrt_spd` when the NKI backend is active **and** `A` is
well-conditioned (condition number ≲ 10⁶). For ill-conditioned SPD, stick
with the eigendecomposition path.

```python
M, iters, res = trnsolver.inv_sqrt_spd_ns(metric, tol=1e-8)
```
