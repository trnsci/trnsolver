# Factorizations & Direct Solvers

## `cholesky(A, upper=False)`

`A = L Lᵀ` (or `Uᵀ U` with `upper=True`) for SPD `A`. Returns `L` (or `U`).

```python
L = trnsolver.cholesky(A_spd)
```

## `lu(A)`

Pivoted LU decomposition. Returns `(P, L, U)` with `P A = L U`.

## `qr(A)`

`A = Q R` with `Q` orthogonal and `R` upper-triangular.

## `solve(A, B)`

Solves `A X = B` via LU.

## `solve_spd(A, B)`

Solves `A X = B` via Cholesky — faster than `solve` when `A` is SPD.

## `inv_spd(A)`

Returns `A⁻¹` for SPD `A` via Cholesky.

## `inv_sqrt_spd(A)`

Returns `A⁻¹ᐟ²` for SPD `A`. Used as the density-fitting metric inverse in DF-MP2 (paired with `gemm` in trnblas).

Currently implemented via eigendecomposition: `A⁻¹ᐟ² = V Λ⁻¹ᐟ² Vᵀ`. A Newton-Schulz GEMM-only variant is on the roadmap once the NKI GEMM in trnblas is validated.

```python
M = trnsolver.inv_sqrt_spd(metric)
```
