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

Returns `A⁻¹ᐟ²` for SPD `A` via eigendecomposition: `A⁻¹ᐟ² = V Λ⁻¹ᐟ² Vᵀ`.
Robust default for general SPD, including ill-conditioned metrics.

```python
M = trnsolver.inv_sqrt_spd(metric)
```

## `inv_sqrt_spd_ns(A, *, max_iters=20, tol=1e-7)`

Returns `(X, iters, residual)` where `X ≈ A⁻¹ᐟ²`, computed via the coupled
Newton-Schulz iteration `T = ½(3I − ZY); Y ← YT; Z ← TZ` starting from a
Frobenius-norm-scaled `Y₀`. All operations are GEMMs, which maps cleanly to
the Trainium Tensor Engine via `trnblas.gemm` once that backend validates.

Choose this over `inv_sqrt_spd` when the NKI backend is active **and** `A` is
well-conditioned (condition number ≲ 10⁶). For ill-conditioned SPD, stick
with the eigendecomposition path.

```python
M, iters, res = trnsolver.inv_sqrt_spd_ns(metric, tol=1e-8)
```
