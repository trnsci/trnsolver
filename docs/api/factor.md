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

This includes A `(*, n, n, n)` with B `(*, n, n)`, where A's last batch dim
equals `n`: `B` is a batch of vectors and `X` is `(*, n, n)`, as in
`torch.linalg.solve`.

!!! note "Changed in 0.23.0 (#130)"
    trnsolver 0.22.1 and earlier read that `B` as `(*, n, k)` right-hand-side
    matrices broadcast over A's batch (result `(*, n, n, n)`; 0.22.1 warned,
    #79). For that result, pass the explicit form, which means the same in
    every release:

    ```python
    X = trnsolver.solve_spd(A, B.unsqueeze(-3))  # 3-D A: one shared (n, k) right-hand side
    X = trnsolver.solve_spd(A, B.unsqueeze(0))  # any A: the 0.22.1 reading
    ```

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
