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

## `inv_sqrt_spd_ns(A, iters=None, *, max_iters=20, tol=1e-7, polynomial="newton", cond=None, scaling=None, precision=None)`

Returns `(X, iters, residual)` where `X ≈ A⁻¹ᐟ²`, computed via the coupled
Newton-Schulz iteration `T = c₁I + c₃ZY; Y ← YT; Z ← TZ` from `Y₀ = A / s`,
`Z₀ = I`, with `s ≥ λ_max(A)` a cheap norm bound. Newton-Schulz is
`(c₁, c₃) = (3/2, −1/2)`. All O(n³) work is matrix products, which maps to the
Trainium Tensor Engine. `residual` is `‖YZ − I‖_F / √n`.

- `iters=None` (adaptive, the original behaviour): up to `max_iters`
  iterations, stopping at `residual < tol`, Frobenius scaling, products through
  `trnblas.gemm` when trnblas is installed.
- `iters=k` (fixed, #149): exactly `k` iterations, no host synchronisation in
  the loop (one static schedule), `scaling="bound"` (the smaller of `‖A‖_F` and
  `‖A‖_∞`). `trnsolver.matfun.ns_iteration_count(cond, tol, polynomial)` gives
  `k` ahead of time from a condition bound alone.
- `polynomial="minimax"`: the greedy minimax odd-cubic step polynomials of
  "Polar Express" (Amsel et al., arXiv 2505.16932) for eigenvalues of `A / s` in
  `[1 / cond, 1]`. The coefficients tend to Newton-Schulz as the iteration
  converges. To 1e-14: 7 / 9 / 12 / 14 iterations at `cond` = 1e2 / 1e4 / 1e6 /
  1e8, against 11 / 17 / 22 / 28 for Newton-Schulz. `cond` bounds `s / λ_min`;
  an underestimate is safe and only slows the smallest eigenvalues.
- `precision=`: how each product is computed (see `density_purification` in
  [Eigensolvers](eigen.md)), e.g. `"ozaki1"` for `trnblas.gemm_fp64` or a list
  of per-iteration specs for staged precision.

Choose this over `inv_sqrt_spd` when the NKI backend is active **and** `A` is
well-conditioned (condition number ≲ 10⁶). For ill-conditioned SPD, stick
with the eigendecomposition path.

```python
M, iters, res = trnsolver.inv_sqrt_spd_ns(metric, tol=1e-8)

from trnsolver.matfun import ns_iteration_count

k = ns_iteration_count(1e6, 1e-13, "minimax")
X, _, res = trnsolver.inv_sqrt_spd_ns(S, k, polynomial="minimax", cond=1e6)
```
