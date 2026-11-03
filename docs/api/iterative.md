# Iterative Solvers

Both solvers operate on the host (PyTorch). The matvec / inner-product hot path would benefit from NKI Level 1/2 kernels in `trnblas`, but the iteration loop itself stays on CPU.

## `cg(A, b, x0=None, tol=1e-6, maxiter=1000, M=None)`

Conjugate Gradient for SPD `A`. `A` may be a tensor or a callable `A(x) → A @ x`.

- **`x0`** — initial guess; defaults to zeros
- **`tol`** — relative residual tolerance
- **`maxiter`** — iteration cap
- **`M`** — optional preconditioner (tensor or callable applying `M⁻¹`)

Returns `(x, iters, residual)`.

```python
x, iters, residual = trnsolver.cg(A_spd, b, tol=1e-8)
```

## `gmres(A, b, x0=None, tol=1e-6, maxiter=200, restart=30)`

GMRES for general (non-symmetric) systems. Same calling convention as `cg`. `restart` controls the Krylov subspace size before restart.

```python
x, iters, residual = trnsolver.gmres(A, b, tol=1e-6)
```

## `jacobi_preconditioner(A)`

Build a diagonal (Jacobi) preconditioner callable `M(r) = r / diag(A)` for use as the `M=` argument of `cg`. Cheap and effective when `A` is diagonally dominant or has widely varying diagonal scales.

```python
M = trnsolver.jacobi_preconditioner(A)
x, iters, res = trnsolver.cg(A, b, M=M, tol=1e-8)
```

Raises `ValueError` if any diagonal entry is within `1e-15` of zero. More-capable preconditioners (IC0, SSOR, block-Jacobi) are tracked in [#16](https://github.com/trnsci/trnsolver/issues/16).

## Block CG on matrix-free operators (#97)

### `cg_block(A, B, M=None, *, tol=1e-6, maxiter=None, check_every=10, X0=None, return_coeffs=False)`

Column-batched preconditioned CG for SPD `A` with an `(n, R)` right-hand side (or `(n,)`). `A` and `M` are callables on `(n, R)` blocks, or tensors. Each iteration makes **one** call `A(P)` on the whole block. Converged columns freeze, but the full block is still passed in, so every call has the same shape. Convergence is per column, `‖rⱼ‖ ≤ tol·‖bⱼ‖`. The host syncs only every `check_every` iterations.

Returns `(X, info)`. `info` has these keys:

- `iterations`: int, the operator calls in the loop.
- `col_iterations`, `converged`, `breakdown`: per-column.
- `residual`: the recurrence residual.
- `true_residual`: `‖B − AX‖/‖B‖`, from one extra call at the end.
- With `return_coeffs=True`, also `alpha`, `beta` (`(iterations, R)`) and `rz0`.

```python
P = trnsolver.pivoted_lowrank_preconditioner(kdiag, krows, rank=100, shift=noise)
X, info = trnsolver.cg_block(kernel_matmul, torch.cat([y, Z], dim=1), P, tol=1e-4)
```

### `cg_block_ir(A, B, A_hi=None, M=None, *, tol=1e-6, inner_tol=1e-4, max_outer=10, **cg_kwargs)`

Mixed-precision iterative refinement, the same shape as `gmres_ir`:

- the inner `cg_block` runs in fp32 on the fast operator `A`;
- the outer residual `B − A_hi(X)` is float64 on the host (`A_hi` defaults to `A` in float64);
- the corrections accumulate in float64.

Returns a float64 host `X` and `info` with these keys: `outer_iterations`, `inner_iterations`, `residual_history` (`(outer + 1, R)`), `true_residual`, `converged`.

### `pivoted_cholesky(diag_fn, rows_fn, rank, *, tol=0.0)`, `pivoted_lowrank_preconditioner(diag_fn, rows_fn, rank, shift, *, tol=0.0)`

Greedy pivoted Cholesky `K ≈ L Lᵀ`, built from `diag_fn() → (n,)` and one row per step, `rows_fn(idx) → (1, n)`. The preconditioner returns a `PivotedLowRankPreconditioner` that applies `(L Lᵀ + shift·I)⁻¹` with two `(n × k)` GEMMs. It has these attributes and methods:

- `.L`, `.shift`, `.W`;
- `logdet()`, giving `log det(L Lᵀ + shift·I)`;
- `sample(num, generator)`, drawing from `N(0, L Lᵀ + shift·I)`.

### `lanczos_tridiag_from_cg(alpha, beta, col_iterations=None)`, `slq_logdet(A, n, num_probes=32, num_iters=50, generator=None, M=None, *, dtype=torch.float32, device=None, tol=1e-10)`

`lanczos_tridiag_from_cg` builds the per-column Lanczos tridiagonals from CG coefficients. `slq_logdet` gives the stochastic Lanczos quadrature estimate of `log det(A)` and returns `(estimate, info)`. `info` has `std_error`, `quadratures` and `cg_info`.
