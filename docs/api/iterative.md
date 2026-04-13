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
