# Iterative Solvers

Both solvers operate on the host (PyTorch). The matvec / inner-product hot path would benefit from NKI Level 1/2 kernels in `trnblas`, but the iteration loop itself stays on CPU.

## `cg(A, b, x0=None, tol=1e-6, maxiter=1000, M=None, *, reduction=None)`

Conjugate Gradient for SPD `A`. `A` may be a tensor or a callable `A(x) → A @ x`.

- **`x0`** — initial guess; defaults to zeros
- **`tol`** — relative residual tolerance
- **`maxiter`** — iteration cap
- **`M`** — optional preconditioner (tensor or callable applying `M⁻¹`)
- **`reduction`** — how `rᴴz`, `pᴴAp` and `‖r‖` reduce over n (#128): `"pairwise"` (default), `"host64"`, `"ff"`, `"exact"`, `"auto"` or `"device"`, as in [`cg_block`](#cg_blocka-b-mnone--tol1e-6-maxiternone-check_every10-x0none-return_coeffsfalse-reductionnone) (see [fp64-class reductions](#fp64-class-reductions-ff-exact-auto) for the last three). `None` reads `TRNSOLVER_CG_REDUCTION`. In the `pairwise` / `host64` / `device` modes, host tensors are multiplied in float64, as before; on the device the product is in the working dtype, then folded and finished in float64 on the host. `"ff"` / `"exact"` take the working dtype on both sides, so the host gives the device's bits.

Returns `(x, iters, residual)`.

```python
x, iters, residual = trnsolver.cg(A_spd, b, tol=1e-8)
```

## `gmres(A, b, x0=None, tol=1e-6, maxiter=200, restart=30, *, reduction=None)`

GMRES for general (non-symmetric) systems. Same calling convention as `cg`. `restart` controls the Krylov subspace size before restart.

`reduction` (#128) applies to the modified Gram-Schmidt inner products and the norms: `"pairwise"` (default), `"host64"`, `"ff"`, `"exact"`, `"auto"` or `"device"`. In every mode but `"device"`, the update `x + V y` is formed as fixed-order axpys instead of a matmul. `None` reads `TRNSOLVER_GMRES_REDUCTION`.

Cost on Trn2 (LNC=2, one logical core, n = 1e5, the GP exemplar's `KernelOperator` applied to one vector, 289 ms per call, 60 iterations):

| solver | `device` ms/iter | `pairwise` ms/iter | `host64` ms/iter |
|---|---|---|---|
| `cg` | 299.7 | 299.9 (+0.1%) | 296.7 |
| `gmres` (restart 30) | 375.9 | 337.6 (−10%) | 324.6 |

`gmres` gets faster because the old path cast both vectors to float64 on the device for every Gram-Schmidt dot.

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

### `cg_block(A, B, M=None, *, tol=1e-6, maxiter=None, check_every=10, X0=None, return_coeffs=False, reduction=None)`

Column-batched preconditioned CG for SPD `A` with an `(n, R)` right-hand side (or `(n,)`). `A` and `M` are callables on `(n, R)` blocks, or tensors. Each iteration makes **one** call `A(P)` on the whole block. Converged columns freeze, but the full block is still passed in, so every call has the same shape. Convergence is per column, `‖rⱼ‖ ≤ tol·‖bⱼ‖`. The stopping test reads back to the host only every `check_every` iterations.

`reduction` sets how the inner products and norms reduce over n (#128). `None` reads `TRNSOLVER_CG_REDUCTION`.

| `reduction` | How | LNC=1 vs LNC=2 | Cost |
|---|---|---|---|
| `"pairwise"` (default) | zero-pad to m · 2^L rows (m ≤ 64), L elementwise adds of contiguous halves on the device, then the m leaves summed in float64 on the host | bitwise equal | two small D2H syncs per iteration |
| `"host64"` | the whole (n, R) product is copied to the host and reduced in float64 | bitwise equal | one (n, R) D2H per reduction |
| `"ff"` | trnblas.fp64 double-word `dot` / `nrm2`, one per column (below) | bitwise equal | one small D2H per column and reduction |
| `"exact"` | trnblas.fp64 exactly rounded `dot` (below) | bitwise equal | as `"ff"`, 1.3–1.5× its cost through n = 1e6 |
| `"auto"` | `"host64"` below 5e5 rows, `"ff"` from there (below) | bitwise equal | |
| `"device"` | device `sum` / `vector_norm` (the previous behaviour) | differs (5e-4 relative at n = 1e5) | fastest; for comparison |

Device reductions over n add in an order that depends on how the compiler splits the work across logical cores. Elementwise ops do not. So every mode but `"device"` gives the same bits at any LNC, provided the operator `A` (and `M`) does too.

#### fp64-class reductions: `"ff"`, `"exact"`, `"auto"`

These modes (trnsci/trnsci#169) hand the working-dtype vectors to [`trnblas.fp64`](https://github.com/trnsci/trnblas) (trnblas ≥ 0.14, a dependency):

- **`"ff"`**: `fp64.dot(accumulate="accurate")` and `fp64.nrm2`. Products are exact (Dekker TwoProd), the lanes accumulate in double-word fp32 (AccurateDWPlusDW), and the 1024 fp32 chunk partials are summed exactly on the host. The order depends on n only, so the result is the same at LNC=1 and LNC=2. The trnblas CPU reference issues the kernels' fp32 operations one for one, so a host tensor gives the device's bits. `nrm2` scales each chunk by a power of two, so the norm does not overflow where fp32 squares would.
- **`"exact"`**: `fp64.dot(accumulate="exact")`, the integer-limb arm. The result is the exactly rounded Σ xᵢ yᵢ of the fp32 data, which is unique, so it is the same on every backend and LNC. Terms outside the device's 126-bit window, and products outside the fp32 range, take an exact host path. Norms are √(exact Σ xᵢ²).
- **`"auto"`**: `"host64"` for n < 5e5 rows, else `"ff"`. The policy depends on n only, not on the device. Set `TRNSOLVER_REDUCTION_FF_MIN_N` to move the threshold (`0` always takes `"ff"`). `cg_block` reports the mode it used in `info["reduction"]`.

Float-float error grows like 2⁻⁴⁸ · cond, where cond = Σ|xᵢ yᵢ| / |x·y|. CG's `rᴴz`, `pᴴAp` and norms are well conditioned. For a near-orthogonal dot, such as a late Gram-Schmidt coefficient in `gmres`, only `"exact"` stays exact. float64 / complex128 data (host only) is already float64: there `"ff"` reduces in float64 as `"host64"` does, and `"exact"` uses `fp64.exact_dot`. A complex vector is reduced as its (re, im) pairs, in one reduction of length 2n.

Measured on Trn2 (window 7, one logical core at LNC=2; trnsci/trnsci#169). Each cell is ms per call (median) and correct bits on the same fp32 CG vectors:

| n | `pairwise` | `host64` | `"ff"` (`fp64.dot` accurate) | `"exact"` |
|---|---|---|---|---|
| 1e5 | 1.44 (29.4 b) | 0.33 (32.1 b) | 0.57 (52.5 b) | 0.76 (53 b) |
| 1e6 | 1.95 (30.7 b) | 1.80 (35.0 b) | 1.37 (53 b) | 2.05 (53 b) |
| 1e7 | 2.10 (27.8 b) | 38.9 (38.0 b) | 2.57 (53 b) | 7.90 (53 b) |

`host64` on the device is limited by the fp32 product it copies back (about 32–38 bits); `"ff"` and `"exact"` form the products exactly. `"ff"` is cheaper than `host64` from about 5e5 rows (1.37 vs 1.80 ms at 1e6, 2.57 vs 38.9 ms at 1e7), which sets `"auto"`'s threshold. `"exact"` costs 1.3–1.5× `"ff"` through n = 1e6 and 3.1× at 1e7, so `"auto"` does not pick it. Use it where a reduction must be exact, at n ≤ 1e6.

`cg_block` runs one trnblas reduction per column. The projected preconditioner's Qᵀ V (k · R dots) keeps the `"pairwise"` blocked product under `"ff"`, `"exact"` and `"auto"`. A preconditioner needs a fixed order, not 53 bits: an error in M slows CG down but does not move the solution. The per-iteration and end-to-end cost in `cg` and `cg_block` is measured by `scripts/bench_cg_reduction_ff.py`, at n = 1e5, 1e6 and 1e7 for each mode.

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

### `pivoted_cholesky(diag_fn, rows_fn, rank, *, tol=0.0, block=1, sub_fn=None, relax=1.0)`, `pivoted_lowrank_preconditioner(diag_fn, rows_fn, rank, shift, *, tol=0.0, apply="auto")`

Greedy pivoted Cholesky `K ≈ L Lᵀ`, built from `diag_fn() → (n,)` and rows `rows_fn(idx) → (len(idx), n)`. With the default `block=1` it asks for one row per pivot.

`block > 1` (#114) batches the rows, so a matrix-free `K` sees one large kernel evaluation per round instead of one per pivot:

- A round takes the `block` largest remaining diagonals as candidates. It picks pivots greedily from their exact `c × c` Schur complement, and accepts a candidate while its remaining diagonal is at least `relax` times the largest one outside the round.
- With `relax=1` these are the pivots the one-row algorithm takes, up to ties and rounding.
- The accepted pivots' rows come from one `rows_fn` call. One GEMM gives their Schur rows and one triangular solve gives the new columns, all in float64 on the host.
- `sub_fn(idx) → K[idx][:, idx]` (`c²` entries) lets the round choose pivots before any row is requested, so `rows_fn` is asked only for accepted pivots. Without it, every candidate row is requested.
- On the GP kernel (RBF, ℓ = 0.15, d = 3, rank 1024), `block=512, relax=0.5` makes 26 calls at n = 10⁵ and 61 at n = 10⁶, against 1024. The CG iteration counts match the one-row build.

```python
def krows(idx):  # K(X[idx], X) on the device: one launch per round
    eye = torch.eye(len(idx), device="neuron")
    return trnsparse.kernel_matmul(X, X[idx], eye, "rbf", ell, s).T


def ksub(idx):  # K[idx][:, idx], c × c on the host
    return trnsparse.kernel_matmul_reference(
        X[idx], X[idx], torch.eye(len(idx), dtype=torch.float64), "rbf", ell, s
    )


L = trnsolver.pivoted_cholesky(op.diag, krows, 1024, block=512, relax=0.5, sub_fn=ksub)
```

**float64: keep K on the host when `rows_fn` gathers.** The build is float64 on the host, but L is only as accurate as the rows `rows_fn` returns, and the library can't see inside the callable. On Trn2 a device-side gather of float64 (`K[idx]`, `index_select`, `torch.cat`) rounds to fp32, with a relative error of about 3.6e-8. So a `rows_fn` that gathers rows from a float64 `K` on the `neuron` device gives a `pivoted_lowrank_preconditioner` (or `pivoted_cholesky`) accurate to about 2e-8, not about 1e-15. For float64, keep `K`, or whatever data `rows_fn` reads, on the host. Plain copies (`.to`, `.clone()`, `.contiguous()`, basic indexing) are exact.

`PivotedLowRankPreconditioner(L, shift, apply=...)` applies `(L Lᵀ + shift·I)⁻¹`. It has these attributes and methods:

- `.L`, `.shift`, `.W` (or `.Q`, `.g` for `"projected"`);
- `logdet()`, giving `log det(L Lᵀ + shift·I)`;
- `sample(num, generator)`, drawing from `N(0, L Lᵀ + shift·I)`.

`apply` takes one of these values:

| `apply` | where | notes |
|---|---|---|
| `"host64"` | float64 on the host | one `(n, R)` round trip per call |
| `"native"` | L's dtype and device | `(V − W WᵀV)/s`, two GEMMs; cancels in fp32, so use it only for float64 L |
| `"projected"` | L's dtype and device | stable in fp32, four GEMMs, no host round trip (#114) |
| `"auto"` (default) | | `"native"` for float64 L, else `"host64"` |

How `"projected"` works:

- It uses an orthonormal basis Q of L's range, from the Gram matrix in float64 on the host, with `L Lᵀ = Q diag(σ²) Qᵀ`.
- It applies `Q (Σ² + sI)⁻¹ QᵀV + (I − QQᵀ)V/s` with the projection done twice.
- One pass leaves an error of about u·√k·‖V‖ along Q, which the 1/s weight amplifies. That is the `"native"` failure. The second pass removes that error.
- On CPU fp32 it takes the same CG iteration counts as `"host64"` on the GP cases that stall `"native"`.
- Its two QᵀV products reduce over n, so they take `reduction=` as `cg_block` does (`PivotedLowRankPreconditioner(L, shift, apply="projected", reduction=None)`; `None` reads `TRNSOLVER_CG_REDUCTION`). With `"pairwise"`, QᵀV is formed as 128-row block products `Q_iᵀ V_i` in one bmm, then the blocks are folded as above and finished in float64 on the host (#128). `"ff"`, `"exact"` and `"auto"` take the `"pairwise"` QᵀV.

### `lanczos_tridiag_from_cg(alpha, beta, col_iterations=None)`, `slq_logdet(A, n, num_probes=32, num_iters=50, generator=None, M=None, *, dtype=torch.float32, device=None, tol=1e-10, reduction=None)`

`lanczos_tridiag_from_cg` builds the per-column Lanczos tridiagonals from CG coefficients. `slq_logdet` gives the stochastic Lanczos quadrature estimate of `log det(A)` and returns `(estimate, info)`. `info` has `std_error`, `quadratures` and `cg_info`.
