# Eigensolvers

## `eigh(A, tol=1e-10, *, method=None)`

Symmetric eigendecomposition `A = V Λ Vᵀ` for real symmetric `A`. Returns `(eigenvalues, eigenvectors)` with eigenvalues in ascending order.

`method=None` reads `TRNSOLVER_EIGH_METHOD` (default `auto`).

- `auto`: on the native `neuron` device, n < `TRNSOLVER_EIGH_NKI_MIN_N` (default 512) uses `torch.linalg.eigh`; otherwise the NKI path (Householder tridiagonalization, host QR iteration). Off-device, `torch.linalg.eigh`.
- `jacobi` (#48): batched Jacobi for `(n, n)` or `(..., n, n)` input. On the native `neuron` device (FP32, n ≤ 128) it runs the SBUF-resident one-sided Jacobi kernel: one launch per sweep, each holding up to 128 tiles of 128 / n matrices (512 at n = 32) in SBUF. Elsewhere it runs a torch implementation (two-sided Jacobi as batched matmuls). Uses the lower triangle, like `torch.linalg.eigh`. For 33 ≤ n ≤ 128 the kernel is block Jacobi: per step the Tensor Engine forms each matrix's Gram matrix and applies one orthogonal matmul per block pair, and the Vector Engine only rotates the 16 × 16 pair Gram blocks.
- Multi-core (opt-in): `TRNSOLVER_EIGH_CORES=<k>|auto` splits a Jacobi batch over logical NeuronCores, one host thread per core (#48). With `TRNSOLVER_EIGH_POOL=proc` as well, a **host** batch goes to persistent worker processes instead, one per core (`TRNSOLVER_EIGH_POOL_CORES`, else `NEURON_RT_VISIBLE_CORES`), through shared memory (#29). The pool is the faster split (Trn2, n = 128, B = 4096, 8 host CPUs: 324 ms on 4 cores against 531 ms with threads and 655 ms for 8-thread batch-parallel LAPACK), but a NeuronCore belongs to one process: the calling process must not use those cores itself, so device-resident input keeps the thread split. Workers start on the first call (about 10 s, plus any kernel compile) and stay up; call `trnsolver.nki.eigh_pool.shutdown()` to free the cores. Scripts that use the pool need an `if __name__ == "__main__":` guard (the workers are spawned). `TRNSOLVER_JACOBI_SWEEPS_PER_LAUNCH=<k>` runs the scalar kernel's (n ≤ 32) first sweeps k per launch: 5–14% faster on one core (Trn2).
- `auto` also runs the Jacobi kernel for FP32 batches on the native `neuron` device inside a measured region (n ≤ 32 with B ≥ 512, 33 ≤ n ≤ 64 with B ≥ 256, 65 ≤ n ≤ 128 with B ≥ 1024, plus B ≥ 256 for n ≥ 96 and B ≥ 64 at n = 128), where it beats the aten cutover. `TRNSOLVER_EIGH_JACOBI_AUTO` overrides the region.

```python
eigenvalues, eigenvectors = trnsolver.eigh(A)
w, V = trnsolver.eigh(A_batch.to("neuron"), method="jacobi")  # (B, n), (B, n, n)
```

## `eigh_generalized(A, B)`

Generalized symmetric eigenproblem `A x = λ B x` with `B` SPD — the form that arises in the SCF iteration as `F C = S C ε`.

Reduces to standard form via Cholesky of `B`:

1. `B = L Lᵀ`
2. `A' = L⁻¹ A L⁻ᵀ`
3. `eigh(A') → (Λ, V')`
4. Back-transform `V = L⁻ᵀ V'`

```python
eigenvalues, eigenvectors = trnsolver.eigh_generalized(F, S)
```

## Backend notes

Householder tridiagonalization is n − 2 dependent BLAS-2 steps and leaves the Tensor Engine idle (#48). A parallel-ordered Jacobi step rotates n/2 disjoint pairs at once, so a batch of matrices advances in lockstep with no long dependency chain. In the kernel the Tensor Engine applies the step's partner permutation (an exact matmul shared by every tile) and the Vector Engine applies the rotations. For measurements against batched CPU LAPACK see #48.

## `occupied_subspace(F, k, *, guess=None, buffer=None, passes=2, tol=None, degree=None, upper="gershgorin", refresh=3e-2)`

Lowest `k` eigenpairs of a symmetric `F` for SCF, where `F` changes little between calls. Returns `(eigenvalues (k,), eigenvectors (n, k), state)`; pass `state` as `guess` on the next call.

- `guess=None` is a cold start: one full float64 host `eigh`, returned exactly.
- With a warm start, each pass runs a degree-8 scaled Chebyshev filter `Y = p(F) X` on a block of `kb = k + buffer` vectors (`buffer` defaults to about 15%, at least 8), splits off the correction `E = Y - X XᵀY`, and does a float64 Rayleigh-Ritz on `span[X, E]` on the host.
- For FP32 `F` on the native `neuron` device, one pass is one NKI launch (`trnsolver/nki/subspace.py`, n·kb ≤ about 1024 × 512). `F` stays resident in SBUF, the filter is Tensor Engine GEMMs, and only the six kb × kb Gram blocks reach the host. Elsewhere the same arithmetic runs as PyTorch (`subspace.filter_reference`).
- `refresh`: when the first pass finds the block far from `F` (relative residual above `refresh`, as in the first SCF iterations), the call returns a full `eigh` instead.
- `tol`: stop early once the wanted residuals are below `tol` relative to the spectral scale.
- `upper`: the spectral upper bound for the filter: Gershgorin (default), `"cached"`, or a number.

The correction form matters in FP32. The filter grows core levels far below the HOMO much faster than the HOMO itself, and a Rayleigh-Ritz on `YᵀY` loses the HOMO. Degrees above about 12 lose accuracy even in the correction form, so the default is 8.

`examples/scf_df_rhf.py --solver subspace` uses it on `XᵀFX`.
