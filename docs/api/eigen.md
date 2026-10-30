# Eigensolvers

## `eigh(A, tol=1e-10, *, method=None)`

Symmetric eigendecomposition `A = V Λ Vᵀ` for real symmetric `A`. Returns `(eigenvalues, eigenvectors)` with eigenvalues in ascending order.

`method=None` reads `TRNSOLVER_EIGH_METHOD` (default `auto`).

- `auto`: on the native `neuron` device, n < `TRNSOLVER_EIGH_NKI_MIN_N` (default 512) uses `torch.linalg.eigh`; otherwise the NKI path (Householder tridiagonalization, host QR iteration). Off-device, `torch.linalg.eigh`.
- `jacobi` (#48): batched Jacobi for `(n, n)` or `(..., n, n)` input. On the native `neuron` device (FP32, n ≤ 128) it runs the SBUF-resident one-sided Jacobi kernel: one launch per sweep, each holding up to 128 tiles of 128 / n matrices (512 at n = 32) in SBUF. Elsewhere it runs a torch implementation (two-sided Jacobi as batched matmuls). Uses the lower triangle, like `torch.linalg.eigh`. For 33 ≤ n ≤ 128 the kernel is block Jacobi: per step the Tensor Engine forms each matrix's Gram matrix and applies one orthogonal matmul per block pair, and the Vector Engine only rotates the 16 × 16 pair Gram blocks.
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
