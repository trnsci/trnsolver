# Eigensolvers

## `eigh(A, tol=1e-10, *, method=None)`

Symmetric eigendecomposition `A = V Λ Vᵀ` for real symmetric `A`. Returns `(eigenvalues, eigenvectors)` with eigenvalues in ascending order.

`method=None` reads `TRNSOLVER_EIGH_METHOD` (default `auto`).

- `auto`: on the native `neuron` device, n < `TRNSOLVER_EIGH_NKI_MIN_N` (default 512) uses `torch.linalg.eigh`; otherwise the NKI path (Householder tridiagonalization, host QR iteration). The NKI path is single-tile (n ≤ 128) and falls back to `torch.linalg.eigh` above that (it raises under `TRNSOLVER_REQUIRE_NKI=1`), so at the default threshold every size runs aten. Host tensors on a TorchNeuron host take the same cutover, i.e. host LAPACK (v0.20.0). Off-device, `torch.linalg.eigh`.
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

The Cholesky fails (`LinAlgError`) when `B` is exactly singular and loses accuracy when it is nearly so, as the overlaps of large or diffuse basis sets are. For those, and to skip the per-call Cholesky in an SCF loop, pass an orthogonalizer:

### `eigh_generalized(A, B, tol=1e-10, *, X=None)`

With `X` (n × m) from [`orthogonalizer`](#orthogonalizers-a-tolerance-for-near-singular-s), it computes `A' = Xᴴ A X` (two GEMMs), `eigh(A')` (m × m) and `V = X V'`, and returns m eigenpairs; `B` is not used and may be `None`. `X` is cast to `A`'s dtype and device, so an FP32 `A` on the device keeps the GEMMs there (`trnblas.gemm` for real 2-D input) while `X` itself is built once, in float64 on the host.

```python
orth = trnsolver.orthogonalizer(S, "canonical", lindep=1e-7)  # once
w, C = trnsolver.eigh_generalized(F, S, X=orth.X)  # every iteration; orth.rank pairs
```

## Orthogonalizers: a tolerance for near-singular S

### `orthogonalizer(S, method="symmetric", lindep=1e-7, *, dtype=None, device=None)`

Returns `Orthogonalizer(X, rank, eigenvalues, pivots)`, with `Xᴴ S X = I_m` on the numerically independent subspace of `S`, where `m = rank ≤ n`. `S` is real symmetric or complex Hermitian, `(n, n)` or a batch `(..., n, n)`. The work is always float64 (complex128) on the host. `dtype` and `device` set only the returned `X`, which is float64 on the host by default.

| `method` | X | Kept when | Cost |
|---|---|---|---|
| `"symmetric"` (default) | Löwdin `S⁻¹ᐟ²` (n × n), when nothing is dropped; otherwise the canonical X | — | one eigh |
| `"canonical"` | `U_keep diag(s_keep⁻¹ᐟ²)` (n × m) | `s_i > lindep · max(s)` | one eigh |
| `"cholesky"` | `X[P, :] = L⁻ᴴ` with `S_PP = L Lᴴ`, zero on the other rows (n × m) | pivoted Cholesky continues while the largest remaining Schur-complement diagonal is `> lindep · max(diag S)` | `O(n m²)`, no eigensolve |

- **Threshold.** `lindep` is relative. A kept problem has condition number at most `1 / lindep` for the eigenvalue methods, so `X` amplifies rounding by at most `1/√lindep`. For a normalised AO basis `max(s) ≥ 1`, so the relative threshold is at least as strict as an absolute one of the same value (Psi4's default is 1e-7, PySCF's 1e-8, both absolute). `lindep=0` keeps every positive eigenvalue (or pivot).
- **Symmetric with drops.** When eigenvalues are dropped, no n × n `X` satisfies `Xᴴ S X = I`, so `"symmetric"` returns the canonical X. This is PySCF's convention. With nothing dropped it is `inv_sqrt_spd(S)`, so existing results do not change.
- **Cholesky threshold.** The Cholesky test is on Schur pivots, which are never smaller than `λ_min(S)`, so at the same `lindep` it keeps more functions than `"canonical"`, and the kept block is worse conditioned. For an FP32 SCF, use about `lindep=1e-5` with it (see below). `pivots` lists the kept basis functions in pivot order. The pivots come from LAPACK `?pstrf` when SciPy is installed, and from a torch loop otherwise.
- **Batches.** For a batch, `rank` is a LongTensor and `X` is `(..., n, max rank)`. For a matrix with a lower rank, the trailing columns are zero (and `pivots` is padded with −1).
- **Autograd.** It is differentiable through `S`. `"symmetric"` without drops, and `"cholesky"` for a fixed pivot set, give `X` itself. Canonical `X` is defined only up to the sign (phase) of each column, so only gauge-invariant functions of it, such as `X Xᴴ`, have a well-defined gradient.

**Measured (#72), DF-RHF, `examples/scf_df_rhf.py`, CPU.** benzene/aug-cc-pVTZ (n = 414, smallest eigenvalue of S 3.4e-7). `E − E_ref` is against PySCF; `E(C)` is the float64 energy at the converged orbitals.

| run | rank | FP64 E − E_ref | FP32 |
|---|---|---|---|
| `symmetric` / `canonical`, lindep 1e-7 | 411 | 1.4e-6 Eh (3 functions dropped) | converged, 23 iterations, E(C) − E_ref 4.7e-6 |
| `symmetric`, lindep 1e-9 (no drop, plain S⁻¹ᐟ²) | 414 | −5.7e-8 | not converged in 80 iterations |
| `generalized` (Cholesky of S) | 414 | −5.7e-8 | not converged in 80 iterations |
| `cholesky`, lindep 1e-7 | 414 | −5.7e-8 | not converged in 80 iterations |
| `cholesky`, lindep 1e-5 | 408 | — | converged, 14 iterations, E(C) − E_ref 8.7e-6 |

In float64 nothing has to be dropped at this size. In FP32 the dropped directions decide convergence.

C20H42/aug-cc-pVDZ (n = 838, smallest eigenvalue 4.1e-7) is milder. At lindep 1e-7 two functions are dropped, and every variant converges in FP32 (26–35 iterations). In float64 the two dropped functions raise the energy by 2.8e-5 Eh against the full basis (canonical −781.9162346 against −781.9162624 Eh for S⁻¹ᐟ², the Cholesky of S, and `cholesky`). The threshold trades that basis-set truncation against FP32 conditioning. Lower it, or use float64, when the truncation matters more.

## Backend notes

Householder tridiagonalization is n − 2 dependent BLAS-2 steps and leaves the Tensor Engine idle (#48). A parallel-ordered Jacobi step rotates n/2 disjoint pairs at once, so a batch of matrices advances in lockstep with no long dependency chain. In the kernel the Tensor Engine applies the step's partner permutation (an exact matmul shared by every tile) and the Vector Engine applies the rotations. For measurements see #48; the fair CPU baseline splits the batch across the same threads, which is 1.6–5.8× faster than one batched `torch.linalg.eigh` call (#54, v0.19.0).
