# Architecture

## Phase roadmap

trnsolver follows the [trnsci 5-phase roadmap](https://trnsci.dev/roadmap/). Each phase has a tracker issue; work is labeled `phase-N-<theme>` across the whole suite for cross-project coordination.

| Phase | Theme | Trnsolver scope | Tracker |
|---|---|---|---|
| 1 | correctness | NKI Jacobi kernel validated on trn1/trn2; eigh_generalized on NKI path; SCF example end-to-end | [#26](https://github.com/trnsci/trnsolver/issues/26) |
| 2 | precision | Iterative refinement for eigh / solve_spd; Kahan summation in CG / GMRES; depends on trnblas double-double GEMM | [#27](https://github.com/trnsci/trnsolver/issues/27) |
| 3 | perf | Newton-Schulz NKI backend, preconditioners (Jacobi ✓, IC0, SSOR, block-Jacobi), NEFF cache reuse | [#28](https://github.com/trnsci/trnsolver/issues/28) |
| 4 | multi-chip | Parallel Jacobi sweeps across NeuronCores; sharded preconditioners | [#29](https://github.com/trnsci/trnsolver/issues/29) |
| 5 | generation | trn2-tuned rotation-block tile size; runtime capability detection in dispatch | [#30](https://github.com/trnsci/trnsolver/issues/30) |

v0.3.0 shipped with the CPU path feature-complete and a stub NKI kernel. v0.4.0 was the Phase 1 ship (Householder-QR `eigh` validated on trn1.2xlarge; #26 and #27 are closed). Since v0.16.0 the primary hardware target is Trn2 on native TorchNeuron.

**Phase 1 iteration is no longer hardware-gated.** Neuron SDK 2.29 (April 2026) brought NKI 0.3.0 Stable with a CPU simulator (`nki.simulate_kernel`); the `test-simulator` CI job exercises kernel correctness on `ubuntu-latest` without a trn1 instance. Hardware runs via `scripts/run_neuron_tests.sh` are now reserved for final real-device validation, not inner-loop iteration.

## Package layout

```
trnsolver/
├── eigen.py        # eigh, eigh_generalized, eigh_randomized
├── _jacobi.py      # torch batched Jacobi (off-device / FP64 / n > 128)
├── factor.py       # cholesky, lu, qr, svd, pinv, schur, solve, solve_spd, inv_spd, inv_sqrt_spd(_ns)
├── orthogonalize.py  # orthogonalizer: symmetric / canonical / pivoted-Cholesky X for S
├── iterative.py    # cg, gmres, gmres_ir, preconditioners
└── nki/
    ├── __init__.py
    ├── dispatch.py # auto/pytorch/nki backend selection, Householder/matmul kernels, launch helper
    ├── jacobi.py   # batched Jacobi kernels (scalar n <= 32, block 33 <= n <= 128)
    ├── eigh_pool.py # per-core worker-process pool (TRNSOLVER_EIGH_POOL=proc)
    └── _native.py  # TorchNeuron custom ops (nki_op)
```

Each high-level operation in `eigen` / `factor` / `iterative` checks the active backend via `nki.dispatch._use_nki()` and either calls into the NKI kernel or falls back to PyTorch.

## SCF → solver mapping

The headline use case is the SCF iteration for quantum chemistry. At each SCF step:

| SCF step | Math | trnsolver call |
|----------|------|----------------|
| Overlap Cholesky | `S = L Lᵀ` | `cholesky(S)` |
| Reduce to standard | `A' = L⁻¹ F L⁻ᵀ` | `eigh_generalized(F, S)` |
| …or, near-singular S | `X = orthogonalizer(S).X` once, `A' = Xᵀ F X` | `eigh_generalized(F, S, X=X)` |
| Eigensolve | `A' V = V Λ` | `eigh(A')` (Jacobi on NKI) |
| Back-transform | `C = L⁻ᵀ V` | `trsm` (in trnblas) |
| Density | `P = C_occ @ C_occᵀ` | `syrk` (in trnblas) |
| Metric inverse | `J⁻¹ᐟ²` | `inv_sqrt_spd(J)` |

## NKI Jacobi strategy

The Jacobi eigensolver maps cleanly to Trainium hardware:

- **One-sided (Hestenes) Jacobi, parallel order**: each step rotates n/2 disjoint column pairs, so a batch of matrices advances in lockstep with no long dependency chain
- **SBUF residency**: up to 128 tiles of 128 / n matrices stay in SBUF for a whole sweep, one launch per sweep
- **Tensor Engine**: the step's partner permutation (an exact matmul shared by every tile); for 33 ≤ n ≤ 128 also the Gram matrix and the block-pair rotations as one matmul
- **Vector Engine**: the rotations (for the block kernel, only on the 16 × 16 pair Gram blocks)
- **Convergence check** after the expected sweep count (6-8 for random matrices), at most 15 sweeps

Jacobi is preferred over Householder tridiagonalization on Trainium because Householder is n − 2 dependent BLAS-2 steps that leave the Tensor Engine idle; batched on device it still lost to CPU LAPACK by about 6× at n = 128 (#48). The trade-off is `O(n³)` per sweep with `O(n)` sweeps — cubic overall but with a large constant. For `n > ~500`, classical Householder + QR iteration on CPU may still be faster.

## Backend dispatch

`trnsolver.nki.dispatch` follows the same pattern as the sister `trnblas` and `trnfft` packages:

- `HAS_NKI` — `True` iff `neuronxcc.nki` imports successfully.
- `set_backend("auto" | "pytorch" | "nki")` — choose dispatch mode.
- `TRNSOLVER_REQUIRE_NKI=1` env var — fail loudly on kernel-path errors instead of silently falling back. Used by the validation suite.

For the suite-wide picture of which NKI kernels are validated on hardware across trnsci, see the [NKI validation status page](https://trnsci.dev/nki_validation_status/).

## Known gaps

- **NKI kernels are single-tile (n ≤ 128).** Larger `eigh` inputs run `torch.linalg.eigh` (host LAPACK through aten on the `neuron` device); the default cutover sends n < 512 there too (#42).
- **One matrix is not split across NeuronCores.** The multi-core options split a batch (#54, #57); splitting a single matrix waits on device-side collectives (#29).
- **CG / GMRES are pure PyTorch.** The inner products and matvecs would benefit from NKI Level 1/2 kernels in trnblas, but the iteration logic stays on the host.
- **`inv_sqrt_spd` uses eigendecomposition.** The all-GEMM Newton-Schulz variant is `inv_sqrt_spd_ns` (GEMMs through `trnblas.gemm` when trnblas is installed).
- **No FP64.** Trainium's Tensor Engine maxes out at FP32. Double-double emulation is not implemented.
