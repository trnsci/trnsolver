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

v0.3.0 shipped with the CPU path feature-complete and a stub NKI kernel. v0.4.0 is the Phase 1 ship.

## Package layout

```
trnsolver/
├── eigen.py        # eigh, eigh_generalized
├── factor.py       # cholesky, lu, qr, solve, solve_spd, inv_sqrt_spd
├── iterative.py    # cg, gmres
└── nki/
    ├── __init__.py
    └── dispatch.py # auto/pytorch/nki backend selection + Jacobi rotation kernel
```

Each high-level operation in `eigen` / `factor` / `iterative` checks the active backend via `nki.dispatch._use_nki()` and either calls into the NKI kernel or falls back to PyTorch.

## SCF → solver mapping

The headline use case is the SCF iteration for quantum chemistry. At each SCF step:

| SCF step | Math | trnsolver call |
|----------|------|----------------|
| Overlap Cholesky | `S = L Lᵀ` | `cholesky(S)` |
| Reduce to standard | `A' = L⁻¹ F L⁻ᵀ` | `eigh_generalized(F, S)` |
| Eigensolve | `A' V = V Λ` | `eigh(A')` (Jacobi on NKI) |
| Back-transform | `C = L⁻ᵀ V` | `trsm` (in trnblas) |
| Density | `P = C_occ @ C_occᵀ` | `syrk` (in trnblas) |
| Metric inverse | `J⁻¹ᐟ²` | `inv_sqrt_spd(J)` |

## NKI Jacobi strategy

The Jacobi eigensolver maps cleanly to Trainium hardware:

- **Each Givens rotation** is a 2-row/col update — rank-2 matmul on the Tensor Engine
- **Off-diagonal max-finding** maps to a Vector Engine reduction
- **Sweep convergence check** is a scalar reduction on the Scalar Engine
- **Eigenvector accumulation** `V' = V @ G` is a batched column update

Jacobi is preferred over Householder tridiagonalization on Trainium because each rotation is a fixed-size operation that maps cleanly to the 128×128 tile, whereas Householder requires growing reflector chains. The trade-off is `O(n³)` per sweep with `O(n)` sweeps — cubic overall but with a large constant. For `n > ~500`, classical Householder + QR iteration on CPU may still be faster.

## Backend dispatch

`trnsolver.nki.dispatch` follows the same pattern as the sister `trnblas` and `trnfft` packages:

- `HAS_NKI` — `True` iff `neuronxcc.nki` imports successfully.
- `set_backend("auto" | "pytorch" | "nki")` — choose dispatch mode.
- `TRNSOLVER_REQUIRE_NKI=1` env var — fail loudly on kernel-path errors instead of silently falling back. Used by the validation suite.

For the suite-wide picture of which NKI kernels are validated on hardware across trnsci, see the [NKI validation status page](https://trnsci.dev/nki_validation_status/).

## Known gaps

- **NKI Jacobi kernel is a stub.** Falls back to `torch.linalg.eigh` until validated on hardware.
- **CG / GMRES are pure PyTorch.** The inner products and matvecs would benefit from NKI Level 1/2 kernels in trnblas, but the iteration logic stays on the host.
- **`inv_sqrt_spd` uses eigendecomposition,** not Newton-Schulz. Newton-Schulz `(X_{k+1} = ½ X_k (3I − A X_k²))` is all GEMMs and would map better to trnblas, but needs good initial scaling.
- **No FP64.** Trainium's Tensor Engine maxes out at FP32. Double-double emulation is not implemented.
