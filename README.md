# trnsolver

[![CI](https://github.com/trnsci/trnsolver/actions/workflows/ci.yml/badge.svg)](https://github.com/trnsci/trnsolver/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/trnsci/trnsolver/graph/badge.svg)](https://codecov.io/gh/trnsci/trnsolver)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI](https://img.shields.io/pypi/v/trnsolver)](https://pypi.org/project/trnsolver/)
[![Python](https://img.shields.io/pypi/pyversions/trnsolver)](https://pypi.org/project/trnsolver/)
[![License](https://img.shields.io/github/license/trnsci/trnsolver)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://trnsci.dev/trnsolver/)

Linear solvers and eigendecomposition for AWS Trainium via NKI.

Eigenvalue problems, matrix factorizations, and iterative solvers for scientific computing on Trainium. The Jacobi eigensolver is the primary NKI acceleration target — each Givens rotation maps to a 2-row matmul on the Tensor Engine. The Newton-Schulz matrix-sqrt-inverse is the secondary target: an all-GEMM iteration whose shape aligns with the Tensor Engine pipeline.

Part of the **trnsci** scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Current phase

trnsolver follows the [trnsci 5-phase roadmap](https://trnsci.dev/roadmap/). Active work is tracked in phase-labeled GitHub issues:

- **[Phase 1 — correctness](https://github.com/trnsci/trnsolver/issues/26)** (active): NKI Jacobi kernel validated on hardware, eigh_generalized on NKI path, SCF example end-to-end. Target release: **v0.4.0**.
- **[Phase 2 — precision](https://github.com/trnsci/trnsolver/issues/27)**: iterative refinement for eigh / solve_spd, Kahan summation in CG / GMRES.
- **[Phase 3 — perf](https://github.com/trnsci/trnsolver/issues/28)**: Newton-Schulz NKI backend, preconditioner suite, NEFF cache reuse.
- **[Phase 4 — multi-chip](https://github.com/trnsci/trnsolver/issues/29)**: parallel Jacobi sweeps across NeuronCores.
- **[Phase 5 — generation](https://github.com/trnsci/trnsolver/issues/30)**: trn2 rotation-block tuning.

## Install

```bash
pip install trnsolver

# With Neuron hardware support
pip install trnsolver[neuron]
```

## Quick example

```python
import torch
import trnsolver

# Symmetric eigenvalue decomposition
w, V = trnsolver.eigh(A)

# Generalized eigenproblem: A x = λ B x  (the SCF problem)
w, V = trnsolver.eigh_generalized(F, S)

# Factorizations + direct solves
L = trnsolver.cholesky(A)
x = trnsolver.solve_spd(A, b)
M = trnsolver.inv_sqrt_spd(A)                            # eigendecomposition-based
M, iters, res = trnsolver.inv_sqrt_spd_ns(A, tol=1e-8)   # Newton-Schulz, all-GEMM

# Pseudoinverse (Moore-Penrose, truncated SVD)
Ap = trnsolver.pinv(A)

# Iterative solvers with preconditioners
precond = trnsolver.jacobi_preconditioner(A)
blk_precond = trnsolver.block_jacobi_preconditioner(A, block_size=16)
ssor_precond = trnsolver.ssor_preconditioner(A, omega=1.0)  # symmetric Gauss-Seidel
x, iters, res = trnsolver.cg(A, b, M=ssor_precond, tol=1e-8)
x, iters, res = trnsolver.gmres(A, b, tol=1e-6)
```

## Why

Trainium has no native LAPACK. Every SCF iteration, every density-fitting metric inversion, every Krylov solve on Trainium currently falls back to torch.linalg on the host CPU or hand-rolled wrappers. `trnsolver` closes that gap: same solver API surface, NKI-accelerated Jacobi on the Tensor Engine, PyTorch fallback everywhere else.

## SCF example

```bash
python examples/scf_eigen.py --demo
python examples/scf_eigen.py --nbasis 50 --nocc 10
```

Demonstrates the self-consistent-field iteration: build Fock matrix → solve generalized eigenproblem `FC = SCε` → build density → check convergence. This is the headline use case for quantum-chemistry workflows, feeding into DF-MP2 via `trnblas`.

## Status

**v0.8.0** — `ssor_preconditioner(A, omega=1.0)` (#28). SSOR preconditioner for SPD systems: two triangular solves + diagonal scaling per application. Outperforms scalar Jacobi on coupled matrices (1D Laplacian, FEM stiffness). ω ∈ (0, 2); ω=1 is symmetric Gauss-Seidel.

**v0.7.0** — BF16/FP16 dtype support across the full public API (#19). All entry points (`cholesky`, `lu`, `qr`, `solve`, `solve_spd`, `inv_spd`, `pinv`, `inv_sqrt_spd`, `inv_sqrt_spd_ns`, `eigh`, `eigh_generalized`, `cg`, `gmres`, `block_jacobi_preconditioner`) accept BF16/FP16 inputs, upcast to FP32 internally, and restore the original dtype on output.

**v0.6.0** — `eigh` subspace rotation refinement: one Rayleigh-Ritz step (V^T A V re-diagonalization) after Householder-QR reduces eigenvector residuals by 1–2 orders of magnitude for n ≥ 64 (#31). `solve_spd` gains `iterative_refinement=True`: mixed-precision FP64 residual + second Cholesky solve for SPD systems with cond up to ~1e7 (#32).

**API coverage:**

| Category | Shipped (v0.5.0) | Deferred |
|----------|------------------|----------|
| Eigensolvers | `eigh`, `eigh_generalized` | `svd` (Jacobi-SVD, Phase 3) |
| Factorizations | `cholesky`, `lu`, `qr`, `pinv` | `schur` (implicit-shift QR, Phase 3) |
| Direct solvers | `solve`, `solve_spd`, `inv_spd`, `inv_sqrt_spd`, `inv_sqrt_spd_ns` | — |
| Iterative | `cg` (w/ preconditioner), `gmres` | — |
| Preconditioners | `jacobi_preconditioner`, `block_jacobi_preconditioner`, `ssor_preconditioner` | — |

**Roadmap:**
- **v0.4.0** — NKI Householder-QR `eigh` validated on trn1.2xlarge (#9, #12, #38)
- **v0.4.1** — `eigh_generalized` NKI triangular-solve path via `trnblas.trsm` (#11)
- **v0.5.0** — Newton-Schulz trnblas.gemm (#14), FP64 CG/GMRES dots + Rayleigh refinement (#27), block-Jacobi (#16), `pinv` (#22)
- **v0.6.0** — eigh subspace rotation refinement (#31), solve_spd iterative refinement (#32)
- **v0.7.0** — BF16/FP16 across the full API (#19) ✓
- **v0.8.0** — SSOR preconditioner (#28) ✓
- **v0.9.0+** — multi-NeuronCore parallel Jacobi (#20)

## Operations

| Category | Operation | Description |
|----------|-----------|-------------|
| Eigen | `eigh` | Symmetric eigendecomposition (Jacobi / torch) |
| Eigen | `eigh_generalized` | Generalized: `Ax = λBx` via Cholesky reduction |
| Factor | `cholesky` | `A = LL^T` |
| Factor | `lu` | `PA = LU` |
| Factor | `qr` | `A = QR` |
| Factor | `pinv` | Moore-Penrose pseudoinverse (truncated SVD) |
| Solve | `solve` | `Ax = b` (LU-based) |
| Solve | `solve_spd` | `Ax = b` (Cholesky, A is SPD) |
| Solve | `inv_spd` | `A^{-1}` for SPD A |
| Solve | `inv_sqrt_spd` | `A^{-1/2}` via eigendecomposition |
| Solve | `inv_sqrt_spd_ns` | `A^{-1/2}` via Newton-Schulz (all-GEMM) |
| Iterative | `cg` | Conjugate Gradient (SPD systems) |
| Iterative | `gmres` | GMRES (general systems) |
| Iterative | `jacobi_preconditioner` | Diagonal preconditioner for CG |
| Iterative | `block_jacobi_preconditioner` | Block-diagonal Cholesky preconditioner for CG |
| Iterative | `ssor_preconditioner` | SSOR / symmetric Gauss-Seidel preconditioner for CG |

## Benchmarks

CPU baselines (torch.linalg, scipy.linalg, trnsolver PyTorch path) run on every CI build; CUDA baselines (`benchmarks/bench_cuda.py`, cuSOLVER via torch.linalg) run on a vintage-matched g5.xlarge A10G instance; NKI numbers are pending v0.4.0 hardware validation. See the [benchmarks page](https://trnsci.dev/trnsolver/benchmarks/) for the latest table and vintage-matching rationale.

## Related projects in the trnsci suite

All six siblings are on PyPI, along with the umbrella meta-package:

| Project | What | Latest |
|---------|------|-------:|
| [trnsci](https://github.com/trnsci/trnsci) | Umbrella meta-package pulling the whole suite | v0.1.0 |
| [trnfft](https://github.com/trnsci/trnfft) | FFT and complex-valued tensors | v0.8.0 |
| [trnblas](https://github.com/trnsci/trnblas) | BLAS Level 1–3 | v0.4.0 |
| [trnrand](https://github.com/trnsci/trnrand) | Philox / Sobol / Halton RNG | v0.1.0 |
| trnsolver | Linear solvers and eigendecomposition | **v0.8.0** |
| [trnsparse](https://github.com/trnsci/trnsparse) | Sparse matrix operations | v0.1.1 |
| [trntensor](https://github.com/trnsci/trntensor) | Tensor contractions (einsum, TT/Tucker) | v0.1.1 |

## License

Apache 2.0 — Copyright 2026 Scott Friedman


## Disclaimer

trnsci is an **independent open-source project**. It is not sponsored by, endorsed by, or affiliated with Amazon.com, Inc., Amazon Web Services, Inc., or Annapurna Labs Ltd.

"AWS", "Amazon", "Trainium", "Inferentia", "NeuronCore", "Neuron SDK", and related identifiers are trademarks of their respective owners and are used here solely for descriptive and interoperability purposes. Use does not imply endorsement, partnership, or any other relationship.

All work, opinions, analyses, benchmark results, architectural commentary, and editorial judgments in this repository and on [trnsci.dev](https://trnsci.dev) are those of the project's contributors. They do not represent the views, positions, or commitments of Amazon, AWS, or Annapurna Labs.

Feedback directed at the Neuron SDK or Trainium hardware is good-faith ecosystem commentary from independent users. It is not privileged information, is not pre-reviewed by AWS, and should not be read as authoritative about product roadmap, behavior, or quality.

For official AWS guidance, see [aws-neuron documentation](https://awsdocs-neuron.readthedocs-hosted.com/) and the [AWS Trainium product page](https://aws.amazon.com/ai/machine-learning/trainium/).
