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

# Iterative solvers with preconditioners
precond = trnsolver.jacobi_preconditioner(A)
x, iters, res = trnsolver.cg(A, b, M=precond, tol=1e-8)
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

**v0.3.0** — PyTorch path is feature-complete. NKI Jacobi kernel is scaffolded but not yet validated on hardware; `set_backend("auto")` falls back to `torch.linalg.eigh` everywhere until v0.4.0 lands.

**API coverage:**

| Category | Shipped (v0.3.0) | Deferred |
|----------|------------------|----------|
| Eigensolvers | `eigh`, `eigh_generalized` | `svd` (Jacobi-SVD target for v0.5.0) |
| Factorizations | `cholesky`, `lu`, `qr` | `schur`, `pinv` (see #22) |
| Direct solvers | `solve`, `solve_spd`, `inv_spd`, `inv_sqrt_spd`, `inv_sqrt_spd_ns` | — |
| Iterative | `cg` (w/ preconditioner), `gmres` | IC0/SSOR/block-Jacobi (#16) |
| Preconditioners | `jacobi_preconditioner` | See #16 |

**Roadmap:**
- **v0.4.0** — NKI Jacobi rotation kernel validated on trn1.2xlarge (#9, #12)
- **v0.5.0** — Newton-Schulz NKI backend via trnblas GEMM (#14, #25), preconditioner expansion (#16), scipy.linalg parity audit (#22)
- **v0.6.0+** — BF16/FP16 across the API (#19), multi-NeuronCore parallel Jacobi sweep (#20)

## Operations

| Category | Operation | Description |
|----------|-----------|-------------|
| Eigen | `eigh` | Symmetric eigendecomposition (Jacobi / torch) |
| Eigen | `eigh_generalized` | Generalized: `Ax = λBx` via Cholesky reduction |
| Factor | `cholesky` | `A = LL^T` |
| Factor | `lu` | `PA = LU` |
| Factor | `qr` | `A = QR` |
| Solve | `solve` | `Ax = b` (LU-based) |
| Solve | `solve_spd` | `Ax = b` (Cholesky, A is SPD) |
| Solve | `inv_spd` | `A^{-1}` for SPD A |
| Solve | `inv_sqrt_spd` | `A^{-1/2}` via eigendecomposition |
| Solve | `inv_sqrt_spd_ns` | `A^{-1/2}` via Newton-Schulz (all-GEMM) |
| Iterative | `cg` | Conjugate Gradient (SPD systems) |
| Iterative | `gmres` | GMRES (general systems) |
| Iterative | `jacobi_preconditioner` | Diagonal preconditioner for CG |

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
| trnsolver | Linear solvers and eigendecomposition | **v0.3.0** |
| [trnsparse](https://github.com/trnsci/trnsparse) | Sparse matrix operations | v0.1.1 |
| [trntensor](https://github.com/trnsci/trntensor) | Tensor contractions (einsum, TT/Tucker) | v0.1.1 |

## License

Apache 2.0 — Copyright 2026 Scott Friedman
