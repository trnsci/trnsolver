# trnsolver

Linear solvers and eigendecomposition for AWS Trainium via NKI.

Eigenvalue problems, matrix factorizations, and iterative solvers for scientific computing on Trainium. The Jacobi eigensolver is the primary NKI acceleration target — each Givens rotation maps to a 2-row matmul on the Tensor Engine.

Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Install

```bash
pip install trnsolver

# With Neuron hardware support
pip install trnsolver[neuron]
```

## Usage

```python
import torch
import trnsolver

# Symmetric eigenvalue decomposition (Jacobi on NKI)
eigenvalues, eigenvectors = trnsolver.eigh(A)

# Generalized eigenproblem: A x = λ B x (the SCF problem)
eigenvalues, eigenvectors = trnsolver.eigh_generalized(F, S)

# Factorizations
L = trnsolver.cholesky(A)
P, L, U = trnsolver.lu(A)
Q, R = trnsolver.qr(A)

# Direct solvers
x = trnsolver.solve(A, b)
x = trnsolver.solve_spd(A, b)        # Cholesky-based (faster for SPD)
A_inv_sqrt = trnsolver.inv_sqrt_spd(A)  # A^{-1/2} for density fitting

# Iterative solvers
x, iters, residual = trnsolver.cg(A, b, tol=1e-8)
x, iters, residual = trnsolver.gmres(A, b, tol=1e-6)
```

## SCF Example

```bash
python examples/scf_eigen.py --demo
python examples/scf_eigen.py --nbasis 50 --nocc 10
```

Demonstrates the self-consistent field iteration: build Fock matrix → solve generalized eigenproblem FC = SCε → build density → check convergence.

## Operations

| Category | Operation | Description |
|----------|-----------|-------------|
| Eigen | `eigh` | Symmetric eigendecomposition (Jacobi / torch) |
| Eigen | `eigh_generalized` | Generalized: Ax = λBx via Cholesky reduction |
| Factor | `cholesky` | A = LL^T |
| Factor | `lu` | PA = LU |
| Factor | `qr` | A = QR |
| Solve | `solve` | Ax = b (LU-based) |
| Solve | `solve_spd` | Ax = b (Cholesky, A is SPD) |
| Solve | `inv_spd` | A^{-1} for SPD A |
| Solve | `inv_sqrt_spd` | A^{-1/2} for density fitting metric |
| Iterative | `cg` | Conjugate Gradient (SPD systems) |
| Iterative | `gmres` | GMRES (general systems) |

## Status

- [x] Jacobi eigensolver with PyTorch backend
- [x] Generalized eigenvalue problem (Cholesky reduction)
- [x] Cholesky, LU, QR factorizations
- [x] Direct and SPD solvers
- [x] CG and GMRES iterative solvers
- [x] SCF example (quantum chemistry use case)
- [ ] NKI Jacobi rotation kernel validation
- [ ] Newton-Schulz A^{-1/2} via trnblas GEMM
- [ ] Benchmarks vs LAPACK/cuSOLVER

## Related Projects

| Project | What |
|---------|------|
| [trnfft](https://github.com/trnsci/trnfft) | FFT + complex ops |
| [trnblas](https://github.com/trnsci/trnblas) | BLAS operations |
| trnsolver | This repo |

## License

Apache 2.0 — Copyright 2026 Scott Friedman
