# trnsolver

Linear solvers and eigendecomposition for AWS Trainium via NKI.

Eigenvalue problems, matrix factorizations, and iterative solvers for scientific computing on Trainium. The Jacobi eigensolver is the primary NKI acceleration target — each Givens rotation maps to a 2-row update on the Tensor Engine.

Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Features

- **Symmetric & generalized eigensolvers** — `eigh`, `eigh_generalized` (Cholesky reduction)
- **Factorizations** — `cholesky`, `lu`, `qr`
- **Direct solvers** — `solve`, `solve_spd`, `inv_spd`, `inv_sqrt_spd`, `inv_sqrt_spd_ns` (Newton-Schulz)
- **Iterative solvers** — `cg` (SPD systems, with preconditioner support), `gmres` (general)
- **Preconditioners** — `jacobi_preconditioner` diagonal; IC0/SSOR/block-Jacobi on roadmap
- **NKI acceleration** — Jacobi rotation and Newton-Schulz are the primary Tensor Engine targets (hardware validation in v0.4.0)
- **SCF-ready** — generalized eigenproblem `FC = SCε` is the headline use case (quantum chemistry)

## Quick example

```python
import torch
import trnsolver

A = torch.randn(64, 64)
A = A @ A.T  # symmetric

eigenvalues, eigenvectors = trnsolver.eigh(A)
```

## License

Apache 2.0 — Copyright 2026 Scott Friedman
