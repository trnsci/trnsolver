# trnsolver

Linear solvers and eigendecomposition for AWS Trainium via NKI.

Eigenvalue problems, matrix factorizations, and iterative solvers for scientific computing on Trainium. The batched Jacobi eigensolver is the primary NKI acceleration target — whole sweeps run SBUF-resident, with the Tensor Engine doing the pair permutations (and, for 33 ≤ n ≤ 128, the Gram and rotation matmuls) and the Vector Engine the rotations.

Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Features

- **Symmetric & generalized eigensolvers** — `eigh` (batched Jacobi with `method="jacobi"`), `eigh_generalized` (Cholesky reduction), `eigh_randomized` (k dominant eigenpairs)
- **Factorizations** — `cholesky`, `lu`, `qr`, `svd`, `pinv`, `schur`
- **Direct solvers** — `solve`, `solve_spd`, `inv_spd`, `inv_sqrt_spd`, `inv_sqrt_spd_ns` (Newton-Schulz)
- **Iterative solvers** — `cg` (SPD systems, with preconditioner support), `gmres` (general), `gmres_ir` (iterative refinement, FP64 outer residual)
- **Preconditioners** — `jacobi_preconditioner` (diagonal), `block_jacobi_preconditioner`, `ssor_preconditioner`
- **NKI acceleration** — batched Jacobi `eigh` kernels for n ≤ 128, validated on Trn2 (native TorchNeuron); opt-in split across logical NeuronCores (`TRNSOLVER_EIGH_CORES`, `TRNSOLVER_EIGH_POOL=proc`)
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
