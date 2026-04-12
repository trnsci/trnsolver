# trnsolver

Linear solvers and eigendecomposition for AWS Trainium via NKI.
Part of the trn-* scientific computing suite by Playground Logic.

## What This Is

Eigenvalue problems, matrix factorizations, and iterative solvers targeting
Trainium's Tensor Engine. Complements trnblas (BLAS) and trnfft (FFT).

**Primary use case:** SCF eigenvalue problem for quantum chemistry with
Prof. Ben Janesko at TCU. The SCF iteration solves FC = SCε at each step —
a generalized symmetric eigenvalue problem reduced to standard form via
Cholesky of the overlap matrix S. This feeds into DF-MP2 (trnblas) for
the correlation energy.

## Architecture

```
trnsolver/
├── trnsolver/
│   ├── __init__.py          # Re-exports all solver operations
│   ├── eigen.py             # eigh, eigh_generalized (Jacobi + torch fallback)
│   ├── factor.py            # cholesky, lu, qr, solve, solve_spd, inv_sqrt_spd
│   ├── iterative.py         # cg, gmres
│   └── nki/
│       ├── __init__.py
│       └── dispatch.py      # auto/pytorch/nki dispatch + Jacobi rotation kernel
├── tests/
│   ├── conftest.py          # Fixtures: symmetric, SPD, random matrices
│   ├── test_eigen.py        # Eigenvalue tests (reconstruction, orthogonality)
│   ├── test_factor.py       # Cholesky, LU, QR, solve tests
│   └── test_iterative.py    # CG, GMRES convergence tests
├── examples/
│   └── scf_eigen.py         # SCF iteration demo (Janesko use case)
├── pyproject.toml
├── README.md
├── LICENSE                  # Apache 2.0
└── CLAUDE.md                # This file
```

## SCF → Solver Mapping

| SCF Step | Math | trnsolver Call |
|----------|------|---------------|
| Overlap Cholesky | S = L L^T | `cholesky(S)` |
| Reduce to standard | A' = L^{-1} F L^{-T} | `eigh_generalized(F, S)` |
| Eigensolve | A' V = V Λ | `eigh(A')` (Jacobi on NKI) |
| Back-transform | C = L^{-T} V | `trsm` (in trnblas) |
| Density | P = C_occ @ C_occ^T | `syrk` (in trnblas) |
| Metric inverse | J^{-1/2} | `inv_sqrt_spd(J)` |

## NKI Jacobi Strategy

The Jacobi eigensolver maps to Trainium hardware:

- **Each Givens rotation** is a 2-row/col update — rank-2 matmul on Tensor Engine
- **Off-diagonal max finding** maps to Vector Engine reduction
- **Sweep convergence check** is a scalar reduction on Scalar Engine
- **Eigenvector accumulation** V' = V @ G is a batched column update

The Jacobi method is preferred over Householder tridiagonalization on
Trainium because each rotation is a fixed-size operation that maps cleanly
to the 128×128 tile, vs Householder which requires growing reflector chains.

## Known Gaps & Design Notes

- **NKI Jacobi kernel is a stub.** Falls back to torch.linalg.eigh until
  validated on hardware. The rotation kernel scaffold is in nki/dispatch.py.

- **Jacobi is O(n³) per sweep with O(n) sweeps** — cubic overall but with
  a large constant. For n > ~500, Householder + QR iteration would be faster.
  The Jacobi method is chosen because it's embarrassingly parallel (rotations
  on disjoint pairs can run concurrently) and maps cleanly to NKI tiles.

- **CG/GMRES are pure PyTorch.** The inner products and matvecs would
  benefit from NKI Level 1/2 kernels in trnblas, but the iteration logic
  stays on the host.

- **inv_sqrt_spd uses eigendecomposition**, not the Newton-Schulz iteration.
  Newton-Schulz (X_{k+1} = 0.5 X_k (3I - A X_k²)) is all GEMMs and would
  map better to trnblas, but needs good initial scaling.

## Dependencies

- `torch>=2.1` — tensor operations and CPU fallback
- `numpy>=1.24` — test reference
- `neuronxcc` — NKI kernels (optional, only on Neuron hardware)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
python examples/scf_eigen.py --demo
```

## Naming Convention

Sibling repos in the trn-* suite:
- `trnfft` — FFT + complex ops (https://github.com/scttfrdmn/trnfft)
- `trnblas` — BLAS operations (https://github.com/scttfrdmn/trnblas)
- `trnsolver` — Linear solvers, eigendecomposition (this repo)
- `trnrand` — Random number generation (planned)

All repos: Python/NKI, Apache 2.0, Playground Logic.
