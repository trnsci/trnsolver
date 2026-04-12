# Quickstart

## Symmetric eigendecomposition

```python
import torch
import trnsolver

A = torch.randn(64, 64)
A = A @ A.T  # symmetric

eigenvalues, eigenvectors = trnsolver.eigh(A)

# Reconstruction check
recon = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
assert torch.allclose(A, recon, atol=1e-5)
```

## Generalized eigenproblem

The SCF problem `FC = SCε` is a generalized symmetric eigenvalue problem. `eigh_generalized` reduces it to standard form via Cholesky of the overlap matrix `S`:

```python
F = build_fock_matrix(...)   # symmetric
S = build_overlap_matrix(...)  # SPD

eigenvalues, eigenvectors = trnsolver.eigh_generalized(F, S)
```

## Factorizations

```python
L = trnsolver.cholesky(A_spd)        # A = LL^T
P, L, U = trnsolver.lu(A)             # PA = LU
Q, R = trnsolver.qr(A)                # A = QR
```

## Direct solvers

```python
x = trnsolver.solve(A, b)              # LU-based
x = trnsolver.solve_spd(A, b)          # Cholesky — faster for SPD A
A_inv_sqrt = trnsolver.inv_sqrt_spd(A) # density-fitting metric inverse
```

## Iterative solvers

```python
x, iters, residual = trnsolver.cg(A_spd, b, tol=1e-8)
x, iters, residual = trnsolver.gmres(A, b, tol=1e-6)
```

## Backend selection

```python
trnsolver.set_backend("auto")     # NKI if available, else PyTorch (default)
trnsolver.set_backend("pytorch")  # force PyTorch fallback
trnsolver.set_backend("nki")      # require NKI; raises if unavailable
```

Run the SCF demo end-to-end:

```bash
python examples/scf_eigen.py --demo
```
