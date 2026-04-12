# Eigensolvers

## `eigh(A, *, max_sweeps=50, tol=1e-10)`

Symmetric eigendecomposition `A = V Λ Vᵀ` for real symmetric `A`. Returns `(eigenvalues, eigenvectors)` with eigenvalues in ascending order.

When the NKI backend is active, dispatches to the Jacobi rotation kernel (currently scaffolded — falls back to `torch.linalg.eigh` until on-hardware validation lands). Pure-PyTorch path uses `torch.linalg.eigh`.

```python
eigenvalues, eigenvectors = trnsolver.eigh(A)
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

## Backend notes

The Jacobi method is preferred over Householder tridiagonalization on Trainium because each Givens rotation maps cleanly to a 128×128 Tensor Engine tile. For details see the [architecture notes](../architecture.md).
