# Benchmarks

Performance results for trnsolver — eigendecomposition, factorizations, and iterative solvers — comparing the PyTorch CPU fallback and NKI Trainium path across canonical workloads.

## Status

Baseline PyTorch-fallback numbers run on every CI build. NKI numbers are pending on-hardware validation on trn1 / trn2 — run `scripts/run_neuron_tests.sh` to generate them locally once a Neuron CI instance is provisioned (see [AWS Setup](aws_setup.md)).

## Reproducing locally

```bash
pytest benchmarks/ --benchmark-only
```

## Results table (placeholder)

| Op | Size | PyTorch (CPU) | NKI (Trainium) | Speedup |
|---|---|---|---|---|
| eigh (symmetric) | 256×256 | TBD | TBD | TBD |
| eigh (symmetric) | 1024×1024 | TBD | TBD | TBD |
| cholesky | 1024×1024 | TBD | TBD | TBD |
| qr | 1024×1024 | TBD | TBD | TBD |
| cg (10k sparse) | — | TBD | TBD | TBD |
| gmres (10k sparse) | — | TBD | TBD | TBD |

Numbers will be populated once the NKI Jacobi rotation kernel validates on trn1 / trn2 and the benchmark harness is wired into CI.
