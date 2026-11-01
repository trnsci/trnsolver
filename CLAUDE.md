# trnsolver

Linear solvers and eigendecomposition for AWS Trainium via NKI.
Part of the trnsci scientific computing suite.

## What This Is

Eigenvalue problems, matrix factorizations, and iterative solvers targeting
Trainium's Tensor Engine. Complements trnblas (BLAS) and trnfft (FFT).

**Primary use case:** SCF eigenvalue problem for quantum chemistry. The SCF
iteration solves FC = SCε at each step — a generalized symmetric eigenvalue
problem reduced to standard form via Cholesky of the overlap matrix S. This
feeds into DF-MP2 (trnblas) for the correlation energy.

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
│       ├── dispatch.py      # NKI kernels, launch helper, dispatch-path selection
│       └── _native.py       # TorchNeuron custom ops (nki_op), loaded lazily
├── tests/
│   ├── conftest.py          # Fixtures: symmetric, SPD, random matrices
│   ├── test_eigen.py        # Eigenvalue tests (reconstruction, orthogonality)
│   ├── test_factor.py       # Cholesky, LU, QR, solve tests
│   └── test_iterative.py    # CG, GMRES convergence tests
├── examples/
│   └── scf_eigen.py         # SCF iteration demo (quantum chemistry)
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

The batched one-sided (Hestenes) Jacobi kernels in `trnsolver/nki/jacobi.py`
map to Trainium hardware:

- **Parallel order:** each step rotates n/2 disjoint column pairs, so a batch
  advances in lockstep; up to 128 tiles of 128 / n matrices stay in SBUF
  for a whole sweep (one launch per sweep)
- **Tensor Engine:** the step's partner permutation (exact matmul); for
  33 <= n <= 128 also the Gram matrix and block-pair rotations
- **Vector Engine:** the rotations (block kernel: 16 x 16 pair Gram blocks)
- **Convergence check** after the expected sweep count, at most 15 sweeps

Jacobi is preferred over Householder tridiagonalization on Trainium because
Householder is n - 2 dependent BLAS-2 steps that leave the Tensor Engine
idle; batched on device it still lost to CPU LAPACK (#48).

## Known Gaps & Design Notes

- **NKI Jacobi kernels are batched and n <= 128.** `trnsolver/nki/jacobi.py`
  (scalar n <= 32, block 33 <= n <= 128), validated on Trn2. Householder-QR
  `eigh` is single-tile (n <= 128); the default aten cutover (n < 512) and
  the > 128 fallback send single matrices to `torch.linalg.eigh`.

- **Jacobi is O(n³) per sweep with O(n) sweeps** — cubic overall but with
  a large constant. For n > ~500, Householder + QR iteration would be faster.
  The Jacobi method is chosen because it's embarrassingly parallel (rotations
  on disjoint pairs can run concurrently) and maps cleanly to NKI tiles.

- **CG/GMRES are pure PyTorch.** The inner products and matvecs would
  benefit from NKI Level 1/2 kernels in trnblas, but the iteration logic
  stays on the host.

- **inv_sqrt_spd uses eigendecomposition.** The all-GEMM Newton-Schulz
  variant is `inv_sqrt_spd_ns` (GEMMs via `trnblas.gemm` when installed).

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
- `trnfft` — FFT + complex ops (https://github.com/trnsci/trnfft)
- `trnblas` — BLAS operations (https://github.com/trnsci/trnblas)
- `trnsolver` — Linear solvers, eigendecomposition (this repo)
- `trnrand` — Random number generation

All repos: Python/NKI, Apache 2.0.

## Blog posts

When asked to draft a blog post for this library for the [trnsci blog](https://trnsci.dev/blog/):

1. Read the editorial brief at [`docs/blog/AUTHOR_BRIEF.md`](https://github.com/trnsci/trnsci/blob/main/docs/blog/AUTHOR_BRIEF.md) in the umbrella repo (trnsci/trnsci). It defines voice (authorless, library-as-subject), stance (architecture-first, transparency-always), and the nine required section headings.

2. Find the prompt block for this library in [`BLOG_PROMPTS.md`](https://github.com/trnsci/trnsci/blob/main/BLOG_PROMPTS.md) at the umbrella repo root. It carries library-specific context and suggested architectural angles.

3. Draft the post following the brief. Open a PR against `trnsci/trnsci` at `docs/blog/posts/<YYYY-MM-DD>-<slug>.md`. Scott (suite director) reviews before merge.

The umbrella repo — not this one — owns the blog. Per-library retrospective posts are unsigned; library is the subject, no byline. See the brief for the full set of rules.
