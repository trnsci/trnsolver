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

# Trn2/Trn3 with TorchNeuron (native PyTorch, Neuron SDK 2.33+)
pip install "trnsolver[torchneuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com

# Trn1 / PyTorch-XLA lineage
pip install "trnsolver[neuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

Kernels dispatch on whichever Neuron PyTorch lineage is installed:

- **TorchNeuron (native)**: Trn2/Trn3, Neuron SDK 2.33+, Python ≥ 3.12,
  `torch` 2.14, `torch-neuronx` 2.14.*. Each kernel is a PyTorch custom op
  (`torch.ops.trnsolver.*`) on the `neuron` device; no `torch_xla`.
- **PyTorch/XLA**: Trn1 (and any host on the `torch-neuronx` 2.9 lineage).
  Kernels dispatch through `torch_xla`, unchanged.

trnsolver picks the native path when the native `torch_neuronx` and a
`neuron` device are present, and the XLA path when `torch_xla` is installed.
Set `TRNSCI_NKI_DISPATCH=native`, `xla` or `cpu` (shared by all trnsci
libraries) to force a path. With an editable install (`pip install -e`), set
`NKI_ENABLE_TRACE_CACHE=0`: the NKI kernel cache only versions installed
wheels and can serve stale kernels. See [installation](docs/installation.md).

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
U, s, Vh = trnsolver.svd(A)  # economy SVD: A = U @ diag(s) @ Vh
x = trnsolver.solve_spd(A, b)
M = trnsolver.inv_sqrt_spd(A)  # eigendecomposition-based
M, iters, res = trnsolver.inv_sqrt_spd_ns(A, tol=1e-8)  # Newton-Schulz, all-GEMM

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

`examples/scf_df_rhf.py` is the end-to-end version: density-fitted restricted Hartree–Fock with the J/K build on packed three-centre integrals (`trnblas`, `trntensor`), DIIS, and `trnsolver.eigh` every iteration, on the host or a NeuronCore. `scripts/bench_scf.py` times each stage against the CPU and float64 (PySCF integrals for real molecules if PySCF is installed; a synthetic s-Gaussian H2 lattice otherwise):

```bash
python examples/scf_df_rhf.py --system h2-lattice:16
python scripts/bench_scf.py --systems h2-lattice:16 --modes cpu,cpu64
python scripts/bench_scf.py --systems pyscf:alkane20 --modes cpu,neuron,neuron-nosync   # on Trn2
```

## Status

**v0.18.0** — Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron; LNC=1 and LNC=2; trnsci/trnsci#35). `eigh(A, method="jacobi")` gains a block Jacobi NKI kernel for 33 ≤ n ≤ 128 (padded sizes 64 and 128), and `eigh(method="auto")` now uses the Jacobi kernel for FP32 batches on the `neuron` device inside a measured (n, B) region, where it beats the aten cutover (#52). Inside that region batched `eigh` is 1.5–2.2× faster than CPU LAPACK; at n=128, B=1024 it takes 321 ms against 668 ms. Outside the region the default cutover is unchanged.

**v0.17.0** — Native TorchNeuron dispatch (Trn2/Trn3) alongside the PyTorch/XLA path (Trn1, maintenance); validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0; trnsci/trnsci#35). New `eigh(A, method="jacobi")`: a batched one-sided Jacobi NKI kernel for FP32 n ≤ 128 on the `neuron` device, with a batched-matmul torch path everywhere else (#50). At n=32, B=1024 it takes 40.2 ms against 60.5 ms for CPU LAPACK. The default `eigh` dispatch and its small-n cutover to `torch.linalg.eigh` are unchanged.

**v0.16.0** — Trn2 validation (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron device; trnsci/trnsci#35). `eigh` on the `neuron` device uses `torch.linalg.eigh` below `TRNSOLVER_EIGH_NKI_MIN_N` (default 512), and the NKI path runs its QR iteration on host tensors (#42). `apply_reflectors_kernel` compiles under NKI 0.7.0 / neuronx-cc 2.28. `cholesky`, `solve`, `solve_spd`, `inv_spd`, `eigh_generalized` and `qr` trace to Neuron-lowerable ops under `torch.compile` (#41); `eigh` (NKI path), `cg`, `gmres` and `gmres_ir` are eager-only.

**v0.15.0** — `gmres_ir(A, b, ...)`. GMRES with iterative refinement: Carson–Higham two-precision framework for general (non-symmetric) Ax = b. FP32 LU(A) computed once as a left preconditioner; each outer step computes a FP64 residual and solves the preconditioned correction via inner GMRES (FP32 Krylov). Extends the `solve_spd(iterative_refinement=True)` pattern from SPD Cholesky to general LU. Reliable for cond(A) ≲ 1e7. Returns `(x, total_inner_iters, fp64_residual)`. BF16 inner matvec (50–100× via Tensor Engine + SR) deferred to v0.16.0.

**v0.14.0** — BF16-native sketch path for `eigh_randomized` (Principle 1). New `matmul_kernel_fp32out` kernel: BF16 inputs → FP32 output via `nc_matmul` PSUM accumulation. When A is BF16 and NKI is active, `eigh_randomized` keeps A in BF16, routes all sketch/projection matmuls through `matmul_kernel_fp32out`, and runs QR/eigh in FP32. This delivers 50–100× BF16 throughput on the Tensor Engine while retaining FP32 accumulation quality — the PSUM upgrade is free.

**v0.13.0** — `eigh_randomized(A, k, oversampling=10, n_iter=0, seed=None)`. Randomized symmetric eigensolver implementing the Nakatsukasa–Tropp sketch-and-Rayleigh-Ritz (sRR) algorithm (SIAM SIMAX 45(2), 2024). O(n²k) vs O(n³) for `eigh`. Returns k eigenvalues of largest absolute magnitude and their corresponding orthonormal eigenvectors.

**v0.12.0** — `apply_reflectors_kernel`. Fused Householder reflector application: the eigenvector assembly step of `eigh` (n-2 iterations of `Q ← Q − 2 v (vᵀQ)`) is now a single NKI dispatch resident entirely in SBUF. SBUF layout trick: maintain Q^T in SBUF so the reduction `p = v^T Q` uses free-dim sum (axis=1, supported) instead of partition-dim sum (unsupported in NKI 0.3.0). All Vector Engine ops; simulator-safe.

**v0.11.0** — `matmul_kernel` (Tensor Engine via `nisa.nc_matmul`) + `matmul_kernel_sim` (Vector Engine, simulator-safe). The Rayleigh-Ritz refinement step in `eigh` now routes all three GEMMs through the Tensor Engine on hardware. Resolves the NKI 0.3.0 free-dim constraint that blocked the earlier vector matvec attempt (#36): a matrix moving operand (n, k) with k > 1 satisfies the constraint. Closes #36.

**v0.10.0** — `schur(A, tol=1e-10)`. Schur decomposition: returns `(Q, T)` with `A = Q @ T @ Q.T`. T is quasi-upper triangular (2×2 blocks for complex eigenvalue pairs). Implemented via Householder Hessenberg reduction + Francis implicit double-shift QR with deflation. Closes the last Factorizations "Deferred" item.

**v0.9.0** — `svd(A, full_matrices=False)`. Singular Value Decomposition: returns `(U, s, Vh)`. Economy SVD by default; BF16/FP16 promoted to FP32. Closes the `svd` item in the API table.

**v0.8.0** — `ssor_preconditioner(A, omega=1.0)` (#28). SSOR preconditioner for SPD systems: two triangular solves + diagonal scaling per application. Outperforms scalar Jacobi on coupled matrices (1D Laplacian, FEM stiffness). ω ∈ (0, 2); ω=1 is symmetric Gauss-Seidel.

**v0.7.0** — BF16/FP16 dtype support across the full public API (#19). All entry points (`cholesky`, `lu`, `qr`, `solve`, `solve_spd`, `inv_spd`, `pinv`, `inv_sqrt_spd`, `inv_sqrt_spd_ns`, `eigh`, `eigh_generalized`, `cg`, `gmres`, `block_jacobi_preconditioner`) accept BF16/FP16 inputs, upcast to FP32 internally, and restore the original dtype on output.

**v0.6.0** — `eigh` subspace rotation refinement: one Rayleigh-Ritz step (V^T A V re-diagonalization) after Householder-QR reduces eigenvector residuals by 1–2 orders of magnitude for n ≥ 64 (#31). `solve_spd` gains `iterative_refinement=True`: mixed-precision FP64 residual + second Cholesky solve for SPD systems with cond up to ~1e7 (#32).

**API coverage:**

| Category | Shipped (v0.18.0) | Deferred |
|----------|-------------------|----------|
| Eigensolvers | `eigh`, `eigh_generalized`, `eigh_randomized` | — |
| Factorizations | `cholesky`, `lu`, `qr`, `svd`, `pinv`, `schur` | — |
| Direct solvers | `solve`, `solve_spd`, `inv_spd`, `inv_sqrt_spd`, `inv_sqrt_spd_ns` | — |
| Iterative | `cg` (w/ preconditioner), `gmres`, `gmres_ir` | — |
| Preconditioners | `jacobi_preconditioner`, `block_jacobi_preconditioner`, `ssor_preconditioner` | — |

**Roadmap:**
- **v0.4.0** — NKI Householder-QR `eigh` validated on trn1.2xlarge (#9, #12, #38)
- **v0.4.1** — `eigh_generalized` NKI triangular-solve path via `trnblas.trsm` (#11)
- **v0.5.0** — Newton-Schulz trnblas.gemm (#14), FP64 CG/GMRES dots + Rayleigh refinement (#27), block-Jacobi (#16), `pinv` (#22)
- **v0.6.0** — eigh subspace rotation refinement (#31), solve_spd iterative refinement (#32)
- **v0.7.0** — BF16/FP16 across the full API (#19) ✓
- **v0.8.0** — SSOR preconditioner (#28) ✓
- **v0.9.0** — `svd` factorization ✓
- **v0.10.0** — `schur` (implicit-shift QR via Householder + Francis double-shift) ✓
- **v0.11.0** — Tensor Engine matmul kernel (`nisa.nc_matmul`), Rayleigh-Ritz on NKI (#36) ✓
- **v0.12.0** — Fused Householder reflector application (`apply_reflectors_kernel`) ✓
- **v0.13.0** — `eigh_randomized` (Nakatsukasa–Tropp sRR, O(n²k), Tensor Engine sketch) ✓
- **v0.14.0** — `matmul_kernel_fp32out` + BF16-native sketch in `eigh_randomized` (Principle 1: BF16→FP32 PSUM, 50–100× at BF16) ✓
- **v0.15.0** — `gmres_ir`: Carson–Higham iterative refinement for general Ax = b (FP32 LU preconditioner + FP64 outer residual) ✓
- **v0.16.0** — Trn2 validation on Neuron SDK 2.33 (native TorchNeuron device): `eigh` small-n cutover, NKI 0.7.0 kernel fixes, `torch.compile`-traceable factorizations ✓
- **v0.17.0** — Native TorchNeuron dispatch; batched Jacobi `eigh(method="jacobi")` kernel on Trn2 ✓
- **v0.18.0** — Block Jacobi kernel for 33 ≤ n ≤ 128; `eigh(method="auto")` uses the Jacobi kernel in a measured region on Trn2 ✓
- **v0.19.0** — BF16 inner matvec for GMRES-IR via `matmul_kernel_fp32out` + SR (Principle 3); block RHS support
- **v0.20.0+** — NKI Hessenberg reduction kernel, multi-NeuronCore parallel Jacobi (#20)

## Operations

| Category | Operation | Description |
|----------|-----------|-------------|
| Eigen | `eigh` | Symmetric eigendecomposition (Jacobi / torch) |
| Eigen | `eigh_generalized` | Generalized: `Ax = λBx` via Cholesky reduction |
| Eigen | `eigh_randomized` | Randomized: k dominant eigenpairs, O(n²k), sRR |
| Factor | `cholesky` | `A = LL^T` |
| Factor | `lu` | `PA = LU` |
| Factor | `qr` | `A = QR` |
| Factor | `svd` | `A = U diag(s) Vh` (economy by default) |
| Factor | `pinv` | Moore-Penrose pseudoinverse (truncated SVD) |
| Solve | `solve` | `Ax = b` (LU-based) |
| Solve | `solve_spd` | `Ax = b` (Cholesky, A is SPD) |
| Solve | `inv_spd` | `A^{-1}` for SPD A |
| Solve | `inv_sqrt_spd` | `A^{-1/2}` via eigendecomposition |
| Solve | `inv_sqrt_spd_ns` | `A^{-1/2}` via Newton-Schulz (all-GEMM) |
| Iterative | `cg` | Conjugate Gradient (SPD systems) |
| Iterative | `gmres` | GMRES (general systems) |
| Iterative | `gmres_ir` | GMRES-IR: iterative refinement, FP64 outer residual |
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
| trnsolver | Linear solvers and eigendecomposition | **v0.18.0** |
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
