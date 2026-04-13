# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] — 2026-04-12

### Added

- `inv_sqrt_spd_ns(A)` — Newton-Schulz iteration for `A^{-1/2}`. All-GEMM,
  returns `(X, iters, residual)`. Progress on #14 (NKI-GEMM backend swap
  lands when trnblas GEMM validates on hardware).
- scipy.linalg baselines (eigh, cholesky, cho_solve) in
  `benchmarks/bench_solver.py` for LAPACK comparison. Progress on #13.
- `benchmarks/bench_cuda.py` — CUDA / cuSOLVER benchmarks, auto-skipped
  when no GPU is available. Vintage-matched methodology (trn1 ↔ A10G,
  trn2 ↔ H100). Progress on #13.
- `infra/terraform/gpu.tf` — opt-in GPU CI instance (default `g5.xlarge`
  A10G, vintage peer of trn1; `p5.4xlarge` for trn2 via override).
  Disabled by default via `enable_gpu_ci = false`.
- `scripts/run_cuda_tests.sh` — SSM-based runner mirroring
  `run_neuron_tests.sh` for the GPU box.

### Changed

- `docs/benchmarks.md` rewritten with vintage-matching methodology and
  real CPU numbers; `docs/aws_setup.md` documents the GPU instance and
  H100 opt-in.
- CI: bumped to `actions/checkout@v6` and `actions/setup-python@v6`
  (Node.js 24 runtime); standalone Deploy Docs workflow removed — docs
  are served from trnsci.dev.
- `pyproject.toml` normalized across the trnsci suite; added
  `Documentation` URL pointing to `trnsci.dev/trnsolver/`.

## [0.2.0] — 2026-04-12

### Added

- `jacobi_preconditioner(A)` in `trnsolver.iterative` — diagonal preconditioner
  callable for `cg`. Partial progress on #16.
- Benchmark suite `benchmarks/bench_solver.py` covering eigh, factorizations,
  solve, CG (with/without Jacobi preconditioner), and GMRES. Closes #8.
- Status badges in README (CI, PyPI, Python versions, license, docs). Closes #4.
- Manual Neuron hardware CI workflow (`.github/workflows/neuron.yml`) —
  `workflow_dispatch`-only, OIDC-based, wraps `scripts/run_neuron_tests.sh`.
  Closes #5.

### Changed

- `cg(A, b, M=...)` tensor preconditioner is now applied as `M @ r` (already-
  inverted operator), not `inv(M) @ r` recomputed every iteration.
- Repository transferred from `scttfrdmn/trnsolver` to `trnsci/trnsolver`.
  Hardcoded URLs updated across docs, pyproject, terraform, and CI.

## [0.1.1] — 2026-04-12

### Changed

- Bumped `neuronxcc` floor from `>=2.15` to `>=2.24` to unify with the
  rest of the trnsci suite. `torch-neuronx` floor bumped to `>=2.9`.

## [0.1.0] - 2026-04-11

### Added

- Initial scaffold of `trnsolver` — linear solvers and eigendecomposition
  for AWS Trainium via NKI. Third sibling in the `trn-*` suite alongside
  `trnblas` and `trnfft`.
- Eigenvalue API: `eigh` (symmetric), `eigh_generalized` (Cholesky reduction
  for `Ax = λBx`).
- Factorizations: `cholesky`, `lu`, `qr`, `solve`, `solve_spd`, `inv_spd`,
  `inv_sqrt_spd`.
- Iterative solvers: `cg`, `gmres`.
- NKI Jacobi rotation kernel scaffold in `trnsolver/nki/dispatch.py` with
  auto/pytorch/nki backend selection. PyTorch fallback active until
  on-hardware validation lands.
- SCF demo (`examples/scf_eigen.py`) for the quantum-chemistry use case:
  build Fock → solve `FC = SCε` → density → convergence loop.
- pytest test suite covering eigen, factorization, and iterative solvers
  on CPU.
- mkdocs-material site, GitHub Actions CI/docs/publish workflows, Terraform
  module for the Neuron CI instance, and `scripts/run_neuron_tests.sh`
  SSM-driven hardware test runner — all mirroring the trnblas/trnfft layout.
