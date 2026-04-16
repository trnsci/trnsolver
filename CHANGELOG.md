# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] — 2026-04-16

### Changed

- **`eigh` switches from classical Jacobi to Householder-QR** on the NKI
  path (#38 option B). Two-stage algorithm: stage 1 is Householder
  tridiagonalization via `matvec_kernel` + `rank2_update_kernel` NKI
  kernels (Vector-Engine for now; Tensor-Engine refactor tracked in #36);
  stage 2 is pure-host implicit-shift QR with deflation. Eigenvectors are
  assembled from stored Householder reflectors + accumulated Givens
  rotations. Simulator regression at rtol=1e-3 matches
  `torch.linalg.eigh`. Closes #9, #10, #38.
- Deleted the Jacobi-era code: `rotate_pairs_kernel`, `_jacobi_eigh_nki`,
  `_call_rotate_pairs`, `_rotation_angles_strided`,
  `_diag_block_fixup_strided`, `trnsolver/_brent_luk.py`. Kept as a
  post-mortem on #9 for architectural record.
- `tests/test_jacobi_neuron.py` renamed to `tests/test_eigh_neuron.py`;
  class `TestJacobiEighNKI` → `TestEighNKI`. Tests are algorithm-agnostic
  (they only call the public `trnsolver.eigh`).

- **NKI namespace migration: `neuronxcc.nki.*` → `nki.*`** (Neuron SDK 2.29 /
  NKI 0.3.0 Stable). `trnsolver/nki/dispatch.py` and `trnsolver/eigen.py`
  import from the canonical top-level `nki` package. Legacy shim not used.
- `[neuron]` extra restored with `nki>=0.3.0`, `neuronxcc>=2.24`,
  `torch-neuronx>=2.9`. These packages ship pre-installed on the Deep
  Learning AMI Neuron venv; the extra is for hosts that need the simulator
  path outside the DLAMI (ubuntu-latest CI runners).
- Pytest marker renamed `simulator` → `nki_simulator` to match suite
  convention. CI job renamed `test-simulator` → `nki-simulator`.
- Test file renamed `tests/test_jacobi_simulator.py` → `tests/test_nki_sim.py`
  and reworked to exercise the **public API** (`trnsolver.eigh`) under
  `TRNSOLVER_USE_SIMULATOR=1`, rather than calling the kernel directly.
  Catches integration bugs in dispatch + host driver, not just kernel-local
  issues.

### Added

- `TRNSOLVER_USE_SIMULATOR=1` env var: dispatch bypasses `torch_xla` and
  routes `rotate_pairs_kernel` through `nki.simulate(kernel)(np_args)` on
  CPU. Seconds-per-iteration feedback for kernel development without AWS
  cost.
- `trnsolver.nki._use_simulator()` helper exported alongside `_use_nki()`.
- `scripts/run_simulator_tests.sh` — SSM runner mirroring
  `run_neuron_tests.sh` but with `TRNSOLVER_USE_SIMULATOR=1` set.
- `docs/developing_kernels.md` — kernel-author guide: three dispatch
  modes, simulator workflow, CI gate matrix, architecture-first reminder.
- `docs/api/nki.md`: env-var table now documents
  `TRNSOLVER_USE_SIMULATOR`.

Closes #39.

### Fixed

- **`torch_xla.sync()` after each `rank2_update_kernel` call** in
  `_householder_tridiag`. Without the barrier, `A_work` (the kernel output)
  carried a growing XLA computation history into the next loop iteration,
  causing a unique traced graph — and a fresh NEFF compile — on every
  Householder step. With the barrier, each `_call_matvec` sees a concrete
  leaf tensor and reuses the cached NEFF. Also resolved the NCC_IDEL901
  delinearization compiler crash that appeared on the 30th+ compilation.
  Exposed by hardware run; simulator was unaffected (numpy path, no XLA).
  Closes #12.

- **Hardware validated** on trn1.2xlarge (NKI 0.3.0 / neuronxcc 2.24):
  14/14 `@pytest.mark.neuron` tests pass in 41 s wall-clock.
  Eigenvalue rtol=1e-3 holds for n ∈ {4, 8, 16, 32, 64, 128}. Closes #26.

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
