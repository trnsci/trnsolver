# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Complex Hermitian and batched input gave wrong results with no error (#64).**
  `cholesky(upper=True)`, `solve_spd` (including `iterative_refinement=True`),
  `inv_spd`, `inv_sqrt_spd`, `eigh_generalized`, `eigh_randomized`, `pinv`
  and `ssor_preconditioner` now follow the conjugate transpose and per-matrix
  semantics of `torch.linalg`. Before, they used `.T`, which drops the
  conjugation and reverses batch dims; a batch of size n kept its shape.
  Relative error was 0.3–0.5 for complex input, and 0.4–2.5 or non-finite
  for a batch of size n. `pinv` used the first matrix's singular values as
  the cutoff for the whole batch and raised on empty input. Real (n, n)
  results are unchanged. Affected: every release through 0.20.0 (`pinv`
  since 0.5.0, `ssor_preconditioner` since 0.8.0). Complex input to
  `eigh_randomized` (eigenvalue error 1.35) now also stays off the real NKI
  matmul kernel.
  `eigh` sends complex and batched input (as from `eigh_generalized`) to
  aten instead of the real single-matrix NKI Householder path, which raised
  for them under `TRNSOLVER_REQUIRE_NKI=1` on a Neuron host.
- **`cg`/`gmres` returned `x0` (or zeros) with residual 0.0 for a zero or
  small-scale right-hand side (#68).** `b = 0` with an `x0` returned `x0`
  in place of the solution 0. Any `‖b‖ < 1e-15` returned without
  iterating, and `cg` stopped a small-scale problem before its first step
  (`|pAp| < 1e-30`). Now only `b == 0` short-circuits (to zeros), and `cg`
  stops on exact breakdown. `jacobi_preconditioner` scaled the columns of
  a block `r` of shape (n, k); it now scales rows. Affected: every
  release through 0.20.0 (`jacobi_preconditioner` since 0.2.0).
- **`lu` documented the wrong permutation convention (#71).** The docstring,
  API docs, quickstart and README said `P @ A = L @ U`; `lu` returns
  `A = P @ L @ U` (the `torch.linalg.lu` convention), which differs whenever
  the pivot permutation is not its own inverse. The docs now match; the
  returned values are unchanged. Affected docs: every release through 0.20.0.

### Added

- **`orthogonalizer(S, method, lindep)` and `eigh_generalized(F, S, X=X)`:
  a tolerance for near-singular overlaps (#72, for #15).** `"symmetric"`
  (Loewdin S^-1/2, the default), `"canonical"` (keeps the eigenvalues of S
  above `lindep` x max, n x m X) and `"cholesky"` (pivoted Cholesky, no
  eigensolve, O(n m^2)). It returns `X` with X^H S X = I on the kept
  subspace and the kept `rank`. It is built once, in float64 on the host,
  for real or complex Hermitian S, batched, and differentiable.
  `eigh_generalized(F, S, X=X)` then needs only two GEMMs, an m x m `eigh`
  and a back-transform, in F's dtype and on F's device, and no Cholesky of
  S. `examples/scf_df_rhf.py` uses them (`--orth`, `--lindep`, and
  `pyscf:<name>@<basis>` systems for diffuse bases). The default results
  are unchanged: every current test system is far from the 1e-7 threshold,
  so `symmetric` is still S^-1/2, and its energies match the previous
  release to all printed digits (fp64 and fp32, h2-lattice, water, benzene,
  C10H22). With a diffuse basis, dropping the dependencies is what lets an
  FP32 SCF converge. benzene/aug-cc-pVTZ (n = 414): with three functions
  dropped, it converges in 23 iterations, and the float64 energy at its
  orbitals is 4.7e-6 Eh from PySCF. Plain S^-1/2, the Cholesky of S, or
  `cholesky` at 1e-7 do not converge in 80 iterations.
  C20H42/aug-cc-pVDZ (n = 838, two functions dropped) converges in FP32
  either way. In float64, dropping the two functions raises the energy by
  2.8e-5 Eh: lower `lindep` when that matters more than FP32 conditioning.
- **Guard against the NKI trace-cache closure bug ([aws-neuron/aws-neuron-sdk#1454](https://github.com/aws-neuron/aws-neuron-sdk/issues/1454),
  trnsci/trnsci#35).** `tests/test_trace_cache_safety.py` (CPU) finds every
  `@nki.jit` kernel and fails if its body, or a helper it calls, reads a
  closure variable or a module global that is not a constant expression
  (reviewed exceptions are allow-listed with a reason), and checks each
  kernel's `__closure__` at runtime against a stub `nki`.
- Tests: `conftest.py` no longer turns the NKI trace cache off. Neuron tests
  run with it on, as under `import torch_neuronx`, with a fresh
  `NKI_TRACE_CACHE_URL` directory per session (no stale editable-install
  traces). `NKI_ENABLE_TRACE_CACHE=0` still turns it off.
- README: known platform issue note for #1454.
- **DF-RHF example: J and K from trntensor's fused packed-integral kernels
  (trntensor#60).** `DFJK(fused=None)` (and `run_scf(jk_fused=...)`,
  `bench_scf.py --jk-k auto|fused|unpack`) builds K with
  `trntensor.df_exchange` and J with `trntensor.df_coulomb` on a `neuron`
  device when trntensor provides them. The packed integrals stay in one
  block per device and are never unpacked. Trn2, LNC=2, one logical core,
  time to convergence against the unpack + einsum + syrk path: C10H22
  352 -> 229 ms, C20H42 756 -> 587 ms, C40H82 9.68 -> 4.71 s (J/K stage
  365 -> 163 ms per iteration). At LNC=1, C40H82 runs in 7.7 s fused; the
  unpack path no longer fits there (12 GB per physical core). Older
  trntensor keeps the unpack path.

### Fixed

- **`TRNSOLVER_JACOBI_ONEZERO` is a kernel argument ([aws-neuron/aws-neuron-sdk#1454](https://github.com/aws-neuron/aws-neuron-sdk/issues/1454)).**
  `jacobi_sweeps_kernel` and `jacobi_block_kernel` read the `_ONEZERO`
  module global from their body, which the persistent NKI trace cache does
  not key on, so a process could run a trace made under the other setting.
  The flag is now their `onezero` argument (a static `int` on the
  `trnsolver::jacobi_sweeps` / `jacobi_block` ops; `dispatch._launch` takes
  `static=`). Both settings compute the same result; only the Tensor
  Engine pass count differs.

## [0.20.0] — 2026-10-31

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: opt-in per-core worker-process pool for batched Jacobi `eigh` (`TRNSOLVER_EIGH_POOL=proc`): 3.65× scaling on 4 cores, 1.75–2.0× faster than 8-thread CPU at n=64–128, B≥1024 (#57); host tensors now take the host-LAPACK cutover on TorchNeuron hosts (an n=64 call was 265 s, now 0.15 ms), and a DF-RHF SCF example built from the suite (#58).

### Added

- **DF-RHF SCF end to end on the trnsci libraries (#15).**
  `examples/scf_df_rhf.py` runs closed-shell density-fitted Hartree-Fock with
  the J/K build on packed three-centre integrals (`trnblas.gemv`/`syrk`,
  `trntensor.einsum`/`unpack_tril`), DIIS, and `trnsolver.eigh` (or
  `eigh_generalized`) per iteration, on the host or a NeuronCore. Systems: a
  synthetic s-Gaussian H2 lattice with closed-form integrals (no
  dependencies), or PySCF integrals for real molecules (water, benzene,
  C_kH_2k+2). `scripts/bench_scf.py` times each stage per iteration, CPU
  against device against float64, and one `eigh` at SCF sizes
  (`--eigh-only`). The float64 loop matches PySCF's DF-RHF energy to
  1e-11 Eh. On Trn2 (one logical core, host pinned to 2 CPUs), time to
  convergence is 9.5x (n_ao = 250), 38x (490) and 51x / 78x (970, packed /
  resident unpacked integrals) faster than the same loop on those 2 CPUs;
  below n_ao ~ 128 the device is slower. `eigh` (host LAPACK through aten)
  is the largest device stage up to n_ao = 250 (32-39%), J/K from 490 up
  (54%, then 72-83% at 970). Options: J/K split over NeuronCores
  (`neuron-jk<k>` modes; 1.5x on the J/K stage with 4 cores), resident
  unpacked integrals (`--jk-square`), and a GEMM-only purification solver
  (`--solver purify`; slower than the host `eigh` at every measured size,
  kept as the measured experiment).
  Tests: `tests/test_scf_df_rhf.py` (CPU).
- **Process-pool multi-core Jacobi `eigh`, opt-in (#29).**
  `TRNSOLVER_EIGH_POOL=proc` (with `TRNSOLVER_EIGH_CORES=<k>|auto`) solves a
  host batch on persistent worker processes, one per logical NeuronCore
  (`TRNSOLVER_EIGH_POOL_CORES`, else `NEURON_RT_VISIBLE_CORES`), instead of
  one host thread per core. Inputs go through per-worker shared-memory
  segments; workers write results straight into a shared output buffer
  that the returned tensors are views of, reused once the caller drops
  them. Workers start on the first call and stay up
  (`trnsolver.nki.eigh_pool.shutdown()` frees the cores); a failed pool
  shuts down and `eigh` falls back to one core. The calling process must
  not use the pool's cores itself, so device-resident input keeps the
  thread split. Trn2, n = 128, B = 4096, 8 host CPUs: 1 / 2 / 4 cores
  1182 / 607 / 324 ms (thread split 1227 / 763 / 531 ms; 8-thread
  batch-parallel LAPACK 655 ms). Default (unset) unchanged.
- `TRNSOLVER_JACOBI_SWEEPS_PER_LAUNCH=<k>`: the scalar kernel (padded
  n <= 32) runs its first sweeps k per launch (5-14% faster on Trn2 at
  n = 32). Default 1, unchanged.
- `scripts/bench_jacobi_multicore.py --modes pool`, and
  `scripts/probe_large_n_split.py`, the step-cost probe behind the
  single-matrix multi-core verdict in #29.

### Fixed

- **`eigh` of a host tensor on a TorchNeuron host no longer runs the NKI
  path (#15).** With the default backend, a CPU tensor skipped the #42
  cutover and drove the launch-bound NKI tridiagonalization through the
  device: 265 s for one n = 64 call on Trn2 (first call, with compiles),
  against 0.15 ms for LAPACK. Host inputs on the native path now take the
  cutover, i.e. host LAPACK. `TRNSOLVER_EIGH_NKI_MIN_N=0`,
  `TRNSOLVER_REQUIRE_NKI=1` and the simulator keep the kernel path.
- `examples/scf_eigen.py`: the closed-shell density is `P = 2 C_occ C_occ^T`
  (was `C_occ C_occ^T`, inconsistent with `F = H + J - K/2`), the energy is
  evaluated at the density the Fock matrix was built from, and the default
  tolerance is 1e-5 (1e-8 is below what fp32 resolves; the demo never
  converged).

## [0.19.0] — 2026-10-31

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: opt-in batched Jacobi `eigh` across logical NeuronCores (`TRNSOLVER_EIGH_CORES`): n=128, B=1024 drops from 310 to 152 ms on four cores (#54); corrects the v0.18.0 CPU-baseline headline.

### Added

- **Batched Jacobi `eigh` across NeuronCores, opt-in (#48).**
  `TRNSOLVER_EIGH_CORES=<k>|auto` splits a `method="jacobi"` batch (and the
  `method="auto"` Jacobi region) into whole-tile chunks, one per logical
  NeuronCore, each solved by its own host thread. `auto` uses every visible
  core while each keeps at least 512 / 256 / 128 matrices (padded n = 32 /
  64 / 128). The default (unset or 1) is the single-core path, unchanged.
  Chunks for other cores go through the host: direct core-to-core copies
  gave no speed-up. The first call for a split shape runs the chunks in
  turn, because NKI tracing (also inside aten ops) is not thread-safe; if
  the split fails, `eigh` falls back to one core. Trn2, n = 128, B = 1024,
  device input: 1 / 2 / 4 cores 309 / 202 / 127 ms (8 host CPUs);
  batch-parallel CPU LAPACK on 8 threads 159 ms.
- `scripts/bench_jacobi_multicore.py`: the 1 / 2 / 4-core, process-per-core
  and CPU (batched, and batch-parallel threads) measurements.

### Corrected

- **The v0.18.0 headline overstated the CPU comparison.** Its "1.5–2.2×
  faster than CPU LAPACK" compared against batched `torch.linalg.eigh`,
  which barely uses extra threads at these sizes. Splitting the batch
  across the same 2 threads is 1.6–5.8× faster. Against that baseline the
  single-core kernel at n = 128, B = 1024 is 310 vs 352 ms (1.13×), not
  321 vs 668 ms (#54).

### Fixed

- Native launches run on the input's NeuronCore (`neuron:k`) instead of
  always on the current device, and the Jacobi driver keeps its work on a
  NeuronCore-resident input's core.

## [0.18.0] — 2026-10-30

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron; LNC=1 and LNC=2) — see trnsci/trnsci#35.

Headline performance: block Jacobi kernel for 33 ≤ n ≤ 128 and a measured `eigh(method="auto")` region; inside it batched `eigh` on Trn2 is 1.5–2.2× faster than CPU LAPACK, e.g. 321 vs 668 ms at n=128, B=1024 (#52). *Corrected in the next release: that baseline is batched `torch.linalg.eigh`; against the batch split over the same threads the gain is about 1.13× (#54).*

### Added

- **Block Jacobi kernel for 33 <= n <= 128 (#48).** `eigh(method="jacobi")`
  on the native `neuron` device now runs `jacobi_block_kernel` (native op
  `trnsolver::jacobi_block`) for padded sizes 64 and 128. Columns form
  blocks of 8; per step, per tile, the Tensor Engine transposes Z^T, forms
  the Gram matrix, compacts each block pair's 16 x 16 Gram block, transposes
  the accumulated rotations into the circle-method block shuffle and applies
  them to [Z^T | V^T] as one matmul. The Vector Engine runs one-sided Jacobi
  on the 16 x 16 Gram blocks only (full round robin on a sweep's first step,
  the 8 cross-pair steps after that). Trn2, B = 1024, end to end: n = 64
  194 -> 75 ms, n = 128 1350 -> 321 ms (CPU LAPACK, 2 threads: 163 / 668 ms).
  `TRNSOLVER_JACOBI_BLOCK=0` restores the scalar kernel.
- **`eigh(method="auto")` picks the Jacobi kernel in a measured region
  (#48).** FP32 batches on the native `neuron` device with (n, B) in
  `eigen._JACOBI_AUTO_REGION` (n <= 32 with B >= 512, 33 <= n <= 64 with
  B >= 256, 65 <= n <= 128 with B >= 1024, n >= 96 with B >= 256, n = 128
  with B >= 64), where it beats the default aten cutover. Outside it the
  cutover is unchanged. `TRNSOLVER_EIGH_JACOBI_AUTO=off|n_min-n_max:B_min,...`
  overrides it.

### Changed

- Jacobi device driver: eigenvectors are returned with unit columns and the
  eigenvalues as Rayleigh quotients `v.z / v.v - sigma`; eigenvectors are
  reordered with one row `index_select` instead of `torch.gather` (102 ms ->
  2 ms on device at B = 1024, n = 128). The scalar kernel's partner
  permutation matmuls set `is_stationary_onezero`.

## [0.17.0] — 2026-10-30

Native TorchNeuron dispatch (Trn2/Trn3) alongside the PyTorch/XLA path (Trn1, maintenance); validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0) — see trnsci/trnsci#35.

Headline performance: batched Jacobi `eigh(A, method="jacobi")` kernel on Trn2, 40.2 ms vs 60.5 ms for CPU LAPACK at n=32, B=1024 (#50); the native `eigh` cutover to `torch.linalg.eigh` below n=512 stays the default.

### Added

- **`eigh(A, method="jacobi")`: batched Jacobi eigensolver (#48).** Accepts
  `(n, n)` or `(..., n, n)`; `TRNSOLVER_EIGH_METHOD=jacobi` selects it when
  `method` is not given. Opt-in: the default `auto` dispatch, including the
  aten cutover for n < 512, is unchanged.
  - On the native `neuron` device (FP32, n <= 128) it runs the one-sided
    (Hestenes) Jacobi kernel in `trnsolver/nki/jacobi.py`: up to 128 tiles
    of 128 / n matrices each (512 matrices at n = 32) held in SBUF for a
    whole sweep, one launch per
    sweep, a convergence check after the expected sweep count (6-8 for
    random matrices), at most 15 sweeps. Native op `trnsolver::jacobi_sweeps`.
  - Each step uses the Tensor Engine as an exact partition permutation
    (the block-diagonal step permutation is the stationary operand, shared
    by every tile) and applies the n/2 rotations on the Vector Engine in the
    `X + K^T X` form, which keeps FP32 orthogonality at about 1e-6.
  - Elsewhere (CPU, FP64, the XLA lineage, n > 128) it runs the torch
    implementation in `trnsolver/_jacobi.py`: parallel-ordered two-sided
    Jacobi with each step as batched matmuls.
  - Tuning: `TRNSOLVER_JACOBI_GROUP` (tiles per launch) and
    `TRNSOLVER_JACOBI_CHAINS` (independent dependency chains per step).
  - Tests: `tests/test_jacobi.py` (schedules, layout, torch path and the
    device host driver against `torch.linalg.eigh` on CPU; kernel vs a torch
    model in the simulator; NKI 0.7 MLIR verify; `@pytest.mark.neuron`).
  - Bench: `scripts/bench_jacobi.py` (CPU LAPACK, torch-on-device, compiled,
    kernel).

- **Native PyTorch dispatch on TorchNeuron (#40, trnsci/trnsci#35).** On
  TorchNeuron (Neuron SDK 2.33+, `torch-neuronx` 2.14, PyTorch 2.14,
  Trn2/Trn3) every hardware NKI kernel is registered as a PyTorch custom op
  with `torch_neuronx.nki_op` and launched on `device="neuron"`, with no
  `torch_xla` import. Ops: `trnsolver::matvec`, `rank2_update`, `matmul`,
  `matmul_fp32out`, `apply_reflectors`. Kernel bodies are unchanged.
  - The dispatch path is detected on the first kernel launch: native when
    `torch_neuronx` exports `nki_op` and a `neuron` device is available, XLA
    when `torch_xla` is installed, otherwise none. The XLA path (Trn1) is
    unchanged: kernels run on `torch_xla.device()` followed by
    `torch_xla.sync()`, as before. The simulator path is unchanged.
  - The custom ops are registered lazily (`trnsolver/nki/_native.py` is
    imported only once the native path is selected), so CPU-only installs
    and the XLA lineage see no import-time side effects.
  - Every host wrapper (`_call_matvec`, `_call_rank2_update`, `_call_matmul`,
    `_call_matmul_fp32out`, `_call_apply_reflectors`) goes through one
    launch helper, `dispatch._launch`.
  - `TRNSCI_NKI_DISPATCH=auto|native|xla|cpu` forces the dispatch path (a
    suite-wide variable). Forcing an unavailable path raises.
  - `TRNSCI_XLA_DEPRECATION=1` makes the XLA path emit one
    `DeprecationWarning`. Off by default; the suite turns it on at
    TorchNeuron GA.
  - New `[torchneuron]` extra (`torch>=2.14`, `torch-neuronx>=2.14`,
    `nki>=0.6`); `[neuron]` stays the PyTorch/XLA lineage.
  - Tests: `conftest.py` sets `NKI_ENABLE_TRACE_CACHE=0` by default;
    CPU-runnable path-selection, routing and lazy-registration tests;
    `@pytest.mark.neuron` native-path tests (not yet run on hardware). The
    two `test_eigh_neuron.py` tests that call kernels on `torch_xla` tensors
    are skipped off the XLA path.

### Changed

- **Householder tridiagonalization keeps its scalar work on the host.**
  `_householder_tridiag` reads column k of the working matrix once per step
  and computes the norm, sign, `v`, `v·v`, `v·w` and `u` on CPU copies; only
  the working matrix and the two kernels stay on the device. The fixup of
  row and column k is two slice writes instead of four. The reflectors are
  accumulated on the host. This replaces about six device `.item()` syncs
  and several element writes per step. The algorithm is unchanged, and CPU
  and simulator results are bit-identical.
- **`gmres` keeps its Hessenberg matrix and least-squares solve on the
  host.** Every entry was already read back as a Python float; the Krylov
  basis stays on `b`'s device. Removes per-element device writes and a
  device `lstsq`. CPU results are bit-identical.
- **`schur` runs on a CPU copy of the input** and returns `Q`, `T` on the
  input's device, like the eigh QR iteration (#42). CPU inputs are
  unchanged.

## [0.16.0] — 2026-10-28

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron device) — see trnsci/trnsci#35.

### Fixed

- **`eigh` on the native `neuron` device uses `torch.linalg.eigh` below
  `TRNSOLVER_EIGH_NKI_MIN_N` (default 512).** On Trn2 the NKI path took
  0.4 s at n=16 and 21 s at n=128, against 0.4-1.1 ms for aten on the same
  device, and grew about 4x per doubling of n. It is launch-bound, not
  compute-bound. Tridiagonalization makes n-2 host-driven steps, each with
  two kernel launches, about six `.item()` reads and an XLA sync. The
  implicit-shift QR iteration then ran element by element on the input's
  device: O(n^2) Givens rotations, each with about 20 single-element reads,
  writes and column updates, so at ~0.2 ms per eager launch the QR stage
  dominated. Tensors on the `neuron` device with n below the threshold now
  go to `torch.linalg.eigh`, and the result stays on the input's device.
  `TRNSOLVER_EIGH_NKI_MIN_N=0` turns the cutover off, and
  `TRNSOLVER_REQUIRE_NKI=1` bypasses it, so the hardware suite still runs the
  kernels. Other devices, including CPU inputs with the NKI backend, are not
  affected. The NKI path is single-tile (n <= 128), so with the default every
  size it supports uses aten. Under `torch.compile` the cutover runs eagerly
  behind the same single graph break as the NKI path. (#42)
- **The NKI `eigh` path runs its QR iteration on CPU tensors.** `diag`,
  `subdiag` and the Givens accumulator are copied to the host before
  `_qr_iterate`, and the accumulator is moved back to the input's device
  before `apply_reflectors_kernel`. CPU inputs already took this path, so
  their results are unchanged. Device-resident inputs no longer pay one
  launch or sync per scalar. (#42)
- **`apply_reflectors_kernel` now compiles under NKI 0.7.0 / neuronx-cc 2.28
  (Trn2).** The kernel loaded all Householder vectors as an SBUF `V^T` and
  sliced row `V_T[k:k+1, :]` per step. That is a 1-partition access at
  partition k, and the newer verifier only allows those to start at partition
  0, 32, 64 or 96 (`'nisa.tensor_copy' op Expected 'src' partition start in
  {0, 32, 64, 96}`). This failed all 19 `TestEighNKI` hardware tests. Each
  vector is now loaded per step with `load_transpose2d(V_refs[:, k:k+1])`, so
  it lands at partition 0, the same idiom `matmul_kernel_sim` uses. The
  algorithm is unchanged and simulator output is bit-identical to the old
  kernel for n ∈ {4, 8, 16, 33, 64, 128}. (#40)
- **`benchmarks/bench_solver.py::TestEigh::test_eigh_nki` skips n > 128.**
  The NKI eigh path is single-tile, and `_householder_tridiag` raises
  `NotImplementedError` for n > 128, so the 256 and 512 cases failed on
  hardware rather than measuring anything. The PyTorch and torch baselines
  still run at every size.
- **`cholesky`, `solve`, `solve_spd`, `inv_spd` and `eigh_generalized` use
  only Neuron-lowerable ops under `torch.compile`.** On Trn2 there were two
  separate failures:
  - `torch.linalg.cholesky` and `torch.linalg.solve` decompose to a `*_ex`
    kernel followed by `aten._linalg_check_errors(info)`. That op only
    raises, and newer PyTorch wraps it in the `with_effects` higher-order op,
    which TorchNeuron's FxImporter does not implement (`BackendCompilerFailed`).
  - The Neuron backend has no lowering for `linalg_cholesky_ex` or
    `linalg_solve_ex` themselves.

  Inside a compiled region (`torch.compiler.is_compiling()`), real inputs now
  use traceable implementations built from elementwise ops, `where`, `mm`,
  `sqrt`, `argmax`, `gather` and `tril`/`triu`. Each is a static loop that
  unrolls at trace time, with no `.item()`:
  - `_cholesky_traceable`: right-looking column Cholesky that reads only the
    lower triangle, like `potrf`.
  - `_solve_lu_traceable`: Gaussian elimination with partial pivoting on
    `[A | B]`. It picks the same pivot as `getrf` (`argmax |a_ik|`) and swaps
    rows with a `gather` permutation.
  - `_solve_triangular_traceable`: forward and back substitution, used by
    `solve_spd` in place of `torch.linalg.solve_triangular`.

  Compiled results match eager to about 1e-6 in fp32 and 1e-13 in fp64 at
  n = 128. Under `torch.compile`, a non-SPD or singular input yields NaN or
  inf instead of raising. Eager calls and complex inputs are unchanged. (#41)
- **`qr` traces to StableHLO-lowerable ops under `torch.compile`.**
  `aten.linalg_qr` has no StableHLO lowering on Neuron. When compiling, real
  inputs now use `_householder_qr`, a reduced Householder QR made of
  elementwise ops, `where` masks, `mm` and `triu`. It has no `.item()`, no
  data-dependent control flow and no in-place slice assignment, and its
  static column loop unrolls at trace time. It uses LAPACK's sign convention
  (`R[k, k] = -sign(A[k, k]) * ||A[k:, k]||`, and the reflector is skipped
  when the sub-column is already zero), so Q and R match `torch.linalg.qr` to
  rounding. Supports batch dimensions. Eager calls and complex inputs still
  use `torch.linalg.qr`. (#41)
- **`eigh` (NKI path), `cg`, `gmres` and `gmres_ir` are documented as
  eager-only and wrapped in `torch.compiler.disable`.** Their convergence and
  deflation logic is data-dependent Python control flow on `.item()` scalars,
  and they return Python numbers. Under `torch.compile` this split the graph
  into many pieces (26 graph breaks for `eigh` at n=128, 4 for `cg`). They now
  run eagerly behind one graph break, and `fullgraph=True` raises a clear
  `Unsupported` error. The algorithms are unchanged. (#41)

### Added

- **`tests/test_compile.py`**: CPU checks with `torch.compile(backend="aot_eager",
  fullgraph=True)`:
  - `cholesky`, `solve` (including a case that needs pivoting and a vector
    right-hand side), `solve_spd`, `inv_spd` and `qr` trace with zero graph
    breaks and match eager in fp32 and fp64.
  - The traced aten graph contains none of `_linalg_check_errors`,
    `linalg_cholesky_ex`, `linalg_solve_ex`, `linalg_solve_triangular`,
    `linalg_lu` or `linalg_qr`.
  - Eager `cholesky` and `solve` still raise `LinAlgError`.
  - The traceable kernels match LAPACK in fp64, including batches, pivoting
    and zero columns.
  - `cg` falls back to eager. With `fullgraph=True` it raises `Unsupported`
    on torch 2.11 and `UserError` on 2.14.

- **`tests/test_nki_compile.py`** (marker `nki_simulator`): runs the NKI
  frontend (trace → MLIR → `verify()`) on CPU for every `@nki.jit` kernel at
  n ∈ {8, 16, 32, 128} for both trn1 and trn2. `nki.simulate` does not run
  the MLIR verifier, so this catches hardware-only lowering errors like #40
  without a Neuron device. Needs `nki`, but not `neuronx-cc`.

## [0.15.0] — 2026-04-22

### Added

- **`gmres_ir(A, b, tol=1e-10, max_outer=5, inner_tol=1e-1, inner_maxiter=50, inner_restart=20)`**
  — GMRES with iterative refinement for general (non-symmetric) Ax = b. Implements
  the Carson–Higham two-precision framework, extending the `solve_spd` iterative
  refinement path (Cholesky, SPD only, one correction step) to GMRES + multiple outer
  iterations over any invertible A.

  Algorithm: compute FP32 LU(A) once; outer loop computes the FP64 residual
  r = b − Ax, then solves the left-preconditioned correction system
  (U⁻¹ L⁻¹ Pᵀ A) δx = U⁻¹ L⁻¹ Pᵀ r via the existing `gmres` callable interface,
  and accumulates x += δx. When the FP32 LU is accurate, the preconditioned
  system is near-identity and the inner GMRES converges in very few steps.

  Precision tiers:
  - **Factorization:** FP32 LU via `torch.linalg.lu` (once, reused per outer step)
  - **Inner GMRES:** FP32 Krylov vectors and modified Gram-Schmidt
  - **Outer residual:** FP64 — captures errors below FP32 machine epsilon
  - **Correction solve:** FP32 triangular solves reusing stored L, U
  - **Reliable for:** cond(A) ≲ 1/eps32 ≈ 1e7

  Returns `(x, total_inner, residual)` where `total_inner` is the total GMRES
  matvec count across all outer steps and `residual` is the final FP64 relative
  residual. BF16/FP16 inputs upcast to FP32 via `_to_fp32`; output restored.

  **Honest non-fits and deferred work:**
  - **BF16 inner path deferred (v0.16.0).** The inner matvec A@v could use
    `matmul_kernel_fp32out` (BF16 Tensor Engine → FP32 PSUM, Principle 1) for
    50–100× throughput. However, Gram-Schmidt dot products need SR (Principle 3,
    Connolly–Higham–Mary 2021) for BF16 stability at n ≥ ~300. Deferred until
    `trnrand` SR lands.
  - **Single-RHS only.** Block RHS (n × k) via column iteration is v0.16.0.
  - **Callable A not supported.** LU requires a concrete matrix; raises `TypeError`
    for callable matvec operators. Use `gmres` directly for matrix-free problems.

## [0.14.0] — 2026-04-21

### Added

- **`matmul_kernel_fp32out(A, V)` and `matmul_kernel_fp32out_sim(A, V)`** in
  `trnsolver.nki.dispatch`. Tensor Engine C = A @ V with FP32 output regardless
  of input dtype. A and V are BF16; nc_matmul accumulates BF16 products to FP32
  via PSUM at no extra cost (Principle 1: PSUM as free FP32 accumulator). The
  result is stored as FP32 — unlike `matmul_kernel`, which downcasts the PSUM
  result back to A.dtype and would discard the accumulated FP32 precision.

  `matmul_kernel_fp32out_sim` is the simulator-safe Vector Engine variant (same
  interface, FP32 output; receives FP32-upcast arrays since numpy has no BF16 dtype).

- **`_call_matmul_fp32out(A, V)`** dispatch helper in `trnsolver.eigen`. Casts
  A and V to BF16 for the Tensor Engine, calls `matmul_kernel_fp32out` on hardware
  (or `matmul_kernel_fp32out_sim` on the simulator with FP32-upcast numpy), and
  returns a FP32 result tensor on `orig_device`. Falls back to
  `torch.matmul(A.float(), V.float())` when NKI is unavailable or n/k > 128.

### Changed

- **`eigh_randomized`** now uses the BF16-native sketch path when A is BF16 and
  the NKI backend is active (`_use_nki()` returns True). Instead of upcasting A to
  FP32, it keeps A in BF16 and routes the sketch (A @ Ω), power-iteration, and
  projection (A @ Q) through `_call_matmul_fp32out`. All post-sketch operations
  (QR of Y, eigh of the small ℓ×ℓ B matrix, eigenvector lift) run in FP32.
  Output is restored to BF16 via `_restore`. On CPU/non-NKI hosts the original
  _to_fp32 upcast path is used unchanged.

  The BF16-native path delivers the 50–100× BF16 throughput on the Tensor Engine
  while maintaining FP32-quality accumulation in the sketch — the exact scenario
  identified in the trnsci technical brief (Principle 1). This is the correctness
  baseline; profiling vs cuSOLVER on vintage-matched hardware is tracked separately.

## [0.13.0] — 2026-04-21

### Added

- **`eigh_randomized(A, k, *, oversampling=10, n_iter=0, seed=None)`** — Randomized
  symmetric eigensolver implementing the Nakatsukasa–Tropp sketch-and-Rayleigh-Ritz
  (sRR) algorithm (SIAM SIMAX 45(2), 2024). Returns the k eigenvalues of largest
  absolute magnitude and their corresponding eigenvectors. Complexity O(n²k) vs
  `eigh`'s O(n³), with 50–100× speedup potential at BF16 on Trainium once the
  native BF16 sketch path lands (v0.14.0 target).

  Algorithm: sketch Ω ∈ R^{n×(k+p)} ~ N(0,1); Y = A@Ω; optional power iteration
  (A@Q each round with re-orthogonalization); Q = orth(Y); B = Q^T A Q (small
  (k+p)×(k+p) matrix); exact eigh(B); select k eigenvalues of largest |value|; lift
  V_k = Q @ V_small. Output eigenvalues are Rayleigh quotients sorted ascending.

  - `_call_matmul` shape guard extended to also short-circuit when `_use_nki()` is
    False, making it safe for callers such as `eigh_randomized` that call it
    unconditionally on arbitrary-size inputs (previously only dimension bounds were
    checked; non-NKI hardware would attempt `import torch_neuronx`).
  - Optional `trnrand` import for the sketch matrix Ω (NKI-accelerated Philox RNG);
    falls back to `torch.randn` when `trnrand` is not installed.
  - `seed` parameter enables reproducible runs; `None` = unseeded.

  **Note:** Targets eigenvalues of largest magnitude, not smallest. For SCF's
  k occupied orbitals (k smallest eigenvalues), use `eigh`. v0.14.0 will add a
  native BF16 sketch path (BF16 A → FP32 PSUM via `nc_matmul`, Principle 1).

### Changed

- `_call_matmul` dispatch helper now short-circuits with `torch.matmul` when
  `_use_nki()` is False (in addition to the existing n > 128 / k > 128 dimension
  guards). This is a pure correctness fix for CPU-only environments; NKI-hardware
  behavior is unchanged.

## [0.12.0] — 2026-04-21

### Added

- **`apply_reflectors_kernel(V_refs, Q_in)`** in `trnsolver.nki.dispatch`. Fused
  Householder reflector application: applies n-2 reflectors to Q in a single NKI
  dispatch instead of the previous host-driven Python loop (n-2 iterations of
  `Q ← Q − 2 v (vᵀQ)` via `torch.outer`). Eliminates n-2 iterations of O(n²)
  PyTorch calls for the eigenvector assembly step of `eigh`, replacing them with
  one Vector Engine kernel resident entirely in SBUF.

  SBUF layout: the kernel maintains Q^T in SBUF so the critical reduction
  `p[j] = Σᵢ vₖ[i]·Q[i,j]` becomes a free-dim sum (axis=1, supported by NKI
  0.3.0) rather than a partition-dim sum (axis=0, unsupported). V_refs is loaded
  transposed (V_T[k, i] = vₖ[i]) enabling static row-slice access within the
  compile-time-unrolled `nl.affine_range` loop. The kernel returns Q^T;
  `_call_apply_reflectors` transposes back before returning to
  `_householder_qr_eigh`.

  All ops are Vector Engine (`load_transpose2d`, `broadcast_to`, `multiply`,
  `sum(axis=1)`) — the kernel is simulator-safe with no separate `_sim` variant;
  `apply_reflectors_kernel_sim = apply_reflectors_kernel`.

- **`_call_apply_reflectors(V_refs, Q)`** dispatch helper in `trnsolver.eigen`.
  Routes to `apply_reflectors_kernel` (hardware) or `apply_reflectors_kernel_sim`
  (simulator), handling the CPU→XLA→CPU round-trip and result transpose.

### Changed

- `_householder_qr_eigh` in `trnsolver.eigen` now routes the eigenvector assembly
  step through `_call_apply_reflectors` (NKI) instead of the host-only
  `_apply_reflectors` loop.

## [0.11.0] — 2026-04-21

### Added

- **`matmul_kernel(A, V)` and `matmul_kernel_sim(A, V)`** in
  `trnsolver.nki.dispatch`. Tensor Engine matrix-matrix product C = A @ V via
  `nisa.nc_matmul` (hardware) and a Vector Engine per-column broadcast+sum loop
  (simulator). The hardware kernel uses A^T as the SBUF stationary operand and V
  as the moving operand (shape (n, k)), satisfying the NKI 0.3.0 free-dim
  constraint (free_dim = k > 1) that blocked the earlier vector matvec attempt
  (#36). The simulator kernel (`matmul_kernel_sim`) handles `TRNSOLVER_USE_SIMULATOR=1`
  because `nki.simulate` strips the `nc_matmul` stationary operand and cannot
  route the hardware kernel. Closes #36.

- **`_call_matmul(A, V)`** dispatch helper in `trnsolver.eigen`. Routes to
  `matmul_kernel` (hardware) or `matmul_kernel_sim` (simulator), handling the
  full CPU→XLA→CPU device round-trip and `torch_xla.sync()` for NEFF cache
  reuse.

- **Rayleigh-Ritz refinement on NKI** (`trnsolver.eigen._householder_qr_eigh`).
  The three GEMMs in the post-tridiagonalization subspace rotation step
  (`A@V`, `V.T@AV`, `V@Q_corr`) now route through `_call_matmul` on the Tensor
  Engine instead of PyTorch on CPU. This is the largest previously-unaccelerated
  hot path in the NKI `eigh` call at n=128.

## [0.10.0] — 2026-04-18

### Added

- **`schur(A, tol=1e-10)`** in `trnsolver.factor`. Schur decomposition returns
  `(Q, T)` with `A = Q @ T @ Q.T`. T is quasi-upper triangular — upper triangular
  with possible 2×2 blocks on the diagonal for complex-conjugate eigenvalue pairs
  of real A; for symmetric A all diagonal blocks are 1×1 (T is the eigenvalue
  matrix). Implemented via (1) upper Hessenberg reduction with successive Householder
  reflectors accumulated into Q, then (2) Francis implicit double-shift QR with
  deflation until the active window is exhausted. Host-only (no NKI dispatch);
  BF16/FP16 inputs promoted to FP32 via the standard `_to_fp32`/`_restore` shim.
  This closes the last item in the Factorizations "Deferred" column of the API table.

## [0.9.0] — 2026-04-17

### Added

- **`svd(A, full_matrices=False)`** in `trnsolver.factor`. Singular Value
  Decomposition: returns `(U, s, Vh)` with `A = U @ diag(s) @ Vh`. Default
  `full_matrices=False` returns the economy/thin decomposition (U is m×k,
  Vh is k×n where k = min(m, n)), matching the convention used internally by
  `pinv`. BF16/FP16 inputs are promoted to FP32 via the standard
  `_to_fp32`/`_restore` shim. This closes the `svd` item from the Factorizations
  row of the API table; the NKI-Jacobi SVD backend is a future Phase 3 perf
  upgrade tracked in #28.

## [0.8.0] — 2026-04-17

### Added

- **`ssor_preconditioner(A, omega=1.0)`** in `trnsolver.iterative`. SSOR
  (Symmetric Successive Over-Relaxation) preconditioner for symmetric positive
  definite A. Applies M^{-1} r via three steps: (1) forward triangular solve
  (D + ω L) t = r, (2) diagonal scaling v = ω(2-ω) diag(A) ⊙ t, (3) backward
  triangular solve (D + ω L^T) z = v. At ω = 1 this reduces to the symmetric
  Gauss-Seidel step. M_SSOR is SPD when A is SPD and ω ∈ (0, 2), so it is safe
  as a CG preconditioner. SSOR outperforms scalar Jacobi on matrices with
  off-diagonal coupling (1D Laplacian, FEM stiffness, density-fitting metrics).
  BF16/FP16 inputs are promoted to FP32 in both the factory and the returned
  closure (consistent with the dtype-promotion pattern in #19). Adds the final
  preconditioner item from the Phase 3 tracker (#28); SSOR benchmark added to
  `benchmarks/bench_solver.py`.

## [0.7.0] — 2026-04-17

### Added

- **BF16/FP16 dtype support across the public API** (#19). All entry points
  (`cholesky`, `lu`, `qr`, `solve`, `solve_spd`, `inv_spd`, `pinv`,
  `inv_sqrt_spd`, `inv_sqrt_spd_ns`, `eigh`, `eigh_generalized`, `cg`,
  `gmres`, `block_jacobi_preconditioner`) now accept `torch.bfloat16` and
  `torch.float16` inputs. A thin promotion shim (`_to_fp32` / `_restore` in
  `factor.py`) upcasts BF16/FP16 to FP32 at each public API boundary,
  runs the existing FP32 computation path unchanged, and downcasts the
  output back to the original dtype before returning. FP32/FP64 inputs
  pass through with zero overhead. `block_jacobi_preconditioner` upcasts
  during factory (block Cholesky in FP32), and the returned closure
  upcasts its residual argument and restores the output dtype.
  New test module `tests/test_dtype.py` exercises every covered entry
  point with both BF16 and FP16 via `pytest.mark.parametrize`. Closes #19.

## [0.6.0] — 2026-04-17

### Changed

- **`eigh` subspace rotation refinement** (#31). After the Householder-QR
  eigenvector assembly, one Rayleigh-Ritz step is now applied: compute
  H = V^T A V (reusing the already-calculated AV), call
  `torch.linalg.eigh(H)`, and rotate V by the resulting eigenvectors
  Q_corr. This reduces `||AV - V diag(w)||_F / (n||A||_F)` by 1–2 orders
  of magnitude for n ≥ 64 (from ~1e-4 to < 1e-5), and re-orthogonalises
  V implicitly via the eigh call. No API change; always-on.
  Supersedes the scalar Rayleigh-quotient pass added in v0.5.0 (same cost,
  strictly better result). Closes #31.

- **`solve_spd(A, B, iterative_refinement=False)`** gains an
  `iterative_refinement` keyword (#32). When `True`, the residual
  R = B − A X is computed in FP64 (mixed-precision), cast back to the
  input dtype, and a second Cholesky solve produces a correction dX.
  Reliably reduces the residual norm for SPD systems with cond(A) up to
  ~1/eps32 ≈ 1e7. Cost: one extra matvec + two triangular solves.
  Default is `False` (backward-compatible). Closes #32.

## [0.5.0] — 2026-04-17

### Added

- **`block_jacobi_preconditioner(A, block_size=16)`** in `trnsolver.iterative`
  (#16). Cholesky-factorizes each diagonal block of size `block_size`
  independently; applying the preconditioner solves the block-diagonal system
  exactly via `torch.cholesky_solve`. More effective than scalar Jacobi for
  matrices with localized coupling (FEM stiffness, density-fitting metric).
  `block_size=n` reduces to a full Cholesky preconditioner (exact solve).
  `block_size=1` is numerically equivalent to `jacobi_preconditioner`.
  Per-block Cholesky solves are independent and map naturally to
  NeuronCore-parallel execution. Closes #16 (block-Jacobi item; IC0
  degenerates to full Cholesky for dense matrices and is not added; SSOR
  deferred to v0.6.0).

- **`pinv(A, rcond=None)`** in `trnsolver.factor` (#22). Moore-Penrose
  pseudoinverse via truncated SVD: `U, s, Vh = svd(A, full_matrices=False)`;
  singular values ≤ `rcond * s[0]` are zeroed. Default `rcond = eps *
  max(m, n)` matches numpy/scipy. Handles real and complex inputs via `.mH`
  (conjugate transpose). Closes #22 (pinv item; schur decomposition requires a
  full implicit-shift QR implementation and is deferred to Phase 3).

### Changed

- **`inv_sqrt_spd_ns` uses `trnblas.gemm`** for the three O(n³) GEMMs in the
  Newton-Schulz loop (`Z@Y`, `Y@T`, `T@Z`, convergence `Y@Z`) when trnblas is
  installed (#14, #25). `trnblas.gemm` dispatches to the blocked NKI GEMM
  kernel on Trainium; falls back to `torch.matmul` otherwise. The O(n²)
  scalar combination `T = 1.5I − 0.5(Z@Y)` stays host-side (no systolic
  advantage). Closes #14.

- **CG and GMRES inner products promoted to FP64** (#27). `torch.dot(r, z)`
  calls in `cg()` (lines 76, 80, 94) and the Modified Gram-Schmidt inner
  products in `gmres()` now cast operands to `torch.float64` before the dot,
  returning a Python float. Working tensors remain FP32; only the scalar
  accumulations gain double precision. Equivalent to Kahan compensation for
  FP32 inputs without a per-element Python loop. Closes #27 (Kahan item).

- **Rayleigh-quotient eigenvalue refinement** in `_householder_qr_eigh` (#27).
  After assembling eigenvectors V from Householder reflectors + Givens
  rotations, one pass `w = (V * (A @ V)).sum(dim=0)` replaces the QR-iteration
  eigenvalue estimates with Rayleigh quotients. These are quadratically
  convergent from the eigenvector approximation and correct FP32 rounding
  drift that accumulates in the tridiagonal QR iteration at n ≥ 64.
  No API change; no extra allocation (element-wise multiply avoids n×n
  intermediate). Closes #27 (refinement item).

## [0.4.1] — 2026-04-17

### Changed

- **`eigh_generalized` triangular solves use `trnblas.trsm`** when trnblas is
  installed (#11). `trnblas.trsm` dispatches to a blocked NKI GEMM path on
  Trainium; falls back to `torch.linalg.solve_triangular` on CPU or when
  trnblas is absent. The inner `eigh` call already dispatches to NKI, so
  this closes the last remaining PyTorch-only step in the NKI generalized
  eigensolver path. Validated on trn1.2xlarge: `test_eigh_generalized` passes
  (n=16, rtol=1e-2) with trnblas installed. Closes #11.

- **`matvec_kernel` Tensor Engine attempt reverted** (#36). `nisa.nc_matmul`
  requires the `moving` operand's free dimension to meet a minimum tile width
  that a (n, 1) vector (free_dim=1) does not satisfy. The neuronxcc-2.24
  compiler and the NKI 0.3.0 simulator both reject the call with
  "nc_matmul() missing value for required argument 'moving'". Kernel reverts
  to the Vector-Engine broadcast+sum idiom validated in v0.4.0. The
  `matvec_kernel_sim` alias is retained for dispatch clarity.
  Tensor Engine matvec is deferred until the SDK documents the minimum
  free-dim constraint or the algorithm batches multiple vectors per call.
  Investigation recorded in #36.

- **Hardware re-validated** on trn1.2xlarge (NKI 0.3.0 / neuronxcc 2.24):
  14/14 `@pytest.mark.neuron` tests pass in 36 s wall-clock. Includes
  `test_eigh_generalized` with trnblas TRSM path. Closes #11. Closes #36
  (matvec investigation complete; Vector Engine path retained).

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
