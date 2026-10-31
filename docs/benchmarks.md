# Benchmarks

Performance numbers for trnsolver across three axes:

1. **CPU baselines** — torch.linalg, scipy.linalg (LAPACK), and the trnsolver PyTorch path. Run on every CI build.
2. **GPU baselines** — `torch.linalg` on CUDA (cuSOLVER / cuBLAS) on a vintage-matched Nvidia instance.
3. **Trainium NKI** — the NKI path of trnsolver on trn1 / trn2. Pending hardware validation.

## Vintage matching

Comparing a 2022 Trainium chip to a 2024 H100 is not informative — both the arithmetic tier and the price gap distort the result. The benchmark table pairs each Trainium generation with the Nvidia GPU of closest architectural generation and approximate price tier:

| Trainium chip | Launch | Nvidia peer | EC2 instance | On-demand cost |
|---|---|---|---|---|
| trn1 (NeuronCore v2) | Oct 2022 | A10G (Ampere, 2021) | `g5.xlarge` | ~$1.01/hr |
| trn2 (NeuronCore v3) | Dec 2024 | H100 (Hopper, 2022) | `p5.4xlarge` | ~$12/hr |

The default GPU baseline is `g5.xlarge` (A10G) because it's the vintage peer of `trn1.2xlarge` and within the same price tier. H100 baselines are an opt-in follow-up on `p5.4xlarge` (1× H100) — roughly 10× the cost of the A10G box, but still far cheaper than `p5.48xlarge` (8× H100, ~$98/hr).

## Reproducing locally (CPU)

```bash
pytest benchmarks/bench_solver.py -v -m "not neuron and not cuda" --benchmark-only
```

## Reproducing on GPU (AWS)

```bash
cd infra/terraform
AWS_PROFILE=aws terraform apply -var=enable_gpu_ci=true \
  -var=vpc_id=vpc-... -var=subnet_id=subnet-...

# Wait for user-data to finish (~5 min), then stop; scripts will wake it.
AWS_PROFILE=aws ./scripts/run_cuda_tests.sh g5
```

## Reproducing on Trainium

```bash
AWS_PROFILE=aws ./scripts/run_neuron_tests.sh trn1
```

## CPU results (laptop reference — n=256)

Run on darwin/Apple-silicon CPU, Python 3.14, torch 2.x, scipy 1.17.1, float32 except where noted.

| Op | trnsolver | torch.linalg | scipy.linalg |
|---|---|---|---|
| `eigh` (symmetric) | 2,240 µs | 2,251 µs | 4,059 µs |
| `cholesky` | 69 µs | 68 µs | 100 µs |
| `solve_spd` | 86 µs | 143 µs (torch.solve) | 21 µs (cho_solve) |
| `qr` | 863 µs | — | — |
| `inv_sqrt_spd` (eig-based) | 2,320 µs | — | — |
| `inv_sqrt_spd_ns` (Newton-Schulz) | 4,782 µs | — | — |
| `cg` (no precond) | 161 µs | — | — |
| `gmres` | 2,549 µs | — | — |

Numbers are for illustration — on the benchmark-target Linux CI hardware scipy.linalg is generally faster than torch.linalg for small dense factorizations, and the trnsolver CPU path should sit between them (thin layer over `torch.linalg`). Reference values live in `results.json` artifacts from CI runs.

## GPU results — A10G / g5.xlarge (cuSOLVER via torch.linalg)

Run on AWS `g5.xlarge` (1× A10G, 24 GB, Ampere). Numbers include an explicit `torch.cuda.synchronize()` so the timer captures kernel execution, not async launch. Mean of 5+ warm rounds, µs:

| Op | n=64 | n=128 | n=256 | n=512 |
|---|---:|---:|---:|---:|
| `cholesky` | 95 | 110 | 166 | 277 |
| `qr` | 283 | 615 | 1,028 | 2,248 |
| `solve_spd` | 195 | 293 | 560 | 1,306 |
| `eigh` | 944 | 2,095 | 5,463 | 15,919 |
| `inv_sqrt_spd` (eig-based) | 1,108 | 2,298 | 6,191 | 17,709 |
| `inv_sqrt_spd_ns` (Newton-Schulz) | 2,979 | 3,005 | **2,973** | 3,492 |

**The headline result**: on GPU, `inv_sqrt_spd_ns` **beats the eigendecomposition-based `inv_sqrt_spd` by 2.1× at n=256 and 5.1× at n=512**. Same story — eigh dominates on CPU because LAPACK is heavily optimized for it, but when you're on an accelerator the all-GEMM shape of Newton-Schulz wins. This is the evidence that the NS path will pay off on Trainium once the trnblas NKI GEMM backend lands.

Reproduce: `AWS_PROFILE=aws ./scripts/run_cuda_tests.sh g5`.

## Trainium results (trn1 / trn2)

_Pending — classical Jacobi was architecturally mismatched to NKI (see [#9](https://github.com/trnsci/trnsolver/issues/9) post-mortem). Phase 1 redesign is iterating against the new NKI 0.3.0 CPU simulator (Neuron SDK 2.29, April 2026) with full-sweep Jacobi / Householder-QR / block-Jacobi candidates; hardware numbers will land once the redesigned kernel validates in simulation._

## DF-RHF SCF end to end (Trn2, #15)

`scripts/bench_scf.py` on trn2.3xlarge, SDK 2.33 native TorchNeuron, LNC=2,
one logical NeuronCore, host threads pinned to one physical core
(`OMP_NUM_THREADS=2 taskset -c 2,8`). PySCF cc-pVDZ / cc-pvdz-jkfit
integrals (`h2-lattice:16` is the synthetic s-Gaussian system). fp32 on
CPU and device; the reference is PySCF's float64 DF-RHF energy on the same
integrals, which the float64 loop (`cpu64`) reproduces to 1e-11 Eh. Warm
times; the first run of each shape pays a one-off compile (6-10 s at water,
up to 455 s at n_ao = 970). The CPU runs keep the unpacked integrals
resident (at n_ao = 970, `--k-cache-gb 24`); one logical core's 24 GB holds
them only without the packed copy (`--jk-square`).

| system | n_ao / naux / nocc | CPU fp32: iters, total | device: iters, total | speedup |
|---|---|---|---|---|
| water | 24 / 116 / 5 | 10, 5.3 ms | 10, 60.2 ms | 0.09x |
| benzene | 114 / 558 / 21 | 11, 113 ms | 10, 130 ms | 0.87x |
| h2-lattice:16 | 128 / 256 / 16 | 10, 93.0 ms | 8, 61.9 ms | 1.5x |
| C10H22 | 250 / 1206 / 41 | 16, 2.82 s | 16, 0.30 s | 9.5x |
| C20H42 | 490 / 2366 / 81 | 15, 28.0 s | 15, 0.74 s | 38x |
| C40H82 | 970 / 4686 / 161 | 22, 493 s | 22, 9.62 s (packed); 22, 6.36 s (`--jk-square`); 19, 4.06 s (`--jk-square`, J/K on 4 cores) | 51x; 78x; 121x |

Energy against the float64 reference: the fp32 total energy is off by
8e-6 to 6e-4 Eh on the device and up to 2e-2 Eh on the CPU (fp32
accumulation in the host BLAS J and K; at C10H22, 1.6e-3 Eh from J and
7e-4 from K), but re-evaluating it in float64 at the fp32 orbitals
(`E(C)`) gives 2e-10 to 5e-7 Eh up to n_ao = 490 and 5e-8 to 1e-7 Eh at 970.

Median per-iteration stage times on the device (ms):

| system | J/K | DIIS | eigh | other | largest |
|---|---|---|---|---|---|
| water | 1.49 | 2.01 | 2.18 | 1.17 | eigh (32%) |
| benzene | 3.84 | 3.22 | 5.48 | 3.05 | eigh (35%) |
| C10H22 | 7.93 | 3.11 | 8.94 | 3.18 | eigh (39%) |
| C20H42 | 27.7 | 5.02 | 16.3 | 1.86 | J/K (54%) |
| C40H82 (packed) | 361 | 6.82 | 65.6 | 3.95 | J/K (83%) |
| C40H82 (`--jk-square`) | 213 | 6.99 | 71.3 | 4.25 | J/K (72%) |
| C40H82 (`--jk-square`, J/K on 4 cores) | 139 | 6.81 | 63.4 | 4.25 | J/K (65%) |

On the CPU, J/K is 74-99% of every iteration from n_ao = 114 up.

J/K split over logical NeuronCores at C40H82 (whole chip, `neuron-jk<k>`
modes, one Python thread issuing to every core, inputs and partial
results staged through the host): per iteration 437 / 325 / 306 ms on
1 / 2 / 4 cores with packed integrals, 289 / 221 / 214 ms with
`--jk-square`; the J/K stage itself goes 213 -> 139 ms on 4 cores (1.5x).
Within one aux block of 532 at n_ao = 970, the gather that unpacks the
packed triangle takes 28 ms, the half-transform GEMM 5.7 ms, the permute
4.4 ms and the SYRK 5.8 ms; the two GEMMs run at about 28 TFLOP/s.

`eigh` here is one n x n solve per iteration. On a `neuron` tensor it is
`torch.linalg.eigh`, which TorchNeuron computes on the host: 1.5 / 2.2 /
4.4 / 17 / 105 ms at n = 64 / 128 / 256 / 512 / 1024, against 0.15 / 0.58 /
2.5 / 12 / 91 ms for host LAPACK on the same two threads. The batched
Jacobi kernel and the process pool are for batches; for one n = 64 / 128
matrix the Jacobi kernel takes 8.6 / 12 ms. A GEMM-only replacement
(canonical purification, `--solver purify`) took 36 / 48 / 111 ms
compiled at n = 250 / 490 / 970.

## Notes

- `inv_sqrt_spd_ns` is Newton-Schulz — all GEMM. On CPU it loses to the eigendecomposition path because eigh is heavily optimized in LAPACK/MKL. On GPU it wins at n ≥ 256 because the workload shape fits the Tensor Core pipeline. Trainium's Tensor Engine should show a similar win once `trnblas` GEMM validates on hardware.
- cuSOLVER is invoked indirectly via `torch.linalg` on a CUDA tensor. We don't link cuSOLVER directly.
