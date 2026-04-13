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

## GPU results (A10G / g5.xlarge)

_Pending — populated by running `AWS_PROFILE=aws ./scripts/run_cuda_tests.sh`._

## Trainium results (trn1 / trn2)

_Pending — populated by running `AWS_PROFILE=aws ./scripts/run_neuron_tests.sh`._

## Notes

- `inv_sqrt_spd_ns` is Newton-Schulz — all GEMM. On CPU it loses to the eigendecomposition path because eigh is heavily optimized in LAPACK/MKL. The shape of the workload is the important thing: once the trnblas NKI GEMM backend validates on Trainium, the NS path should win on trn1 / trn2.
- cuSOLVER is invoked indirectly via `torch.linalg` on a CUDA tensor. We don't link cuSOLVER directly.
