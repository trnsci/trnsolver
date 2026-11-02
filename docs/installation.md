# Installation

## Basic install

```bash
pip install trnsolver
```

## With Neuron hardware support

```bash
# Trn2/Trn3 with TorchNeuron (native PyTorch, Neuron SDK 2.33+)
pip install "trnsolver[torchneuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com

# Trn1 / PyTorch-XLA lineage
pip install "trnsolver[neuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

`[torchneuron]` pins `torch>=2.14`, `torch-neuronx>=2.14` and `nki>=0.6`
(torch-neuronx 2.14 requires Python ≥ 3.12). `[neuron]` is the XLA lineage.

### Two Neuron PyTorch lineages

`torch-neuronx` ships in two lineages, and trnsolver dispatches its NKI
kernels on whichever one is installed:

| Lineage | Instances | Stack | Kernel dispatch |
|---|---|---|---|
| TorchNeuron (native) | Trn2, Trn3 | Neuron SDK 2.33+, Python ≥ 3.12, `torch` 2.14, `torch-neuronx` 2.14.* | Each kernel is a PyTorch custom op (`torch.ops.trnsolver.matvec`, `rank2_update`, `matmul`, `matmul_fp32out`, `apply_reflectors`, `jacobi_sweeps`, `jacobi_block`) registered with `torch_neuronx.nki_op`; tensors live on `device="neuron"`. |
| PyTorch/XLA | Trn1 (also Inf2) | `torch-neuronx` 2.9 lineage, `torch_xla` | Unchanged: kernels run on the `torch_xla` device. |

Selection is automatic, on the first kernel launch:

1. **native**: `torch_neuronx` exports `nki_op` and a `neuron` device is
   available (`torch_neuronx.device_count() > 0`). This path never imports
   `torch_xla`.
2. **xla**: `torch_xla` is installed.
3. otherwise there is no device path; use `set_backend("pytorch")` or
   `TRNSOLVER_USE_SIMULATOR=1`.

The `trnsolver::` custom ops are registered only when the native path is
selected. To force a path, set `TRNSCI_NKI_DISPATCH` before the first launch
(shared by every trnsci library):

| Value | Effect |
|---|---|
| `auto` (default) | The order above. |
| `native` | Use TorchNeuron; raise `RuntimeError` if no `neuron` device is available. |
| `xla` | Use torch_xla even if TorchNeuron is present; raise `RuntimeError` if torch_xla is missing. |
| `cpu` | No device path. |

`TRNSCI_XLA_DEPRECATION=1` makes the XLA path emit one `DeprecationWarning`.
It is off by default.

### Editable installs and the NKI kernel cache

TorchNeuron enables the NKI trace cache by default
(`NKI_ENABLE_TRACE_CACHE=1`). The cache only versions source that belongs to
an installed wheel (a `site-packages` tree with `.dist-info`), so an editable
install can get a stale kernel after you edit `dispatch.py`. When developing
with `pip install -e`, turn it off before starting Python:

```bash
export NKI_ENABLE_TRACE_CACHE=0
```

The test suite keeps the cache on, as users run it, but gives each session
a fresh, empty cache directory (`NKI_TRACE_CACHE_URL`), so edits never hit
a stale entry.

The cache also ignores constants captured in a kernel's closure
([aws-neuron/aws-neuron-sdk#1454](https://github.com/aws-neuron/aws-neuron-sdk/issues/1454));
trnsolver's kernels take every specialising value as an argument
(`tests/test_trace_cache_safety.py`). Set `NKI_ENABLE_TRACE_CACHE=0` if
you write NKI kernels that a factory specialises through a closure.

See the TorchNeuron
[environment variables](https://github.com/aws-neuron/torch-neuronx/blob/main/docs/environment_variables.md)
reference.

### Deep Learning AMI (XLA lineage)

On a Trainium/Inferentia instance, install into the AMI's pre-built Neuron venv (which already contains `neuronxcc`, since it's not on public PyPI):

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
pip install trnsolver[dev]
```

The `trnsolver[neuron]` extra is only useful when building a custom Neuron environment from scratch.

## Development install

```bash
git clone https://github.com/trnsci/trnsolver.git
cd trnsolver
pip install -e ".[dev]"
pytest tests/ -v
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- NumPy >= 1.24
- Trainium hardware (optional): `[torchneuron]` — `torch>=2.14`, `torch-neuronx>=2.14`, `nki>=0.6`, Python ≥ 3.12; or `[neuron]` (XLA lineage) — `nki>=0.3.0`, `neuronxcc>=2.24`, `torch-neuronx>=2.9`

## Hardware compatibility

The NKI kernels, including the batched Jacobi `eigh` kernels, are validated on Trn2 (Neuron SDK 2.33, native TorchNeuron; trnsci/trnsci#35); the Householder-QR `eigh` path was validated on trn1.2xlarge in v0.4.0. Without NKI, or when a kernel path fails (unless `TRNSOLVER_REQUIRE_NKI=1`), every operation runs its PyTorch path. CPU/PyTorch correctness is preserved across all platforms.

For a pre-built CI instance, see [AWS Setup](aws_setup.md) and the Terraform module in `infra/terraform/`.
