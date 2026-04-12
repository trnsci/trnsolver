# Installation

## Basic install

```bash
pip install trnsolver
```

## With Neuron hardware support

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
- neuronxcc >= 2.15 (optional, for Trainium hardware)
- torch-neuronx >= 2.1 (optional, for Trainium hardware)

## Hardware compatibility

The NKI Jacobi rotation kernel is scaffolded but not yet validated on hardware — `trnsolver.eigh` falls back to `torch.linalg.eigh` whenever the NKI path isn't selected. CPU/PyTorch correctness is preserved across all platforms.

For a pre-built CI instance, see [AWS Setup](aws_setup.md) and the Terraform module in `infra/terraform/`.
