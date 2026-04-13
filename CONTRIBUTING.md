# Contributing to trnsolver

Thanks for your interest in contributing. This document covers the whole suite — the same guidelines apply to the umbrella and to each of the six sub-projects (`trnfft`, `trnblas`, `trnrand`, `trnsolver`, `trnsparse`, `trntensor`).

## Where does my change belong?

- **Scoped to a single library** — bug fix, new feature, added kernel in one sub-project → open a PR in that sub-project's repo.
- **Cross-cutting** — integration examples, umbrella docs, CUDA-rosetta content, coordinated version bumps, shared tooling → open a PR against [trnsci/trnsci](https://github.com/trnsci/trnsci).

If you aren't sure, open the PR where it's easiest and we'll move it.

## Development setup

```bash
git clone git@github.com:trnsci/trnsci.git
cd trnsci
# Clone sibling sub-projects if you don't already have them:
for p in trnfft trnblas trnrand trnsolver trnsparse trntensor; do
  [ -d "$p" ] || git clone "git@github.com:trnsci/$p.git"
done
make install-dev
make test-all
```

A single sub-project in isolation:

```bash
cd trnsolver
uv pip install -e ".[dev]"          # or: pip install -e ".[dev]"
pytest tests/ -v -m "not neuron and not simulator"
```

`uv` is preferred — much faster than pip, same interface. `uv sync` installs from the committed `uv.lock` for a reproducible dev environment.

**Neuron packages (`neuronx-cc`, `torch-neuronx`) are NOT installed via pip.** They ship pre-installed in the Deep Learning AMI Neuron venv at `/opt/aws_neuronx_venv_pytorch_*/`. The `infra/terraform/main.tf` AMI filter + `most_recent=true` picks up whatever DLAMI is current; SDK upgrades are transparent.

## Lint and format

Ruff enforces style and sorts imports. Run locally:

```bash
ruff check .          # lint
ruff format .         # format
```

Pre-commit hooks run both automatically on `git commit`:

```bash
pip install pre-commit
pre-commit install
```

CI enforces the same checks (`.github/workflows/ci.yml::lint`), so skipping hooks locally just moves the failure to CI.

## Tests

Three test channels, in order of iteration cost:

- `pytest tests/ -v -m "not neuron and not simulator"` — pure CPU unit tests. Fast (< 1 s). Must pass on every PR.
- `pytest tests/ -v -m simulator` — NKI kernels run on CPU via `nki.simulate_kernel` (Neuron SDK 2.29+, NKI 0.3.0 Stable). No Trainium hardware needed. The simulator is part of the DLAMI's `aws_neuronx_venv`; when running locally on a Linux x86_64 box, a CI-style bootstrap works:
  ```bash
  uv pip install --system "neuronx-cc" --extra-index-url https://pip.repos.neuron.amazonaws.com
  ```
  Linux x86_64 only. GitHub Actions `test-simulator` job runs this automatically on every push. Use this as the **inner loop** for NKI kernel development — seconds per iteration instead of minutes on hardware.
- `pytest tests/ -v -m neuron` — real Trainium hardware. Burn only for final validation. Invoke via `AWS_PROFILE=aws ./scripts/run_neuron_tests.sh trn1` — the script wakes the pre-provisioned trn1 instance, runs the suite over SSM, prints results. Not part of standard CI.

## Conventions

- **License header** — new files get the Apache 2.0 notice and `Copyright 2026 Scott Friedman` where a notice is customary (long source files, not one-line configs).
- **Python** — ≥ 3.10; `torch` ≥ 2.1; `numpy` ≥ 1.24.
- **No emoji** in source, commits, or docs unless the feature is literally about emoji.
- **Docstrings** only where they help a reader who doesn't have the context you had when writing. Don't restate the signature.
- **No throwaway files** in the repo root — work-in-progress notes belong in PR descriptions or `docs/`.

## Commit style

Conventional-ish prefixes: `fix:`, `feat:`, `docs:`, `test:`, `bench:`, `chore:`, `refactor:`. Keep the subject line ≤ 72 characters. Longer context goes in the body.

## Pull requests

Before opening a PR:

- [ ] `pytest tests/ -v -m "not neuron"` passes
- [ ] If you changed public API or behaviour, `CHANGELOG.md` has an `[Unreleased]` entry
- [ ] If you added a public symbol, the project's `docs/api.md` lists it
- [ ] PR description says *what* changed and *why*, not just *what*

## Issues

Label each issue one of: `bug`, `feature`, `question`, `docs`, `infra`. Bugs include repro steps and expected-vs-actual output. Features describe the use case.

## Code of conduct

By participating, you agree to abide by the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). Report concerns to the contact listed there.

## License

Contributions are Apache 2.0, per the repository's [LICENSE](LICENSE). You attest you have the right to submit the code.
