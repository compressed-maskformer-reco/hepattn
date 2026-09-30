# hepattn

End-to-end ML for particle physics reconstruction: a single encoder–decoder
transformer (MaskFormer-style) applied to many reconstruction tasks — hit
filtering, tracking, vertexing, particle flow, muon tracking — across different
detectors. See [README.md](README.md) for the paper list and task overview.

## Repository layout

- `src/hepattn/models/` — core building blocks: `encoder.py`, `decoder.py`,
  `maskformer.py`, `attention.py`, `task.py`, `matcher.py`, `loss.py`, etc.
- `src/hepattn/experiments/` — one subdirectory per experiment/detector
  (`clic`, `atlas_muon`, `trackml`, `tide`, `pixel`, `itk`, `cld`, `colliderml`).
  Each holds its own configs, data readers, and run instructions.
- `src/hepattn/callbacks/`, `src/hepattn/flex/`, `src/hepattn/utils/` —
  training callbacks, FlexAttention helpers, and shared utilities.
- `tests/` — pytest suite (see markers below).

## Environment & common commands

The project uses **pixi** (see `pyproject.toml` for environments: `default`,
`cpu`, `isambard`, `clic`, `tide`, `ci`).

- Enter the env: `pixi shell` (or `pixi shell -e <env>`)
- Run all tests: `pytest`
- CI-equivalent tests (no GPU, no external data): `pytest -m 'not gpu and not requiresdata'`
- Lint & format before committing: `ruff check --fix .` then `ruff format .`

## Terminology

- **constituent** — input entity fed to the encoder/decoder (e.g. a detector hit).
- **object** — reconstructed output from the decoder (e.g. a particle/track).
- **input** / **input_object** — generic term for any module input.
- **output** — generic term for any module output.

## Documentation & instruction files

- [README.md](README.md) — setup, environments, running tests, contributing.
- Per-experiment READMEs under `src/hepattn/experiments/*/` — how to run each experiment.
- **[CLIC particle-flow explainer](src/hepattn/experiments/clic/EXPLAINER.md)** —
  a detailed, worked walkthrough of the MaskFormer architecture, the four task
  heads, Hungarian matching (training-loss vs. evaluation-metric), the data file
  formats (`raw` vs `infer`), and the evaluation pipeline. The best starting
  point for understanding how the model actually works end to end.
