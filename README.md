<h1 align="center"><b>DR-estimator-OPE-large-action</b><br>Doubly robust OPE with large action spaces</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.12-blue" alt="Python" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-526EAF.svg?logo=opensourceinitiative&logoColor=white" alt="License: MIT" /></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" /></a>
  <a href="https://arxiv.org/abs/2308.03443"><img src="https://img.shields.io/badge/paper-arxiv.2308.03443-B31B1B.svg" alt="arXiv" /></a>
  <a href="https://attend.ieee.org/ssci-2023/"><img src="https://img.shields.io/badge/IEEE-SSCI%202023-00629B.svg" alt="IEEE SSCI 2023" /></a>
</p>

Code for the paper [*Doubly Robust Estimator for Off-Policy Evaluation with Large Action Spaces*](https://arxiv.org/abs/2308.03443) ([IEEE SSCI 2023](https://attend.ieee.org/ssci-2023/)) by Tatsuhiro Shimizu and Laura Forastiere.

## Documentation

| Resource | Description |
| -------- | ----------- |
| [src/synthetic/hydra_conf/config.yaml](src/synthetic/hydra_conf/config.yaml) | App defaults (dataset, experiment, scale, Hydra `chdir`) |
| [src/synthetic/hydra_conf/experiment/](src/synthetic/hydra_conf/experiment/) | One YAML per sweep axis (beta, `n_actions`, epsilon, …) |
| [src/synthetic/hydra_conf/scale/](src/synthetic/hydra_conf/scale/) | Budget presets: `fastest`, `faster`, `bestest`, `slowest` |
| [tests/](tests/) | `pytest` (fast unit tests + optional `@pytest.mark.integration`) |


## Installation and quick start

```bash
git clone <this-repo-url>
cd DR-estimator-OPE-large-action
# uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Run a small sweep (defaults: `experiment=beta`, `scale=fastest`):

```bash
uv run python -m synthetic.run_experiment
```

Hydra writes under `outputs/` (gitignored). Override on the CLI, for example:

```bash
uv run python -m synthetic.run_experiment experiment=n_actions scale=faster
uv run python -m synthetic.run_experiment experiment=beta n_seeds=5 dataset.n_actions=500
```

**Note:** Sweep lists live in `experiment/*.yaml` as `sweep_values` (not `values`, which clashes with OmegaConf).

## Docker

**Build and run (CLI):**

```bash
docker build -t dr-ope-synthetic .
docker run --rm -v "$(pwd)/outputs:/app/outputs" dr-ope-synthetic experiment=beta scale=fastest
```

**Faster smoke:**

```bash
docker run --rm -v "$(pwd)/outputs:/app/outputs" dr-ope-synthetic experiment=beta scale=fastest n_seeds=1
```

**Compose:**

```bash
docker compose build
docker compose run --rm synthetic experiment=beta scale=fastest
```


## Experiment presets

| Group | Role |
| ----- | ---- |
| `experiment=beta` (default) | Sweeps `beta`; others: `n_actions`, `epsilon`, `n_rounds`, `n_cat_dim`, … |
| `scale=fastest` | Small `n_seeds`, `n_test`, `n_train` for quick runs |
| `scale=faster` / `bestest` / `slowest` | Larger budgets (see YAML) |


## Tests and lint

```bash
uv sync --group dev
uv run pytest -m "not integration"    # fast (default in CI)
uv run pytest                          # includes integration (trains RF in `run_ope`)
uv run ruff check src tests
uv run mypy src/synthetic
```

## Citation

```
@article{shimizu2023doubly,
  title={Doubly Robust Estimator for Off-Policy Evaluation with Large Action Spaces},
  author={Shimizu, Tatsuhiro and Forastiere, Laura},
  journal={arXiv preprint arXiv:2308.03443},
  year={2023}
}
```

## Related

- [Open Bandit Pipeline (zr-obp)](https://github.com/st-tech/zr-obp)