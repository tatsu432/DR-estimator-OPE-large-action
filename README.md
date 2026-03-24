# DR-estimator-OPE-large-action

**Doubly robust OPE with large action spaces** — synthetic experiments extending [Open Bandit Pipeline (OBP) / zr-obp](https://github.com/st-tech/zr-obp).

Code for the paper [*Doubly Robust Estimator for Off-Policy Evaluation with Large Action Spaces*](https://arxiv.org/abs/2308.03443) (IEEE SSCI 2023) by Tatsuhiro Shimizu and Laura Forastiere.

Python · MIT License · Hydra · Docker · uv

## Documentation

| Resource | Description |
| -------- | ----------- |
| This README | Install, Hydra CLI, Docker / Compose, tests, CI |
| [Dockerfile](Dockerfile) | Reproducible image: `README.md` in build context, [`docker/cpu_only_sync.sh`](docker/cpu_only_sync.sh) then CPU PyTorch; `uv run --no-sync` entrypoint, non-root user |
| [compose.yaml](compose.yaml) | Compose service with host-mounted `outputs/` |
| [src/synthetic/hydra_conf/config.yaml](src/synthetic/hydra_conf/config.yaml) | App defaults (dataset, experiment, scale, Hydra `chdir`) |
| [src/synthetic/hydra_conf/experiment/](src/synthetic/hydra_conf/experiment/) | One YAML per sweep axis (beta, `n_actions`, epsilon, …) |
| [src/synthetic/hydra_conf/scale/](src/synthetic/hydra_conf/scale/) | Budget presets: `fastest`, `faster`, `bestest`, `slowest` |
| [tests/](tests/) | `pytest` (fast unit tests + optional `@pytest.mark.integration`) |
| [.github/workflows/lint.yml](.github/workflows/lint.yml) | Ruff, Mypy, fast pytest on push/PR |
| [.github/workflows/docker.yml](.github/workflows/docker.yml) | Build image and run a one-seed smoke job |

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

**Artifacts:** With `hydra.job.chdir=true`, each run stores `df/result_df.csv` under `outputs/<experiment>/<timestamp>/`. If `output.save_legacy_csv=true`, a copy is also written to `<repo_root>/<experiment.output_subdir>/df/result_df.csv` (e.g. `varying_beta_data/df/`).

## Docker

The image copies `README.md` (required by Hatchling), runs [`docker/cpu_only_sync.sh`](docker/cpu_only_sync.sh) (`uv sync --frozen --no-dev` with `--no-install-package` for `torch` and every CUDA-related sibling in the lockfile: `triton`, `cuda-*`, `nvidia-*`), then installs **CPU-only** `torch` (version pinned to match `uv.lock`) from the [PyTorch CPU wheel index](https://download.pytorch.org/whl/cpu). Sync must come first: a prior `uv pip install torch` would be removed when `uv sync` reconciles the environment. If you regenerate the lockfile and new CUDA package names appear, add them to that script. The default user is `appuser` (uid `10001`). Set `MPLBACKEND=Agg` for headless plots.

**Build and run (CLI):**

```bash
docker build -t dr-ope-synthetic .
docker run --rm -v "$(pwd)/outputs:/app/outputs" dr-ope-synthetic experiment=beta scale=fastest
```

**Faster smoke (e.g. CI):**

```bash
docker run --rm -v "$(pwd)/outputs:/app/outputs" dr-ope-synthetic experiment=beta scale=fastest n_seeds=1
```

**Compose:**

```bash
docker compose build
docker compose run --rm synthetic experiment=beta scale=fastest
```

Append Hydra overrides after the image / service name; they replace `CMD` but not `ENTRYPOINT`.

**Build fails with `No space left on device`:** Free Docker disk first: **Docker Desktop → Settings → Resources → increase Disk image size**, then `docker system prune -a` (drops unused images/layers). The install `RUN` uses a **BuildKit tmpfs mount on `/tmp`** (`size` ≈ 3 GiB RAM during that step) so PyTorch wheel extraction does not fill the overlay; the final image still needs room for `.venv` (~1–2 GiB). The image sets `UV_NO_CACHE=1` and omits `UV_LINK_MODE=copy`. If **`mkdir` on `/app` fails**, the disk image is full — prune or expand before rebuilding. After a failed build, **`docker compose run` may still use an older image**; rebuild successfully before relying on the new image.

## Experiment presets

| Group | Role |
| ----- | ---- |
| `experiment=beta` (default) | Sweeps `beta`; others: `n_actions`, `epsilon`, `n_rounds`, `n_cat_dim`, … |
| `scale=fastest` | Small `n_seeds`, `n_test`, `n_train` for quick runs |
| `scale=faster` / `bestest` / `slowest` | Larger budgets (see YAML) |

## OBP / zr-obp notes

- Uses PyPI `obp` for `OffPolicyEvaluation`, `RegressionModel`, IPS/DR/DM.
- `SyntheticBanditDatasetWithActionEmbeds` is **vendored** in [`src/synthetic/synthetic_bandit_with_action_embeds.py`](src/synthetic/synthetic_bandit_with_action_embeds.py) because the published `obp` wheel does not ship it.
- `embed_selection=true` is **not** supported on PyPI `obp` (no MIPS). It raises `NotImplementedError`; use full [zr-obp](https://github.com/st-tech/zr-obp) if you need that path.

## Tests and lint

```bash
uv sync --group dev
uv run pytest -m "not integration"    # fast (default in CI)
uv run pytest                          # includes integration (trains RF in `run_ope`)
uv run ruff check src tests
uv run mypy src/synthetic
```

## Abstract

We study Off-Policy Evaluation (OPE) in contextual bandit settings with large action spaces. Parametric estimators can be biased; importance-weighted ones can have high variance. [MIPS](https://arxiv.org/abs/2202.06317) reduces variance using action embeddings. We propose a **marginalized doubly robust (MDR)** estimator; theory and experiments support its behavior relative to IPS/MIPS/DM/DR.

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
- Similar packaging/docs patterns: [BDCM](https://github.com/tatsu432/BDCM) (Hydra, Docker, uv, CI)
