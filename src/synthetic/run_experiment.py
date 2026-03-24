"""Hydra CLI for synthetic OPE sweeps."""

import hydra
from omegaconf import DictConfig

from synthetic.experiment_runner import run_sweep_experiment


@hydra.main(version_base=None, config_path="hydra_conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_sweep_experiment(cfg)


if __name__ == "__main__":
    main()
