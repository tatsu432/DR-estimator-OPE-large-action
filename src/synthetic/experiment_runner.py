"""Single entry path for all sweep experiments (DRY)."""

from __future__ import annotations

import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pandas import DataFrame
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from synthetic.ope import run_ope
from synthetic.plots import plot_line
from synthetic.policy import gen_eps_greedy
from synthetic.reward_function_registry import resolve_reward_function
from synthetic.synthetic_bandit_with_action_embeds import (
    SyntheticBanditDatasetWithActionEmbeds,
)

logger = getLogger(__name__)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _dataset_kwargs_from_cfg(cfg: DictConfig) -> dict[str, Any]:
    d_kw = OmegaConf.to_container(cfg.dataset, resolve=True)
    assert isinstance(d_kw, dict)
    rf_key = d_kw.pop("reward_function")
    assert isinstance(rf_key, str)
    d_kw["reward_function"] = resolve_reward_function(rf_key)
    return cast(dict[str, Any], d_kw)


def build_dataset_and_rounds(
    cfg: DictConfig, sweep_value: Any
) -> tuple[SyntheticBanditDatasetWithActionEmbeds, float, int]:
    """Return (dataset, evaluation-policy epsilon, validation log size)."""
    d_kw = _dataset_kwargs_from_cfg(cfg)
    policy_eps = float(cfg.policy.eps)
    n_val = int(cfg.n_train)

    mode = cfg.experiment.mode
    if mode == "dataset_field":
        field = str(cfg.experiment.field)
        d_kw[field] = sweep_value
    elif mode == "policy_eps":
        policy_eps = float(sweep_value)
    elif mode == "val_n_rounds":
        n_val = int(sweep_value)
    else:
        raise ValueError(f"Unknown experiment.mode: {mode}")

    dataset = SyntheticBanditDatasetWithActionEmbeds(**d_kw)
    return dataset, policy_eps, n_val


def summarize_estimates(
    estimated_policy_value_list: list[dict[str, Any]],
    policy_value: float,
    x_col: str,
    x_value: Any,
) -> DataFrame:
    result_df = (
        DataFrame(DataFrame(estimated_policy_value_list).stack())
        .reset_index(1)
        .rename(columns={"level_1": "est", 0: "value"})
    )
    result_df[x_col] = x_value
    result_df["se"] = (result_df.value - policy_value) ** 2
    result_df["bias"] = 0.0
    result_df["variance"] = 0.0
    sample_mean = DataFrame(result_df.groupby(["est"]).mean().value).reset_index()
    for est_ in sample_mean["est"]:
        estimates = result_df.loc[result_df["est"] == est_, "value"].values
        mean_estimates = sample_mean.loc[sample_mean["est"] == est_, "value"].values
        mean_estimates = np.ones_like(estimates) * mean_estimates
        result_df.loc[result_df["est"] == est_, "bias"] = (policy_value - mean_estimates) ** 2
        result_df.loc[result_df["est"] == est_, "variance"] = (estimates - mean_estimates) ** 2
    return result_df


def run_sweep_experiment(cfg: DictConfig) -> None:
    logger.info("cwd=%s", Path.cwd())
    start = time.time()

    sweep_values = list(cfg.experiment.sweep_values)
    x_col = str(cfg.experiment.result_column)
    xlabel = str(cfg.experiment.xlabel)
    xticklabels = sweep_values
    markersize = int(cfg.markersize)
    random_state = int(cfg.random_state)

    out_df = Path("df")
    out_df.mkdir(parents=True, exist_ok=True)

    result_parts: list[DataFrame] = []
    for sweep_value in sweep_values:
        dataset, policy_eps, n_val = build_dataset_and_rounds(cfg, sweep_value)

        test_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=int(cfg.n_test))
        action_dist_test = gen_eps_greedy(
            expected_reward=test_bandit_data["expected_reward"],
            is_optimal=bool(cfg.policy.is_optimal),
            eps=policy_eps,
        )
        policy_value = dataset.calc_ground_truth_policy_value(
            expected_reward=test_bandit_data["expected_reward"],
            action_dist=action_dist_test,
        )

        estimated_policy_value_list: list[dict[str, Any]] = []
        for seed_i in tqdm(
            range(int(cfg.n_seeds)),
            desc=f"{xlabel}: {sweep_value}",
        ):
            val_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=n_val)
            action_dist_val = gen_eps_greedy(
                expected_reward=val_bandit_data["expected_reward"],
                is_optimal=bool(cfg.policy.is_optimal),
                eps=policy_eps,
            )
            estimated_policy_values = run_ope(
                dataset=dataset,
                round=seed_i,
                val_bandit_data=val_bandit_data,
                action_dist_val=action_dist_val,
                embed_selection=bool(cfg.embed_selection),
                random_state=random_state,
            )
            estimated_policy_value_list.append(estimated_policy_values)

        result_parts.append(
            summarize_estimates(
                estimated_policy_value_list,
                float(policy_value),
                x_col,
                sweep_value,
            )
        )

    result_df = pd.concat(result_parts).reset_index(level=0)
    result_df.to_csv(out_df / "result_df.csv")

    if bool(cfg.output.save_legacy_csv):
        legacy = Path(get_original_cwd()) / cfg.experiment.output_subdir / "df"
        legacy.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(legacy / "result_df.csv")

    elapsed = (time.time() - start) / 60
    print(f"execution time: {elapsed} mins")

    for flag_share_y_scale in (True, False):
        for flag_log_scale in (False, True):
            plot_line(
                result_df=result_df,
                x=x_col,
                xlabel=xlabel,
                xticklabels=xticklabels,
                flag_log_scale=flag_log_scale,
                flag_share_y_scale=flag_share_y_scale,
                markersize=markersize,
            )
