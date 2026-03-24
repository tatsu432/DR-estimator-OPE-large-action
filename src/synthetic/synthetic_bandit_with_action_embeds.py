# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.
#
# Vendored from https://github.com/st-tech/zr-obp (obp/dataset/synthetic_embed.py) and adapted
# to PyPI ``obp``: the published wheel omits this dataset and ``sample_action_fast``. Parent
# ``SyntheticBanditDataset`` init is inlined so we do not depend on the upstream subclass API.

"""Synthetic contextual bandit data with discrete action embeddings (large-action OPE)."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from obp.dataset.base import BaseBanditDataset
from obp.dataset.reward_type import RewardType
from obp.dataset.synthetic import (
    linear_reward_function,
    logistic_reward_function,
)
from obp.types import BanditFeedback
from obp.utils import softmax
from sklearn.utils import check_random_state, check_scalar


def _sample_action_fast(
    action_dist: np.ndarray, random_state: int | None = None
) -> np.ndarray:
    """Row-wise categorical sampling (zr-obp ``sample_action_fast``)."""
    random_ = check_random_state(random_state)
    uniform_rvs = random_.uniform(size=action_dist.shape[0])[:, np.newaxis]
    cum_action_dist = action_dist.cumsum(axis=1)
    flg = cum_action_dist > uniform_rvs
    return np.asarray(flg.argmax(axis=1), dtype=np.int64)


@dataclass
class SyntheticBanditDatasetWithActionEmbeds(BaseBanditDataset):
    """Synthesize bandit data with action/item category embeddings (OBP / zr-obp semantics)."""

    n_actions: int
    dim_context: int = 1
    reward_type: str = RewardType.BINARY.value
    reward_function: Callable[..., np.ndarray] | None = None
    reward_std: int | float = 1.0
    behavior_policy_function: Callable[..., np.ndarray] | None = None
    beta: int | float = 0.0
    n_cat_per_dim: int = 10
    latent_param_mat_dim: int = 5
    n_cat_dim: int = 3
    p_e_a_param_std: int | float = 1.0
    n_unobserved_cat_dim: int = 0
    n_irrelevant_cat_dim: int = 0
    n_deficient_actions: int = 0
    random_state: int = 12345
    dataset_name: str = "synthetic_bandit_dataset_with_action_embed"

    def __post_init__(self) -> None:
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.dim_context, "dim_context", int, min_val=1)
        check_scalar(self.beta, "beta", (int, float))
        check_scalar(
            self.n_deficient_actions,
            "n_deficient_actions",
            int,
            min_val=0,
            max_val=self.n_actions - 1,
        )
        if self.random_state is None:
            raise ValueError("`random_state` must be given")
        self.random_ = check_random_state(self.random_state)
        if RewardType(self.reward_type) not in [
            RewardType.BINARY,
            RewardType.CONTINUOUS,
        ]:
            raise ValueError(
                f"`reward_type` must be either '{RewardType.BINARY.value}' or "
                f"'{RewardType.CONTINUOUS.value}', but {self.reward_type} is given.'"
            )
        check_scalar(self.reward_std, "reward_std", (int, float), min_val=0)
        if self.reward_function is None:
            self.expected_reward = self.random_.uniform(size=self.n_actions)
        if RewardType(self.reward_type) == RewardType.CONTINUOUS:
            self.reward_min = 0
            self.reward_max = 1e10
        self.action_context = np.eye(self.n_actions, dtype=int)

        check_scalar(self.n_cat_per_dim, "n_cat_per_dim", int, min_val=1)
        check_scalar(self.latent_param_mat_dim, "latent_param_mat_dim", int, min_val=1)
        check_scalar(self.n_cat_dim, "n_cat_dim", int, min_val=1)
        check_scalar(self.p_e_a_param_std, "p_e_a_param_std", (int, float), min_val=0.0)
        check_scalar(
            self.n_unobserved_cat_dim,
            "n_unobserved_cat_dim",
            int,
            min_val=0,
            max_val=self.n_cat_dim,
        )
        check_scalar(
            self.n_irrelevant_cat_dim,
            "n_irrelevant_cat_dim",
            int,
            min_val=0,
            max_val=self.n_cat_dim,
        )
        self.n_cat_dim += 1
        self.n_unobserved_cat_dim += 1
        self.n_irrelevant_cat_dim += 1
        self._define_action_embed()

        if self.reward_function is None:
            if RewardType(self.reward_type) == RewardType.BINARY:
                self.reward_function = logistic_reward_function
            elif RewardType(self.reward_type) == RewardType.CONTINUOUS:
                self.reward_function = linear_reward_function

    def _define_action_embed(self) -> None:
        self.latent_cat_param = self.random_.normal(
            size=(self.n_cat_dim, self.n_cat_per_dim, self.latent_param_mat_dim)
        )
        self.p_e_a = softmax(
            self.random_.normal(
                scale=self.p_e_a_param_std,
                size=(self.n_actions, self.n_cat_per_dim, self.n_cat_dim),
            ),
        )
        self.action_context_reg = np.zeros((self.n_actions, self.n_cat_dim), dtype=int)
        for d in np.arange(self.n_cat_dim):
            self.action_context_reg[:, d] = _sample_action_fast(
                self.p_e_a[np.arange(self.n_actions, dtype=int), :, d],
                random_state=int(self.random_state + d),
            )

    @property
    def len_list(self) -> int:
        return 1

    def calc_ground_truth_policy_value(
        self, expected_reward: np.ndarray, action_dist: np.ndarray
    ) -> float:
        if not isinstance(expected_reward, np.ndarray):
            raise ValueError("expected_reward must be ndarray")
        if not isinstance(action_dist, np.ndarray):
            raise ValueError("action_dist must be ndarray")
        if action_dist.ndim != 3:
            raise ValueError(
                f"action_dist must be 3-dimensional, but is {action_dist.ndim}."
            )
        if expected_reward.shape[0] != action_dist.shape[0]:
            raise ValueError(
                "the size of axis 0 of expected_reward must be the same as that of action_dist"
            )
        if expected_reward.shape[1] != action_dist.shape[1]:
            raise ValueError(
                "the size of axis 1 of expected_reward must be the same as that of action_dist"
            )
        return float(
            np.average(expected_reward, weights=action_dist[:, :, 0], axis=1).mean()
        )

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> BanditFeedback:
        check_scalar(n_rounds, "n_rounds", int, min_val=1)
        contexts = self.random_.normal(size=(n_rounds, self.dim_context))
        cat_dim_importance = np.zeros(self.n_cat_dim)
        cat_dim_importance[self.n_irrelevant_cat_dim :] = self.random_.dirichlet(
            alpha=self.random_.uniform(size=self.n_cat_dim - self.n_irrelevant_cat_dim),
            size=1,
        )
        cat_dim_importance = cat_dim_importance.reshape((1, 1, self.n_cat_dim))

        q_x_e = np.zeros((n_rounds, self.n_cat_per_dim, self.n_cat_dim))
        q_x_a = np.zeros((n_rounds, self.n_actions, self.n_cat_dim))
        assert self.reward_function is not None
        for d in np.arange(self.n_cat_dim):
            q_x_e[:, :, d] = self.reward_function(
                context=contexts,
                action_context=self.latent_cat_param[d],
                random_state=self.random_state + d,
            )
            q_x_a[:, :, d] = q_x_e[:, :, d] @ self.p_e_a[:, :, d].T
        q_x_a = (q_x_a * cat_dim_importance).sum(2)

        if self.behavior_policy_function is None:
            pi_b_logits = q_x_a
        else:
            pi_b_logits = self.behavior_policy_function(
                context=contexts,
                action_context=self.action_context,
                random_state=self.random_state,
            )
        if self.n_deficient_actions > 0:
            pi_b = np.zeros_like(q_x_a)
            n_supported_actions = self.n_actions - self.n_deficient_actions
            supported_actions = np.argsort(
                self.random_.gumbel(size=(n_rounds, self.n_actions)), axis=1
            )[:, ::-1][:, :n_supported_actions]
            supported_actions_idx = (
                np.tile(np.arange(n_rounds), (n_supported_actions, 1)).T,
                supported_actions,
            )
            pi_b[supported_actions_idx] = softmax(
                self.beta * pi_b_logits[supported_actions_idx]
            )
        else:
            pi_b = softmax(self.beta * pi_b_logits)
        actions = _sample_action_fast(pi_b, random_state=int(self.random_state))

        action_embed = np.zeros((n_rounds, self.n_cat_dim), dtype=int)
        for d in np.arange(self.n_cat_dim):
            action_embed[:, d] = _sample_action_fast(
                self.p_e_a[actions, :, d],
                random_state=int(d),
            )

        expected_rewards_factual = np.zeros(n_rounds)
        for d in np.arange(self.n_cat_dim):
            expected_rewards_factual += (
                cat_dim_importance[0, 0, d]
                * q_x_e[np.arange(n_rounds), action_embed[:, d], d]
            )
        if RewardType(self.reward_type) == RewardType.BINARY:
            rewards = self.random_.binomial(n=1, p=expected_rewards_factual)
        elif RewardType(self.reward_type) == RewardType.CONTINUOUS:
            rewards = self.random_.normal(
                loc=expected_rewards_factual, scale=self.reward_std, size=n_rounds
            )
        else:
            raise NotImplementedError

        return dict(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            action_context=self.action_context_reg[:, self.n_unobserved_cat_dim :],
            action_embed=action_embed[:, self.n_unobserved_cat_dim :],
            context=contexts,
            action=actions,
            position=None,
            reward=rewards,
            expected_reward=q_x_a,
            q_x_e=q_x_e[:, :, self.n_unobserved_cat_dim :],
            p_e_a=self.p_e_a[:, :, self.n_unobserved_cat_dim :],
            pi_b=pi_b[:, :, np.newaxis],
            pscore=pi_b[np.arange(n_rounds), actions],
        )
