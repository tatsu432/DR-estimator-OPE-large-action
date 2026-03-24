from typing import Any, cast

import numpy as np
from obp.ope import DirectMethod as DM  # type: ignore
from obp.ope import DoublyRobust as DR
from obp.ope import InverseProbabilityWeighting as IPS
from obp.ope import OffPolicyEvaluation, RegressionModel
from sklearn.ensemble import RandomForestRegressor  # type: ignore

from synthetic.regression_model_mdr import RegressionModelMDR


def _flatten_policy_probs(
    pi_b_3d: np.ndarray, action_dist_val: np.ndarray, n: int, n_actions: int
) -> tuple[np.ndarray, np.ndarray]:
    pi_b = np.zeros((n, n_actions))
    action_dist = np.zeros((n, n_actions))
    for i in range(n):
        for j in range(n_actions):
            pi_b[i][j] = pi_b_3d[i][j][0]
            action_dist[i][j] = action_dist_val[i][j][0]
    return pi_b, action_dist


def _marginal_embedding_weights(
    pi_b: np.ndarray,
    action_dist: np.ndarray,
    p_e_a: np.ndarray,
    action_embed: np.ndarray,
    n: int,
) -> np.ndarray:
    p_e_pi_b = np.ones(n)
    p_e_pi_e = np.ones(n)
    for d in np.arange(p_e_a.shape[-1]):
        p_e_pi_b_d = pi_b[np.arange(n), :] @ p_e_a[:, :, d]
        p_e_pi_b *= p_e_pi_b_d[np.arange(n), action_embed[:, d]]
        p_e_pi_e_d = action_dist[np.arange(n), :] @ p_e_a[:, :, d]
        p_e_pi_e *= p_e_pi_e_d[np.arange(n), action_embed[:, d]]
    return p_e_pi_e / p_e_pi_b


def run_ope(
    dataset: Any,
    round: int,
    val_bandit_data: dict[str, Any],
    action_dist_val: np.ndarray,
    embed_selection: bool = False,
    random_state: int = 12345,
) -> dict[str, Any]:
    if embed_selection:
        raise NotImplementedError(
            "embed_selection=True requires MarginalizedInverseProbabilityWeighting from "
            "full zr-obp (https://github.com/st-tech/zr-obp); PyPI 'obp' does not ship MIPS."
        )

    reg_model = RegressionModel(
        n_actions=dataset.n_actions,
        action_context=val_bandit_data["action_context"],
        base_model=RandomForestRegressor(
            n_estimators=10, max_samples=0.8, random_state=random_state + round
        ),
    )

    estimated_rewards = reg_model.fit_predict(
        context=val_bandit_data["context"],
        action=val_bandit_data["action"],
        reward=val_bandit_data["reward"],
        n_folds=2,
        random_state=random_state + round,
    )

    ope_estimators = [
        IPS(estimator_name="IPS"),
        DR(estimator_name="DR"),
        DM(estimator_name="DM"),
    ]

    ope = OffPolicyEvaluation(
        bandit_feedback=val_bandit_data,
        ope_estimators=ope_estimators,
    )
    estimated_policy_values = ope.estimate_policy_values(
        action_dist=action_dist_val,
        estimated_rewards_by_reg_model=estimated_rewards,
    )

    p_e_a = val_bandit_data["p_e_a"]
    n = val_bandit_data["n_rounds"]
    n_actions = len(val_bandit_data["pi_b"][0])
    pi_b, action_dist = _flatten_policy_probs(
        val_bandit_data["pi_b"], action_dist_val, n, n_actions
    )
    w_x_e = _marginal_embedding_weights(
        pi_b,
        action_dist,
        p_e_a,
        val_bandit_data["action_embed"],
        n,
    )

    V_MIPS = float(np.mean(w_x_e * val_bandit_data["reward"]))
    estimated_policy_values["MIPS"] = V_MIPS

    reg_model_mdr = RegressionModelMDR(
        n_actions=dataset.n_actions,
        action_context=val_bandit_data["action_context"],
        base_model=RandomForestRegressor(
            n_estimators=10, max_samples=0.8, random_state=random_state + round
        ),
    )

    estimated_rewards_mdr = reg_model_mdr.fit_predict(
        context=val_bandit_data["context"],
        action=val_bandit_data["action"],
        embedding=val_bandit_data["action_embed"],
        reward=val_bandit_data["reward"],
        n_folds=2,
        random_state=random_state + round,
    )

    q_xi_ai_ei = estimated_rewards_mdr[
        np.arange(val_bandit_data["n_rounds"]), val_bandit_data["action"]
    ]

    V_MDR = estimated_policy_values["DM"] + V_MIPS - np.mean(w_x_e * q_xi_ai_ei)
    estimated_policy_values["MDR"] = float(V_MDR)

    return cast(dict[str, Any], estimated_policy_values)
