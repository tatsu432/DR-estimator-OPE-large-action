from typing import Dict
from typing import Optional

import numpy as np
from obp.ope import DirectMethod as DM
from obp.ope import DoublyRobust as DR
from obp.ope import DoublyRobustWithShrinkageTuning as DRos
from obp.ope import InverseProbabilityWeighting as IPS
from obp.ope import MarginalizedInverseProbabilityWeighting as MIPS
from obp.ope import OffPolicyEvaluation
from obp.ope import SubGaussianDoublyRobustTuning as SGDR
from obp.ope import SwitchDoublyRobustTuning as SwitchDR


def run_ope(
    # observed data D
    val_bandit_data: Dict,
    # evaluation policy \pi(a|x)
    action_dist_val: np.ndarray,
    # \hat{q(x, a)}
    estimated_rewards: Optional[np.ndarray] = None,
    estimated_rewards_mrdr: Optional[np.ndarray] = None,
    embed_selection: bool = False,
) -> np.ndarray:

    if embed_selection is False:
        # lambdas for switch sorts of estimators
        # lambdas = [10, 50, 100, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, np.inf]
        # lambdas_sg = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0]
        
        ope_estimators = [
            IPS(estimator_name="IPS"),
            DR(estimator_name="DR"),
            DM(estimator_name="DM"),
            MIPS(
                n_actions=val_bandit_data["n_actions"],
                embedding_selection_method=None,
                estimator_name="MIPS",
            ),
        ]
    else:
        ope_estimators = [
            MIPS(
                n_actions=val_bandit_data["n_actions"],
                embedding_selection_method=None,
                estimator_name="MIPS (true)",
            ),
            MIPS(
                n_actions=val_bandit_data["n_actions"],
                embedding_selection_method="greedy",
                min_emb_dim=5,
                estimator_name="MIPS (slope)",
            ),
        ]

    ope = OffPolicyEvaluation(
        # observed data D
        bandit_feedback=val_bandit_data,
        # list of estimators
        ope_estimators=ope_estimators,
    )
    if embed_selection is False:
        estimated_policy_values = ope.estimate_policy_values(
            action_dist=action_dist_val,
            estimated_rewards_by_reg_model=estimated_rewards,
            action_embed=val_bandit_data["action_embed"],
            pi_b=val_bandit_data["pi_b"],
            p_e_a={"MIPS": val_bandit_data["p_e_a"]},
        )
    else:
        estimated_policy_values = ope.estimate_policy_values(
            action_dist=action_dist_val,
            estimated_rewards_by_reg_model=estimated_rewards,
            action_embed=val_bandit_data["action_embed"],
            pi_b=val_bandit_data["pi_b"],
            p_e_a={
                "MIPS (true)": val_bandit_data["p_e_a"],
                "MIPS (slope)": val_bandit_data["p_e_a"],
            },
        )
    

    return estimated_policy_values
