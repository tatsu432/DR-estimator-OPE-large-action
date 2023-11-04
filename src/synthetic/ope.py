from typing import Dict
from typing import Optional

import numpy as np
from obp.ope import DirectMethod as DM
from obp.ope import DoublyRobust as DR
from obp.ope import InverseProbabilityWeighting as IPS
from obp.ope import MarginalizedInverseProbabilityWeighting as MIPS
from obp.ope import OffPolicyEvaluation

from obp.ope import RegressionModel
from sklearn.ensemble import RandomForestRegressor

from regression_model import RegressionModelMDR

def run_ope(
    dataset: Dict,
    round: int, 
    # observed data D
    val_bandit_data: Dict,
    # evaluation policy \pi(a|x)
    action_dist_val: np.ndarray,
    embed_selection: bool = False,
) -> np.ndarray:
    

    ## OPE using validation data
    
    # Machine learning model to estimate the reward function (:math:`q(x,a):= \mathbb{E}[r|x,a]`).
    reg_model = RegressionModel(
        # Number of actions.
        n_actions=dataset.n_actions,
        # Context vectors characterizing actions (i.e., a vector representation or an embedding of each action).
        action_context=val_bandit_data["action_context"],
        # A machine learning model used to estimate the reward function.
        base_model=RandomForestRegressor(
            n_estimators=10,
            max_samples=0.8,
            random_state=12345 + round
        ),
    )

    # Fit the regression model on given logged bandit data and estimate the expected rewards on the same data.
    # Returns
    #  q_hat: array-like, shape (n_rounds, n_actions, len_list)
    #  Expected rewards of new data estimated by the regression model.
    estimated_rewards = reg_model.fit_predict(
        context=val_bandit_data["context"],  # context; x
        action=val_bandit_data["action"],  # action; a
        reward=val_bandit_data["reward"],  # reward; r
        # Number of folds in the cross-fitting procedure.
        n_folds=2,
        random_state=12345 + round
    )

    if embed_selection is False:
        ope_estimators = [
            IPS(estimator_name="IPS"),
            DR(estimator_name="DR"),
            DM(estimator_name="DM"),
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

    # MIPS
    # P(e|x, a) transition kernel
    p_e_a = val_bandit_data["p_e_a"]
    # behavior policy \pi_b(a|x) and evaluation policy \pi_e(a|x)
    # change the shape of them so that we can do matrix multiplication later
    pi_b = np.zeros((len(val_bandit_data["pi_b"]), len(val_bandit_data["pi_b"][0])))
    action_dist = np.zeros((len(val_bandit_data["pi_b"]), len(val_bandit_data["pi_b"][0])))
    for i in range(len(val_bandit_data["pi_b"])):
        for j in range(len(val_bandit_data["pi_b"][0])):
            pi_b[i][j] = val_bandit_data["pi_b"][i][j][0]
            action_dist[i][j] = action_dist_val[i][j][0]
    # n
    n = val_bandit_data["n_rounds"]
    # e
    action_embed = val_bandit_data["action_embed"]
    # initialize P(e|x, pi_b) and P(e|x, pi_e) by 1
    p_e_pi_b = np.ones(n)
    p_e_pi_e = np.ones(n)
    # for each d in [0, 1, 2, \cdots, d_e - 1]
    for d in np.arange(p_e_a.shape[-1]):
        # P(e|x, pi_b)
        p_e_pi_b_d = pi_b[np.arange(n), :] @ p_e_a[:, :, d]
        p_e_pi_b *= p_e_pi_b_d[np.arange(n), action_embed[:, d]]
        # P(e|x, pi_e)
        p_e_pi_e_d = action_dist[np.arange(n), :] @ p_e_a[:, :, d]
        p_e_pi_e *= p_e_pi_e_d[np.arange(n), action_embed[:, d]]
    # marginalize importance weight w(x_i, e_i)
    w_x_e = p_e_pi_e / p_e_pi_b

    V_MIPS = np.mean(w_x_e * val_bandit_data["reward"])

    estimated_policy_values["MIPS"] = V_MIPS



    # MDR
    # Machine learning model to estimate the reward function (:math:`q(x,a):= \mathbb{E}[r|x,a]`).
    reg_model_mdr = RegressionModelMDR(
        # Number of actions.
        n_actions=dataset.n_actions,
        # Context vectors characterizing actions (i.e., a vector representation or an embedding of each action).
        action_context=val_bandit_data["action_context"],
        # A machine learning model used to estimate the reward function.
        base_model=RandomForestRegressor(
            n_estimators=10,
            max_samples=0.8,
            random_state=12345 + round
        ),
    )

    # Fit the regression model on given logged bandit data and estimate the expected rewards on the same data.
    # Returns
    #  q_hat: array-like, shape (n_rounds, n_actions, len_list)
    #  Expected rewards of new data estimated by the regression model.
    estimated_rewards_mdr = reg_model_mdr.fit_predict(
        context=val_bandit_data["context"],  # context; x
        action=val_bandit_data["action"],  # action; a
        embedding=val_bandit_data["action_embed"], 
        reward=val_bandit_data["reward"],  # reward; r
        # Number of folds in the cross-fitting procedure.
        n_folds=2,
        random_state=12345 + round
    )

    q_xi_ai_ei = estimated_rewards_mdr[np.arange(val_bandit_data["n_rounds"]), val_bandit_data["action"]]
        
    # MDR
    # This is the MDR that I propose in the paper
    V_MDR = estimated_policy_values["DM"] + V_MIPS - np.mean(w_x_e * q_xi_ai_ei)
            
    
    # add MDR to the dictionary of estimator and estimated value 
    # estimated_policy_values["MDR1"] = V_MDR1
    estimated_policy_values["MDR"] = V_MDR
    

    return estimated_policy_values
