import numpy as np
from obp.dataset.synthetic import linear_reward_function

from synthetic.policy import gen_eps_greedy
from synthetic.synthetic_bandit_with_action_embeds import SyntheticBanditDatasetWithActionEmbeds


def test_obtain_batch_bandit_feedback_keys() -> None:
    dataset = SyntheticBanditDatasetWithActionEmbeds(
        n_actions=12,
        dim_context=4,
        beta=-1.0,
        reward_type="continuous",
        reward_function=linear_reward_function,
        reward_std=1.0,
        random_state=42,
    )
    fb = dataset.obtain_batch_bandit_feedback(n_rounds=25)
    assert fb["n_rounds"] == 25
    assert fb["n_actions"] == 12
    for key in (
        "context",
        "action",
        "reward",
        "expected_reward",
        "action_embed",
        "action_context",
        "p_e_a",
        "pi_b",
        "pscore",
    ):
        assert key in fb
    assert isinstance(fb["reward"], np.ndarray)
    assert fb["reward"].shape == (25,)


def test_calc_ground_truth_policy_value() -> None:
    dataset = SyntheticBanditDatasetWithActionEmbeds(
        n_actions=8,
        dim_context=3,
        beta=0.0,
        reward_type="continuous",
        reward_function=linear_reward_function,
        random_state=0,
    )
    fb = dataset.obtain_batch_bandit_feedback(n_rounds=40)
    pol = gen_eps_greedy(
        expected_reward=fb["expected_reward"], is_optimal=True, eps=0.1
    )
    v = dataset.calc_ground_truth_policy_value(
        expected_reward=fb["expected_reward"],
        action_dist=pol,
    )
    assert isinstance(v, float)
    assert np.isfinite(v)
