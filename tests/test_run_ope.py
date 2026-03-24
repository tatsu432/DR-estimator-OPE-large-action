import numpy as np
import pytest
from obp.dataset.synthetic import linear_reward_function

from synthetic.ope import run_ope
from synthetic.policy import gen_eps_greedy
from synthetic.synthetic_bandit_with_action_embeds import (
    SyntheticBanditDatasetWithActionEmbeds,
)


@pytest.mark.integration
def test_run_ope_produces_estimator_keys() -> None:
    dataset = SyntheticBanditDatasetWithActionEmbeds(
        n_actions=25,
        dim_context=5,
        beta=-1.0,
        reward_type="continuous",
        reward_function=linear_reward_function,
        reward_std=1.5,
        random_state=7,
    )
    val = dataset.obtain_batch_bandit_feedback(n_rounds=40)
    action_dist = gen_eps_greedy(
        expected_reward=val["expected_reward"],
        is_optimal=True,
        eps=0.05,
    )
    out = run_ope(
        dataset,
        0,
        val,
        action_dist,
        embed_selection=False,
        random_state=7,
    )
    for name in ("IPS", "DR", "DM", "MIPS", "MDR"):
        assert name in out
        v = float(np.asarray(out[name]).item())
        assert np.isfinite(v)
