import pytest

from synthetic.reward_function_registry import REWARD_FUNCTIONS, resolve_reward_function


def test_resolve_linear() -> None:
    fn = resolve_reward_function("linear")
    assert callable(fn)
    assert fn is REWARD_FUNCTIONS["linear"]


def test_resolve_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown reward_function"):
        resolve_reward_function("not_a_key")
