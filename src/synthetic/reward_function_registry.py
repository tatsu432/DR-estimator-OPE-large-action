"""Map config keys to OBP reward callables (YAML-friendly)."""

from collections.abc import Callable

from obp.dataset import linear_reward_function

REWARD_FUNCTIONS: dict[str, Callable] = {
    "linear": linear_reward_function,
}


def resolve_reward_function(key: str) -> Callable:
    if key not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward_function {key!r}; choose one of {sorted(REWARD_FUNCTIONS)}"
        )
    return REWARD_FUNCTIONS[key]
