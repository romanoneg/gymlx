"""Template for creating custom gymlx environments.

Copy this file as a starting point for your own environment.
See gymlx/envs/classic_control/cartpole.py for a complete example.
"""
from gymlx.envs.env import Env
from gymlx.spaces import Box, Discrete
import mlx.core as mx
from typing import NamedTuple, Any


class EnvState(NamedTuple):
    """Mutable state that changes each step. Always include time for truncation."""
    # your_state_var: mx.array
    time: mx.array


class EnvParams(NamedTuple):
    """Immutable configuration. Always include max_steps_in_episode."""
    # your_param: float = 1.0
    max_steps_in_episode: int = 200


class CustomEnv(Env[EnvState, EnvParams]):
    """Rename this class to your environment name."""

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        """Return default EnvParams instance."""
        return EnvParams()

    @property
    def observation_space(self) -> Box:
        """Return Box or Discrete describing observation bounds/shape."""
        raise NotImplementedError

    @property
    def action_space(self) -> Discrete:
        """Return Discrete(n=...) or Box(...) for your action space."""
        raise NotImplementedError

    @property
    def num_actions(self) -> int:
        """Number of discrete actions, or action dimension for continuous."""
        raise NotImplementedError

    def is_terminal(self, state: EnvState, params: EnvParams) -> mx.array:
        """Return bool array: True if goal reached or failure condition."""
        raise NotImplementedError

    def is_truncated(self, state: EnvState, params: EnvParams) -> mx.array:
        """Return bool array: True if time limit reached."""
        return state.time >= params.max_steps_in_episode

    def get_obs(self, state: EnvState, params: EnvParams) -> mx.array:
        """Transform state into observation array."""
        raise NotImplementedError

    def reset_state(self, key: mx.array, params: EnvParams) -> EnvState:
        """Return initial EnvState. Use key for random initialization."""
        raise NotImplementedError

    def step_env(
        self,
        key: mx.array,
        state: EnvState,
        action: int | float | mx.array,
        params: EnvParams
    ) -> tuple[mx.array, mx.array, mx.array, EnvState, dict[Any, Any]]:
        """Core dynamics. Returns (reward, terminated, truncated, new_state, info).

        - Check prev_done to zero reward if stepping from done state
        - Increment state.time in new_state
        - Use mx ops only (no Python if/else on array values)
        """
        raise NotImplementedError
