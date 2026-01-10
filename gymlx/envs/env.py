"""Base environment class for gymlx."""
import mlx.core as mx
import mlx.utils
from typing import NamedTuple, Generic, TypeVar, Any
from abc import ABC, abstractmethod

TEnvState = TypeVar("TEnvState", bound=NamedTuple)
TEnvParams = TypeVar("TEnvParams", bound=NamedTuple)


class Env(ABC, Generic[TEnvState, TEnvParams]):
    """Base class for gymlx environments.

    gymlx uses a functional API where state is explicitly passed and returned.
    This enables vmap and compile for massive parallelization on GPU.

    Key differences from gymnasium:
    - State is explicit: step(key, state, action) -> (obs, reward, term, trunc, state, info)
    - Auto-reset: when done, state is automatically reset (no manual reset needed in loops)
    - Random keys: MLX random keys must be passed for reproducibility

    Usage:
        env = gymlx.make("CartPole-v1")
        key = mx.random.key(0)
        obs, state, info = env.reset(key)

        # Single step
        obs, reward, term, trunc, state, info = env.step(key, state, action)

        # Vectorized (1000 parallel envs)
        vmap_reset = mx.compile(mx.vmap(env.reset))
        vmap_step = mx.compile(mx.vmap(env.step))
        keys = mx.random.split(key, 1000)
        obs, state, info = vmap_reset(keys)
    """

    def __init__(self):
        return

    def step(self, key: mx.array, state: TEnvState, action: int | float | mx.array, params: TEnvParams | None = None
             ) -> tuple[mx.array, mx.array, mx.array, mx.array, TEnvState, dict[Any, Any]]:
        """Execute one environment step with auto-reset.

        Args:
            key: MLX random key for stochastic transitions
            state: Current environment state
            action: Action to take
            params: Optional environment parameters (uses default_params if None)

        Returns:
            obs: Observation array
            reward: Scalar reward
            terminated: Whether episode ended due to terminal state
            truncated: Whether episode ended due to time limit
            state: Next state (auto-reset if done)
            info: Additional information dict
        """
        params = self.default_params if params is None else params
        step_key, reset_key = mx.random.split(key)

        obs, reward, terminated, truncated, state, info = self.step_env(step_key, state, action, params)
        done = mx.logical_or(terminated, truncated)

        res_obs, res_state, _ = self.reset(reset_key, params)

        obs = mx.where(done, res_obs, obs)
        state = mlx.utils.tree_map(
            lambda t, f: mx.where(done, t, f),
            res_state, state
            )

        return (obs, reward, terminated, truncated, state, info)

    def reset(self, key: mx.array, params: TEnvParams | None = None
              ) -> tuple[mx.array, TEnvState, dict[Any, Any]]:
        """Reset environment to initial state.

        Args:
            key: MLX random key for initial state sampling
            params: Optional environment parameters (uses default_params if None)

        Returns:
            obs: Initial observation
            state: Initial state
            info: Additional information dict (empty by default)
        """
        if params is None:
            params = self.default_params
        obs, state, info = self.reset_env(key, params)
        return obs, state, info

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Number of discrete actions (for discrete action spaces)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_space(self):
        """Observation space specification (Box or Discrete)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self):
        """Action space specification (Box or Discrete)."""
        raise NotImplementedError

    @abstractmethod
    def step_env(self, key: mx.array, state: TEnvState, action: int | float | mx.array, params: TEnvParams
                 ) -> tuple[mx.array, mx.array, mx.array, mx.array, TEnvState, dict[Any, Any]]:
        """Internal step implementation (override in subclass)."""
        raise NotImplementedError

    @abstractmethod
    def reset_env(self, key: mx.array, params: TEnvParams) -> tuple[mx.array, TEnvState, dict[Any, Any]]:
        """Internal reset implementation (override in subclass)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_params(self) -> TEnvParams:
        """Default environment parameters."""
        raise NotImplementedError
