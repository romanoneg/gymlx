"""Base environment class for gymlx."""
import mlx.core as mx
from typing import NamedTuple, Generic, TypeVar, Any
from abc import ABC, abstractmethod

TEnvState = TypeVar("TEnvState", bound=NamedTuple)
TEnvParams = TypeVar("TEnvParams", bound=NamedTuple)


class Env(ABC, Generic[TEnvState, TEnvParams]):
    """Base class for gymlx environments.

    gymlx uses a functional API where state is explicitly passed and returned.
    This enables vmap and compile for massive parallelization on GPU.

    Key features:
    - State is explicit: step(state, action) -> (obs, reward, term, trunc, state, info)
    - Auto-reset: when done, state is automatically reset (controllable)
    - Optional keys: Pass MLX random keys for reproducibility, or omit to use global RNG

    Usage (with explicit key):
        env = gymlx.make("CartPole-v1")
        key = mx.random.key(0)
        obs, state, info = env.reset(key)
        key, subkey = mx.random.split(key)
        obs, reward, term, trunc, state, info = env.step(state, action, key=subkey)

    Usage (keyless - uses global RNG):
        env = gymlx.make("CartPole-v1")
        obs, state, info = env.reset()  # Uses global RNG
        obs, reward, term, trunc, state, info = env.step(state, action)  # Uses global RNG

    Vectorized (1000 parallel envs):
        vmap_reset = mx.compile(mx.vmap(env.reset))
        vmap_step = mx.compile(mx.vmap(env.step))
        keys = mx.random.split(mx.random.key(0), 1000)
        obs, states, info = vmap_reset(keys)

        # Pass keys as third positional argument for reproducibility:
        step_keys = mx.random.split(mx.random.key(1), 1000)
        obs, reward, term, trunc, states, info = vmap_step(states, actions, step_keys)
    """

    def __init__(self):
        return

    def step(self, state: TEnvState, action: int | float | mx.array,
             key: mx.array = None, params: TEnvParams | None = None,
             auto_reset: bool = True
             ) -> tuple[mx.array, mx.array, mx.array, mx.array, TEnvState, dict[Any, Any]]:
        """Execute one environment step.

        Args:
            state: Environment state
            action: Action to take
            key: Optional MLX random key. If None, uses global RNG.
            params: Optional environment parameters (uses default_params if None)
            auto_reset: If True (default), automatically reset done environments.

        Returns:
            obs: Observation array
            reward: Scalar reward
            terminated: Whether episode ended due to terminal state
            truncated: Whether episode ended due to time limit
            state: Next state (auto-reset if done and auto_reset=True)
            info: Additional information dict
        """
        params = self.default_params if params is None else params

        if auto_reset:
            # Split key only if provided; otherwise both use global RNG
            if key is not None:
                step_key, reset_key = mx.random.split(key)
            else:
                step_key, reset_key = None, None
            reward, terminated, truncated, step_state, info = self.step_env(state, action, step_key, params)
            done = terminated | truncated

            reset_state = self.reset_state(reset_key, params)

            # Select between step state and reset state
            state = type(reset_state)(*(
                mx.where(done, getattr(reset_state, f), getattr(step_state, f))
                for f in reset_state._fields
            ))
        else:
            # no need to split key since no reset is happening
            reward, terminated, truncated, state, info = self.step_env(state, action, key, params)

        obs = self.get_obs(state, params)
        return (obs, reward, terminated, truncated, state, info)

    def reset(self, key: mx.array = None, params: TEnvParams | None = None
              ) -> tuple[mx.array, TEnvState, dict[Any, Any]]:
        """Reset environment to initial state.

        Args:
            key: Optional MLX random key for initial state sampling.
                 If None, uses global RNG state.
            params: Optional environment parameters (uses default_params if None)

        Returns:
            obs: Initial observation
            state: Initial state
            info: Additional information dict (empty by default)
        """
        if params is None:
            params = self.default_params
        state = self.reset_state(key, params)
        obs = self.get_obs(state, params)
        return obs, state, {}

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
    def step_env(self, state: TEnvState, action: int | float | mx.array, key: mx.array, params: TEnvParams
                 ) -> tuple[mx.array, mx.array, mx.array, TEnvState, dict[Any, Any]]:
        """Internal step implementation (override in subclass).

        Returns: (reward, terminated, truncated, state, info) - observation is computed separately via get_obs.
        Note: key may be None when using global RNG.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_state(self, key: mx.array, params: TEnvParams) -> TEnvState:
        """Reset and return initial state only (no observation computation)."""
        raise NotImplementedError

    @abstractmethod
    def get_obs(self, state: TEnvState, params: TEnvParams) -> mx.array:
        """Compute observation from state. Called once after state selection."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_params(self) -> TEnvParams:
        """Default environment parameters."""
        raise NotImplementedError
