"""MountainCar environment for gymlx.

Classic control task where a car must build momentum by oscillating
back and forth to reach the goal at the top of a hill.
"""
from ..env import Env
from ...spaces import Box, Discrete
import mlx.core as mx
from typing import NamedTuple, Any


class EnvState(NamedTuple):
    """MountainCar environment state."""
    position: mx.array
    velocity: mx.array
    time: mx.array


class EnvParams(NamedTuple):
    """MountainCar environment parameters."""
    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    goal_position: float = 0.5
    goal_velocity: float = 0.0
    force: float = 0.001
    gravity: float = 0.0025
    max_steps_in_episode: int = 200


class MountainCar(Env[EnvState, EnvParams]):
    """MountainCar-v0 environment.

    A car is placed in a valley and must reach the goal at the top of
    the right hill. The car's engine is too weak to climb directly,
    so it must build momentum by oscillating.

    Observation: [position, velocity]
    Actions: 0 (left), 1 (none), 2 (right)
    Reward: -1 per step until goal reached
    """

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @property
    def observation_space(self) -> Box:
        return Box(
            low=mx.array([-1.2, -0.07]),
            high=mx.array([0.6, 0.07]),
            shape=(2,),
            dtype=mx.float32
        )

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=3)

    @property
    def num_actions(self) -> int:
        return 3

    def is_terminal(self, state: EnvState, params: EnvParams) -> mx.array:
        """Check if car reached the goal."""
        return state.position >= params.goal_position

    def is_truncated(self, state: EnvState, params: EnvParams) -> mx.array:
        """Check if episode should be truncated (time limit)."""
        return state.time >= params.max_steps_in_episode

    def get_obs(self, state: EnvState, params: EnvParams) -> mx.array:
        """Compute observation from state."""
        return mx.array([state.position, state.velocity])

    def reset_state(self, key: mx.array, params: EnvParams) -> EnvState:
        """Reset and return initial state."""
        position = mx.random.uniform(low=-0.6, high=-0.4, shape=(), key=key)
        return EnvState(position, mx.array(0.0), mx.array(0))

    def step_env(self, state: EnvState, action: int | float | mx.array, key: mx.array, params: EnvParams
                 ) -> tuple[mx.array, mx.array, mx.array, EnvState, dict[Any, Any]]:
        # Check if stepping from an already-done state
        prev_done = self.is_terminal(state, params) | self.is_truncated(state, params)

        force = (action.astype(mx.float32) - 1) * params.force

        velocity = state.velocity + force - params.gravity * mx.cos(3 * state.position)
        velocity = mx.clip(velocity, -params.max_speed, params.max_speed)

        position = state.position + velocity
        position = mx.clip(position, params.min_position, params.max_position)

        velocity = mx.where((position == params.min_position) & (velocity < 0), 0.0, velocity)

        new_state = EnvState(position, velocity, state.time + 1)
        terminated = self.is_terminal(new_state, params)
        truncated = self.is_truncated(new_state, params)

        # Gymnasium: -1 every step. Zero if stepping from done state.
        reward = -1.0 * (1 - prev_done)
        return (reward, terminated, truncated, new_state, {})
