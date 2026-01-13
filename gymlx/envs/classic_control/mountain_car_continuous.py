"""MountainCarContinuous environment for gymlx.

Continuous action variant of MountainCar where the agent controls
the force applied to the car directly.
"""
from ..env import Env
from ...spaces import Box
import mlx.core as mx
from typing import NamedTuple, Any


class EnvState(NamedTuple):
    """MountainCarContinuous environment state."""
    position: mx.array
    velocity: mx.array
    time: mx.array


class EnvParams(NamedTuple):
    """MountainCarContinuous environment parameters."""
    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    goal_position: float = 0.45
    goal_velocity: float = 0.0
    power: float = 0.0015
    gravity: float = 0.0025
    max_steps_in_episode: int = 999


class MountainCarContinuous(Env[EnvState, EnvParams]):
    """MountainCarContinuous-v0 environment.

    Continuous action variant of MountainCar. The agent applies force
    directly to the car to reach the goal at the top of the hill.

    Observation: [position, velocity]
    Actions: Continuous force in [-1, 1]
    Reward: +100 for reaching goal, -0.1*action^2 per step
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
    def action_space(self) -> Box:
        return Box(low=mx.array([-1.0]), high=mx.array([1.0]), shape=(1,), dtype=mx.float32)

    @property
    def num_actions(self) -> int:
        return 1

    def is_terminal(self, state: EnvState, params: EnvParams) -> mx.array:
        """Check if car reached the goal with sufficient velocity."""
        return (state.position >= params.goal_position) & (state.velocity >= params.goal_velocity)

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

    def step_env(self, key: mx.array, state: EnvState, action: int | float | mx.array, params: EnvParams
                 ) -> tuple[mx.array, mx.array, mx.array, EnvState, dict[Any, Any]]:
        # Check if stepping from an already-done state
        prev_done = self.is_terminal(state, params) | self.is_truncated(state, params)

        force = mx.clip(action, -1.0, 1.0).reshape(())

        velocity = state.velocity + force * params.power - params.gravity * mx.cos(3 * state.position)
        velocity = mx.clip(velocity, -params.max_speed, params.max_speed)

        position = state.position + velocity
        position = mx.clip(position, params.min_position, params.max_position)

        # Zero velocity at left boundary
        velocity = mx.where((position == params.min_position) & (velocity < 0), 0.0, velocity)

        new_state = EnvState(position, velocity, state.time + 1)
        terminated = self.is_terminal(new_state, params)
        truncated = self.is_truncated(new_state, params)

        # Gymnasium: -0.1*action² + 100 at goal. Zero if stepping from done state.
        reward = -0.1 * force * force
        reward = mx.where(terminated, reward + 100.0, reward)
        reward = reward * (1 - prev_done)

        return (reward, terminated, truncated, new_state, {})
