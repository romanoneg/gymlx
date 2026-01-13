"""Pendulum environment for gymlx.

Classic control task where a pendulum starts hanging down and must be
swung up and balanced. Continuous action space for torque control.
"""
from ..env import Env
from ...spaces import Box
import mlx.core as mx
from typing import NamedTuple, Any


class EnvState(NamedTuple):
    """Pendulum environment state."""
    theta: mx.array
    theta_dot: mx.array
    time: mx.array


class EnvParams(NamedTuple):
    """Pendulum environment parameters."""
    max_speed: float = 8.0
    max_torque: float = 2.0
    dt: float = 0.05
    g: float = 10.0
    m: float = 1.0
    l: float = 1.0
    max_steps_in_episode: int = 200


class Pendulum(Env[EnvState, EnvParams]):
    """Pendulum-v1 environment.

    Swing up a pendulum and balance it upright using continuous torque.
    The pendulum starts at a random angle with random velocity.

    Observation: [cos(theta), sin(theta), theta_dot]
    Actions: Continuous torque in [-2, 2]
    Reward: -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2)
    """

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @property
    def observation_space(self) -> Box:
        return Box(
            low=mx.array([-1.0, -1.0, -8.0]),
            high=mx.array([1.0, 1.0, 8.0]),
            shape=(3,),
            dtype=mx.float32
        )

    @property
    def action_space(self) -> Box:
        return Box(low=mx.array([-2.0]), high=mx.array([2.0]), shape=(1,), dtype=mx.float32)

    @property
    def num_actions(self) -> int:
        return 1

    def angle_normalize(self, x):
        """Normalize angle to [-pi, pi]."""
        return ((x + mx.pi) % (2 * mx.pi)) - mx.pi

    def is_terminal(self, state: EnvState, params: EnvParams) -> mx.array:
        """Pendulum never terminates (only truncates)."""
        return mx.array(False)

    def is_truncated(self, state: EnvState, params: EnvParams) -> mx.array:
        """Check if episode should be truncated (time limit)."""
        return state.time >= params.max_steps_in_episode

    def get_obs(self, state: EnvState, params: EnvParams) -> mx.array:
        """Compute observation from state."""
        return mx.array([mx.cos(state.theta), mx.sin(state.theta), state.theta_dot])

    def reset_state(self, key: mx.array, params: EnvParams) -> EnvState:
        """Reset and return initial state."""
        if key is not None:
            key_theta, key_vel = mx.random.split(key)
        else:
            key_theta, key_vel = None, None
        theta = mx.random.uniform(low=-mx.pi, high=mx.pi, shape=(), key=key_theta)
        theta_dot = mx.random.uniform(low=-1.0, high=1.0, shape=(), key=key_vel)
        return EnvState(theta, theta_dot, mx.array(0))

    def step_env(self, key: mx.array, state: EnvState, action: int | float | mx.array, params: EnvParams
                 ) -> tuple[mx.array, mx.array, mx.array, EnvState, dict[Any, Any]]:
        # Check if stepping from an already-done state
        prev_done = self.is_terminal(state, params) | self.is_truncated(state, params)

        theta = state.theta
        theta_dot = state.theta_dot

        torque = mx.clip(action, -params.max_torque, params.max_torque).squeeze()

        # Cost function
        theta_norm = self.angle_normalize(theta)
        costs = theta_norm ** 2 + 0.1 * theta_dot ** 2 + 0.001 * torque ** 2

        # Physics: pendulum dynamics
        gravity_term = -3 * params.g / (2 * params.l) * mx.sin(theta + mx.pi)
        torque_term = 3.0 / (params.m * params.l ** 2) * torque
        new_theta_dot = theta_dot + (gravity_term + torque_term) * params.dt
        new_theta_dot = mx.clip(new_theta_dot, -params.max_speed, params.max_speed)
        new_theta = theta + new_theta_dot * params.dt

        new_state = EnvState(new_theta, new_theta_dot, state.time + 1)
        terminated = self.is_terminal(new_state, params)
        truncated = self.is_truncated(new_state, params)

        # Gymnasium: cost-based reward. Zero if stepping from done state.
        reward = -costs.squeeze() * (1 - prev_done)
        return (reward, terminated, truncated, new_state, {})
