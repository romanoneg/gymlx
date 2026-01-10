from .env import Env
from ..spaces import Box
import mlx.core as mx
from typing import NamedTuple, Any


class EnvState(NamedTuple):
    theta: mx.array
    theta_dot: mx.array
    time: mx.array


class EnvParams(NamedTuple):
    max_speed = 8.0
    max_torque = 2.0
    dt = 0.05
    g = 10.0
    m = 1.0
    l = 1.0
    max_steps_in_episode = 200


class Pendulum(Env[EnvState, EnvParams]):

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
        return ((x + mx.pi) % (2 * mx.pi)) - mx.pi

    def is_terminal(self, state, params):
        return mx.array(False)

    def is_truncated(self, state, params):
        return state.time >= params.max_steps_in_episode

    def step_env(self, key: mx.array, state: EnvState, action: int | float | mx.array, params: EnvParams
                 ) -> tuple[mx.array, mx.array, mx.array, mx.array, EnvState, dict[Any, Any]]:
        theta = state.theta
        theta_dot = state.theta_dot

        torque = mx.clip(action, -params.max_torque, params.max_torque)

        costs = self.angle_normalize(theta) ** 2 + 0.1 * theta_dot ** 2 + 0.001 * (torque ** 2)

        new_theta_dot = theta_dot + (
            -3 * params.g / (2 * params.l) * mx.sin(theta + mx.pi) + 3.0 / (params.m * params.l ** 2) * torque
        ) * params.dt

        new_theta_dot = mx.clip(new_theta_dot, -params.max_speed, params.max_speed)
        new_theta = theta + new_theta_dot * params.dt

        state = EnvState(new_theta, new_theta_dot, state.time + 1)

        terminated = self.is_terminal(state, params)
        truncated = self.is_truncated(state, params)

        reward = -costs.reshape(())

        obs = mx.array([mx.cos(new_theta), mx.sin(new_theta), new_theta_dot])

        return (obs, reward, terminated, truncated, state, {})

    def reset_env(self, key: mx.array, params: EnvParams) -> tuple[mx.array, EnvState, dict[Any, Any]]:
        key_theta, key_vel = mx.random.split(key)
        theta = mx.random.uniform(low=-mx.pi, high=mx.pi, shape=(), key=key_theta)
        theta_dot = mx.random.uniform(low=-1.0, high=1.0, shape=(), key=key_vel)

        state = EnvState(theta, theta_dot, mx.array(0))
        obs = mx.array([mx.cos(theta), mx.sin(theta), theta_dot])

        return obs, state, {}
