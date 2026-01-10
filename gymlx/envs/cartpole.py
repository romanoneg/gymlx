from .env import Env
from ..spaces import Box, Discrete
import mlx.core as mx
from typing import NamedTuple, Any


class EnvState(NamedTuple):
    x: mx.array
    v: mx.array
    theta: mx.array
    theta_dot: mx.array
    time: mx.array


class EnvParams(NamedTuple):
    g = -9.8
    mass_cart = 1
    mass_pole = 0.1
    length_pole = 0.5
    friction_cart_ground = 0.005
    friction_pole_cart = 0.000002
    force_magnitude = 10
    total_mass = 1.1
    polemass_length = 0.05
    tau = 0.02
    theta_threshold_radians = 12 * 2 * mx.pi / 360
    x_threshold = 2.4
    max_steps_in_episode = 500


class CartPole(Env[EnvState, EnvParams]):

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @property
    def observation_space(self) -> Box:
        high = mx.array([4.8, mx.inf, 0.418, mx.inf])
        return Box(low=-high, high=high, shape=(4,), dtype=mx.float32)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=2)

    @property
    def num_actions(self) -> int:
        return 2

    def is_terminal(self, state, params):
        done1 = mx.logical_or(
            state.x < -params.x_threshold,
            state.x > params.x_threshold,
        )
        done2 = mx.logical_or(
            state.theta < -params.theta_threshold_radians,
            state.theta > params.theta_threshold_radians,
        )
        return mx.logical_or(done1, done2)

    def is_truncated(self, state, params):
        return state.time >= params.max_steps_in_episode

    def step_env(self, key: mx.array, state: EnvState, action: int | float | mx.array, params: EnvParams
                 ) -> tuple[mx.array, mx.array, mx.array, mx.array, EnvState, dict[Any, Any]]:
        prev_terminal = mx.logical_or(self.is_terminal(state, params), self.is_truncated(state, params))
        force = params.force_magnitude * action - params.force_magnitude * (1 - action)
        costheta = mx.cos(state.theta)
        sintheta = mx.sin(state.theta)

        temp = (
            force + params.polemass_length * state.theta_dot ** 2 * sintheta
        ) / params.total_mass
        thetaacc = (params.g * sintheta - costheta * temp) / (
            params.length_pole
            * (4.0 / 3.0 - params.mass_pole * costheta ** 2 / params.total_mass)
        )
        xacc = (temp - params.polemass_length * thetaacc * costheta / params.total_mass)

        x = state.x + params.tau * state.v
        x_dot = state.v + params.tau * xacc
        theta = state.theta + params.tau * state.theta_dot
        theta_dot = state.theta_dot + params.tau * thetaacc

        reward = 1.0 - prev_terminal

        state = EnvState(x, x_dot, theta, theta_dot, state.time + 1)
        terminated = self.is_terminal(state, params)
        truncated = self.is_truncated(state, params)

        return (
            mx.array([x, x_dot, theta, theta_dot]),
            reward,
            terminated,
            truncated,
            state,
            {}
        )

    def reset_env(self, key: mx.array, params: EnvParams) -> tuple[mx.array, EnvState, dict[Any, Any]]:
        state = mx.random.uniform(low=-0.05, high=0.05, shape=(4,), key=key)
        return state, EnvState(*state, time=mx.array(0)), {}
