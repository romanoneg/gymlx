from .env import Env
from ..spaces import Box, Discrete
import mlx.core as mx
from typing import NamedTuple, Any


class EnvState(NamedTuple):
    theta1: mx.array
    theta2: mx.array
    theta1_dot: mx.array
    theta2_dot: mx.array
    time: mx.array


class EnvParams(NamedTuple):
    dt = 0.2
    link_length_1 = 1.0
    link_length_2 = 1.0
    link_mass_1 = 1.0
    link_mass_2 = 1.0
    link_com_pos_1 = 0.5
    link_com_pos_2 = 0.5
    link_moi = 1.0
    max_vel_1 = 4 * 3.141592653589793
    max_vel_2 = 9 * 3.141592653589793
    max_steps_in_episode = 500


class Acrobot(Env[EnvState, EnvParams]):

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @property
    def observation_space(self) -> Box:
        high = mx.array([1.0, 1.0, 1.0, 1.0, 12.566371, 28.274334])
        return Box(low=-high, high=high, shape=(6,), dtype=mx.float32)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=3)

    @property
    def num_actions(self) -> int:
        return 3

    def wrap(self, x, m, M):
        diff = M - m
        return (x - m) % diff + m

    def is_terminal(self, state, params):
        return -mx.cos(state.theta1) - mx.cos(state.theta2 + state.theta1) > 1.0

    def is_truncated(self, state, params):
        return state.time >= params.max_steps_in_episode

    def dsdt(self, state_augmented, params):
        m1 = params.link_mass_1
        m2 = params.link_mass_2
        l1 = params.link_length_1
        lc1 = params.link_com_pos_1
        lc2 = params.link_com_pos_2
        I1 = params.link_moi
        I2 = params.link_moi
        g = 9.8

        theta1 = state_augmented[0]
        theta2 = state_augmented[1]
        dtheta1 = state_augmented[2]
        dtheta2 = state_augmented[3]
        a = state_augmented[4]

        c2 = mx.cos(theta2)
        s2 = mx.sin(theta2)

        d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * c2) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * c2) + I2

        phi2 = m2 * lc2 * g * mx.cos(theta1 + theta2 - mx.pi / 2.0)
        phi1 = (
            - m2 * l1 * lc2 * dtheta2 ** 2 * s2
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * s2
            + (m1 * lc1 + m2 * l1) * g * mx.cos(theta1 - mx.pi / 2.0)
            + phi2
        )

        ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * s2 - phi2) / (
            m2 * lc2 ** 2 + I2 - d2 ** 2 / d1
        )
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        return mx.stack([dtheta1, dtheta2, ddtheta1, ddtheta2, mx.array(0.0)])

    def step_env(self, key: mx.array, state: EnvState, action: int | float | mx.array, params: EnvParams
                 ) -> tuple[mx.array, mx.array, mx.array, mx.array, EnvState, dict[Any, Any]]:
        torque = (action - 1).astype(mx.float32)

        y0 = mx.stack([state.theta1, state.theta2, state.theta1_dot, state.theta2_dot, torque])

        k1 = self.dsdt(y0, params)
        k2 = self.dsdt(y0 + params.dt * k1 / 2, params)
        k3 = self.dsdt(y0 + params.dt * k2 / 2, params)
        k4 = self.dsdt(y0 + params.dt * k3, params)

        yout = y0 + params.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        theta1 = self.wrap(yout[0], -mx.pi, mx.pi)
        theta2 = self.wrap(yout[1], -mx.pi, mx.pi)
        theta1_dot = mx.clip(yout[2], -params.max_vel_1, params.max_vel_1)
        theta2_dot = mx.clip(yout[3], -params.max_vel_2, params.max_vel_2)

        next_state = EnvState(theta1, theta2, theta1_dot, theta2_dot, state.time + 1)

        terminated = self.is_terminal(next_state, params)
        truncated = self.is_truncated(next_state, params)

        reward = mx.where(terminated, mx.array(0.0), mx.array(-1.0))

        obs = mx.array([
            mx.cos(theta1), mx.sin(theta1),
            mx.cos(theta2), mx.sin(theta2),
            theta1_dot, theta2_dot
        ])

        return (obs, reward, terminated, truncated, next_state, {})

    def reset_env(self, key: mx.array, params: EnvParams) -> tuple[mx.array, EnvState, dict[Any, Any]]:
        state_vec = mx.random.uniform(low=-0.1, high=0.1, shape=(4,), key=key)

        state = EnvState(state_vec[0], state_vec[1], state_vec[2], state_vec[3], mx.array(0))

        obs = mx.array([
            mx.cos(state.theta1), mx.sin(state.theta1),
            mx.cos(state.theta2), mx.sin(state.theta2),
            state.theta1_dot, state.theta2_dot
        ])

        return obs, state, {}
