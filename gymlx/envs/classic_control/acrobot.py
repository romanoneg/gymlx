"""Acrobot environment for gymlx.

Classic control task where a two-link robot arm must swing its tip
above a threshold height. Only the second joint is actuated.
"""
from ..env import Env
from ...spaces import Box, Discrete
import mlx.core as mx
from typing import NamedTuple, Any


class EnvState(NamedTuple):
    """Acrobot environment state."""
    theta1: mx.array
    theta2: mx.array
    theta1_dot: mx.array
    theta2_dot: mx.array
    time: mx.array


class EnvParams(NamedTuple):
    """Acrobot environment parameters."""
    dt: float = 0.2
    link_length_1: float = 1.0
    link_length_2: float = 1.0
    link_mass_1: float = 1.0
    link_mass_2: float = 1.0
    link_com_pos_1: float = 0.5
    link_com_pos_2: float = 0.5
    link_moi: float = 1.0
    max_vel_1: float = 4 * mx.pi
    max_vel_2: float = 9 * mx.pi
    max_steps_in_episode: int = 500


class Acrobot(Env[EnvState, EnvParams]):
    """Acrobot-v1 environment.

    A two-link pendulum with only the second joint actuated.
    Goal: Swing the tip above the threshold height (1.0 above the base).

    Observation: [cos(t1), sin(t1), cos(t2), sin(t2), t1_dot, t2_dot]
    Actions: 0 (negative torque), 1 (none), 2 (positive torque)
    Reward: -1 per step until goal reached (0 at termination)
    """

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @property
    def observation_space(self) -> Box:
        high = mx.array([1.0, 1.0, 1.0, 1.0, 4 * mx.pi, 9 * mx.pi])
        return Box(low=-high, high=high, shape=(6,), dtype=mx.float32)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=3)

    @property
    def num_actions(self) -> int:
        return 3

    def is_terminal(self, state: EnvState, params: EnvParams) -> mx.array:
        """Check if tip of acrobot is above threshold height."""
        return -mx.cos(state.theta1) - mx.cos(state.theta2 + state.theta1) > 1.0

    def is_truncated(self, state: EnvState, params: EnvParams) -> mx.array:
        """Check if episode should be truncated (time limit)."""
        return state.time >= params.max_steps_in_episode

    def get_obs(self, state: EnvState, params: EnvParams) -> mx.array:
        """Compute observation from state."""
        return mx.array([
            mx.cos(state.theta1), mx.sin(state.theta1),
            mx.cos(state.theta2), mx.sin(state.theta2),
            state.theta1_dot, state.theta2_dot
        ])

    def reset_state(self, key: mx.array, params: EnvParams) -> EnvState:
        """Reset and return initial state."""
        vals = mx.random.uniform(low=-0.1, high=0.1, shape=(4,), key=key)
        return EnvState(vals[0], vals[1], vals[2], vals[3], mx.array(0))

    def _dsdt(self, theta1, theta2, dtheta1, dtheta2, torque, params):
        """Compute derivatives for acrobot dynamics."""
        m1, m2 = params.link_mass_1, params.link_mass_2
        l1 = params.link_length_1
        lc1, lc2 = params.link_com_pos_1, params.link_com_pos_2
        I1 = I2 = params.link_moi
        g = 9.8

        c2 = mx.cos(theta2)
        s2 = mx.sin(theta2)

        d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2) + I1 + I2
        d2 = m2 * (lc2**2 + l1 * lc2 * c2) + I2

        phi2 = m2 * lc2 * g * mx.sin(theta1 + theta2)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * s2
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * s2
            + (m1 * lc1 + m2 * l1) * g * mx.sin(theta1)
            + phi2
        )

        ddtheta2 = (torque + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * s2 - phi2) / (
            m2 * lc2**2 + I2 - d2**2 / d1
        )
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        return ddtheta1, ddtheta2

    def step_env(self, state: EnvState, action: int | float | mx.array, key: mx.array, params: EnvParams
                 ) -> tuple[mx.array, mx.array, mx.array, EnvState, dict[Any, Any]]:
        # Check if stepping from an already-done state
        prev_done = self.is_terminal(state, params) | self.is_truncated(state, params)

        torque = (action - 1).astype(mx.float32)
        dt = params.dt

        t1, t2 = state.theta1, state.theta2
        dt1, dt2 = state.theta1_dot, state.theta2_dot

        # RK4 integration
        k1_dt1, k1_dt2 = self._dsdt(t1, t2, dt1, dt2, torque, params)

        k2_dt1, k2_dt2 = self._dsdt(
            t1 + dt/2 * dt1, t2 + dt/2 * dt2,
            dt1 + dt/2 * k1_dt1, dt2 + dt/2 * k1_dt2,
            torque, params
        )

        k3_dt1, k3_dt2 = self._dsdt(
            t1 + dt/2 * (dt1 + dt/2 * k1_dt1), t2 + dt/2 * (dt2 + dt/2 * k1_dt2),
            dt1 + dt/2 * k2_dt1, dt2 + dt/2 * k2_dt2,
            torque, params
        )

        k4_dt1, k4_dt2 = self._dsdt(
            t1 + dt * (dt1 + dt/2 * k2_dt1), t2 + dt * (dt2 + dt/2 * k2_dt2),
            dt1 + dt * k3_dt1, dt2 + dt * k3_dt2,
            torque, params
        )

        # RK4 combination
        new_t1 = t1 + dt * (dt1 + 2*(dt1 + dt/2*k1_dt1) + 2*(dt1 + dt/2*k2_dt1) + (dt1 + dt*k3_dt1)) / 6
        new_t2 = t2 + dt * (dt2 + 2*(dt2 + dt/2*k1_dt2) + 2*(dt2 + dt/2*k2_dt2) + (dt2 + dt*k3_dt2)) / 6
        new_dt1 = dt1 + dt * (k1_dt1 + 2*k2_dt1 + 2*k3_dt1 + k4_dt1) / 6
        new_dt2 = dt2 + dt * (k1_dt2 + 2*k2_dt2 + 2*k3_dt2 + k4_dt2) / 6

        # Wrap angles to [-pi, pi]
        new_t1 = ((new_t1 + mx.pi) % (2 * mx.pi)) - mx.pi
        new_t2 = ((new_t2 + mx.pi) % (2 * mx.pi)) - mx.pi

        # Clip velocities
        new_dt1 = mx.clip(new_dt1, -params.max_vel_1, params.max_vel_1)
        new_dt2 = mx.clip(new_dt2, -params.max_vel_2, params.max_vel_2)

        new_state = EnvState(new_t1, new_t2, new_dt1, new_dt2, state.time + 1)
        terminated = self.is_terminal(new_state, params)
        truncated = self.is_truncated(new_state, params)

        # Gymnasium: -1 per step, 0 at goal. Zero if stepping from done state.
        reward = -1.0 * (1 - (prev_done | terminated))
        return (reward, terminated, truncated, new_state, {})
