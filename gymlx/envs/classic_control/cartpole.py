"""CartPole environment for gymlx.

Classic control task where a pole is balanced on a cart by applying
left/right forces. The episode ends when the pole falls or the cart
moves out of bounds.
"""
from ..env import Env
from ...spaces import Box, Discrete
import mlx.core as mx
from typing import NamedTuple, Any


class EnvState(NamedTuple):
    """CartPole environment state."""
    x: mx.array
    v: mx.array
    theta: mx.array
    theta_dot: mx.array
    time: mx.array


class EnvParams(NamedTuple):
    """CartPole environment parameters."""
    g: float = -9.8
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    length_pole: float = 0.5
    friction_cart_ground: float = 0.005
    friction_pole_cart: float = 0.000002
    force_magnitude: float = 10.0
    total_mass: float = 1.1 # mass_cart + mass_pole
    polemass_length: float = 0.05 # mass_pole * length
    tau: float = 0.02
    theta_threshold_radians: float = 12 * 2 * mx.pi / 360
    x_threshold: float = 2.4
    max_steps_in_episode: int = 500


class CartPole(Env[EnvState, EnvParams]):
    """CartPole-v1 environment.

    A pole is attached to a cart moving along a frictionless track.
    Apply force of +1 or -1 to push the cart left or right.
    Goal: Keep the pole balanced upright.

    Observation: [x, x_dot, theta, theta_dot]
    Actions: 0 (left), 1 (right)
    Reward: +1 for each step the pole remains upright
    """

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @property
    def observation_space(self) -> Box:
        high = mx.array([4.8, mx.inf, 0.41887903, mx.inf])
        return Box(low=-high, high=high, shape=(4,), dtype=mx.float32)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=2)

    @property
    def num_actions(self) -> int:
        return 2

    def is_terminal(self, state: EnvState, params: EnvParams) -> mx.array:
        """Check if state is terminal (pole fell or cart out of bounds)."""
        return (mx.abs(state.x) > params.x_threshold) | (mx.abs(state.theta) > params.theta_threshold_radians)

    def is_truncated(self, state: EnvState, params: EnvParams) -> mx.array:
        """Check if episode should be truncated (time limit)."""
        return state.time >= params.max_steps_in_episode

    def get_obs(self, state: EnvState, params: EnvParams) -> mx.array:
        """Compute observation from state."""
        return mx.array([state.x, state.v, state.theta, state.theta_dot])

    def reset_state(self, key: mx.array, params: EnvParams) -> EnvState:
        """Reset and return initial state."""
        vals = mx.random.uniform(low=-0.05, high=0.05, shape=(4,), key=key)
        return EnvState(*vals, mx.array(0))

    def step_env(self, state: EnvState, action: int | float | mx.array, key: mx.array, params: EnvParams
                 ) -> tuple[mx.array, mx.array, mx.array, EnvState, dict[Any, Any]]:
        # Check if stepping from an already-done state
        prev_done = self.is_terminal(state, params) | self.is_truncated(state, params)

        force = params.force_magnitude * action - params.force_magnitude * (1 - action)
        costheta = mx.cos(state.theta)
        sintheta = mx.sin(state.theta)

        # Physics calculations
        temp = (force + params.polemass_length * state.theta_dot ** 2 * sintheta) / params.total_mass
        thetaacc = (params.g * sintheta - costheta * temp) / (
            params.length_pole * (4.0 / 3.0 - params.mass_pole * costheta ** 2 / params.total_mass)
        )
        xacc = temp - params.polemass_length * thetaacc * costheta / params.total_mass

        # Euler integration
        x = state.x + params.tau * state.v
        x_dot = state.v + params.tau * xacc
        theta = state.theta + params.tau * state.theta_dot
        theta_dot = state.theta_dot + params.tau * thetaacc

        new_state = EnvState(x, x_dot, theta, theta_dot, state.time + 1)
        terminated = self.is_terminal(new_state, params)
        truncated = self.is_truncated(new_state, params)

        # Gymnasium: +1 every step. Zero if stepping from done state.
        reward = 1 - prev_done
        return (reward, terminated, truncated, new_state, {})
