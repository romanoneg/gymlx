from .env import Env
from ..spaces import Box, Discrete
import mlx.core as mx
from typing import NamedTuple, Any


class EnvState(NamedTuple):
    position: mx.array
    velocity: mx.array
    time: mx.array


class EnvParams(NamedTuple):
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    goal_position = 0.5
    goal_velocity = 0.0
    force = 0.001
    gravity = 0.0025
    max_steps_in_episode = 200


class MountainCar(Env[EnvState, EnvParams]):

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

    def is_terminal(self, state, params):
        return state.position >= params.goal_position

    def is_truncated(self, state, params):
        return state.time >= params.max_steps_in_episode

    def step_env(self, key: mx.array, state: EnvState, action: int | float | mx.array, params: EnvParams
                 ) -> tuple[mx.array, mx.array, mx.array, mx.array, EnvState, dict[Any, Any]]:
        force = (action - 1) * params.force

        velocity = state.velocity + force + mx.cos(3 * state.position) * (-params.gravity)
        velocity = mx.clip(velocity, -params.max_speed, params.max_speed)

        position = state.position + velocity
        position = mx.clip(position, params.min_position, params.max_position)

        velocity = mx.where(
            (position == params.min_position) & (velocity < 0),
            mx.array(0.0),
            velocity
        )

        reward = mx.array(-1.0)

        state = EnvState(position, velocity, state.time + 1)
        terminated = self.is_terminal(state, params)
        truncated = self.is_truncated(state, params)

        return (
            mx.array([position, velocity]),
            reward,
            terminated,
            truncated,
            state,
            {}
        )

    def reset_env(self, key: mx.array, params: EnvParams) -> tuple[mx.array, EnvState, dict[Any, Any]]:
        position = mx.random.uniform(low=-0.6, high=-0.4, shape=(), key=key)
        velocity = mx.array(0.0)
        state = EnvState(position, velocity, mx.array(0))
        return mx.array([position, velocity]), state, {}
