from .env import Env
from ..mxdataclass import mxDataclass
from mlx.utils import tree_flatten
import mlx.core as mx
from collections import namedtuple
from typing import NamedTuple

class _hiddenState(NamedTuple):
    x: mx.array = mx.array([0])
    v: mx.array = mx.array([0])
    theta: mx.array = mx.array([0])
    theta_dot: mx.array = mx.array([0])
    time: mx.array = mx.array([0])

class GameState(NamedTuple):
    observation: mx.array = mx.array([1])
    rewards: mx.array = mx.array([0])
    terminated: bool = mx.array([False])
    truncated: bool = mx.array([False])
    hiddenState: _hiddenState = _hiddenState()

class CartPole(Env):
    
    def __init__(self) -> Env:
        super().__init__()

        # environment constants from https://ieeexplore.ieee.org/document/6313077
        self.g = -9.8
        self.mass_cart = 1
        self.mass_pole = 0.1
        self.total_mass = self.mass_pole + self.mass_cart
        self.length_pole = 0.5
        self.polemass_length = self.mass_pole * self.length_pole
        self.friction_cart_ground = 0.005
        self.friction_pole_cart = 0.000002
        self.force_magnitude = 10

        # extra env variables
        self.tau = 0.02
        self.theta_threshold_radians = 12 * 2 * mx.pi / 360
        self.x_threshold = 2.4
        self.max_steps_in_episode = 500

    def is_terminal(self, state):
        done1 = mx.logical_or(
            state.x < -self.x_threshold,
            state.x > self.x_threshold,
        )
        done2 = mx.logical_or(
            state.theta < -self.theta_threshold_radians,
            state.theta > self.theta_threshold_radians,
        )
        done = mx.logical_or(done1, done2)
        return done

    def is_truncated(self, state):
        return state.time >= self.max_steps_in_episode

    def step(self, key: mx.array, game_state: GameState, action: int, ):

        state = game_state.hiddenState
        
        """Performs step transitions in the environment."""
        prev_terminal = mx.logical_or(self.is_terminal(state),self.is_truncated(state))
        force = self.force_magnitude * action - self.force_magnitude * (1 - action)
        costheta = mx.cos(state.theta)
        sintheta = mx.sin(state.theta)

        temp = (
            force + self.polemass_length * state.theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.g * sintheta - costheta * temp) / (
            self.length_pole
            * (4.0 / 3.0 - self.mass_pole * costheta ** 2 / self.total_mass)
        )
        xacc = (temp - self.polemass_length * thetaacc * costheta / self.total_mass)

        x = state.x + self.tau * state.v
        x_dot = state.v + self.tau * xacc
        theta = state.theta + self.tau * state.theta_dot
        theta_dot = state.theta_dot + self.tau * thetaacc

        # Important: Reward is based on termination is previous step transition
        reward = 1.0 - prev_terminal

        # Update state dict and evaluate termination conditions
        state = _hiddenState(x, x_dot, theta, theta_dot, state[time] + 1)
        terminated = self.is_terminal(state)
        truncated = self.is_truncated(state)

        return GameState(
            observation=mx.array([x, x_dot, theta, theta_dot]), #obs
            rewards=reward, #reward
            terminated=terminated, #terminated
            truncated=truncated, #truncated
            hiddenState=state # state
        )

    def reset():
            return

if __name__=='__main__':

    state = GameState()
    batch_state = GameState(observation=mx.array([1,1]),
                        rewards=mx.array([1,1]),
                        terminated=mx.array([False, False]),
                        truncated=mx.array([False, False]),
                        hiddenState=_hiddenState(
                            x=mx.array([0,0]),
                            v=mx.array([0,0]),
                            theta=mx.array([0,0]),
                            theta_dot=mx.array([0,0]),
                            time=mx.array([0,0])
                        )
                       )

    # def transform(batched_fn):
    #     def return_func(batch):
    #         t, hidden_t = type(batch), type(batch.hiddenState) 
    #         pp_batch = t(*batched_fn(batch))  # preprocessed batch 
    #         return pp_batch._replace(hiddenState=hidden_t(*pp_batch.hiddenState))
    #     return return_func
    
    batch_axes = GameState(
            observation=0,
            rewards=0,
            terminated=0,
            truncated=0,
            hiddenState=_hiddenState(
                x=0,
                v=0,
                theta=0,
                theta_dot=0,
                time=0
            )
        )
   
    env = CartPole()   
    step_transform = mx.vmap(env.step, in_axes=batch_axes, out_axes=0)
    # step_transform = transform(step)

    #--------------------------- testing ------------------------------------------

    print(f"{env.step(state)=}")
    print(f"{step_transform(batch_state)=}")

