from .env import Env
import mlx.core as mx
from typing import NamedTuple, Any

class EnvState(NamedTuple):
    x: mx.array 
    v: mx.array
    theta: mx.array
    theta_dot: mx.array
    time: mx.array

class EnvParams(NamedTuple):
    # environment constants from https://ieeexplore.ieee.org/document/6313077
    g = -9.8
    mass_cart = 1
    mass_pole = 0.1
    length_pole = 0.5
    friction_cart_ground = 0.005
    friction_pole_cart = 0.000002
    force_magnitude = 10
    
    # Derived constants pre-calculated
    total_mass = 1.1 # mass_cart + mass_pole
    polemass_length = 0.05 # mass_pole * length_pole

    # extra env variables
    tau = 0.02
    theta_threshold_radians = 12 * 2 * mx.pi / 360
    x_threshold = 2.4
    max_steps_in_episode = 500

class CartPole(Env[EnvState, EnvParams]):
    
    def __init__(self):
        super().__init__()
    
    @property
    def default_params(self) -> EnvParams: # type: ignore
        return EnvParams()

    def is_terminal(self, state, params):
        done1 = mx.logical_or(
            state.x < -params.x_threshold,
            state.x > params.x_threshold,
        )
        done2 = mx.logical_or(
            state.theta < -params.theta_threshold_radians,
            state.theta > params.theta_threshold_radians,
        )
        done = mx.logical_or(done1, done2)
        return done

    def is_truncated(self, state, params):
        return state.time >= params.max_steps_in_episode

    def step_env(self, key: mx.array, state: EnvState, action: int | float | mx.array, params: EnvParams 
                 ) -> tuple[mx.array, mx.array, mx.array, mx.array, EnvState, dict[Any, Any]]:

        """Performs step transitions in the environment."""
        prev_terminal = mx.logical_or(self.is_terminal(state, params),self.is_truncated(state, params))
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

        # Important: Reward is based on termination is previous step transition
        reward = 1.0 - prev_terminal

        # Update state dict and evaluate termination conditions
        state = EnvState(x, x_dot, theta, theta_dot, state.time + 1)
        terminated = self.is_terminal(state, params)
        truncated = self.is_truncated(state, params)

        return (
            mx.array([x, x_dot, theta, theta_dot]), #obs
            reward, #reward
            terminated, #terminated
            truncated, #truncated
            state, # state
            {"discount": mx.array(1)} # TODO: placeholder
        )

    def reset_env(self, key: mx.array, params: EnvParams) -> tuple[mx.array, EnvState]:
        state = mx.random.uniform(low=-0.05, high=0.05, shape=(4,), key=key)
        return state, EnvState(*state, time=mx.array(0))
    
    def num_actions(self): # type: ignore
        return 2
    
    def observation_space(self): # type: ignore
        return None

    def action_space(self): # type: ignore
        return None

if __name__=='__main__':

    env = CartPole()   
    key = mx.random.key(42)
    res_key, step_key = mx.random.split(key, 2)
    obs, state = env.reset(res_key)
    obs, rewards, terminated, truncated, state, info = env.step(step_key, state, mx.array(0))

    vmap_key = mx.random.key(42)
    res_key, step_key = mx.random.split(key, 2)
    res_key = mx.random.split(res_key, 2000)
    step_key = mx.random.split(step_key, 2000)

    vmap_reset = mx.compile(mx.vmap(env.reset))
    vmap_step = mx.compile(mx.vmap(env.step))

    obs, state = vmap_reset(res_key)
    print(f"{obs=}")
    print(f"{state=}")
    import time

    with mx.stream(mx.gpu):
        start = time.perf_counter()
        CHUNK = 300 
        for _ in range(1_000_000 // CHUNK):
            for _ in range(CHUNK):
                obs, rewards, terminated, truncated, state, info = vmap_step(step_key, state, mx.zeros(2000))
            mx.eval(state)
    print(f"Final time per 1M: {time.perf_counter() - start}")
    # print(f"{state=}")
    # print(f"{obs=}")
    # print(f"{info=}")
