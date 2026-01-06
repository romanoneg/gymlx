from .env import Env
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
    def default_params(self) -> EnvParams: # type: ignore
        return EnvParams()

    def is_terminal(self, state, params):
        # Terminate if the car reaches the goal position and velocity condition
        # Standard Gym only checks position >= goal_position
        return state.position >= params.goal_position

    def is_truncated(self, state, params):
        return state.time >= params.max_steps_in_episode

    def step_env(self, key: mx.array, state: EnvState, action: int | float | mx.array, params: EnvParams 
                 ) -> tuple[mx.array, mx.array, mx.array, mx.array, EnvState, dict[Any, Any]]:

        """Performs step transitions in the environment."""
        prev_terminal = mx.logical_or(self.is_terminal(state, params), self.is_truncated(state, params))
        
        # Action is discrete: 0 (left), 1 (nothing), 2 (right)
        # Convert to -1, 0, 1
        force = (action - 1) * params.force
        
        velocity = state.velocity + force + mx.cos(3 * state.position) * (-params.gravity)
        velocity = mx.clip(velocity, -params.max_speed, params.max_speed)
        
        position = state.position + velocity
        position = mx.clip(position, params.min_position, params.max_position)
        
        # Reset velocity to 0 if we hit the left boundary
        velocity = mx.where(
            (position == params.min_position) & (velocity < 0), 
            mx.array(0.0), 
            velocity
        )

        # Reward is -1 per step, unless terminal (but Gym is always -1 until done)
        # However, checking CartPole implementation:
        # "Important: Reward is based on termination is previous step transition"
        # CartPole gives +1 for every step it stays alive.
        # MountainCar gives -1 for every step.
        reward = -1.0 
        
        # If previously terminated, we shouldn't accumulate more reward or change state meaningfully, 
        # but the base Env class handles reset logic via mx.where. 
        # However, CartPole calculates `reward = 1.0 - prev_terminal` to zero out reward if already done.
        # For MountainCar, if we are done, we probably shouldn't return -1?
        # Standard Gym returns -1 until termination. If terminated, the episode ends. 
        # In this vectorised auto-reset setting, if it was terminal last step, it has just been reset.
        # So we return -1 for the new step in the new episode.
        # BUT, if `prev_terminal` is true, it means `state` passed in was terminal.
        # The `Env.step` wraps this:
        # obs, reward, terminated, truncated, state, info = self.step_env(...)
        # done = ...
        # res_obs, res_state = self.reset(...)
        # obs = mx.where(done, res_obs, obs)
        # So `step_env` computes the NEXT state from the CURRENT state.
        
        # In CartPole: `reward = 1.0 - prev_terminal`. 
        # If input state was terminal, `prev_terminal` is 1, so reward is 0. 
        # This seems to imply "if the state passed in was already done, this transition doesn't yield reward".
        # But `Env.step` doesn't use `prev_terminal` to mask reward. 
        # It relies on the implementation.
        # Let's stick to -1.0. If `Env.step` resets, the reward returned here corresponds to the transition FROM the state passed in.
        # If the state passed in was terminal, technically we shouldn't be stepping it, but `Env` logic is:
        # Step -> Check Done -> If Done, Next State = Reset State.
        # So the reward returned is for the step that CAUSES termination or the step AFTER?
        # In CartPole `step_env`: `prev_terminal` checks if the *input* state was terminal.
        # If input state was terminal, reward is 0.
        # Let's verify CartPole logic. 
        # `prev_terminal = mx.logical_or(self.is_terminal(state, params),self.is_truncated(state, params))`
        # `reward = 1.0 - prev_terminal`
        # So if input state is bad, reward is 0.
        
        # For MountainCar, goal is to reach flag. Reward is -1 per step.
        # If input state is terminal (reached goal), maybe reward should be 0? 
        # But typically you get 0 reward upon reaching goal and -1 otherwise? 
        # No, MountainCar is -1 always. Terminating gives you the release from -1s.
        # Some implementations give 0 on the step you reach the goal?
        # Gymnax MountainCar: reward = -1.0.
        # Let's stick to -1.0.
        
        state = EnvState(position, velocity, state.time + 1)
        terminated = self.is_terminal(state, params)
        truncated = self.is_truncated(state, params)

        return (
            mx.array([position, velocity]), # obs
            mx.array(reward), # reward
            terminated, # terminated
            truncated, # truncated
            state, # state
            {"discount": mx.array(1)} 
        )

    def reset_env(self, key: mx.array, params: EnvParams) -> tuple[mx.array, EnvState]:
        # Random position in [-0.6, -0.4], zero velocity
        position = mx.random.uniform(low=-0.6, high=-0.4, shape=(), key=key)
        velocity = mx.array(0.0)
        state = EnvState(position, velocity, mx.array(0))
        return mx.array([position, velocity]), state
    
    def num_actions(self): # type: ignore
        return 3
    
    def observation_space(self): # type: ignore
        return None

    def action_space(self): # type: ignore
        return None
