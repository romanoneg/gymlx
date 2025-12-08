import mlx.core as mx
import mlx.utils
from typing import NamedTuple, Generic, TypeVar, Any
from abc import ABC, abstractmethod 

# Type Variables for EnvStates and EnvParams, for each environment
TEnvState = TypeVar("TEnvState", bound=NamedTuple)
TEnvParams = TypeVar("TEnvParams", bound=NamedTuple)

class Env(ABC, Generic[TEnvState, TEnvParams]):

    def __init__(self):
        return

    def step(self, key: mx.array, state: TEnvState, action: int | float | mx.array, params: TEnvParams | None = None
             ) -> tuple[mx.array, mx.array, mx.array, mx.array, TEnvState, dict[Any, Any]]:

        params = self.default_params if params is None else params
        step_key, reset_key = mx.random.split(key)

        obs, reward, terminated, truncated, state, info = self.step_env(step_key, state, action, params)
        done = mx.logical_or(terminated, truncated)

        res_obs, res_state = self.reset(reset_key, params)

        obs = mx.where(done, res_obs, obs)
        state = mlx.utils.tree_map(
            lambda t, f: mx.where(done, t, f), 
            res_state, state
            )  
        
        return (obs, reward, terminated, truncated, state, info)

    def reset(self, key: mx.array, params: TEnvParams | None = None
              ) -> tuple[mx.array, TEnvState]:
        if params is None: 
            params = self.default_params
        obs, states = self.reset_env(key, params)
        return obs, states
    
    @property
    @abstractmethod
    def num_actions(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_space(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self):
        raise NotImplementedError

    @abstractmethod
    def step_env(self, key: mx.array, state: TEnvState, action: int | float | mx.array, params: TEnvParams
                 ) -> tuple[mx.array, mx.array, mx.array, mx.array, TEnvState, dict[Any, Any]]:
        raise NotImplementedError
    
    @abstractmethod
    def reset_env(self, key:mx.array, params: TEnvParams) -> tuple[mx.array, TEnvState]:
        raise NotImplemented
    
    @property
    @abstractmethod
    def default_params(self) -> TEnvParams:
        raise NotImplementedError

    # def render TODO


