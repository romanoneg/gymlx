import mlx.core as mx
from typing import NamedTuple
from abc import ABC 

class BaseEnvParams(NamedTuple):
    maxTimesteps: int = 500

class Env(ABC):

    def __init__(self):
        return

    def step(self):
        return

    def reset(self):
        return

    @property
    @abstractmethod
    def observation_space(self):
        pass

    @property
    @abstractmethod
    def action_space(self):
        pass

    

    # def render TODO


