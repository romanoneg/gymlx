"""gymlx - Hardware-accelerated gymnasium environments using MLX."""
import mlx.core as mx

from .envs import (
    Env,
    CartPole, CartPoleState, CartPoleParams,
    MountainCar, MountainCarState, MountainCarParams,
    MountainCarContinuous, MountainCarContinuousState, MountainCarContinuousParams,
    Pendulum, PendulumState, PendulumParams,
    Acrobot, AcrobotState, AcrobotParams,
)
from .spaces import Box, Discrete

_registry = {
    "CartPole-v1": CartPole,
    "MountainCar-v0": MountainCar,
    "Pendulum-v1": Pendulum,
    "Acrobot-v1": Acrobot,
    "MountainCarContinuous-v0": MountainCarContinuous,
}


def make(env_id: str) -> Env:
    """Create an environment by ID (gymnasium-like API)."""
    if env_id not in _registry:
        raise ValueError(f"Unknown environment: {env_id}. Available: {list(_registry.keys())}")
    return _registry[env_id]()


def make_key(seed: int) -> mx.array:
    """Create an MLX random key from a seed."""
    return mx.random.key(seed)


def split_key(key: mx.array, n: int = 2) -> mx.array:
    """Split an MLX random key into n keys."""
    return mx.random.split(key, n)
