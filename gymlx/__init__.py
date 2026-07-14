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

__all__ = [
    # Base class
    "Env",
    # Spaces
    "Box", "Discrete",
    # CartPole
    "CartPole", "CartPoleState", "CartPoleParams",
    # MountainCar
    "MountainCar", "MountainCarState", "MountainCarParams",
    # MountainCarContinuous
    "MountainCarContinuous", "MountainCarContinuousState", "MountainCarContinuousParams",
    # Pendulum
    "Pendulum", "PendulumState", "PendulumParams",
    # Acrobot
    "Acrobot", "AcrobotState", "AcrobotParams",
    # Factory functions
    "make", "list_envs",
]

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


def list_envs() -> list[str]:
    """List all available environment IDs.

    Returns:
        List of environment ID strings that can be passed to gymlx.make().

    Example:
        >>> gymlx.list_envs()
        ['CartPole-v1', 'MountainCar-v0', 'Pendulum-v1', 'Acrobot-v1', 'MountainCarContinuous-v0']
    """
    return list(_registry.keys())
