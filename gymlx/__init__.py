from gymlx.envs import Env, CartPole, MountainCar, Pendulum, Acrobot, MountainCarContinuous

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
