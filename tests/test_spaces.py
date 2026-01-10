"""Tests for observation and action space consistency."""
import pytest
import mlx.core as mx
import gymlx
from gymlx.spaces import Box, Discrete


ENV_IDS = list(gymlx._registry.keys())


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_observation_shape_matches_space(env_id):
    """Verify observation shape matches observation_space.shape."""
    env = gymlx.make(env_id)
    key = mx.random.key(0)
    obs, state, info = env.reset(key)
    mx.eval(obs)

    assert obs.shape == env.observation_space.shape, (
        f"{env_id}: obs.shape {obs.shape} != observation_space.shape {env.observation_space.shape}"
    )


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_observation_space_has_required_fields(env_id):
    """Verify observation_space has shape and dtype."""
    env = gymlx.make(env_id)
    space = env.observation_space

    assert isinstance(space, Box), f"{env_id}: observation_space should be Box"
    assert hasattr(space, "shape")
    assert hasattr(space, "dtype")
    assert hasattr(space, "low")
    assert hasattr(space, "high")


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_action_space_type(env_id):
    """Verify action_space is Box or Discrete."""
    env = gymlx.make(env_id)
    space = env.action_space

    assert isinstance(space, (Box, Discrete)), f"{env_id}: action_space should be Box or Discrete"


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_num_actions_matches_action_space(env_id):
    """Verify num_actions is consistent with action_space."""
    env = gymlx.make(env_id)

    if isinstance(env.action_space, Discrete):
        assert env.num_actions == env.action_space.n
    elif isinstance(env.action_space, Box):
        assert env.num_actions == env.action_space.shape[0]
