"""Tests for environment functionality."""
import pytest
import mlx.core as mx
import gymlx


ENV_IDS = list(gymlx._registry.keys())


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_reset_returns_tuple_of_three(env_id):
    """Verify reset returns (obs, state, info)."""
    env = gymlx.make(env_id)
    key = mx.random.key(0)
    result = env.reset(key)

    assert len(result) == 3, f"{env_id}: reset should return 3 values"
    obs, state, info = result
    assert isinstance(info, dict), f"{env_id}: info should be dict"


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_step_returns_tuple_of_six(env_id):
    """Verify step returns (obs, reward, terminated, truncated, state, info)."""
    env = gymlx.make(env_id)
    key = mx.random.key(0)
    obs, state, info = env.reset(key)

    action = mx.array(0)
    result = env.step(state, action)

    assert len(result) == 6, f"{env_id}: step should return 6 values"


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_step_produces_valid_reward(env_id):
    """Verify step produces a scalar reward."""
    env = gymlx.make(env_id)
    key = mx.random.key(0)
    obs, state, info = env.reset(key)

    action = mx.array(0)
    obs, reward, terminated, truncated, state, info = env.step(state, action)
    mx.eval(reward)

    assert reward.ndim == 0 or reward.shape == (), f"{env_id}: reward should be scalar"


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_terminated_truncated_are_boolean(env_id):
    """Verify terminated and truncated are boolean-like."""
    env = gymlx.make(env_id)
    key = mx.random.key(0)
    obs, state, info = env.reset(key)

    action = mx.array(0)
    obs, reward, terminated, truncated, state, info = env.step(state, action)
    mx.eval(terminated, truncated)

    assert terminated.dtype == mx.bool_ or terminated.ndim == 0
    assert truncated.dtype == mx.bool_ or truncated.ndim == 0


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_multiple_steps_work(env_id):
    """Verify multiple steps can be run without error."""
    env = gymlx.make(env_id)
    key = mx.random.key(0)
    obs, state, info = env.reset(key)

    for i in range(10):
        action = mx.array(0)
        obs, reward, terminated, truncated, state, info = env.step(state, action)

    mx.eval(obs, state)
