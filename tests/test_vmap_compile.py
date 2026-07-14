"""Tests for vmap and compile functionality."""
import pytest
import mlx.core as mx
import gymlx


ENV_IDS = list(gymlx._registry.keys())
BATCH_SIZE = 100


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_vmap_reset(env_id):
    """Verify vmap works on reset."""
    env = gymlx.make(env_id)
    keys = mx.random.split(mx.random.key(0), BATCH_SIZE)

    vmap_reset = mx.vmap(env.reset)
    obs, state, info = vmap_reset(keys)
    mx.eval(obs, state)

    assert obs.shape[0] == BATCH_SIZE, f"{env_id}: batched obs should have batch dim"


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_vmap_step(env_id):
    """Verify vmap works on step."""
    env = gymlx.make(env_id)
    keys = mx.random.split(mx.random.key(0), BATCH_SIZE)

    vmap_reset = mx.vmap(env.reset)
    obs, state, info = vmap_reset(keys)

    actions = mx.zeros(BATCH_SIZE, dtype=mx.int32)
    vmap_step = mx.vmap(env.step)
    obs, reward, terminated, truncated, state, info = vmap_step(state, actions)
    mx.eval(obs, reward, state)

    assert obs.shape[0] == BATCH_SIZE
    assert reward.shape[0] == BATCH_SIZE


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_compile_reset(env_id):
    """Verify compile works on reset."""
    env = gymlx.make(env_id)
    key = mx.random.key(0)

    compiled_reset = mx.compile(env.reset)
    obs, state, info = compiled_reset(key)
    mx.eval(obs, state)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_compile_step(env_id):
    """Verify compile works on step."""
    env = gymlx.make(env_id)
    key = mx.random.key(0)
    obs, state, info = env.reset(key)

    compiled_step = mx.compile(env.step)
    action = mx.array(0)
    obs, reward, terminated, truncated, state, info = compiled_step(state, action)
    mx.eval(obs, state)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_vmap_compile_combined(env_id):
    """Verify vmap + compile work together."""
    env = gymlx.make(env_id)
    keys = mx.random.split(mx.random.key(0), BATCH_SIZE)

    vmap_reset = mx.compile(mx.vmap(env.reset))
    vmap_step = mx.compile(mx.vmap(env.step))

    obs, state, info = vmap_reset(keys)
    mx.eval(obs, state)

    actions = mx.zeros(BATCH_SIZE, dtype=mx.int32)
    for _ in range(5):
        obs, reward, terminated, truncated, state, info = vmap_step(state, actions)

    mx.eval(obs, state)
    assert obs.shape[0] == BATCH_SIZE
