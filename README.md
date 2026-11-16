# gymlx

Classic-control gym environments written in Apple's [MLX](https://github.com/ml-explore/mlx), so thousands of them can run in parallel on an Apple Silicon GPU.

```bash
git clone https://github.com/romanoneg/gymlx && cd gymlx
pip install -e .
```

## Built-in environments

`CartPole-v1`, `Pendulum-v1`, `Acrobot-v1`, `MountainCar-v0`, `MountainCarContinuous-v0`

The API is functional: state is passed in and returned, and finished episodes reset automatically.

```python
import mlx.core as mx
import gymlx

env = gymlx.make("CartPole-v1")
obs, state, info = env.reset(mx.random.key(0))
obs, reward, terminated, truncated, state, info = env.step(state, mx.array(1))
```

To run many at once, wrap `reset` and `step` in `mx.vmap` and `mx.compile`:

```python
num_envs = 65_536
reset = mx.compile(mx.vmap(env.reset))
step = mx.compile(mx.vmap(env.step))

obs, state, _ = reset(mx.random.split(mx.random.key(0), num_envs))
for _ in range(1000):
    actions = mx.random.randint(0, 2, shape=(num_envs,))
    obs, reward, terminated, truncated, state, _ = step(state, actions)
mx.eval(state)
```

`step` also takes `key=` (reproducible resets), `params=` (override physics constants) and `auto_reset=False`.

## Writing your own environment

Subclass `Env` with a state and a params `NamedTuple`, then define three methods. Use only `mx` operations (no Python `if` on array values) and return every value as an `mx.array`.

```python
from typing import NamedTuple
import mlx.core as mx
from gymlx import Env, Box, Discrete

class State(NamedTuple):
    x: mx.array
    time: mx.array

class Params(NamedTuple):
    goal: float = 1.0
    max_steps_in_episode: int = 100

class Walk(Env[State, Params]):
    """Step left or right along a line until you reach the goal."""
    default_params = Params()
    observation_space = Box(low=mx.array([-10.0]), high=mx.array([10.0]), shape=(1,))
    action_space = Discrete(n=2)
    num_actions = 2

    def reset_state(self, key, params):
        return State(mx.random.uniform(-0.5, 0.5, shape=(), key=key), mx.array(0))

    def get_obs(self, state, params):
        return mx.array([state.x])

    def step_env(self, state, action, key, params):
        state = State(state.x + 0.1 * (2 * action - 1), state.time + 1)
        terminated = state.x >= params.goal
        truncated = state.time >= params.max_steps_in_episode
        return mx.array(-1.0), terminated, truncated, state, {}
```

`Walk()` then works with `reset`, `step`, `mx.vmap` and `mx.compile` exactly like the built-ins.

## Benchmarks

Seconds per 1M steps (lower is better) with random actions and auto-reset, on a MacBook Air M2 (16 GB, MLX 0.32). The baseline is [gymnax](https://github.com/RobertTLange/gymnax) (JAX) on an NVIDIA A100 with 2,000 envs, as published in its README.

| Environment | gymnax, A100, 2k envs | gymlx, M2, 2k envs | gymlx, M2, best | Envs at best | Steps/s at best |
|---|---|---|---|---|---|
| CartPole-v1 | 0.05 | 0.054 | **0.0039** | 65,536 | 258M |
| Pendulum-v1 | 0.07 | 0.042 | **0.0030** | 131,072 | 329M |
| Acrobot-v1 | 0.07 | 0.135 | **0.0092** | 65,536 | 108M |
| MountainCar-v0 | 0.07 | 0.038 | **0.0018** | 131,072 | 571M |
| MountainCarContinuous-v0 | 0.09 | 0.039 | **0.0019** | 131,072 | 537M |

At the same 2,000 envs the M2 ranges from on par (CartPole) to about 2x faster than the A100, except Acrobot, which is about 2x slower. Throughput peaks at 65k–131k envs, where gymlx is 8–48x faster than the 2k-env gymnax figures; gymnax was not measured at those batch sizes.

Run your own with `gymlx-benchmark --env CartPole-v1 --num-envs 131072` (uses a fixed action rather than random ones).
