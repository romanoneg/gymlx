"""Benchmark utilities for gymlx environments."""
import mlx.core as mx
import time
import argparse
from . import make, _registry


def benchmark(env_id: str = "CartPole-v1", num_envs: int = 131072, total_steps: int = 10_000_000, chunk_size: int = 500):
    """Run benchmark for seconds per 1M steps.

    Args:
        env_id: Environment identifier
        num_envs: Number of parallel environments (optimal around 65k-131k for M-series)
        total_steps: Total environment steps to run
        chunk_size: Number of steps to batch before syncing (reduces kernel launch overhead)
    """
    env = make(env_id)

    key = mx.random.key(42)
    reset_key, step_key = mx.random.split(key)
    reset_keys = mx.random.split(reset_key, num_envs)
    step_keys = mx.random.split(step_key, num_envs)

    vmap_reset = mx.compile(mx.vmap(env.reset))
    vmap_step = mx.compile(mx.vmap(lambda s, a, k: env.step(s, a, key=k)))

    obs, state, info = vmap_reset(reset_keys)
    mx.eval(obs, state)

    actions = mx.zeros(num_envs, dtype=mx.int32)

    # Warmup
    for _ in range(10):
        obs, reward, terminated, truncated, state, info = vmap_step(state, actions, step_keys)
        step_keys = mx.random.split(step_keys[0], num_envs)
    mx.eval(state)

    # Calculate iterations
    steps_per_iter = num_envs
    total_iters = total_steps // steps_per_iter
    num_chunks = max(1, total_iters // chunk_size)
    actual_chunk = total_iters // num_chunks

    with mx.stream(mx.gpu):
        start = time.perf_counter()
        for _ in range(num_chunks):
            for _ in range(actual_chunk):
                obs, reward, terminated, truncated, state, info = vmap_step(state, actions, step_keys)
                step_keys = mx.random.split(step_keys[0], num_envs)
            mx.eval(state)
        elapsed = time.perf_counter() - start

    actual_steps = num_chunks * actual_chunk * steps_per_iter
    steps_per_sec = actual_steps / elapsed

    print(f"Environment: {env_id}")
    print(f"Num parallel envs: {num_envs:,}")
    print(f"Total steps: {actual_steps:,}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Steps/sec: {steps_per_sec:,.0f}")
    print(f"Seconds per 1M steps: {1_000_000 / steps_per_sec:.4f}s")

    return steps_per_sec


def main():
    """CLI entry point for benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark gymlx environments")
    parser.add_argument("--env", type=str, default="CartPole-v1", choices=list(_registry.keys()), help="Environment ID")
    parser.add_argument("--num-envs", type=int, default=131072, help="Number of parallel environments")
    parser.add_argument("--total-steps", type=int, default=10_000_000, help="Total steps to run")
    parser.add_argument("--chunk-size", type=int, default=500, help="Steps per chunk before sync")

    args = parser.parse_args()
    benchmark(args.env, args.num_envs, args.total_steps, args.chunk_size)


if __name__ == "__main__":
    main()
