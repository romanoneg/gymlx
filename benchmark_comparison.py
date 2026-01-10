"""Benchmark gymlx environments with random actions and compile time measurement."""
import mlx.core as mx
import time
import sys
sys.path.insert(0, '/Users/romanoneg/Projects/gymlx')
import gymlx

# JAX gymnax benchmark reference (A100 GPU, 2000 workers)
JAX_BENCHMARKS = {
    "CartPole-v1": 0.05,
    "Pendulum-v1": 0.07,
    "Acrobot-v1": 0.07,
    "MountainCar-v0": 0.07,
    "MountainCarContinuous-v0": 0.09,
}

ENVIRONMENTS = list(JAX_BENCHMARKS.keys())


def benchmark_env(env_id: str, num_envs: int = 131072, total_steps: int = 10_000_000, chunk_size: int = 500):
    """Benchmark a single environment with random actions.

    Returns:
        dict with compile_time, run_time, steps_per_sec, sec_per_1m_steps
    """
    env = gymlx.make(env_id)
    action_space = env.action_space

    # Check if discrete or continuous action space
    is_discrete = hasattr(action_space, 'n')

    key = mx.random.key(42)
    reset_key, step_key, action_key = mx.random.split(key, 3)
    reset_keys = mx.random.split(reset_key, num_envs)

    # Vmapped functions
    vmap_reset = mx.vmap(env.reset)
    vmap_step = mx.vmap(env.step)

    # Initial reset (uncompiled)
    obs, state, info = vmap_reset(reset_keys)
    mx.eval(obs, state)

    # Generate random actions based on action space
    if is_discrete:
        actions = mx.random.randint(0, action_space.n, shape=(num_envs,), key=action_key)
    else:
        # Continuous action space
        actions = mx.random.uniform(
            low=action_space.low[0].item(),
            high=action_space.high[0].item(),
            shape=(num_envs, *action_space.shape),
            key=action_key
        )

    step_keys = mx.random.split(step_key, num_envs)
    mx.eval(actions, step_keys)

    # Compile step function by tracing
    compile_start = time.perf_counter()
    compiled_step = mx.compile(vmap_step)
    # Force compilation by running once
    obs, reward, terminated, truncated, state, info = compiled_step(step_keys, state, actions)
    mx.eval(state)
    compile_time = time.perf_counter() - compile_start

    # Warmup with random actions
    for i in range(10):
        action_key = mx.random.split(action_key)[0]
        if is_discrete:
            actions = mx.random.randint(0, action_space.n, shape=(num_envs,), key=action_key)
        else:
            actions = mx.random.uniform(
                low=action_space.low[0].item(),
                high=action_space.high[0].item(),
                shape=(num_envs, *action_space.shape),
                key=action_key
            )
        step_key = mx.random.split(step_key)[0]
        step_keys = mx.random.split(step_key, num_envs)
        obs, reward, terminated, truncated, state, info = compiled_step(step_keys, state, actions)
    mx.eval(state)

    # Calculate iterations
    steps_per_iter = num_envs
    total_iters = total_steps // steps_per_iter
    num_chunks = max(1, total_iters // chunk_size)
    actual_chunk = total_iters // num_chunks

    # Pre-generate all action keys for the benchmark
    # This avoids random number generation overhead in the timed section
    all_action_keys = mx.random.split(action_key, num_chunks * actual_chunk)
    all_step_keys = mx.random.split(step_key, num_chunks * actual_chunk)
    mx.eval(all_action_keys, all_step_keys)

    # Timed benchmark
    with mx.stream(mx.gpu):
        start = time.perf_counter()
        iter_idx = 0
        for chunk in range(num_chunks):
            for _ in range(actual_chunk):
                # Get pre-generated keys
                ak = all_action_keys[iter_idx]
                sk = all_step_keys[iter_idx]
                iter_idx += 1

                # Generate random actions
                if is_discrete:
                    actions = mx.random.randint(0, action_space.n, shape=(num_envs,), key=ak)
                else:
                    actions = mx.random.uniform(
                        low=action_space.low[0].item(),
                        high=action_space.high[0].item(),
                        shape=(num_envs, *action_space.shape),
                        key=ak
                    )
                step_keys = mx.random.split(sk, num_envs)
                obs, reward, terminated, truncated, state, info = compiled_step(step_keys, state, actions)
            mx.eval(state)
        elapsed = time.perf_counter() - start

    actual_steps = num_chunks * actual_chunk * steps_per_iter
    steps_per_sec = actual_steps / elapsed
    sec_per_1m = 1_000_000 / steps_per_sec

    return {
        "compile_time": compile_time,
        "run_time": elapsed,
        "steps_per_sec": steps_per_sec,
        "sec_per_1m_steps": sec_per_1m,
        "actual_steps": actual_steps,
    }


def main():
    """Run benchmarks for all environments and print comparison table."""
    print("=" * 80)
    print("gymlx Benchmark Comparison with JAX gymnax")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Total steps per benchmark: 10,000,000")
    print(f"  - Random actions: Yes")
    print(f"  - JAX reference: NVIDIA A100 GPU, 2000 workers")
    print(f"  - gymlx: Apple Silicon (MLX)")
    print()

    results_2k = {}
    results_131k = {}

    # Benchmark with 2000 envs (same as JAX)
    print("Running benchmarks with 2,000 environments (JAX-comparable)...")
    for env_id in ENVIRONMENTS:
        print(f"  {env_id}...", end=" ", flush=True)
        result = benchmark_env(env_id, num_envs=2000)
        results_2k[env_id] = result
        print(f"{result['sec_per_1m_steps']:.4f}s/1M")

    # Benchmark with 131072 envs (optimal for MLX)
    print("\nRunning benchmarks with 131,072 environments (MLX-optimal)...")
    for env_id in ENVIRONMENTS:
        print(f"  {env_id}...", end=" ", flush=True)
        result = benchmark_env(env_id, num_envs=131072)
        results_131k[env_id] = result
        print(f"{result['sec_per_1m_steps']:.4f}s/1M")

    # Print results table
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    print()
    print(f"{'Environment':<26} {'Compile':<10} {'JAX A100':<12} {'gymlx 2k':<12} {'gymlx 131k':<12} {'2k Ratio':<10} {'131k Ratio':<10}")
    print(f"{'':26} {'(s)':10} {'(s/1M)':12} {'(s/1M)':12} {'(s/1M)':12} {'':10} {'':<10}")
    print("-" * 100)

    for env_id in ENVIRONMENTS:
        r2k = results_2k[env_id]
        r131k = results_131k[env_id]
        jax_time = JAX_BENCHMARKS[env_id]
        ratio_2k = r2k['sec_per_1m_steps'] / jax_time
        ratio_131k = r131k['sec_per_1m_steps'] / jax_time
        print(f"{env_id:<26} {r131k['compile_time']:<10.4f} {jax_time:<12.2f} {r2k['sec_per_1m_steps']:<12.4f} {r131k['sec_per_1m_steps']:<12.4f} {ratio_2k:<10.2f}x {ratio_131k:<10.2f}x")

    print("-" * 100)
    print()
    print("Notes:")
    print("  - JAX A100: NVIDIA A100 GPU with 2,000 workers (from gymnax repo)")
    print("  - gymlx 2k: Apple Silicon with 2,000 parallel environments (apples-to-apples)")
    print("  - gymlx 131k: Apple Silicon with 131,072 parallel environments (MLX-optimal)")
    print("  - Ratio < 1.0 means gymlx is faster than JAX")
    print()


if __name__ == "__main__":
    main()
