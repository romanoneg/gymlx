import mlx.core as mx
from gymlx.envs import CartPole, MountainCar, Pendulum, Acrobot, MountainCarContinuous
import time

def test_env(env_class, name, continuous=False):
    print(f"Testing {name}...")
    env = env_class()
    key = mx.random.key(42)
    
    # Test single step
    print("  Single step check...")
    res_key, step_key = mx.random.split(key, 2)
    obs, state = env.reset(res_key)
    
    if continuous:
        action = mx.array([0.0]) # Scalar action usually, but array for continuous often expected
        if name == "Pendulum":
             action = mx.array(0.0) # Or scalar
        elif name == "MountainCarContinuous":
             action = mx.array([0.0])
    else:
        action = mx.array(0)

    obs, reward, terminated, truncated, state, info = env.step(step_key, state, action)
    mx.eval(obs, reward, terminated, truncated, state)
    print("  Single step passed.")

    # Test VMAP + Compile
    print("  VMAP + Compile check...")
    batch_size = 100
    res_key = mx.random.split(key, batch_size)
    step_key = mx.random.split(key, batch_size)

    vmap_reset = mx.compile(mx.vmap(env.reset))
    vmap_step = mx.compile(mx.vmap(env.step))

    # Reset
    obs, state = vmap_reset(res_key)
    mx.eval(obs, state)
    
    # Step
    if continuous:
        actions = mx.zeros((batch_size, 1)) if name == "MountainCarContinuous" else mx.zeros((batch_size,))
        if name == "Pendulum":
             actions = mx.random.uniform(-2, 2, (batch_size,))
        elif name == "MountainCarContinuous":
             actions = mx.random.uniform(-1, 1, (batch_size,)) # Scalar actions but maybe shape matters?
             # step_env expects: force = mx.clip(action, ...)
             # If action is (Batch, 1), force is (Batch, 1). 
             # velocity = velocity + force * ... 
             # velocity is (Batch,). (Batch,) + (Batch,1) -> (Batch, Batch) broadcasting?
             # Check implementation of MCC.
    else:
        actions = mx.random.randint(0, env.num_actions(), (batch_size,))

    start = time.perf_counter()
    obs, reward, terminated, truncated, state, info = vmap_step(step_key, state, actions)
    mx.eval(obs, reward, terminated, truncated, state)
    end = time.perf_counter()
    print(f"  VMAP step time (first run): {end - start:.4f}s")
    
    start = time.perf_counter()
    obs, reward, terminated, truncated, state, info = vmap_step(step_key, state, actions)
    mx.eval(obs, reward, terminated, truncated, state)
    end = time.perf_counter()
    print(f"  VMAP step time (second run): {end - start:.4f}s")
    print(f"  {name} Passed.\n")

if __name__ == "__main__":
    test_env(CartPole, "CartPole")
    test_env(MountainCar, "MountainCar")
    test_env(Pendulum, "Pendulum", continuous=True)
    test_env(Acrobot, "Acrobot")
    test_env(MountainCarContinuous, "MountainCarContinuous", continuous=True)
