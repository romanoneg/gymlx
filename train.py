"""Example usage of gymlx with mlx neural network with compiling and vmap"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import gymlx

ENV_NAME = "CartPole-v1" # A Discrete env
NUM_ENVS = 65536
STEPS_PER_ROLLOUT = 1024
NUM_ROLLOUTS = 128
GAMMA = 0.99

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return self.fc3(x)

def train():
    env = gymlx.make(ENV_NAME)
    
    policy = PolicyNetwork(obs_dim=env.observation_space.shape[0], num_actions=env.num_actions)
    optimizer = optim.Adam(learning_rate=1e-3)
    opt_state = optimizer.state

    def policy_loss(params, obs, actions, returns):
        policy.update(params) # Allows for func transformations to use the passed params
        log_probs = nn.log_softmax(policy(obs))
        action_log_probs = mx.take_along_axis(log_probs, actions[:, None], axis=1).squeeze(-1)
        return -mx.mean(action_log_probs * returns) # simple REINFORCE objective

    vmap_step = mx.vmap(env.step)
    vmap_reset = mx.vmap(env.reset)
    loss_and_grad = mx.value_and_grad(policy_loss)

    @mx.compile
    def batch_train_step(params, opt_state, key):
        """This entire function gets compiled, this is the mlx equivalent of jax.lax.scan"""

        all_obs, all_actions, all_rewards, all_dones = [], [], [], []
        episode_returns, completed_returns = mx.zeros(NUM_ENVS), mx.zeros(NUM_ENVS)

        episode_key, reset_key = mx.random.split(key)
        all_reset_keys = mx.random.split(reset_key, NUM_ENVS)

        obs, states, info = vmap_reset(all_reset_keys)

        # runs a complete rollout over all parallel envs for STEPS_PER_ROLLOUT num of steps
        for _ in range(STEPS_PER_ROLLOUT):
            act_key, step_key, episode_key = mx.random.split(episode_key, 3)

            policy.update(params)
            actions = mx.random.categorical(policy(obs), key=act_key)

            all_step_keys = mx.random.split(step_key, NUM_ENVS)
            next_obs, rewards, terminated, truncated, states, _ = vmap_step(states, actions, all_step_keys)
            dones = (terminated | truncated)

            episode_returns += rewards
            completed_returns = mx.where(dones, episode_returns, completed_returns)
            episode_returns = mx.where(dones, mx.zeros_like(episode_returns), episode_returns)

            all_obs.append(obs)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_dones.append(dones)

            obs = next_obs

        all_obs = mx.stack(all_obs)
        all_actions = mx.stack(all_actions)
        all_rewards = mx.stack(all_rewards)
        all_dones = mx.stack(all_dones)

        returns_list = [None] * STEPS_PER_ROLLOUT
        G = mx.zeros(NUM_ENVS)
        for t in reversed(range(STEPS_PER_ROLLOUT)):
            G = all_rewards[t] + GAMMA * G * (1 - all_dones[t])
            returns_list[t] = G
        all_returns = mx.stack(returns_list)

        loss, grads = loss_and_grad(params, all_obs.reshape(-1, *all_obs.shape[2:]),
                                    all_actions.reshape(-1, *all_actions.shape[2:]),
                                    all_returns.reshape(-1, *all_returns.shape[2:]))
        
        optimizer.state = opt_state
        policy.update(params)
        optimizer.update(policy, grads)
        new_params = policy.parameters()
        new_opt_state = optimizer.state

        return new_params, new_opt_state, loss, all_obs, all_actions, all_dones, all_rewards, completed_returns

    key = mx.random.key(42)
    params = policy.parameters()

    print(f"Batch \t | num_episodes \t | avg_return \t | loss")
    
    for batch in range(NUM_ROLLOUTS):

        batch_key, key = mx.random.split(key)
        # Need to get num_episodes
        params, opt_state, loss, obs, actions, done, rewards, ep_returns = batch_train_step(
            params, opt_state, batch_key
        )
        mx.eval(params, opt_state, ep_returns)

        avg_return = mx.sum(ep_returns) / mx.maximum(mx.sum(done),1)

        if (batch + 1) % 10 == 0:
            print(f"{batch + 1} \t | {mx.sum(done)} \t| {avg_return} \t | {loss}")

if __name__ == "__main__":
    train()








