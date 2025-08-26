import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm

from utils import *
from torch.nn.utils import parameters_to_vector
from models.MLP import CartpoleMLP

def ppo_update(dataset, update_epochs, batch_size):
    for _ in range(update_epochs):
        for obs_tensor, act_tensor, logp_tensor, adv_tensor, ret_tensor in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            logits = policy_net(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(act_tensor)
            ratio = torch.exp(new_logp - logp_tensor)
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv_tensor
            policy_loss = -torch.min(surr1, surr2).mean()

            value_preds = value_net(obs_tensor).squeeze()
            value_loss = nn.functional.mse_loss(value_preds, ret_tensor)

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
            

def collect_trajectory(env, step_rollout):
    obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
    obs, info = env.reset()
    timesteps = 0
    while timesteps < step_rollout:
        action, log_prob, dist = select_action(obs, policy_net, device)
        value = value_net(torch.tensor(obs, dtype=torch.float32)).item()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        obs_buf.append(obs)
        act_buf.append(action)
        logp_buf.append(log_prob)
        rew_buf.append(reward)
        done_buf.append(done)
        val_buf.append(value)

        obs = next_obs
        timesteps += 1
        if done:
            obs, info = env.reset()
    return obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, obs

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    learning_rate = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    update_epochs = 10
    step_rollout = 128
    batch_size = 64
    ppo_epochs = 2000

    # Create the CartPole environment
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Policy and value networks
    policy_net = CartpoleMLP(obs_dim, act_dim).to(device)
    policy_net.train()
    value_net = nn.Sequential(
        nn.Linear(obs_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)
    value_net.train()
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)

    obs, info = env.reset()
    episode_rewards = []
    timesteps = 0
    params_trajectory = []
    for _ in tqdm.tqdm(range(ppo_epochs), desc="Training Original Policy"):
        # Collect trajectory
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, obs = collect_trajectory(env, step_rollout, value_net, policy_net, device)

        # Process trajectory
        dataset = process_trajectory(
            value_net, obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, obs
        )

        # PPO update
        ppo_update(dataset, update_epochs, batch_size, policy_net, value_net, policy_optimizer, value_optimizer, clip_epsilon)

        # Save parameters
        params_trajectory.append(parameters_to_vector(policy_net.parameters()).detach().cpu().numpy())

    # Save the trained models
    torch.save(policy_net.state_dict(), "saved_models/originals/ppo_cartpole_policy.pt")
    torch.save(value_net.state_dict(), "saved_models/originals/ppo_cartpole_value.pt")

    # Save the parameter trajectory
    traj_np = np.array(params_trajectory)
    print(traj_np.shape)
    np.save("saved_trajectory/param_trajectory_cartpole_mlp.npy", traj_np)
