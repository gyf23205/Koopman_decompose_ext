import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import wandb

from utils import *
from torch.nn.utils import parameters_to_vector
from models.MLP import CartpoleMLP
from envs.CartpoleSwing import CartPoleSwing

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
    obs, info = env.reset(options={'low': -0.2, 'high': 0.2})
    timesteps = 0
    while timesteps < step_rollout:
        action, log_prob, dist = select_action(obs, policy_net, device)
        # Move obs to the same device as value_net
        value = value_net(torch.tensor(obs, dtype=torch.float32).to(device)).item()
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
            obs, info = env.reset(options={'low': -0.2, 'high': 0.2})
    return obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, obs

def validate(env, policy_net, num_episodes=10):
    """Run validation episodes and return average reward."""
    with torch.no_grad():
        episode_rewards = []
        for _ in range(num_episodes):
            obs, info = env.reset(options={'low': -0.2, 'high': 0.2})
            done = False
            total_reward = 0
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                logits = policy_net(obs_tensor)
                action = torch.argmax(logits).item()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            episode_rewards.append(total_reward)
    
    avg_reward = np.mean(episode_rewards)
    return avg_reward

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    learning_rate = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    update_epochs = 10
    step_rollout = 512
    batch_size = 128
    ppo_epochs = 1000
    # w_a = 0.08
    # w_u = 0.5
    # w_s = 1.2

    # Create the CartPole environment
    base = gym.make("CartPole-v1")
    env = CartPoleSwing(base, w_a=1.0, w_u=0.3, w_s=3.0)  # Further increased swing weight to emphasize boundary reaching
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

    # Initialize wandb for tracking
    wandb.login(key='1888b9830153065d084181ffc29812cd1011b84b')
    wandb.init(project="Koopman_ext", name="train_cartpole")
    
    obs, info = env.reset(options={'low': -0.2, 'high': 0.2})
    episode_rewards = []
    validation_rewards = []
    timesteps = 0
    best_val_reward = -float('inf')
    
    for epoch in tqdm.tqdm(range(ppo_epochs), desc="Training Original Policy"):
        # Collect trajectory
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, obs = collect_trajectory(env, step_rollout)

        # Process trajectory
        dataset = process_trajectory(
            value_net, obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, obs, device
        )

        # PPO update
        ppo_update(dataset, update_epochs, batch_size)
        
        # Validation every 20 epochs
        if (epoch + 1) % 20 == 0:
            val_reward = validate(env, policy_net)
            validation_rewards.append(val_reward)
            print(f"Epoch {epoch+1}: Validation reward: {val_reward:.2f}")
            wandb.log({"validation_reward": val_reward, "epoch": epoch+1})
            
            # Save best model
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                torch.save(policy_net.state_dict(), "saved_models_re/originals/ppo_cartpole_swing_policy_best.pt")
                torch.save(value_net.state_dict(), "saved_models_re/originals/ppo_cartpole_swing_value_best.pt")
                print(f"New best model with reward {best_val_reward:.2f}")

    # Save the final trained models
    torch.save(policy_net.state_dict(), "saved_models_re/originals/ppo_cartpole_swing_policy.pt")
    torch.save(value_net.state_dict(), "saved_models_re/originals/ppo_cartpole_swing_value.pt")
    
    # Save validation rewards history
    np.save("saved_models_re/validation_rewards_cartpole_swing.npy", np.array(validation_rewards))
    
    # Close wandb
    wandb.finish()
