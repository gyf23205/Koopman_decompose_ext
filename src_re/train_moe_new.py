import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
from models.MoE import MoE
# import wandb
import sys
sys.path.append("/home/yifan/git/Koopman_decompose_ext/src_re/models")  # Add the directory to the Python path


from utils import *
from torch.nn.utils import parameters_to_vector
from models.MLP import CartpoleMLP
from envs.CartpoleSpin import CartPoleSpin
import torch.nn.functional as F
from torch.serialization import add_safe_globals, safe_globals
from models.Autoencoder import KoopmanAutoencoder

def ppo_update(dataset, update_epochs, batch_size):
    for _ in range(update_epochs):
        for obs_tensor, act_tensor, logp_tensor, adv_tensor, ret_tensor in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            if obs_tensor.shape[-1] < padded_dim:
                pad_size = padded_dim - obs_tensor.shape[-1]
                obs_tensor_padded = torch.nn.functional.pad(obs_tensor, (0, pad_size), "constant", 1)
            with torch.no_grad():
                x_hat, z, z_pred = kae(obs_tensor_padded)
            # extends = torch.diag(torch.ones(act_dim, dtype=weights.dtype, device=weights.device)).tile(weights.shape[0], 1, 1)
            experts_outputs = get_experts_outputs(kae, z, p, act_dim)
            experts_outputs = torch.softmax(experts_outputs, dim=-1)
            extended_experts_outputs = extend_experts_outputs(experts_outputs, act_dim)
            weights = F.softmax(policy_net(obs_tensor), dim=-1)
            probs = torch.sum(weights.view(batch_size, n_dom_modes, 1) * extended_experts_outputs, dim=1)
            dist = torch.distributions.Categorical(probs=probs)
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
    obs, info = env.reset(options={'low': low, 'high': high})
    timesteps = 0
    while timesteps < step_rollout:
        # action, log_prob, dist = select_action(obs, policy_net, device)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        if obs_tensor.shape[-1] < padded_dim:
            pad_size = padded_dim - obs_tensor.shape[-1]
            obs_tensor_padded = torch.nn.functional.pad(obs_tensor, (0, pad_size), "constant", 1).unsqueeze(0)
        with torch.no_grad():
            x_hat, z, z_pred = kae(obs_tensor_padded)

        experts_outputs = get_experts_outputs(kae, z, p, act_dim)
        experts_outputs = torch.softmax(experts_outputs, dim=-1)
        extended_experts_outputs = extend_experts_outputs(experts_outputs, act_dim)

        weights = F.softmax(policy_net(obs_tensor), dim=-1)
        # extends = torch.diag(torch.ones(act_dim, dtype=weights.dtype, device=weights.device)).tile(1, 1, 1)
        probs = torch.sum(weights.view(1, n_dom_modes, 1) * extended_experts_outputs, dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=device)).item()
        
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
            obs, info = env.reset(options={'low': low, 'high': high})
    return obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, obs

def validate(env, policy_net, num_episodes=10):
    """Run validation episodes and return average reward."""
    with torch.no_grad():
        episode_rewards = []
        for _ in range(num_episodes):
            obs, info = env.reset(options={'low': low, 'high': high})
            done = False
            total_reward = 0
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                if obs_tensor.shape[-1] < padded_dim:
                    pad_size = padded_dim - obs_tensor.shape[-1]
                    obs_tensor_padded = torch.nn.functional.pad(obs_tensor, (0, pad_size), "constant", 1).unsqueeze(0)
                with torch.no_grad():
                    x_hat, z, z_pred = kae(obs_tensor_padded)

                experts_outputs = get_experts_outputs(kae, z, p, act_dim)
                experts_outputs = torch.softmax(experts_outputs, dim=-1)
                extended_experts_outputs = extend_experts_outputs(experts_outputs, act_dim)

                weights = F.softmax(policy_net(obs_tensor), dim=-1)
                probs = torch.sum(weights.view(1, n_dom_modes, 1) * extended_experts_outputs, dim=1)
                action = torch.argmax(probs).item()
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
    ppo_epochs = 500
    # w_t = 1.0
    # w_v = 0.2
    w_a = 0.08
    w_c = 0.5
    w_s = 1.07
    w_s_initial = 0.1  # Start with lower w_s
    w_s_final = 1.07   # 1.07 Final target w_s
    w_s_current = w_s_initial
    # w_b = 0.2
    # w_e_initial = 0.6
    low = -0.2
    high = 0.2
    padded_dim = 64
    p = 2
    save = True
    num_layers = 2  # For naming saved models
    n_dom_modes = 6

    # Create the CartPole environment
    base = gym.make("CartPole-v1")
    env = CartPoleSpin(base, w_a=w_a, w_c=w_c, w_s=w_s_current)  # Further increased swing weight to emphasize boundary reaching
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n


    # Load Koopman Autoencoder
    with safe_globals([KoopmanAutoencoder]):
        kae = torch.load(
            "saved_models_re/KAEs/KAE_[4, 64, 32, 0.5, -1, 0.5, 2, 'CARTPOLE_v1']_2025-09-25.pth",
            map_location=device,
            weights_only=False  # Explicitly allow loading the full object
        )
    kae.eval()
    # Policy and value networks
    # policy_net = MoE(obs_dim, num_experts=n_dom_modes, num_layers=num_layers).to(device)
    policy_net = CartpoleMLP(obs_dim, 6).to(device)
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
    # wandb.login(key='1888b9830153065d084181ffc29812cd1011b84b')
    # wandb.init(project="Koopman_ext", name="train_cartpole")

    obs, info = env.reset(options={'low': low, 'high': high})
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
            # wandb.log({"validation_reward": val_reward, "epoch": epoch+1})
            
            # Save best model
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                if save:
                    # Save MoE model
                    torch.save(policy_net.state_dict(), "saved_models_re/MoEs/moe_cartpole_mlp_retrain_value_spin_{}layers.pt".format(num_layers))
                    print(f"New best reward: {best_val_reward} at epoch {epoch + 1}, model saved.")

    # Close wandb
    # wandb.finish()
