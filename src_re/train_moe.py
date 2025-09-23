import gymnasium as gym
import torch
import numpy as np
# import matplotlib.pyplot as plt
import wandb
from datetime import datetime

# from matplotlib.animation import FuncAnimation
from models.Autoencoder import KoopmanAutoencoder
from models.Autoencoder_functions import *
from models.MLP import ValueMLP
from models.MoE import MoE
from utils import *
from envs.CartpoleSpin import CartPoleSpin
# from torch.nn.utils import parameters_to_vector
import sys
sys.path.append("/home/yifan/git/Koopman_decompose_ext/src_re/models")  # Add the directory to the Python path


def ppo_update(dataset):
    for _ in range(update_epochs):
        for obs_tensor, act_tensor, logp_tensor, adv_tensor, ret_tensor in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            if obs_tensor.shape[-1] < padded_dim:
                pad_size = padded_dim - obs_tensor.shape[-1]
                obs_tensor_padded = torch.nn.functional.pad(obs_tensor, (0, pad_size), "constant", 1)
            with torch.no_grad():
                x_hat, z, z_pred = kae(obs_tensor_padded)
            N_O = z.shape[-1]
            experts_outputs = get_experts_outputs(kae, z, p, act_dim=act_dim)
            weights = moe(obs_tensor)
            # sparse_weights = weights  # Using original weights without top-k sparsification
            
            # Top-2 gating: select top 2 experts and renormalize weights
            top_weights, top_indices = torch.topk(weights, top_k, dim=-1)
            top_weights = torch.softmax(top_weights, dim=-1)  # Renormalize top-2 weights
            
            # Create sparse weights tensor
            mask = torch.zeros_like(weights).scatter_(-1, top_indices, 1)
            # sparse_weights = torch.zeros_like(weights)
            # sparse_weights.scatter_(-1, top_indices, top_weights)
            sparse_weights = weights * mask
            mean_weights = weights.mean(0)
            mean_usage = mask.mean(0)
            loss_balance = (mean_weights * mean_usage).sum()
            entropy = -torch.sum(sparse_weights * torch.log(sparse_weights + 1e-10), dim=-1).mean()

            logits = torch.sum(sparse_weights.unsqueeze(-1) * experts_outputs, dim=1)
            dist = torch.distributions.Categorical(logits=logits.view(-1, act_dim))
            new_logp = dist.log_prob(act_tensor)
            ratio = torch.exp(new_logp - logp_tensor)
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv_tensor
            task_loss = -torch.min(surr1, surr2).mean()

            value_preds = value_net(obs_tensor).squeeze()
            value_loss = nn.functional.mse_loss(value_preds, ret_tensor)

            loss = w_t * task_loss + w_v * value_loss  + w_b * loss_balance - w_e * entropy

            value_optimizer.zero_grad()
            moe_optimizer.zero_grad()

            loss.backward()

            # value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=5.0)
            value_optimizer.step()

            if ep >= pretrain_epochs:
                # task_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(moe.parameters(), max_norm=5.0)
                moe_optimizer.step()

            grad_norm_moe = compute_grad_norm(moe)
            grad_norm_value = compute_grad_norm(value_net)
            wandb.log({"MoE Grad Norm": grad_norm_moe,
                       "Value Grad Norm": grad_norm_value,
                       "MoE Loss": task_loss.item(),
                       "Value Loss": value_loss.item(),
                        "Balance Loss": loss_balance.item(),
                        "Entropy": entropy.item(),
                       "loss_all": loss.item() + value_loss.item()})

def collect_trajectory(env, value_net):
    obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
    obs, info = env.reset(options={'low': low, 'high': high})
    timesteps = 0
    while timesteps < step_rollout:
        # Move obs to the same device as the MoE model
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        if obs_tensor.shape[-1] < padded_dim:
            pad_size = padded_dim - obs_tensor.shape[-1]
            obs_tensor_padded = torch.nn.functional.pad(obs_tensor, (0, pad_size), "constant", 1).unsqueeze(0)
        with torch.no_grad():
            x_hat, z, z_pred = kae(obs_tensor_padded)
        N_O = z.shape[-1]
        experts_outputs = get_experts_outputs(kae, z, p, act_dim)
        weights = moe(obs_tensor)
        weights = weights + torch.randn_like(weights) * 1e-2
        # sparse_weights = weights  # Using original weights without top-k sparsification

        # Top-2 gating: select top 2 experts and renormalize weights
        top_weights, top_indices = torch.topk(weights, top_k, dim=-1)
        top_weights = torch.softmax(top_weights, dim=-1)  # Renormalize top-2 weights
        
        # Create sparse weights tensor
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.scatter_(-1, top_indices, top_weights)

        logits = sparse_weights.view(1, 1, -1) @ experts_outputs
        dist = torch.distributions.Categorical(logits=logits.view(-1, act_dim))
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=device)).item()
        value = value_net(torch.tensor(obs, dtype=torch.float32).to(device))
        next_obs, reward, terminated, truncated, info = env.step(action)  # Pass scalar action
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

def val(moe, epoches=5):
    with torch.no_grad():
        episode_rewards = []
        episode_entropy = []
        for _ in range(epoches):
            obs, info = env.reset(options={'low': low, 'high': high})
            done = False
            total_reward = 0
            total_entropy = 0
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                if obs_tensor.shape[-1] < padded_dim:
                    pad_size = padded_dim - obs_tensor.shape[-1]
                    obs_tensor_padded = torch.nn.functional.pad(obs_tensor, (0, pad_size), "constant", 1).unsqueeze(0)
                with torch.no_grad():
                    x_hat, z, z_pred = kae(obs_tensor_padded)
                N_O = z.shape[-1]
                experts_outputs = get_experts_outputs(kae, z, p, act_dim)
                weights = moe(obs_tensor)
                # sparse_weights = weights  # Using original weights without top-k sparsification
                
                # Top-2 gating: select top 2 experts and renormalize weights
                top_weights, top_indices = torch.topk(weights, top_k, dim=-1)
                top_weights = torch.softmax(top_weights, dim=-1)  # Renormalize top-2 weights
                
                # Create sparse weights tensor
                sparse_weights = torch.zeros_like(weights)
                sparse_weights.scatter_(-1, top_indices, top_weights)
                
                entropy = -torch.sum(sparse_weights * torch.log(sparse_weights + 1e-10), dim=-1).mean()
                total_entropy += entropy.item()

                logits = sparse_weights.view(1, 1, -1) @ experts_outputs
                action = torch.argmax(logits).item()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            episode_rewards.append(total_reward)
            episode_entropy.append(total_entropy)
    return np.mean(episode_rewards), np.mean(episode_entropy)

if __name__ == "__main__":
    # Use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Currently using... '+str(device))

    # Set training to be deterministic
    seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    

    # Allowlist the KoopmanAutoencoder class
    # torch.serialization.add_safe_globals([KoopmanAutoencoder])

    wandb.login(key='1888b9830153065d084181ffc29812cd1011b84b')

    # wandb.init(project="Koopman_ext",
    #         name = "MoE_sweep_Cartpole_spin_weights")

    # Hyperparameters
    w_t = 1.0
    w_v = 0.2
    # w_a = 0.08
    # w_c = 0.5
    # w_s = 1.07
    # w_s_initial = 0.1  # Start with lower w_s
    # w_s_final = 1.07   # 1.07 Final target w_s
    # w_s_current = w_s_initial
    w_b = 0.2
    w_e_initial = 0.6
    low = -0.2
    high = 0.2

    save = True
    learning_rate = 1e-5
    step_rollout = 512
    batch_size = 128 # 128
    clip_epsilon = 0.2 # 0.38
    num_layers = 2 # 3
    p = 3
    top_k = 2
    kae_size = 48

    padded_dim = 128
    pretrain_epochs = 0
    ppo_epochs = 800 # 500 
    update_epochs = 5
    n_dom_modes = 8
    frames = []  # To store frames for the GIF
    n_frames = []

    base = gym.make("CartPole-v1")
    env = base
    # env = CartPoleSpin(base, w_a=w_a, w_c=w_c, w_s=w_s_current)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    current_date = datetime.now().strftime("%Y-%m-%d")
    wandb.login(key='1888b9830153065d084181ffc29812cd1011b84b')
    wandb.init(project="Koopman_ext",
            name = 'spin2invert' + str(current_date),
            config={
                "learning_rate": learning_rate,
                "ppo_epochs": ppo_epochs,
                "step_rollout": step_rollout,
                "update_epochs": update_epochs,
                "batch_size": batch_size,
                "clip_epsilon": clip_epsilon,
                "n_dom_modes": n_dom_modes,
                "num_layers": num_layers,
                'p': p, 
                "padded_dim": padded_dim,
                # "w_a": w_a,
                # "w_c": w_c,
                # "w_s_initial": w_s_initial,
                # "w_s_final": w_s_final,
                "w_b": w_b,
                "w_e": w_e_initial,
                "top_k": top_k,
                "pretrain_epochs": pretrain_epochs
        })

    # networks
    kae = KoopmanAutoencoder(padded_dim, kae_size, n_dom_modes, device).to(device)
    kae.load_state_dict(torch.load("saved_models_re/KAEs/KAE_state_dict_[8, 128, 48, 0.5, 0.4, 0.1, 3, 'CARTPOLE_v1']_2025-09-17.pt", weights_only=True, map_location=device))
    kae.eval()
    # Retraining the value network
    value_net = ValueMLP(obs_dim).to(device)
    value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)
    # value_net.load_state_dict(torch.load("saved_models_re/originals/ppo_cartpole_value.pt", map_location=device), strict=False)
    # Add "model." before all parameter names
    # state_dict = value_net.state_dict()
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     new_state_dict["model." + k] = v
    # value_net.load_state_dict(new_state_dict, strict=False)
    # value_net.eval()

    moe = MoE(obs_dim, num_experts=n_dom_modes, num_layers=num_layers).to(device)
    moe_optimizer = optim.Adam(moe.parameters(), lr=learning_rate)
    schelduler = optim.lr_scheduler.StepLR(moe_optimizer, step_size=100, gamma=0.5, verbose=True)


    obs, info = env.reset(options={'low': low, 'high': high})
    timesteps = 0
    params_trajectory = []
    reward_best = -np.inf
    for ep in tqdm(range(ppo_epochs), desc="Training MoE"):  # Use tqdm directly
        # Gradually increase w_s during training
        progress = np.min([ep / (ppo_epochs * 0.6), 1]) 
        # progress = 1.0
        # w_s_current = w_s_initial + (w_s_final - w_s_initial) * progress
        # env.update_w_s(w_s_current)
        w_e = w_e_initial * (1 - progress)  # Decrease w_e from initial to 0
        
        # Log current w_s
        # wandb.log({"w_s_current": w_s_current})

        with torch.no_grad():
            # Collect trajectory
            obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, obs = collect_trajectory(env, value_net)

        # Process trajectory
        dataset = process_trajectory(
            value_net, obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, obs, device
        )

        # PPO update
        ppo_update(dataset)

        # Learning rate scheduling
        if (ep + 1) % 2 == 0:
            schelduler.step()
        wandb.log({"Learning Rate": schelduler.get_last_lr()[0]})

        # Validation
        if (ep + 1) % 10 == 0:
            reward_temp, entropy = val(moe, epoches=5)
            reward_temp_normalized = reward_temp
            # reward_temp_normalized = reward_temp / (w_a + w_c + w_s_current)  # Use current w_s for normalization
            print(f"Validation reward at epoch {ep + 1}: {reward_temp} (normalized: {reward_temp_normalized})")
            wandb.log({"Validation Reward": reward_temp})
            print(f"Validation entropy at epoch {ep + 1}: {entropy}")
            wandb.log({"Validation Entropy": entropy})
            if reward_temp_normalized > reward_best:
                reward_best = reward_temp_normalized
                if save:
                    # Save MoE model
                    torch.save(moe.state_dict(), "saved_models_re/MoEs/moe_cartpole_mlp_retrain_value_spin_{}layers.pt".format(num_layers))
                    print(f"New best reward: {reward_best} at epoch {ep + 1}, model saved.")
    wandb.log({"reward_best_normalized": reward_best})  # Use final w_s for final normalization
    wandb.finish()