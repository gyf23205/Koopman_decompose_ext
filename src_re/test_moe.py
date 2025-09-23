import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from models.Autoencoder import KoopmanAutoencoder
from models.Autoencoder_functions import *
# from models.MLP import CartpoleMLP
from models.MoE import MoE
from envs.CartpoleSpin import CartPoleSpin
from utils import *
import sys
sys.path.append("/home/yifan/git/Koopman_decompose_ext/src_re/models") 


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
    base = gym.make("CartPole-v1", render_mode="rgb_array")
    # w_a = 0.0
    # w_c = 0.5
    # w_s = 1.2
    # env = CartPoleSpin(base, w_a, w_c, w_s)
    env = base
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    num_layers = 2

    n_dom_modes = 8
    padded_dim = 128
    kae_size = 48
    p = 3
    kae = KoopmanAutoencoder(padded_dim, kae_size, n_dom_modes, device).to(device)
    kae.load_state_dict(torch.load("saved_models_re/KAEs/KAE_state_dict_[8, 128, 48, 0.5, 0.4, 0.1, 3, 'CARTPOLE_v1']_2025-09-17.pt", weights_only=True, map_location=device))
    kae.eval()
    # params_traj = torch.tensor(np.load("saved_trajectory/param_trajectory_cartpole_mlp.npy"), device=device, dtype=torch.float32)
    # x_hat, z, z_pred = kae(params_traj)
    # N_O = z.shape[-1]
    # param_sub_all, eigvals = compute_theta_sub_all(kae, z, kae.K)
    moe = MoE(obs_dim, num_experts=n_dom_modes, num_layers=num_layers).to(device)
    moe.load_state_dict(torch.load("saved_models_re/MoEs/moe_cartpole_mlp_retrain_value_spin_{}layers.pt".format(num_layers)))
    moe.eval()
    # policy_net = CartpoleMLP(obs_dim, act_dim).to(device)

    num_episodes = 20
    episode_rewards = []
    frames = []  # To store frames for the GIF
    n_frames = []

    for _ in range(num_episodes):
        f = []
        obs, info = env.reset(options={'low': -0.2, 'high': 0.2})
        done = False
        total_reward = 0
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
            weights = torch.ones_like(weights)
            sparse_weights = weights  # Using original weights without top-k sparsification
            
            # # Top-2 gating: select top 2 experts and renormalize weights
            # top_k = 2
            # top_weights, top_indices = torch.topk(weights, top_k, dim=-1)
            # top_weights = torch.softmax(top_weights, dim=-1)  # Renormalize top-2 weights
            
            # # Create sparse weights tensor
            # sparse_weights = torch.zeros_like(weights)
            # sparse_weights.scatter_(-1, top_indices, top_weights)
            
            # # Compute entropy of the sparse weights
            # entropy = -torch.sum(sparse_weights * torch.log(sparse_weights + 1e-10), dim=-1).mean()

            logits = sparse_weights.view(1, 1, -1) @ experts_outputs
            action = torch.argmax(logits).item()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Record the frame
            f.append(env.render())
        episode_rewards.append(total_reward)
        # episode_rewards.append(total_reward/(w_a + w_c + w_s))
        frames.append(f)
        n_frames.append(len(f))
    env.close()

    # Padding the shorter frames
    max_length = max(n_frames)
    for i in range(len(frames)):
        while len(frames[i]) < max_length:
            frames[i].append(frames[i][-1])  # Repeat the last frame to pad

    print(f"Average reward over {num_episodes} episodes: {np.mean(episode_rewards)}")
    print(f"Rewards per episode: {episode_rewards}")

    # Save all frames as a single GIF with all episodes in subplots and timestamps
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))  # Create a 4x5 grid for 20 episodes
    axes = axes.flatten()  # Flatten the axes for easier indexing

    # Turn off axes for all subplots
    for ax in axes:
        ax.axis("off")

    # Initialize images and text for each subplot
    imgs = [ax.imshow(frames[i][0]) for i, ax in enumerate(axes)]
    texts = [
        ax.text(
            0.5, 0.9, "", color="white", fontsize=8, ha="center", va="center",
            transform=ax.transAxes, bbox=dict(facecolor="black", alpha=0.7, edgecolor="none")
        )
        for ax in axes
    ]

    def update(frame_idx):
        for i, (img, text) in enumerate(zip(imgs, texts)):
            img.set_data(frames[i][frame_idx])
            elapsed_time = frame_idx * (1 / 30)  # Assuming 30 FPS
            text.set_text(f"Time: {elapsed_time:.2f}s")
        return imgs + texts

    ani = FuncAnimation(fig, update, frames=max(n_frames), interval=33)  # ~30 FPS
    ani.save("imgs_re/cartpole_mlp_reconstruct.gif", writer="pillow", fps=30)
    print("GIF saved as imgs_re/cartpole_mlp_reconstruct.gif")