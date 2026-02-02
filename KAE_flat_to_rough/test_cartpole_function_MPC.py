import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Autoencoder import *
from Autoencoder_functions import *
import itertools
import copy

import sys
sys.path.append("../src_re") 
from models.MLP import CartpoleMLP


def test_cartpole_kae_function_MPC(kae, observable_dim, padded_dim, p, device, T = 3, num_episodes = 20, save_imgs = False):
    # mode_number == -1: Reconstruction, otherwise literally desired mode's number
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    with torch.no_grad():
        episode_rewards = []
        frames = []  # To store frames for the GIF
        n_frames = []

        for _ in range(num_episodes):
            f = []
            obs, info = env.reset(options={'low':-0.2, 'high':0.2})
            done = False
            total_reward = 0
            best_seq, best_return = None, -1e9
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

                # MPC ver #####                
                for seq in itertools.product(range(observable_dim), repeat=T):
                    r, control_input = replace_policy_to_kae_MPC(env, seq, obs, obs_tensor, obs_dim, act_dim, observable_dim, padded_dim, kae, p, device)
                    if r > best_return:
                        best_return = r
                        best_seq = seq
                        best_control_input = control_input
                # print("best seq: "+str(best_seq)+" w best return : "+str(best_return))
                ########

                obs, reward, terminated, truncated, info = env.step(best_control_input)
                total_reward += reward
                done = terminated or truncated

                # Record the frame
                f.append(env.render())

            episode_rewards.append(total_reward)
            frames.append(f)
            n_frames.append(len(f))
        env.close()

        # Padding the shorter frames
        max_length = max(n_frames)
        for i in range(len(frames)):
            while len(frames[i]) < max_length:
                frames[i].append(frames[i][-1])  # Repeat the last frame to pad

    if save_imgs:
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
        ani.save("imgs/cartpole_MPC_.gif", writer="pillow", fps=30)
        print("GIF saved as imgs/cartpole_cartpole_MPC.gif")

    return(np.mean(episode_rewards))
    # print(f"Average reward over {num_episodes} episodes: {np.mean(episode_rewards)}")
    # print(f"Rewards per episode: {episode_rewards}")

def replace_policy_to_kae_MPC(env, seq, obs, obs_tensor, obs_dim, act_dim, observable_dim, padded_dim, model, p, device):
    """Simulate using Gym physics, but restore state afterward."""
    total_reward = 0.0
    done = False
    saved_state = np.copy(env.unwrapped.state)

    counter = 0
    for idx in seq:
        # action = controllers[idx](env.state)
        # stt_decompose_mode(kae, z, z_next, mode_number, p, propagation = True, conjugate = False)
        pad_in = torch.ones(padded_dim - obs_dim, device=device)
        aug_input = torch.cat([obs_tensor, pad_in], dim=0)
        _,z,_ = model(aug_input)
        logits = stt_decompose_mode(model, z,_, idx, p, propagation = True)
        logits = logits[:act_dim]
        logits_real = logits.real
        action = torch.argmax(logits_real).item()
        if counter == 0:
            control_input = action

        _, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        counter += 1 
        if terminated or truncated:
            done = True
            break

    env.unwrapped.state = np.copy(saved_state)
    return total_reward, control_input

