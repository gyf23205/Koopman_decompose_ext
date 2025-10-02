import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from models.MLP import CartpoleMLP
from envs.CartpoleSpin import CartPoleSpin

base = gym.make("CartPole-v1", render_mode="rgb_array")
env = CartPoleSpin(base)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

policy_net = CartpoleMLP(obs_dim, act_dim)
policy_net.load_state_dict(torch.load("saved_models_re/originals/ppo_cartpole_swing_policy_best.pt"))
policy_net.eval()

num_episodes = 20
episode_rewards = []
frames = []  # To store frames for the GIF
n_frames = []

for _ in range(num_episodes):
    f = []
    obs, info = env.reset(options={"low": -0.2, "high": 0.2})
    done = False
    total_reward = 0
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        logits = policy_net(obs_tensor)
        action = torch.argmax(logits).item()
        obs, reward, terminated, truncated, info = env.step(action)
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
ani.save("imgs_re/cartpole_all_episodes_spin.gif", writer="pillow", fps=30)
print("GIF saved as imgs_re/cartpole_all_episodes_spin.gif")