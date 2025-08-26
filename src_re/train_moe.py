import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from datetime import datetime

from matplotlib.animation import FuncAnimation
from models.Autoencoder import KoopmanAutoencoder
from models.Autoencoder_functions import *
from models.MLP import CartpoleMLP, ValueMLP
from models.MoE import MoE
from utils import *
from torch.nn.utils import parameters_to_vector

def ppo_update(dataset):
    for _ in range(update_epochs):
        for obs_tensor, act_tensor, logp_tensor, adv_tensor, ret_tensor in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            weights = moe(obs_tensor)
            loss = 0.0
            for i in range(batch_size):
                param_recon = torch.squeeze(param_sub_all[:, :n_dom_modes] @ weights[i, :].unsqueeze(1))
                logits = functional_forward(obs_tensor, param_recon, model_shape, param_shapes)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act_tensor)
                ratio = torch.exp(new_logp - logp_tensor)
                surr1 = ratio * adv_tensor
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv_tensor
                policy_loss = -torch.min(surr1, surr2).mean()

                loss += policy_loss
            
            moe_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            moe_optimizer.step()
            grad_norm = compute_grad_norm(moe)
            wandb.log({"MoE Grad Norm": grad_norm,
                       "MoE Loss": loss.item()})

def collect_trajectory(env, value_net, policy_net):
    obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
    obs, info = env.reset()
    timesteps = 0
    while timesteps < step_rollout:
        # Move obs to the same device as the MoE model
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        weights = moe(obs_tensor)
        param_sub_kae = torch.squeeze(param_sub_all[:, :n_dom_modes] @ weights.unsqueeze(1))
        nn.utils.vector_to_parameters(param_sub_kae, policy_net.parameters())
        action, log_prob, dist = select_action(obs, policy_net, device)
        value = value_net(torch.tensor(obs, dtype=torch.float32).to(device)).item()  # Move obs to the correct device
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
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    kae = KoopmanAutoencoder(load=True, load_path="saved_models_re/KAEs/kae_cartpole_mlp.pt").to(device)
    kae.eval()

    # Hyperparameters
    learning_rate = 3e-4
    ppo_epochs = 500
    step_rollout = 128
    update_epochs = 10
    batch_size = 64
    clip_epsilon = 0.2
    n_dom_modes = 32
    params_traj = torch.tensor(np.load("saved_trajectory_re/param_trajectory_cartpole_mlp.npy"), device=device, dtype=torch.float32)
    x_hat, z, z_pred = kae(params_traj)
    N_O = z.shape[-1]
    param_sub_all, eigvals = compute_theta_sub_all(kae, z, kae.K)
    # param_sub_kae = torch.sum(param_sub_all[:, :n_dom_modes], dim=1)
    policy_net = CartpoleMLP(obs_dim, act_dim).to(device)
    # nn.utils.vector_to_parameters(param_sub_kae, policy_net.parameters())

    num_episodes = 20
    episode_rewards = []
    frames = []  # To store frames for the GIF
    n_frames = []

    current_date = datetime.now().strftime("%Y-%m-%d")
    wandb.login(key='1888b9830153065d084181ffc29812cd1011b84b')
    wandb.init(project="Koopman_ext",
            name = str(current_date),
            config={
                "learning_rate": learning_rate,
                "ppo_epochs": ppo_epochs,
                "step_rollout": step_rollout,
                "update_epochs": update_epochs,
                "batch_size": batch_size,
                "clip_epsilon": clip_epsilon,
                "n_dom_modes": n_dom_modes
        })

    # networks
    policy_net = CartpoleMLP(obs_dim, act_dim).to(device)
    policy_net.eval()
    model_shape, param_shapes = extract_model_structure_and_shapes(policy_net)
    value_net = ValueMLP(obs_dim).to(device)
    value_net.load_state_dict(torch.load("saved_models_re/originals/ppo_cartpole_value.pt", map_location=device), strict=False)
    # Add "model." before all parameter names
    state_dict = value_net.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict["model." + k] = v
    value_net.load_state_dict(new_state_dict, strict=False)
    value_net.eval()

    moe = MoE(obs_dim, num_experts=n_dom_modes).to(device)
    moe_optimizer = optim.Adam(moe.parameters(), lr=learning_rate)
    schelduler = optim.lr_scheduler.StepLR(moe_optimizer, step_size=100, gamma=0.5, verbose=True)


    obs, info = env.reset()
    episode_rewards = []
    timesteps = 0
    params_trajectory = []
    for _ in tqdm(range(ppo_epochs), desc="Training MoE"):  # Use tqdm directly
        with torch.no_grad():
            # Collect trajectory
            obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, obs = collect_trajectory(env, value_net, policy_net)

        # Process trajectory
        dataset = process_trajectory(
            value_net, obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, obs, device
        )

        # PPO update
        ppo_update(dataset)

        schelduler.step()
        wandb.log({"Learning Rate": schelduler.get_last_lr()[0]})

    # Save MoE model
    torch.save(moe.state_dict(), "saved_models_re/MoEs/moe_cartpole_mlp.pt")