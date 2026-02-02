# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_koopman.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --num_env=100 --model=temp1

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# MODIFICATIONS #####################
sys.path.append("/home/joonwon/github/Koopman_decompose_ext/KAE")
# my_model_path = "/home/joonwon/github/Koopman_decompose_ext/KAE/waypoints/temp_2.pth"
# my_model_folder = "/home/joonwon/github/Koopman_decompose_ext/KAE/waypoints/"
from Autoencoder import *
from Autoencoder_functions import *

# KAE params
# p = 1
# padded_dim = 512
obs_dim = 235
# observable_dim = 364
act_dim = 12
# normalize_on = True

# import gymnasium as gym
# from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
# from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--model", type=str, default=None, help="Koopman model.")
parser.add_argument("--pad_dim", type=int, default=512, help="Padded dimension.")
parser.add_argument("--obv_dim", type=int, default=512, help="Observable dimension.")
parser.add_argument("--p", type=int, default=1, help="p value.")
parser.add_argument("--normalizer_on", action="store_true", default=False, help="normalizer.")
parser.add_argument("--koopman_recon", action="store_true", default=False, help="reconstruction using Koopman itself.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # LOAD MODEL PART - MODIFIED ######################
    model_file_name = args_cli.model
    if model_file_name == "original":
        my_model_folder = "/home/joonwon/github/Koopman_decompose_ext/KAE/results/"
        my_model_path = my_model_folder + "walking_policy_new.pth"
        my_model_dict = torch.load(my_model_path, weights_only = False)
        my_model = my_model_dict["actor"]
    else:
        my_model_folder = "/home/joonwon/github/Koopman_decompose_ext/KAE/waypoints/"
        my_model_path = my_model_folder + model_file_name + '.pth'
        my_model = torch.load(my_model_path, weights_only = False)
    #my_model = my_model.to(env_cfg.sim.device)
    my_model = my_model.to('cuda:0')
    my_model.eval()
    print("[DEBUG] Koopman model loaded successfully and set to eval().")

    print("[DEBUG] Koopman model successfully loaded.")
    print("[DEBUG] Koopman model successfully loaded.")
    print("[DEBUG] Koopman model successfully loaded.")
    print("[DEBUG] Koopman model successfully loaded.")
    print("[DEBUG] Koopman model successfully loaded.")
    print("[DEBUG] Koopman model successfully loaded.")
    print("[DEBUG] Koopman model successfully loaded.")
    print("[DEBUG] Koopman model successfully loaded.")
    print("[DEBUG] Koopman model successfully loaded.")
    print("[DEBUG] Koopman model successfully loaded.")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    def my_policy(my_model, p, observable_dim, z, z_next, koopman_on):
        """
        Convert IsaacLab observation to tensor, run your model, return actions.
        Modify this if your model expects a specific input format.
        """
            # actions = my_model(obs_tensor)

        if koopman_on:
            K = my_model.K
            # latent_prop = (K**p)@z
            # latent_prop = torch.matmul(z, (my_model.K**p).T)
            latent_prop = z @ ((K**p).T)
            y_hat = my_model.decoder(latent_prop)
            actions = y_hat[:,:act_dim]
        else:
            # print('wrong')
            actions = stt_decompose_reconstruction_isaac(my_model, z, z_next, observable_dim, p, act_dim, propagation = True)

        # print(actions)
        # print(actions.size())

        return actions

    def my_policy_original(obs, my_model):

        print("INTENTIONALLY DISABLED")
        actions = my_model(obs)
        print(actions.size())
        # print(actions)

        return actions

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    #######################################################

    dt = env.unwrapped.step_dt
    
    # reset environment
    # obs = env.get_observations()

    obs, _ = env.reset()    
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        padded_dim = args_cli.pad_dim
        observable_dim = args_cli.obv_dim

        p = args_cli.p
        normalize_on = args_cli.normalizer_on
        koopman_on = args_cli.koopman_recon
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # actions = policy(obs)

            # env stepping
            # obs, _, _, _ = env.step(actions)

            ##### MODIFY HERE #####
            obs_tensor_bf = torch.as_tensor(obs["policy"], dtype=torch.float32, device=env_cfg.sim.device)
            # print(torch.max(obs_tensor,axis=0))
            num_envs, obs_dim = obs_tensor_bf.shape
            obs_tensor_af = normalizer(obs_tensor_bf)
            
            if model_file_name == "original":
                padded_dim = obs_dim
                if normalize_on:
                    actions = my_policy_original(obs_tensor_af, my_model)
                else:
                    actions = my_policy_original(obs_tensor_bf, my_model)

            else:

                pad_obs = torch.ones((num_envs, padded_dim - obs_dim), device=env_cfg.sim.device)
                if normalize_on:
                    aug_obs = torch.cat([obs_tensor_af, pad_obs], dim=1)
                else: 
                    aug_obs = torch.cat([obs_tensor_bf, pad_obs], dim=1)
                
                x_hat, z, _ = my_model(aug_obs)                   # z: [num_envs, latent_dim]
                # z_next = torch.matmul(z, my_model.K.T)        # batched Koopman propagation
                z_next = z @ (my_model.K).T
                if normalize_on:
                    actions = my_policy(my_model, p, observable_dim, z, z_next, koopman_on)
                else:
                    actions = my_policy(my_model, p, observable_dim, z, z_next, koopman_on)

            # import pickle
            # # print(actions.size())
            # save_obs = "/home/joonwon/results/obs_log.pkl"
            # save_act = "/home/joonwon/results/act_log.pkl"
            # actions_record = actions.detach().cpu()
            # obs_record = obs_tensor_bf.detach().cpu()

            # with open(save_obs, "ab") as f:
            #     pickle.dump(obs_record, f)
            #     f.flush()
            #     os.fsync(f.fileno())

            # print("obs record: "+str(obs_record.size()))

            # with open(save_act, "ab") as f:
            #     pickle.dump(actions_record, f)
            #     f.flush()
            #     os.fsync(f.fileno())

            # print("act record: "+str(actions_record.size()))

            # env stepping
            obs, _, _, _ = env.step(actions)
            #######################

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()  
    simulation_app.close()
    print("[DEBUG] closed app")
