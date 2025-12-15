import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.rl.evaluation import evaluate_agent, log_atari_minigrid_stats, log_procgen_stats
from src.rl.utils.serialization import save_ppo_agent, serialize_ppo_cfg
from src.types import EnvType


def train_ppo(
    agent,
    envs: gym.vector.SyncVectorEnv,
    cfg: DictConfig,
    run_name: str,
    env_type: EnvType,
    device: torch.device,
) -> None:
    """Train script for a PPO agent."""

    if cfg.experiment.track:
        import wandb

        wandb.init(
            name=run_name,
            project=cfg.experiment.wandb_project_name,
            entity=cfg.experiment.wandb_entity,
            tags=cfg.experiment.tag,
            sync_tensorboard=True,
            config=OmegaConf.to_container(cfg, resolve=True),
            save_code=False,
        )
    experiment_dir = Path(f"runs/{run_name}")
    writer = SummaryWriter(experiment_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format(
            "\n".join([f"|{key}|{value}|" for key, value in OmegaConf.to_container(cfg, resolve=True).items()])
        ),
    )

    # ALGO Logic: Storage setup
    obs = torch.zeros((cfg.num_steps, cfg.num_envs, *envs.single_observation_space.shape)).to(device)
    actions = torch.zeros((cfg.num_steps, cfg.num_envs, *envs.single_action_space.shape)).to(device)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    dones = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    values = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    if env_type == "procgen":
        next_obs = torch.Tensor(envs.reset()).to(device)
    else:
        # ALE and Minigrid use the newer gym interface
        next_obs, _ = envs.reset(seed=cfg.experiment.seed)
        next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(cfg.num_envs).to(device)

    smoothed_rewards = deque(maxlen=cfg.num_envs)
    smoothed_ep_length = deque(maxlen=cfg.num_envs)

    # Counters for model saving
    last_agent_save_step = 0

    if cfg.save_agent:
        serialize_ppo_cfg(cfg, experiment_dir)
        save_ppo_agent(agent=agent, save_path=experiment_dir, cfg=cfg, global_step=global_step)

    pbar = trange(1, cfg.num_iterations + 1, leave=True, ascii=True)
    for iteration in pbar:
        for step in range(0, cfg.num_steps):
            global_step += cfg.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # For categorial value function: Convert to scalar value
                action, value, _, logprob, *_ = agent.get_action_and_value(next_obs, categorical_value=False)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            if env_type == "procgen":
                next_obs, reward, next_done, infos = envs.step(action.cpu().numpy())
            else:
                # ALE and Minigrid properly handle termination and truncation
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Logging for Minigrid and Procgen differs because ProcGen uses the old gym interface
            if env_type == "procgen":
                smoothed_rewards, smoothed_ep_length = log_procgen_stats(
                    infos=infos,
                    writer=writer,
                    global_step=global_step,
                    smoothed_rewards=smoothed_rewards,
                    smoothed_ep_length=smoothed_ep_length,
                    num_envs=cfg.num_envs,
                )
            if (env_type == "atari" or env_type == "minigrid") and infos and "episode" in infos:
                smoothed_rewards, smoothed_ep_length = log_atari_minigrid_stats(
                    infos=infos,
                    writer=writer,
                    global_step=global_step,
                    smoothed_rewards=smoothed_rewards,
                    smoothed_ep_length=smoothed_ep_length,
                    num_envs=cfg.num_envs,
                )

        # Update the agent for update_epochs
        loss_dict = agent.update(
            obs=obs,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones=dones,
            values=values,
            next_obs=next_obs,
            next_done=next_done,
            envs=envs,
            device=device,
        )

        if cfg.save_agent and global_step >= last_agent_save_step + cfg.save_interval:
            # save_ppo_agent(agent=agent, save_path=experiment_dir, cfg=cfg, global_step=global_step)
            last_agent_save_step = global_step

        # Evaluate agent on train and test distribution
        if env_type == "procgen":
            with torch.inference_mode():
                agent.eval()
                evaluation_metrics = evaluate_agent(
                    agent=agent,
                    num_envs=cfg.eval_num_envs,
                    eval_train=cfg.eval_train,
                    eval_test=cfg.eval_test,
                    cfg=cfg,
                    is_final=False,
                    device=device,
                )
                loss_dict.update(evaluation_metrics)
                agent.train()

        sps = int(global_step / (time.time() - start_time))
        # Only update progress bar when queues are not empty
        if smoothed_rewards and smoothed_ep_length:
            pbar.set_description(
                f"SPS: {sps} | Step: {global_step} | Episodic return: {np.mean(smoothed_rewards):.2f}"
                f" | Episode length: {np.mean(smoothed_ep_length):.2f}"
            )
        loss_dict["charts/SPS"] = sps
        # Log all metrics
        for k, v in loss_dict.items():
            writer.add_scalar(k, v, global_step)

    # Final evaluation before exiting
    if env_type == "procgen":
        with torch.inference_mode():
            agent.eval()
            evaluation_metrics = evaluate_agent(
                agent=agent,
                num_envs=100,
                eval_train=True,
                eval_test=True,
                cfg=cfg,
                is_final=True,
                device=device,
            )
            for k, v in evaluation_metrics.items():
                writer.add_scalar(k, v, global_step)

    if cfg.save_agent:
        serialize_ppo_cfg(cfg=cfg, save_path=experiment_dir)
        save_ppo_agent(agent=agent, save_path=experiment_dir, cfg=cfg, global_step=global_step)

    pbar.close()
    envs.close()
    writer.close()
    if cfg.experiment.track:
        wandb.finish()
