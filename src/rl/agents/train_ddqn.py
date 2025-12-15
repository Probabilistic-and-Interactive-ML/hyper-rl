import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.rl.agents.buffers import OffPolicyBuffer
from src.rl.evaluation import log_atari_minigrid_stats
from src.rl.utils.train import linear_eps_schedule

from .ddqn import DDQNAgent


def train_ddqn(
    agent: DDQNAgent,
    envs: gym.vector.SyncVectorEnv,
    cfg: DictConfig,
    run_name: str,
    device: torch.device,
) -> None:
    """Train script for a DDQN agent."""

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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format("\n".join([f"|{key}|{value}|" for key, value in vars(cfg).items()])),
    )

    rb = OffPolicyBuffer(
        buffer_size=cfg.buffer_size,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device,
        n_envs=cfg.num_envs,
        handle_timeout_termination=False,
    )

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    obs, _ = envs.reset(seed=cfg.experiment.seed)

    # NOTE: This assumes one env
    autoreset = False
    smoothed_rewards = deque(maxlen=cfg.num_envs)
    smoothed_ep_length = deque(maxlen=cfg.num_envs)

    pbar = trange(cfg.total_timesteps, leave=True, ascii=True)
    # TRY NOT TO MODIFY: start the game
    for global_step in pbar:
        # ALGO LOGIC: put action logic here
        epsilon = linear_eps_schedule(cfg.start_e, cfg.end_e, cfg.exploration_fraction * cfg.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values, _ = agent.get_q_values(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # ALE and Minigrid properly handle termination and truncation
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        if not autoreset:
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        autoreset = np.logical_or(terminations, truncations)[0]

        # ALGO LOGIC: training.
        if global_step > cfg.learning_starts:
            if global_step % cfg.train_frequency == 0:
                data = rb.sample(cfg.batch_size)
                loss_dict, representation_metrics = agent.update(
                    obs=data.observations,
                    actions=data.actions,
                    rewards=data.rewards,
                    next_obs=data.next_observations,
                    dones=data.dones,
                    step=global_step,
                )

                # log training stats
                loss_dict.update({"global_step": global_step})
                loss_dict.update({"epsilon": epsilon})

                # Log encoder metrics only when representation metrics are computed
                if representation_metrics:
                    for k, v in representation_metrics.items():
                        writer.add_scalar(k, v, global_step)

                if infos and "episode" in infos:
                    smoothed_rewards, smoothed_ep_length = log_atari_minigrid_stats(
                        infos=infos,
                        writer=writer,
                        global_step=global_step,
                        smoothed_rewards=smoothed_rewards,
                        smoothed_ep_length=smoothed_ep_length,
                        num_envs=cfg.num_envs,
                    )

                    # Log loss only when logging rewards
                    for k, v in loss_dict.items():
                        writer.add_scalar(k, v, global_step)

            # update target network
            if global_step % cfg.target_network_frequency == 0:
                agent.update_target_network()

        sps = int(global_step / (time.time() - start_time))
        if global_step % 100 == 0 and smoothed_rewards and smoothed_ep_length:
            pbar.set_description(
                f"SPS: {sps} | Step: {global_step} | Episodic return: {np.mean(smoothed_rewards):.2f}"
                f" | Episode length: {np.mean(smoothed_ep_length):.2f}"
            )

    pbar.close()
    envs.close()
    if cfg.experiment.track:
        wandb.finish()
