import gym
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from procgen import ProcgenEnv


def _evaluate_single_distribution(
    agent: nn.Module,
    env,
    stochastic: bool,
    max_steps: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Helper function to evaluate agent on a single distribution of levels.
    Only counts rewards until the first termination for each environment.

    Args:
        agent: The reinforcement learning agent
        env: ProcGen environment instance
        max_steps: Maximum number of steps for evaluation

    Returns:
        Dictionary containing evaluation metrics for this distribution
    """
    episode_rewards = []
    episode_lengths = []

    obs = env.reset()
    current_rewards = np.zeros(env.num_envs)
    current_lengths = np.zeros(env.num_envs)
    first_termination = np.zeros(env.num_envs, dtype=bool)  # Track if environment has terminated
    active_envs = env.num_envs  # Count of environments still being evaluated

    for _ in range(max_steps):
        if active_envs == 0:  # All environments have terminated
            break

        # Convert observation to tensor and get agent action
        obs_tensor = torch.Tensor(obs).to(device)
        action, _ = agent.get_eval_action(obs_tensor, stochastic=stochastic)

        # Step environment
        obs, rewards, dones, infos = env.step(action.cpu().numpy())

        # Only accumulate rewards for environments that haven't terminated yet
        current_rewards += np.where(first_termination, 0, rewards)
        current_lengths += np.where(first_termination, 0, 1)

        # Handle terminations
        for i, done in enumerate(dones):
            if done and not first_termination[i]:
                first_termination[i] = True
                active_envs -= 1
                episode_rewards.append(current_rewards[i])
                episode_lengths.append(current_lengths[i])

    # Handle environments that didn't terminate within max_steps
    for i in range(env.num_envs):
        if not first_termination[i]:
            episode_rewards.append(current_rewards[i])
            episode_lengths.append(current_lengths[i])

    # Compute metrics
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_episode_length": np.mean(episode_lengths),
        "termination_rate": np.mean(first_termination),
    }

    return metrics


def evaluate_agent(
    agent: nn.Module,
    num_envs: int,
    eval_train: bool,
    eval_test: bool,
    cfg: DictConfig,
    is_final: bool,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluates a reinforcement learning agent on ProcGen environments across train
    and test distribution of levels.

    Args:
        agent: The reinforcement learning agent to evaluate
        cfg: Dictionary containing evaluation parameters including:
            - n_envs: Number of parallel environments
            - num_levels: Number of levels in the distribution
            - start_level: Starting level index
            - distribution_mode: Difficulty mode ('easy', 'hard', etc.)
            - env_name: Name of the ProcGen environment
            - max_steps: Maximum number of evaluation steps

    Returns:
        Dictionary containing evaluation metrics including mean rewards and
        completion rates for both training and test distributions
    """
    metrics = {}

    # Create training distribution environment
    if eval_train:
        train_envs = ProcgenEnv(
            num_envs=num_envs,
            env_name=cfg.env_id,
            start_level=0,
            num_levels=cfg.num_levels,
            distribution_mode=cfg.level_distribution,
        )
        train_envs = gym.wrappers.TransformObservation(train_envs, lambda obs: obs["rgb"])

        # Evaluate on training distribution
        train_metrics = _evaluate_single_distribution(
            agent=agent, env=train_envs, stochastic=cfg.stochastic_eval, max_steps=cfg.eval_max_steps, device=device
        )
        prefix = "train_final" if is_final else "train"
        metrics.update({f"{prefix}/{k}": v for k, v in train_metrics.items()})
        train_envs.close()

    if eval_test:
        # Create test distribution environment
        # Evaluation is done on all levels EXCLUDING the train levels (their setup uses start_level=0)
        test_envs = ProcgenEnv(
            num_envs=num_envs,
            env_name=cfg.env_id,
            start_level=0,
            num_levels=0,
            distribution_mode=cfg.level_distribution,
        )
        test_envs = gym.wrappers.TransformObservation(test_envs, lambda obs: obs["rgb"])

        # Evaluate on test distribution
        test_metrics = _evaluate_single_distribution(
            agent=agent, env=test_envs, stochastic=cfg.stochastic_eval, max_steps=cfg.eval_max_steps, device=device
        )

        prefix = "test_final" if is_final else "test"
        metrics.update({f"{prefix}/{k}": v for k, v in test_metrics.items()})
        test_envs.close()

    return metrics
