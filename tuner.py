"""Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/tuner_example.py"""

from ast import Dict
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal, get_args

import optuna
import procgen
import tyro

from src.rl.utils.hparam_tuner import Tuner

ENV_REWARD_RANGES = {
    "bigfish": [1, 25],  # 40 is the real upper bound
    "bossfight": [0.5, 13],
    "caveflyer": [3.5, 12],
    "chaser": [0.5, 13],
    "climber": [2, 12.6],
    "coinrun": [5, 10],
    "dodgeball": [1.5, 19],
    "fruitbot": [-1.5, 32.4],
    "heist": [3.5, 10],
    "jumper": [3, 10],
    "leaper": [3, 10],
    "maze": [5, 10],
    "miner": [1.5, 13],
    "ninja": [3.5, 10],
    "plunder": [2, 12.6],
    "starpilot": [2.5, 64],
}


@dataclass
class Args:
    agent: Literal["ppo", "dqn"] = "ppo"
    """ Type of RL agent. Used to infer the params to tune. """
    target_script: str = "run_ppo.py"
    """ Script to run the agent. """
    env_type: Literal["atari", "minigrid", "procgen"] = "procgen"
    """ Type of environment to run the agent on. Used to infer the params to tune. """
    environment: str = "dodgeball"
    """ Environment to run the agent. Used to infer max and min rewards. """
    gpu: int = 0
    """ GPU to run the agent on. """

    # Optuna config
    study_name: str = "HLGaussProcGen"
    """ Name of the Optuna study. """
    num_trials: int = 200
    """ Number of trials to run. """
    num_seeds: int = 3
    """ Number of seeds to run for each trial. """
    storage: str = "sqlite:///procgen_hlgauss_noreg.db"
    """ Storage URL for the Optuna study. """


def params_fn_ppo(trial: optuna.Trial, env: str) -> Dict:
    """Hyperparameter optimization config for a PPO agent."""
    if env in procgen.env.ENV_NAMES:
        return {
            "experiment.num_threads": 1,
            "optimizer.learning_rate": trial.suggest_float("optimizer.learning_rate", 0.0001, 0.005),
            "optimizer.adam_eps": trial.suggest_float("optimizer.adam_eps", 0.00001, 0.003),
            "optimizer.encoder_weight_decay": 0.0,
            "num_minibatches": 8,
            "update_epochs": 3,
            "num_steps": 256,
            "vf_coef": 0.5,
            "clip_coef": 0.2,
            "ent_coef": 0.01,
            "max_grad_norm": 0.5,
            "total_timesteps": 25_000_000,
            "num_envs": 64,
            "num_levels": 200,
            "level_distribution": "easy",
            "embedding_dim": 32,
            "shared_encoder": True,
            "last_layer_tanh": True,
            "policy.manifold": "hyperboloid",
            "policy.manifold_dtype": "float64",
            "policy.curvature": 1.0,
            "policy.feature_scaling": "learnable",
            "policy.regularization": "rms",
            "policy.forward_pass": "HNNpp_MLR",
            "policy.small_weights": False,
            "value_fn.manifold": "hyperboloid",
            "value_fn.manifold_dtype": "float64",
            "value_fn.curvature": 1.0,
            "value_fn.feature_scaling": "learnable",
            "value_fn.regularization": "rms",
            "value_fn.forward_pass": "HNNpp_MLR",
            "value_fn.small_weights": False,
            "value_fn.loss_fn": "hlgauss",
            "value_fn.loss_min_value": -10.0,
            "value_fn.loss_max_value": 10.0,
            "value_fn.loss_num_bins": 51,
            # Don't compute unnecessary metrics
            "eval_train": False,
            "eval_test": False,
            "stochastic_eval": True,
            "compute_embedding_metrics": False,
        }
    elif env == "atari":
        # TODO: Add parametes for Atari
        raise ValueError("Atari doesn't work yet.")
    elif "MiniGrid" in env:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 5e-4, 5e-1, log=True),
            "adam_eps": trial.suggest_float("adam_eps", 1e-5, 1e-3, log=True),
            "num-minibatches": 16,
            "update_epochs": 12,
            "num_steps": 128,
            "vf_coef": trial.suggest_float("vf_coef", 0, 3),
            "clip_coef": trial.suggest_float("clip_coef", 0, 1),
            # "ent-coef": trial.suggest_float("ent-coef", 0, 0.01),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0, 5),
            "total_timesteps": 100000,
            "num_envs": 32,
            "embedding_dim": 8,
            "vf_loss.fn": "hlgauss",
            "hyperbolic.curvature": 1.0,
            "hyperbolic.manifold": "poincare",
            "hyperbolic.regularization": "none",
            "hyperbolic.feature_scaling": "none",
            "hyperbolic.actor_forward_pass": "HNNpp_MLR",
            "hyperbolic.critic_forward_pass": "HNNpp_MLR",
        }
    else:
        raise ValueError(f"Unknown environment: {env}")


def params_fn_dqn(trial: optuna.Trial, env: str) -> Dict:
    """Hyperparameter optimization config for a DQN agent."""
    # TODO: Add parameters for Gridworld, Atari
    return {
        "learning-rate": trial.suggest_float("learning-rate", 0.0003, 0.003, log=True),
        "buffer-size": trial.suggest_categorical("buffer-size", [10000, 100000, 1000000]),
        "learning-starts": trial.suggest_categorical("learning-starts", [1000, 10000, 100000]),
        "target-update-freq": trial.suggest_categorical("target-update-freq", [100, 1000, 10000]),
        "exploration-fraction": trial.suggest_float("exploration-fraction", 0.1, 0.5),
        "exploration-final-eps": trial.suggest_float("exploration-final-eps", 0.01, 0.1),
        "total-timesteps": 1500000,
    }


def get_params_fn(agent: Literal["ppo", "dqn"], env: str) -> Callable[[optuna.Trial], Dict]:
    """Get the hyperparameter optimization function for the given agent and environment."""

    assert agent in get_args(Literal["ppo", "dqn"])

    if agent == "ppo":
        params_fn = partial(params_fn_ppo, env=env)
        return params_fn
    elif agent == "dqn":
        params_fn = partial(params_fn_dqn, env=env)
        return params_fn_dqn
    else:
        raise ValueError(f"Unknown agent: {agent}")


if __name__ == "__main__":
    args = tyro.cli(Args)

    tuner = Tuner(
        script=args.target_script,
        env_type=args.env_type,
        agent_type=args.agent,
        metric="test_final/mean_reward",
        metric_last_n_average_window=1,
        direction="maximize",
        aggregation_type="average",
        # TODO: This should be able to handle a list of environments
        target_scores={
            args.environment: ENV_REWARD_RANGES[args.environment],
        },
        # target_scores={args.environment: [0, 1]},
        params_fn=get_params_fn(args.agent, args.environment),
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=args.storage,
        # wandb_kwargs={"project": f"Hyperbolic RL Tuning | {args.agent}_{args.environment}"},
        study_name=args.study_name,
        gpu=args.gpu,
    )
    tuner.tune(
        num_trials=args.num_trials,
        num_seeds=args.num_seeds,
    )
