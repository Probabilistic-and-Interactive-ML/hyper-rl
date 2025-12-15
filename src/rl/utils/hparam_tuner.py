"""Adapated from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/tuner.py"""

import os
import runpy
import sys
import time
from collections.abc import Callable
from typing import Any, Literal, Optional

import numpy as np
import optuna
from rich import print
from tensorboard.backend.event_processing import event_accumulator

import wandb


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Tuner:
    def __init__(
        self,
        script: str,
        env_type: Literal["atari", "minigrid", "procgen"],
        agent_type: Literal["ppo", "dqn"],
        metric: str,
        target_scores: dict[str, Optional[list[float]]],
        params_fn: Callable[[optuna.Trial], dict],
        direction: str = "maximize",
        aggregation_type: str = "average",
        metric_last_n_average_window: int = 50,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        storage: str = "sqlite:///hyperbolic_rl_hpopt.db",
        study_name: str = "",
        wandb_kwargs: dict[str, Any] = {},
        gpu: int = -1,
    ) -> None:
        self.script = script
        self.env_type = env_type
        self.agent_type = agent_type
        self.metric = metric
        self.target_scores = target_scores
        if len(self.target_scores) > 1:
            if None in self.target_scores.values():
                raise ValueError(
                    "If there are multiple environments, the target scores must be specified for each environment."
                )

        self.params_fn = params_fn
        self.direction = direction
        self.aggregation_type = aggregation_type
        if self.aggregation_type == "average":
            self.aggregation_fn = np.average
        elif self.aggregation_type == "median":
            self.aggregation_fn = np.median
        elif self.aggregation_type == "max":
            self.aggregation_fn = np.max
        elif self.aggregation_type == "min":
            self.aggregation_fn = np.min
        else:
            raise ValueError(f"Unknown aggregation type {self.aggregation_type}")
        self.metric_last_n_average_window = metric_last_n_average_window
        self.pruner = pruner
        self.sampler = sampler
        self.storage = storage
        self.study_name = study_name
        if len(self.study_name) == 0:
            self.study_name = f"tuner_{int(time.time())}"
        self.wandb_kwargs = wandb_kwargs

        self.gpu = gpu

    def tune(self, num_trials: int, num_seeds: int) -> None:
        def objective(trial: optuna.Trial):
            params = self.params_fn(trial)
            params[f"envs/{self.agent_type}"] = self.env_type
            run = None
            if len(self.wandb_kwargs.keys()) > 0:
                run = wandb.init(
                    **self.wandb_kwargs,
                    config=params,
                    name=f"{self.study_name}_{trial.number}",
                    group=self.study_name,
                    save_code=True,
                    reinit=True,
                )

            algo_command = [f"{key}={value}" for key, value in params.items()]
            normalized_scoress = []
            for seed in range(num_seeds):
                normalized_scores = []
                for env_id in self.target_scores.keys():
                    sys.argv = [
                        *algo_command,
                        f"env_id={env_id}",
                        f"experiment.seed={seed}",
                        f"experiment.gpu={self.gpu}",
                    ]
                    with HiddenPrints():
                        experiment = runpy.run_path(path_name=self.script, run_name="__main__")

                    # read metric from tensorboard
                    ea = event_accumulator.EventAccumulator(f"runs/{experiment.get('run_name')}")
                    ea.Reload()
                    metric_values = [
                        scalar_event.value for scalar_event in ea.Scalars(self.metric)[-self.metric_last_n_average_window :]
                    ]
                    print(
                        f"The average episodic return on {env_id} is {np.average(metric_values)} "
                        f"averaged over the last {self.metric_last_n_average_window} episodes."
                    )
                    if self.target_scores[env_id] is not None:
                        normalized_scores += [
                            (np.average(metric_values) - self.target_scores[env_id][0])
                            / (self.target_scores[env_id][1] - self.target_scores[env_id][0])
                        ]
                    else:
                        normalized_scores += [np.average(metric_values)]
                    if run:
                        run.log({f"{env_id}_return": np.average(metric_values)})

                normalized_scoress += [normalized_scores]
                aggregated_normalized_score = self.aggregation_fn(normalized_scores)
                print(f"The {self.aggregation_type} normalized score is {aggregated_normalized_score} with num_seeds={seed}")
                trial.report(aggregated_normalized_score, step=seed)
                if run:
                    run.log({"aggregated_normalized_score": aggregated_normalized_score})
                if trial.should_prune():
                    if run:
                        run.finish(quiet=True)
                    raise optuna.TrialPruned()

            if run:
                run.finish(quiet=True)
            return np.average(
                self.aggregation_fn(normalized_scoress, axis=1)
            )  # we alaways return the average of the aggregated normalized scores

        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=self.storage,
            pruner=self.pruner,
            sampler=self.sampler,
            load_if_exists=True,
        )
        print("==========================================================================================")
        print("run another tuner with the following command:")
        print(f"uv run tuner.py --study-name {self.study_name}")
        print("==========================================================================================")
        study.optimize(objective, n_trials=num_trials)
        print(f"The best trial obtains a normalized score of {study.best_trial.value}", study.best_trial.params)
        return study.best_trial
