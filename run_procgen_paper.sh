#!/usr/bin/env bash
# Procgen PPO launcher using Hydra's JoblibLauncher for parallel multiruns.
# NOTE: If you don't use `uv`, replace `uv run` with `python` below.

CONFIG_DIR="config/procgen_paper"
CONFIG_NAME="hyperpp"
GPU_ID=1

set -euo pipefail

uv run run_ppo.py \
    -cd=$CONFIG_DIR -cn=$CONFIG_NAME experiment.gpu=$GPU_ID experiment.track=true \
    experiment.wandb_project_name="Hyperbolic ProcGen Metrics" experiment.tag=hyperpp \
    env_id=bigfish compute_embedding_metrics=true \
    hydra/launcher=joblib_jyn_rey +hydra/sweeps=procgen_seeds2
