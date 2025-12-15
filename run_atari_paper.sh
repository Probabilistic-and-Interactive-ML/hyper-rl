#!/usr/bin/env bash
# Procgen DDQN launcher using Hydra's JoblibLauncher for parallel multiruns.
# NOTE: If you don't use `uv`, replace `uv run` with `python` below.

CONFIG_DIR="config/atari_paper"
CONFIG_NAME="hyper_paper" # hyper_paper, hyperpp
GPU_ID=0

set -euo pipefail

# PhoenixNoFrameskip-v4
# BattleZoneNoFrameskip-v4
# NameThisGameNoFrameskip-v4
# DoubleDunkNoFrameskip-v4
# QbertNoFrameskip-v4
uv run run_ddqn.py \
    -cd=$CONFIG_DIR -cn=$CONFIG_NAME experiment.gpu=$GPU_ID experiment.track=true \
    env_id="NameThisGameNoFrameskip-v4" \
    hydra/launcher=joblib_atari +hydra/sweeps=atari_seeds
