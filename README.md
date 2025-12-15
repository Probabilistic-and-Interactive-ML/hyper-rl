# Understanding and Improving Hyperbolic Deep Reinforcement Learning

## Installation

1. **Install uv**: Download from [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv) or install via:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create virtual environment**:

   ```bash
   uv venv
   ```

3. **Install dependencies**:

   ```bash
   uv sync
   ```

4. **Make scripts executable**:

   ```bash
   chmod +x run_atari_paper.sh run_procgen_paper.sh
   ```

5. **Wandb tracking**:

   Replace `your-entity` in all config files with your wandb entity name:

   ```bash
   find config -name "*.yaml" -exec sed -i 's/your-entity/YOUR_ENTITY_NAME/g' {} +
   ```

   Replace `YOUR_ENTITY_NAME` with your actual wandb entity.

## Running Experiments

### Main Paper Results

- **Atari experiments**: `./run_atari_paper.sh`
- **Procgen experiments**: `./run_procgen_paper.sh`

### Configuration Folders

#### `config/atari_paper/`

Contains configs for Atari benchmark with DDQN:

- `hyperpp.yaml` - **HYPER++** (ours): Hyperboloid model with RMSNorm, learned scaling, and categorical loss
- `hyper_paper.yaml` - **Hyper+S-RYM**: Poincaré Ball with SpectralNorm and 1/√d scaling
- `euclidean.yaml` - **Euclidean baseline**: Standard Euclidean representations

To run a specific config:

```bash
uv run run_ddqn.py -cd=config/atari_paper -cn=hyperpp experiment.gpu=0 env_id="NameThisGameNoFrameskip-v4"
```

Atari-5 environments: `NameThisGameNoFrameskip-v4`, `PhoenixNoFrameskip-v4`, `BattleZoneNoFrameskip-v4`, `QbertNoFrameskip-v4`, `DobleDunkNoFrameskip-v4`

#### `config/procgen_paper/`

Contains configs for all 16 ProcGen environments with PPO:

- `hyperpp.yaml` - **HYPER++** (ours)
- `hyper_paper.yaml` - **Hyper+S-RYM**
- `euclidean_baseline.yaml` - **Euclidean baseline**

To run a specific config:

```bash
uv run run_ppo.py -cd=config/procgen_paper -cn=hyperpp experiment.gpu=0 env_id=bigfish
```

Available Procgen environments: `bigfish`, `bossfight`, `caveflyer`, `chaser`, `climber`, `coinrun`, `dodgeball`, `fruitbot`, `heist`, `jumper`, `leaper`, `maze`, `miner`, `ninja`, `plunder`, `starpilot`

#### `config/ppo_ablations/`

Ablation studies for individual HYPER++ components:

- `procgen_hyperpp.yaml` - Full HYPER++ (baseline for ablations)
- `procgen_hyperpp_no_rms.yaml` - Removing RMSNorm
- `procgen_hyperpp_noscale.yaml` - Removing learned scaling
- `procgen_hyperpp_nohlgauss.yaml` - Using MSE instead of categorical loss
- `procgen_hyperpp_poincare.yaml` - Using Poincaré Ball instead of Hyperboloid


```bash
uv run run_ppo.py -cd=config/ppo_ablations -cn=procgen_hyperpp_no_rms experiment.gpu=0 env_id=bigfish
```

### Key Parameters to Modify

Common parameters you might want to adjust:

**Environment & GPU:**

- `env_id=<name>` - Change environment (see lists above)
- `experiment.gpu=<id>` - GPU device ID (e.g., `0`, `1`, `2`, etc.)
- `experiment.track=<bool>` - Activate/deactivate wandb tracking

**Training:**

- `num_envs=<int>` - Number of parallel environments (default: 64 for Procgen)
- `total_timesteps=<int>` - Training duration (default: 25M for Procgen, 10M for Atari)
