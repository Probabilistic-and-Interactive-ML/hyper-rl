import logging
import time

import ale_py
import gymnasium as gym
import hydra
from omegaconf import DictConfig

from src.rl.agents.ddqn import DDQNAgent
from src.rl.agents.train_ddqn import train_ddqn
from src.rl.environments import make_atari
from src.rl.environments.make_functions import make_minigrid
from src.rl.utils.train import set_cuda_configuration, set_seeds


@hydra.main(version_base=None, config_path="config", config_name="dqn")
def main(cfg: DictConfig) -> None:
    # Logging setup
    logging.basicConfig(level=cfg.logging_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    run_name = f"{cfg.env_id}__{cfg.experiment.exp_name}__{cfg.experiment.seed}__{int(time.time())}"
    cfg.experiment.run_name = run_name

    # Seeds and device
    set_seeds(cfg.experiment.seed, torch_deterministic=cfg.experiment.torch_deterministic)
    device = set_cuda_configuration(cfg.experiment.gpu)

    # Env setup
    if cfg.env_type == "minigrid":
        envs = gym.vector.SyncVectorEnv(
            [
                make_minigrid(
                    env_id=cfg.env_id,
                    dense_reward=cfg.dense_reward,
                    disable_orientation=cfg.disable_orientation,
                    idx=i,
                    capture_video=cfg.experiment.capture_video,
                    run_name=run_name,
                )
                for i in range(cfg.num_envs)
            ],
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    elif cfg.env_type == "atari":
        gym.register_envs(ale_py)
        envs = gym.vector.SyncVectorEnv(
            [make_atari(cfg.env_id, i, cfg.experiment.capture_video, run_name) for i in range(cfg.num_envs)],
        )
    else:
        raise ValueError(f"Unknown env_type: {cfg.env_type}")

    agent = DDQNAgent(
        env_type=cfg.env_type,
        envs=envs,
        gamma=cfg.gamma,
        tau=cfg.tau,
        encoder_log_frequency=cfg.encoder_log_frequency,
        encoder_cfg=cfg.encoder,
        manifold_cfg=cfg.manifold,
        q_cfg=cfg.value_fn,
        optim_cfg=cfg.optimizer,
        compute_embedding_metrics=cfg.compute_embedding_metrics,
        device=device,
    ).to(device)

    train_ddqn(agent=agent, envs=envs, cfg=cfg, run_name=run_name, device=device)


if __name__ == "__main__":
    main()
