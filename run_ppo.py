import logging
import time

import ale_py
import gymnasium as gym
import hydra
from omegaconf import DictConfig
from threadpoolctl import threadpool_limits

from src.rl.agents.ppo_discrete import DiscretePPOAgent
from src.rl.agents.train_ppo import train_ppo
from src.rl.environments import make_atari
from src.rl.environments.make_functions import make_minigrid, make_procgen
from src.rl.utils.train import set_cuda_configuration, set_seeds


@hydra.main(version_base=None, config_path="config", config_name="ppo")
def main(cfg: DictConfig) -> None:
    # Logging setup
    logging.basicConfig(level=cfg.logging_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Derived fields
    cfg.batch_size = int(cfg.num_envs * cfg.num_steps)
    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.num_iterations = cfg.total_timesteps // cfg.batch_size
    run_name = f"{cfg.env_id}__{cfg.experiment.exp_name}__{cfg.experiment.seed}__{int(time.time())}"
    cfg.experiment.run_name = run_name

    # Seeds and device
    set_seeds(cfg.experiment.seed, torch_deterministic=cfg.experiment.torch_deterministic)
    device = set_cuda_configuration(cfg.experiment.gpu)

    # Env setup
    if cfg.env_type == "procgen":
        envs = make_procgen(
            env_id=cfg.env_id,
            num_envs=cfg.num_envs,
            level_distribution=cfg.level_distribution,
            start_level=0,
            num_levels=cfg.num_levels,
            capture_video=cfg.experiment.capture_video,
            gamma=cfg.gamma,
            run_name=run_name,
        )
    elif cfg.env_type == "minigrid":
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

    agent = DiscretePPOAgent(
        env_type=cfg.env_type,
        envs=envs,
        gamma=cfg.gamma,
        num_steps=cfg.num_steps,
        gae_lambda=cfg.gae_lambda,
        batch_size=cfg.batch_size,
        minibatch_size=cfg.minibatch_size,
        update_epochs=cfg.update_epochs,
        clip_coef=cfg.clip_coef,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        target_kl=cfg.target_kl,
        norm_adv=cfg.norm_adv,
        compute_embedding_metrics=cfg.compute_embedding_metrics,
        manifold_cfg=cfg.manifold,
        encoder_cfg=cfg.encoder,
        actor_cfg=cfg.policy,
        critic_cfg=cfg.value_fn,
        optim_cfg=cfg.optimizer,
        device=device,
    ).to(device)

    with threadpool_limits(limits=cfg.experiment.num_threads, user_api="openmp"):
        train_ppo(agent=agent, envs=envs, cfg=cfg, run_name=run_name, env_type=cfg.env_type, device=device)

    # Hack to expose the run name for the hyperparameter tuner
    globals()["run_name"] = run_name


if __name__ == "__main__":
    main()
