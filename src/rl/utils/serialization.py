from pathlib import Path

import gymnasium
import torch
from omegaconf import DictConfig, OmegaConf

from ..agents.ppo_discrete import DiscretePPOAgent


def save_ppo_agent(agent: DiscretePPOAgent, save_path: str | Path, cfg: DictConfig, global_step: int) -> None:
    """Save the complete agent. Does not store hyperparameters."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    model_filename = (
        f"{cfg.experiment.seed}_{cfg.policy.manifold}_{cfg.value_fn.manifold}_PPO_agent_{cfg.env_id}_{global_step}.pt"
    )
    full_save_path = save_path / model_filename
    torch.save(agent.state_dict(), full_save_path)


def load_ppo_agent(
    save_path: str | Path,
    cfg: DictConfig,
    envs: gymnasium.vector.SyncVectorEnv,
    device: torch.device,
) -> DiscretePPOAgent:
    """Load a saved agent."""
    save_path = Path(save_path)  # Ensure it's a Path object
    if save_path.is_dir():  # If a directory is given, assume the model is 'agent_model.pt' inside it
        model_path = save_path / "agent_model.pt"
    else:  # Assume save_path is the full path to the model file
        model_path = save_path

    if not model_path.exists():
        raise FileNotFoundError(f"Agent model not found at {model_path}")

    # Instantiate the agent
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
        embedding_dim=cfg.embedding_dim,
        shared_encoder=cfg.shared_encoder,
        last_layer_tanh=cfg.last_layer_tanh,
        feat_reg_coef=cfg.feat_reg_coef,
        compute_embedding_metrics=cfg.compute_embedding_metrics,
        actor_cfg=cfg.policy,
        critic_cfg=cfg.value_fn,
        optim_cfg=cfg.optimizer,
        device=device,
    ).to(device)

    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.to(device)  # Ensure agent is on the correct device
    return agent


def serialize_ppo_cfg(cfg: DictConfig, save_path: str | Path):
    """Serialize the config to a JSON file."""
    OmegaConf.save(cfg, Path(save_path) / "config.yaml", resolve=True)


def deserialize_ppo_cfg(path: str | Path) -> DictConfig:
    """Load the config from a JSON file."""
    return OmegaConf.load(Path(path) / "config.yaml")
