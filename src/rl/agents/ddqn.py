from typing import TypeVar

import torch
import torch.nn as nn
from gym.vector import SyncVectorEnv
from omegaconf import DictConfig

from src.hyperbolic_math.src.manifolds import Euclidean, Hyperboloid
from src.rl.evaluation import log_encoder_metrics
from src.rl.evaluation.logging import compute_representation_metrics
from src.rl.utils.train import get_optimizer
from src.types import EnvType

from .networks import build_encoder, layer_init
from .utils import get_hyperbolic_layer, get_loss_fn, get_manifold_by_name

# Enables typing the agent like this: DDQNAgent["atari"]
AgentTypes = TypeVar("T", bound=EnvType)


class QNetwork(nn.Module):
    def __init__(
        self,
        env_type: EnvType,
        envs: SyncVectorEnv,
        output_dim: int,
        num_actions: int,
        use_categorical_q: bool,
        encoder_cfg: DictConfig,
        manifold_cfg: DictConfig,
        q_cfg: DictConfig,
    ):
        super().__init__()
        self.env_type = env_type
        self.embedding_dim = encoder_cfg.embedding_dim
        self.output_dim = output_dim
        self.num_actions = num_actions
        self.use_categorical_q = use_categorical_q

        # Build the encoder
        self.encoder = build_encoder(env_type=env_type, encoder_cfg=encoder_cfg, curvature=manifold_cfg.curvature, envs=envs)

        # Build the Q-head: if using a hyperbolic manifold then build a hyperbolic head,
        # otherwise use a standard linear layer.
        if manifold_cfg.manifold in {"poincare", "hyperboloid"}:
            manifold = get_manifold_by_name(manifold_cfg.manifold)
            self.manifold = manifold(torch.tensor([manifold_cfg.curvature]), False, dtype=manifold_cfg.manifold_dtype)
            self.q_head = get_hyperbolic_layer(
                cfg=q_cfg,
                manifold=self.manifold,
                fw_method=q_cfg.forward_pass,
                embedding_dim=self.embedding_dim,
                output_dim=self.output_dim,
            )
        else:
            self.manifold = Euclidean(dtype="float32")
            self.q_head = layer_init(nn.Linear(self.embedding_dim, self.output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        if isinstance(self.manifold, Hyperboloid):
            embedding = torch.cat([torch.zeros_like(embedding[..., :1]), embedding], dim=-1)
        q_vals = self.q_head(embedding)
        if self.use_categorical_q:
            # Reshape to match the number of actions and bins
            q_vals = q_vals.view(q_vals.shape[0], self.num_actions, -1)
        return q_vals, embedding


class DDQNAgent[AgentTypes: EnvType](nn.Module):
    def __init__(
        self,
        env_type: EnvType,
        envs: SyncVectorEnv,
        gamma: float,
        tau: float,
        encoder_log_frequency: int,
        encoder_cfg: DictConfig,
        manifold_cfg: DictConfig,
        q_cfg: DictConfig,
        optim_cfg: DictConfig,
        compute_embedding_metrics: DictConfig,
        device: torch.device,
    ):
        super().__init__()
        self.gamma = gamma
        self.env_type = env_type
        self.num_actions = envs.single_action_space.n
        self.embedding_dim = encoder_cfg.embedding_dim
        self.tau = tau

        self.loss_fn = get_loss_fn(q_cfg, device)
        self.use_categorical_q = q_cfg.loss_fn == "hlgauss"

        self.encoder_log_frequency = encoder_log_frequency
        self.compute_embedding_metrics = compute_embedding_metrics

        q_output_dim = self.num_actions * q_cfg.loss_num_bins if self.use_categorical_q else self.num_actions
        self.q_network = QNetwork(
            env_type=env_type,
            envs=envs,
            output_dim=q_output_dim,
            num_actions=self.num_actions,
            use_categorical_q=self.use_categorical_q,
            encoder_cfg=encoder_cfg,
            manifold_cfg=manifold_cfg,
            q_cfg=q_cfg,
        )

        use_riemann = manifold_cfg.manifold in {"poincare", "hyperboloid"} and "HRL_forward" in q_cfg.forward_pass
        self.optimizer = get_optimizer(
            encoder_params=self.q_network.encoder.parameters(),
            head_params=self.q_network.q_head.parameters(),
            cfg=optim_cfg,
            use_riemannian=use_riemann,
        )

        self.target_network = QNetwork(
            env_type=env_type,
            envs=envs,
            output_dim=q_output_dim,
            num_actions=self.num_actions,
            use_categorical_q=self.use_categorical_q,
            encoder_cfg=encoder_cfg,
            manifold_cfg=manifold_cfg,
            q_cfg=q_cfg,
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_eval_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_values, embedding = self.q_network.encoder(x)
        if self.use_categorical_q:
            q_values = self.loss_fn.logits_to_value(q_values)
        _, action = torch.max(q_values, dim=-1)
        return action, embedding

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        q_values, embedding = self.q_network(x)
        if self.use_categorical_q:
            q_values = self.loss_fn.logits_to_value(q_values)
        return q_values, embedding

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ) -> dict[str, float]:
        """Update the Q-network using a batch of data."""

        with torch.no_grad():
            # Get value estimates from the target network
            target_vals, _ = self.target_network(next_obs)
            # Select actions through the policy network
            q_vals, _ = self.q_network(obs)
            if self.use_categorical_q:
                # Convert the probability distribution to value estimates
                target_vals = self.loss_fn.logits_to_value(target_vals)
                q_vals = self.loss_fn.logits_to_value(q_vals)
            policy_actions = q_vals.argmax(dim=1)
            target_max = target_vals[range(len(target_vals)), policy_actions]
            # Calculate Q-target
            td_target = rewards.flatten() + self.gamma * target_max * (1 - dones.flatten())

        old_q_values, embedding = self.q_network(obs)

        if self.use_categorical_q:
            old_val = old_q_values[range(len(old_q_values)), actions.squeeze(), :]
            # Clip the target to the min/max value of the loss function to prevent NaN
            td_target.clamp_(self.loss_fn.min_value, self.loss_fn.max_value)
        else:
            old_val = old_q_values[range(len(old_q_values)), actions.squeeze()]
        loss = self.loss_fn(old_val, td_target)

        self.optimizer.zero_grad()
        grad_norms = nn.utils.clip_grad_norm_(self.parameters(), 1e8)
        loss.backward()
        self.optimizer.step()

        loss_dict = {
            "losses/td_loss": loss,
            "losses/q_values": old_val.mean().item(),
            "losses/grad_norm": grad_norms,
        }

        representation_metrics = {}

        if step // self.encoder_log_frequency > (step - 1) // self.encoder_log_frequency:
            with torch.inference_mode():
                encoder_metrics = log_encoder_metrics(self.q_network.encoder, panel_name="encoder")
                embedding_metrics = compute_representation_metrics(
                    encoder=self.q_network.encoder,
                    embedding=embedding,
                    value_estimate=old_val[:, None] if not self.use_categorical_q else old_val,
                    entropy=None,
                    manifold=self.q_network.manifold,
                    encoder_name="encoder",
                    compute_expensive_metrics=self.compute_embedding_metrics,
                )
                representation_metrics.update(encoder_metrics)
                representation_metrics.update(embedding_metrics)

        return loss_dict, representation_metrics

    def update_target_network(self):
        for target_network_param, q_network_param in zip(
            self.target_network.parameters(), self.q_network.parameters(), strict=True
        ):
            target_network_param.data.copy_(self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data)
