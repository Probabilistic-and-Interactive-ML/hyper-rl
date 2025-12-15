from typing import TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.vector import SyncVectorEnv
from omegaconf import DictConfig

from src.hyperbolic_math.src.manifolds import Euclidean, Hyperboloid
from src.rl.evaluation import compute_representation_metrics, log_actor_critic_metrics, log_encoder_metrics
from src.rl.utils.train import get_optimizer
from src.types import EnvType

from .networks import build_encoder, layer_init
from .utils import get_hyperbolic_layer, get_loss_fn, get_manifold_by_name

# Enables typing the agent like this: DiscretePPOAgent["atari"]
AgentTypes = TypeVar("T", bound=EnvType)


class DiscretePPOAgent[AgentTypes: EnvType](nn.Module):
    def __init__(
        self,
        env_type: EnvType,
        envs: SyncVectorEnv,
        gamma: float,
        num_steps: int,
        gae_lambda: float,
        batch_size: int,
        minibatch_size: int,
        update_epochs: int,
        clip_coef: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        target_kl: float,
        norm_adv: bool,
        compute_embedding_metrics: bool,
        manifold_cfg: DictConfig,
        encoder_cfg: DictConfig,
        actor_cfg: DictConfig,
        critic_cfg: DictConfig,
        optim_cfg: DictConfig,
        device: torch.device,
    ):
        super().__init__()

        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.update_epochs = update_epochs
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.loss_fn = get_loss_fn(critic_cfg, device)
        self.use_categorical_vf = critic_cfg.loss_fn == "hlgauss" or critic_cfg.loss_fn == "c51"
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.norm_adv = norm_adv

        self.env_type = env_type
        self.num_actions = envs.single_action_space.n
        self.embedding_dim = encoder_cfg.embedding_dim

        self.compute_embedding_metrics = compute_embedding_metrics

        # Single shared encoder
        self.encoder = build_encoder(env_type=env_type, encoder_cfg=encoder_cfg, curvature=manifold_cfg.curvature, envs=envs)

        # Handle categorical value function
        vf_output_dim = critic_cfg.loss_num_bins if self.use_categorical_vf else 1

        if manifold_cfg.manifold == "euclidean":
            self.manifold = Euclidean(dtype="float32")
            self.critic = layer_init(nn.Linear(self.embedding_dim, vf_output_dim))
            self.actor = layer_init(nn.Linear(self.embedding_dim, self.num_actions))
        else:
            manifold = get_manifold_by_name(manifold_cfg.manifold)
            self.manifold = manifold(torch.tensor([manifold_cfg.curvature]), False, dtype=manifold_cfg.manifold_dtype)
            self.critic = get_hyperbolic_layer(
                cfg=critic_cfg,
                manifold=self.manifold,
                fw_method=critic_cfg.forward_pass,
                embedding_dim=self.embedding_dim,
                output_dim=vf_output_dim,
            )
            self.actor = get_hyperbolic_layer(
                cfg=actor_cfg,
                manifold=self.manifold,
                fw_method=actor_cfg.forward_pass,
                embedding_dim=self.embedding_dim,
                output_dim=self.num_actions,
            )

        self.use_riemann = (manifold_cfg.manifold in {"poincare", "hyperboloid"}) and (
            "HRL_forward" in critic_cfg.forward_pass or "HRL_forward" in actor_cfg.forward_pass
        )

        self.optimizer = get_optimizer(
            encoder_params=self.encoder.parameters(),
            head_params=list(self.actor.parameters()) + list(self.critic.parameters()),
            cfg=optim_cfg,
            use_riemannian=self.use_riemann,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent embeddings.

        When encoders are shared, this returns the shared embedding.
        """
        hidden = self.encoder(x)
        if isinstance(self.manifold, Hyperboloid):
            hidden = torch.cat([torch.zeros_like(hidden[..., :1]), hidden], dim=-1)
        return hidden

    def get_eval_action(self, x: torch.Tensor, stochastic: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the best action for evaluating the agent."""
        # Use actor encoder for action selection
        hidden = self.encode(x)
        logits = self.actor(hidden)

        if stochastic:
            # Sample from policy distribution
            gumbel = -torch.empty_like(logits).exponential_().log()
            action = (logits + gumbel).argmax(dim=-1)
        else:
            # argmax action
            probs = F.softmax(logits, dim=-1)
            _, action = probs.max(dim=-1)
        return action, hidden

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        # Use critic encoder for value estimation
        hidden = self.encode(x)
        value = self.critic(hidden)
        if self.use_categorical_vf:
            # For categorical value function, we need to convert logits to probabilities
            value = self.loss_fn.logits_to_value(value)
        return value

    def get_action_and_value(
        self, x: torch.Tensor, action: Union[torch.Tensor, None] = None, categorical_value: bool = True
    ) -> tuple[torch.Tensor, ...]:
        # Use actor encoder for policy and critic encoder for value when not shared
        hidden = self.encode(x)
        logits = self.actor(hidden)
        value = self.critic(hidden)

        if self.use_categorical_vf and not categorical_value:
            # For categorical value function, we need to convert logits to probabilities
            value = self.loss_fn.logits_to_value(value)

        # Optimized action sampling w/o Categorical object
        if action is None:
            gumbel = -torch.empty_like(logits).exponential_().log()
            action = (logits + gumbel).argmax(dim=-1)

        logp_all = F.log_softmax(logits, dim=-1)
        logp = logp_all.gather(1, action.unsqueeze(-1)).squeeze(-1)
        probs = logp_all.exp()
        entropy = -(logp_all * probs).sum(dim=-1)

        return action, value, hidden, logp, entropy, logits

    @torch.no_grad()
    def calculate_adv_and_returns(
        self,
        next_obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_done: torch.Tensor,
        values: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate advantages and returns for PPO."""
        # bootstrap value if not done
        next_value = self.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        # Calculate discounted returns for num_steps length trajectories using a reverse loop for efficiency
        lastgaelam = 0
        # Performance implications should not be as big as for off-policy, though
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

        return advantages, returns

    def calculate_value_loss(self, newvalue: torch.Tensor, b_returns: torch.Tensor, mb_inds: torch.Tensor) -> torch.Tensor:
        """Calculate the value loss for a discrete PPO agent."""
        if self.use_categorical_vf:
            # Clip the returns to the min/max value of the loss function to prevent NaN
            target = torch.clamp(b_returns[mb_inds], self.loss_fn.min_value, self.loss_fn.max_value)
            v_loss = self.loss_fn(newvalue, target)
        else:
            # Handle different manifold dtype than float32
            target = b_returns[mb_inds].to(newvalue.dtype)
            v_loss = 0.5 * self.loss_fn(newvalue.squeeze(), target)

        return v_loss

    def calculate_policy_loss(self, b_advantages: torch.Tensor, mb_inds: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        """Calculate the policy loss for a discrete PPO agent."""

        mb_advantages = b_advantages[mb_inds]
        if self.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        return torch.max(pg_loss1, pg_loss2).mean()

    def calculate_entropy_loss(self, entropy: torch.Tensor) -> torch.Tensor:
        """Calculate the entropy loss for a discrete PPO agent."""
        return entropy.mean()

    @torch.inference_mode()
    def get_encoder_metrics(
        self,
        embedding: torch.Tensor,
        v_pred: torch.Tensor,
        entropy: torch.Tensor,
    ) -> dict[str, float]:
        """Compute encoder and embedding representation metrics.

        Args:
            actor_embedding: Embeddings from the actor encoder for the current batch.
            critic_embedding: Embeddings from the critic encoder for the current batch.
            newvalue: Value estimates corresponding to the current batch.
            entropy: Policy entropy values for the current batch.
            obs_batch: Observation batch (already sliced for the minibatch).

        Returns:
            A dictionary with encoder and representation metrics.
        """
        # Shared encoder, so both embeddings are the same
        encoder_metrics = log_encoder_metrics(self.encoder, panel_name="encoder")
        embedding_metrics = compute_representation_metrics(
            encoder=self.encoder,
            embedding=embedding,
            value_estimate=v_pred,
            entropy=entropy,
            manifold=self.manifold,
            encoder_name="encoder",
            compute_expensive_metrics=self.compute_embedding_metrics,
        )
        encoder_metrics.update(embedding_metrics)

        return encoder_metrics

    def take_optim_step(self, pg_loss, entropy_loss, v_loss) -> dict[str, float]:
        loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
        # Take optimization step
        self.optimizer.zero_grad()
        loss.backward()
        shared_grad_norms = nn.utils.clip_grad_norm_(self.parameters(), 1e8)
        shared_grad_norms_clipped = nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        grad_norms = {
            "losses/grad_norm": shared_grad_norms.item(),
            "losses/grad_norm_clipped": shared_grad_norms_clipped.item(),
        }

        return grad_norms

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_obs: torch.Tensor,
        next_done: torch.Tensor,
        envs,
        device: torch.device,
    ) -> dict[str, float]:
        """Update the policy agent."""
        # flatten the batch
        b_obs = obs.reshape((-1, *envs.single_observation_space.shape))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, *envs.single_action_space.shape))
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []

        for _ in range(self.update_epochs):
            # Recalculate advantages every epoch (See: https://openreview.net/forum?id=nIAxjsniDzg)
            advantages, returns = self.calculate_adv_and_returns(next_obs, rewards, dones, next_done, values, device)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                _, newvalue, hidden, newlogprob, entropy, logits = self.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                # Importance sampling ratios
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Unbiased KL estimation
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                # Policy loss
                pg_loss = self.calculate_policy_loss(b_advantages, mb_inds, ratio)
                # Entropy loss
                entropy_loss = self.calculate_entropy_loss(entropy)

                # Value loss
                v_loss = self.calculate_value_loss(newvalue, b_returns, mb_inds)

                # Take optimization step
                grad_norms = self.take_optim_step(pg_loss=pg_loss, entropy_loss=entropy_loss, v_loss=v_loss)

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        # Explained variation of the value function
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        agent_metrics = log_actor_critic_metrics(self.actor, "actor")
        critic_metrics = log_actor_critic_metrics(self.critic, "critic")
        agent_metrics.update(critic_metrics)

        if self.use_categorical_vf:
            # Newvalue are logits
            v_pred = self.loss_fn.logits_to_value(newvalue)[:, None]
        else:
            v_pred = newvalue
        encoder_metrics = self.get_encoder_metrics(
            embedding=hidden,
            v_pred=v_pred,
            entropy=entropy,
        )

        # Return information about the agent's training
        loss_dict = {
            "value_loss/loss": v_loss.item(),
            "value_loss/returns_max": b_returns.max().item(),
            "value_loss/returns_min": b_returns.min().item(),
            "value_loss/estimate_max": b_values.max().item(),
            "value_loss/estimate_min": b_values.min().item(),
            "value_loss/advantage_mean": b_advantages.mean().item(),
            "value_loss/explained_variance": explained_var,
            "policy_loss/loss": pg_loss.item(),
            "policy_loss/entropy": entropy.mean().item(),
            "policy_loss/entropy_variance": entropy.var().item(),
            "policy_loss/logits_min": logits.min().item(),
            "policy_loss/logits_max": logits.max().item(),
            "policy_loss/logits_norm": torch.linalg.norm(logits, dim=-1).mean().item(),
            "policy_loss/logits_variance": logits.var(dim=1).mean().item(),
            "policy_loss/old_approx_kl": old_approx_kl.item(),
            "policy_loss/approx_kl": approx_kl.item(),
            "policy_loss/clipfrac": np.mean(clipfracs),
            **grad_norms,
            **encoder_metrics,
            **agent_metrics,
        }

        # Log HLGauss value function logits statistics if applicable
        if self.use_categorical_vf:
            loss_dict["value_loss/vf_logits_min"] = newvalue.min().item()
            loss_dict["value_loss/vf_logits_max"] = newvalue.max().item()
            loss_dict["value_loss/vf_logits_norm"] = torch.linalg.norm(newvalue, dim=-1).mean().item()

        return loss_dict
