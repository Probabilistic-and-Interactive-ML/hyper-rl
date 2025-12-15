import numpy as np
import torch
import torch.nn as nn
from gym.vector import SyncVectorEnv
from omegaconf import DictConfig
from torch.distributions import Categorical

from src.hyperbolic_math.src.manifolds.euclidean import Euclidean
from src.rl.agents.networks import layer_init
from src.rl.agents.ppo_discrete import DiscretePPOAgent
from src.rl.agents.utils import get_hyperbolic_layer
from src.rl.evaluation import log_actor_critic_metrics
from src.types import EnvType


def flatten01(arr):
    """Flatten first two dimensions of array (num_steps, num_envs, ...) -> (num_steps*num_envs, ...)."""
    return arr.reshape((-1, *arr.shape[2:]))


def unflatten01(arr, targetshape):
    """Unflatten first dimension back to (num_steps, num_envs, ...) shape."""
    return arr.reshape((*targetshape, *arr.shape[1:]))


class DiscretePPGAgent(DiscretePPOAgent):
    """PPG agent with auxiliary phase for deep value learning."""

    def __init__(
        self,
        # PPO parameters (inherited)
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
        # PPG-specific parameters
        n_iteration: int,
        e_policy: int,
        e_auxiliary: int,
        beta_clone: float,
        n_aux_grad_accum: int,
        num_aux_rollouts: int,
        aux_v_loss_scale: float,
    ):
        """Initialize PPG agent.

        Args:
            n_iteration: Number of policy phase iterations per phase
            e_policy: Epochs per policy update
            e_auxiliary: Epochs for auxiliary phase
            beta_clone: Behavior cloning coefficient
            n_aux_grad_accum: Gradient accumulation steps
            num_aux_rollouts: Auxiliary minibatch size (number of rollouts per batch)
        """
        # Initialize PPO components
        super().__init__(
            env_type=env_type,
            envs=envs,
            gamma=gamma,
            num_steps=num_steps,
            gae_lambda=gae_lambda,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            update_epochs=update_epochs,
            clip_coef=clip_coef,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            norm_adv=norm_adv,
            compute_embedding_metrics=compute_embedding_metrics,
            manifold_cfg=manifold_cfg,
            encoder_cfg=encoder_cfg,
            actor_cfg=actor_cfg,
            critic_cfg=critic_cfg,
            optim_cfg=optim_cfg,
            device=device,
        )

        # Store PPG-specific hyperparameters
        self.n_iteration = n_iteration
        self.e_policy = e_policy
        self.e_auxiliary = e_auxiliary
        self.beta_clone = beta_clone
        self.n_aux_grad_accum = n_aux_grad_accum
        self.num_aux_rollouts = num_aux_rollouts
        self.aux_v_loss_scale = aux_v_loss_scale

        # Build auxiliary critic
        self.aux_critic = self._build_aux_critic(critic_cfg)

        # Re-initialize optimizers to include aux_critic parameters
        self._reinitialize_optimizers(optim_cfg)

    def _build_aux_critic(self, critic_cfg: DictConfig) -> nn.Module:
        """Build auxiliary critic with same architecture as main critic.

        Returns:
            Auxiliary critic module
        """
        vf_output_dim = critic_cfg.loss_num_bins if self.use_categorical_vf else 1

        if isinstance(self.manifold, Euclidean):
            return layer_init(nn.Linear(self.embedding_dim, vf_output_dim))
        else:
            return get_hyperbolic_layer(
                cfg=critic_cfg,
                manifold=self.manifold,  # already initialized in parent
                fw_method=critic_cfg.forward_pass,
                embedding_dim=self.embedding_dim,
                output_dim=vf_output_dim,
            )

    def _reinitialize_optimizers(self, optim_cfg: DictConfig) -> None:
        """Re-initialize optimizers to include auxiliary critic parameters.

        CRITICAL: Must be called after super().__init__() because parent
        initializes optimizers without knowledge of aux_critic.
        """
        from src.rl.utils.train import get_optimizer

        self.optimizer = get_optimizer(
            encoder_params=self.encoder.parameters(),
            head_params=list(self.actor.parameters()) + list(self.critic.parameters()) + list(self.aux_critic.parameters()),
            cfg=optim_cfg,
            use_riemannian=self.use_riemann,
        )

    def policy_phase_update(
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
        """Policy phase update: Standard PPO optimization with e_policy epochs.

        PPG uses full-batch advantage normalization (adv_norm_fullbatch=True)
        as opposed to minibatch normalization. This matches the CleanRL reference.

        This method reimplements the PPO update loop with full-batch normalization
        instead of calling the parent's update() to avoid modifying the base PPO class.

        Args:
            obs: Observations
            actions: Actions taken
            logprobs: Log probabilities of actions
            rewards: Rewards received
            dones: Done flags
            values: Value estimates
            next_obs: Next observations
            next_done: Next done flags
            envs: Environments
            device: Device to use

        Returns:
            Dictionary of metrics with "policy/" prefix
        """
        # Flatten the batch
        b_obs = obs.reshape((-1, *envs.single_observation_space.shape))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, *envs.single_action_space.shape))
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []

        for _ in range(self.e_policy):
            # Recalculate advantages every epoch (See: https://openreview.net/forum?id=nIAxjsniDzg)
            advantages, returns = self.calculate_adv_and_returns(next_obs, rewards, dones, next_done, values, device)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)

            # PPG: Full-batch advantage normalization (adv_norm_fullbatch=True)
            # Normalize using statistics from the entire batch BEFORE splitting into minibatches
            if self.norm_adv:
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

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
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                # Policy loss (advantages already normalized at full-batch level)
                mb_advantages = b_advantages[mb_inds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Entropy loss
                entropy_loss = self.calculate_entropy_loss(entropy)

                # Value loss
                v_loss = self.calculate_value_loss(newvalue, b_returns, mb_inds)

                # Take optimization step
                grad_norms = self.take_optim_step(
                    pg_loss=pg_loss,
                    entropy_loss=entropy_loss,
                    v_loss=v_loss,
                )

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
            v_pred = self.loss_fn.probs_to_value(newvalue)[:, None]
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

        # Prefix metrics with "policy/" for clarity
        return loss_dict

    def auxiliary_phase_update(
        self,
        aux_obs: torch.Tensor,
        aux_returns: torch.Tensor,
        device: torch.device,
    ) -> dict[str, float]:
        """Auxiliary phase: Deep value function refinement with behavior cloning.

        Args:
            aux_obs: Stored observations [num_steps, aux_batch_rollouts, *obs_shape]
            aux_returns: Stored returns [num_steps, aux_batch_rollouts]
            device: Device to use

        Returns:
            Dictionary of metrics with "auxiliary/" prefix
        """
        # Calculate total number of rollouts stored (matches CleanRL reference)
        aux_batch_rollouts = aux_obs.shape[1]  # num_envs * n_iteration

        # Sequential indices for building old policy (matches CleanRL reference)
        aux_inds = np.arange(aux_batch_rollouts)

        # Build old policy incrementally with memory-efficient CPU offloading (matches CleanRL reference)
        # Shape: [num_steps, aux_batch_rollouts, num_actions]
        aux_pi = torch.zeros((self.num_steps, aux_batch_rollouts, self.num_actions))

        with torch.no_grad():
            for i, start in enumerate(range(0, aux_batch_rollouts, self.num_aux_rollouts)):
                end = start + self.num_aux_rollouts
                aux_minibatch_ind = aux_inds[start:end]

                # Extract subset: (num_steps, minibatch_size, *obs_shape)
                m_aux_obs = aux_obs[:, aux_minibatch_ind].to(torch.float32).to(device)
                m_obs_shape = m_aux_obs.shape
                # Flatten to (num_steps * minibatch_size, *obs_shape)
                m_aux_obs = flatten01(m_aux_obs)

                # Get policy logits
                hidden = self.encode(m_aux_obs)
                pi_logits = self.actor(hidden).to(torch.float32).cpu().clone()

                # Unflatten back to (num_steps, minibatch_size, num_actions) and store
                aux_pi[:, aux_minibatch_ind] = unflatten01(pi_logits, m_obs_shape[:2])
                del m_aux_obs

        # Auxiliary training loop (matches CleanRL reference)
        all_losses = []
        for auxiliary_update in range(1, self.e_auxiliary + 1):
            # Shuffle indices each epoch (matches CleanRL reference)
            np.random.shuffle(aux_inds)

            for i, start in enumerate(range(0, aux_batch_rollouts, self.num_aux_rollouts)):
                end = start + self.num_aux_rollouts
                aux_minibatch_ind = aux_inds[start:end]

                # Extract minibatch: (num_steps, minibatch_size, *obs_shape)
                m_aux_obs = aux_obs[:, aux_minibatch_ind].to(torch.float32).to(device)
                m_obs_shape = m_aux_obs.shape
                # Flatten to (num_steps * minibatch_size, *obs_shape)
                m_aux_obs = flatten01(m_aux_obs)

                # Extract returns and flatten
                m_aux_returns = aux_returns[:, aux_minibatch_ind].to(torch.float32).to(device)
                m_aux_returns = flatten01(m_aux_returns)

                # Forward passes
                hidden = self.encode(m_aux_obs)
                # CRITICAL: Detach main critic to prevent encoder gradients
                new_value = self.critic(hidden.detach()).squeeze()
                new_aux_value = self.aux_critic(hidden).squeeze()

                new_logits = self.actor(hidden)

                # Compute KL loss using pre-computed old policy logits
                old_pi_logits = flatten01(aux_pi[:, aux_minibatch_ind]).to(device)
                old_pi = Categorical(logits=old_pi_logits)
                new_pi = Categorical(logits=new_logits)
                kl_loss = torch.distributions.kl_divergence(old_pi, new_pi).mean()

                # Compute value losses
                if self.use_categorical_vf:
                    mb_returns_clamped = torch.clamp(m_aux_returns, self.loss_fn.min_value, self.loss_fn.max_value)
                    real_value_loss = self.aux_v_loss_scale * self.loss_fn(new_value, mb_returns_clamped)
                    aux_value_loss = self.aux_v_loss_scale * self.loss_fn(new_aux_value, mb_returns_clamped)
                else:
                    real_value_loss = 0.5 * ((new_value - m_aux_returns) ** 2).mean()
                    aux_value_loss = 0.5 * ((new_aux_value - m_aux_returns) ** 2).mean()

                # Joint loss (matches CleanRL reference)
                joint_loss = aux_value_loss + self.beta_clone * kl_loss
                loss = (joint_loss + real_value_loss) / self.n_aux_grad_accum

                # Backward pass (gradients accumulate)
                loss.backward()

                # Step optimizer with gradient accumulation (matches CleanRL reference)
                if (i + 1) % self.n_aux_grad_accum == 0:
                    nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Log losses (unscaled for clarity)
                all_losses.append(
                    {
                        "kl_loss": kl_loss.item(),
                        "real_value_loss": real_value_loss.item(),
                        "aux_value_loss": aux_value_loss.item(),
                        "total_loss": loss.item() * self.n_aux_grad_accum,
                    }
                )

        # Aggregate and return metrics
        avg_losses = {f"auxiliary/{k}": np.mean([l[k] for l in all_losses]) for k in all_losses[0].keys()}

        return avg_losses
