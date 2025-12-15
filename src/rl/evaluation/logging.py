from collections import deque
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
import torch.linalg as la
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.hyperbolic_math.src.manifolds import Euclidean, Hyperboloid, Manifold, PoincareBall
from src.hyperbolic_math.src.nn_layers import HyperbolicRegressionPoincare, HyperbolicRegressionPoincareHDRL
from src.hyperbolic_math.src.utils.helpers import get_delta
from src.hyperbolic_math.src.utils.math_utils import atanh


def compute_feature_metrics(
    y_pred: torch.Tensor,
    embeddings: torch.Tensor,
    manifold: Manifold,
    sample_size: int | None = None,
    add_eps: float = 1e-2,
) -> torch.Tensor:
    # Ensure shapes
    y = y_pred.view(-1)
    n = y.shape[0]

    assert n > 1, "At least two samples are required to compute feature diversity."
    assert embeddings.shape[0] == n, "Embeddings must match the number of predictions."

    if sample_size is not None and n > sample_size:
        # Randomly sample a subset of points to compute the metrics on
        idx = torch.randperm(n, device=y.device)[:sample_size]
        y = y[idx]
        embeddings = embeddings[idx]
        n = sample_size

    # Condensed pairwise distances for predictions using pdist
    dv = torch.pdist(y.view(n, 1), p=2)
    # Maximum distance for predictions
    dv_max = dv.max()

    # TODO: Check that this is correct for hyperbolic spaces
    # Condensed pairwise distances for embeddings
    if isinstance(manifold, Euclidean):
        ds = torch.pdist(embeddings, p=2)
    elif isinstance(manifold, PoincareBall):
        # Use index combinations to vectorize geodesic distances without NxN allocation
        idx = torch.combinations(torch.arange(n, device=embeddings.device), r=2, with_replacement=False)
        xi = embeddings.index_select(0, idx[:, 0])
        xj = embeddings.index_select(0, idx[:, 1])
        ds = manifold.dist(xi, xj, axis=-1).squeeze(-1).view(-1)
    elif isinstance(manifold, Hyperboloid):
        # Use index combinations to vectorize geodesic distances without NxN allocation
        idx = torch.combinations(torch.arange(n, device=embeddings.device), r=2, with_replacement=False)
        xi = embeddings.index_select(0, idx[:, 0])
        xj = embeddings.index_select(0, idx[:, 1])
        ds = manifold.dist(xi, xj, axis=-1).squeeze(-1).view(-1)

    # DIVERSITY
    ds_max = ds.max()

    # Normalize condensed vectors and compute ratio
    dv_n = dv / (dv_max + 1e-8)
    ds_n = ds / (ds_max + 1e-8)
    ratio = (dv_n / (ds_n + add_eps)).clamp(max=1.0)

    # Convert mean over i<j to mean over all i,j (diagonal contributes zeros)
    # sum_upper = ratio.sum()
    # diversity = 1.0 - (2.0 * sum_upper) / (n * n)
    diversity = 1.0 - ratio.mean()

    # COMPLEXITY REDUCTION
    # L_rep := mean_{i<j} (dv/ds), L_max := max_{i<j} (dv/ds), Score := 1 - L_rep / L_max
    cr_ratio = dv / (ds + add_eps)
    L_max = cr_ratio.max()
    L_rep = cr_ratio.mean()
    complexity_reduction = 1.0 - (L_rep / (L_max))

    return diversity, complexity_reduction


@torch.inference_mode()
def compute_representation_metrics(
    encoder: nn.Module,
    embedding: torch.Tensor,
    value_estimate: torch.Tensor,
    entropy: torch.Tensor,
    manifold: Manifold,
    encoder_name: str,
    compute_expensive_metrics: bool,
) -> dict[str, float]:
    """Compute various metrics related to the representation of the agent's embedding."""

    metrics = {}

    # Log embedding norms: Euclidean norm pre/post scaling, Norm after mapping to the manifold
    representation_norm = torch.linalg.norm(embedding, dim=-1)
    # Unscaled embedding norms
    if getattr(encoder.scaling, "type", None) == "dim":
        unscaled_embedding = embedding * np.sqrt(embedding.shape[-1])
    elif getattr(encoder.scaling, "type", None) == "unit_ball":
        unscaled_embedding = embedding * representation_norm[:, None]
    else:
        unscaled_embedding = embedding
    unscaled_norm = torch.linalg.norm(unscaled_embedding, dim=-1)
    manifold_embedding = manifold.expmap_0(embedding)
    manifold_norm_dist0 = manifold.dist_0(manifold_embedding)
    manifold_norm_l2 = torch.linalg.norm(manifold_embedding, dim=-1, keepdim=True)

    metrics[f"{encoder_name}_representation/embedding_norm"] = representation_norm.mean().item()
    metrics[f"{encoder_name}_representation/unscaled_embedding_norm"] = unscaled_norm.mean().item()
    metrics[f"{encoder_name}_representation/manifold_embedding_dist0_norm"] = manifold_norm_dist0.mean().item()
    metrics[f"{encoder_name}_representation/manifold_embedding_l2_norm"] = manifold_norm_l2.mean().item()
    if isinstance(manifold, PoincareBall):
        conformal_factor = manifold._lambda(manifold_embedding).mean().item()
        metrics[f"{encoder_name}_representation/conformal_factor"] = conformal_factor

    if encoder.scaling.type == "learnable":
        metrics[f"{encoder_name}_representation/feat_scaling_mean"] = encoder.scaling.scale_mean.item()
        metrics[f"{encoder_name}_representation/feat_scaling_std"] = encoder.scaling.scale_std.item()

    # Correlation between embedding and value magnitudes
    value_corr_dist0 = torch.corrcoef(torch.cat([manifold_norm_dist0.detach(), torch.abs(value_estimate).detach()], dim=1).T)[
        0, 1
    ].item()
    value_corr_l2 = torch.corrcoef(torch.cat([manifold_norm_l2.detach(), torch.abs(value_estimate).detach()], dim=1).T)[
        0, 1
    ].item()
    metrics[f"{encoder_name}_correlations/value_corr_dist0"] = value_corr_dist0
    metrics[f"{encoder_name}_correlations/value_corr_l2"] = value_corr_l2

    if entropy is not None:
        # Correlation between embedding magnitude and policy entropy
        entropy_corr_dist0 = torch.corrcoef(torch.cat([manifold_norm_dist0.detach(), entropy[:, None].detach()], dim=1).T)[
            0, 1
        ].item()
        entropy_corr_l2 = torch.corrcoef(torch.cat([manifold_norm_l2.detach(), entropy[:, None].detach()], dim=1).T)[
            0, 1
        ].item()
        metrics[f"{encoder_name}_correlations/entropy_corr_dist0"] = entropy_corr_dist0
        metrics[f"{encoder_name}_correlations/entropy_corr_l2"] = entropy_corr_l2

    if compute_expensive_metrics:
        # Log delta-hyperbolicity using the Gromov product
        delta, diameter, relative_delta = get_delta(embedding, manifold, sample_size=512)
        delta_hyp, diameter_hyp, relative_delta_hyp = get_delta(manifold_embedding, manifold, sample_size=512)

        diversity, complexity_reduction = compute_feature_metrics(
            value_estimate, manifold_embedding, manifold, sample_size=512
        )

        # NOTE: Commented out for final runs to save computation time
        # Dormant neurons and effective rank
        # feat_rank = effective_rank(embedding).item()
        # dormant_fraction, dormant_count = dormant_neurons(obs_batch, encoder)

        # metrics[f"{encoder_name}_representation/effective_rank"] = feat_rank
        # metrics[f"{encoder_name}_representation/dormant_fraction"] = dormant_fraction
        # metrics[f"{encoder_name}_representation/dormant_count"] = dormant_count
        metrics[f"{encoder_name}_representation/feature_diversity"] = diversity.item()
        metrics[f"{encoder_name}_representation/complexity_reduction"] = complexity_reduction.item()
        metrics[f"{encoder_name}_representation/delta_euclidean"] = delta
        metrics[f"{encoder_name}_representation/delta_relative_euclidean"] = relative_delta
        metrics[f"{encoder_name}_representation/delta_diam_euclidean"] = diameter
        metrics[f"{encoder_name}_representation/delta_hyper"] = delta_hyp
        metrics[f"{encoder_name}_representation/delta_relative_hyper"] = relative_delta_hyp
        metrics[f"{encoder_name}_representation/delta_diam_hyper"] = diameter_hyp

    return metrics


@torch.jit.script
def _poincare_l1_dist0(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Compute the l1 distance of a point on the Poincare ball using the Mobius-dist implementation."""
    sqrt_c = c.sqrt()
    dist_c = atanh(sqrt_c * x.norm(p=1, dim=-1, keepdim=True))
    res = 2 * dist_c / sqrt_c

    return res


def log_atari_minigrid_stats(
    infos: dict[str, Any],
    writer: SummaryWriter,
    global_step: int,
    smoothed_rewards: deque[float],
    smoothed_ep_length: deque[float],
    num_envs: int,
) -> tuple[deque, deque]:
    """Log episode statistics Atari and Minigrid."""

    ep_reward_vec = infos["episode"]["r"]
    ep_length_vec = infos["episode"]["l"]

    done_ep_idx = np.nonzero(ep_reward_vec)
    ep_rewards = ep_reward_vec[done_ep_idx]
    ep_lengths = ep_length_vec[done_ep_idx]

    for ep_reward, ep_length in zip(ep_rewards, ep_lengths, strict=True):
        smoothed_rewards.append(ep_reward)
        smoothed_ep_length.append(ep_length)
        smooth_rew = np.mean(smoothed_rewards) if smoothed_rewards else 0
        writer.add_scalar("charts/episodic_return", ep_reward, global_step)
        writer.add_scalar("charts/episodic_length", ep_length, global_step)
        writer.add_scalar(f"charts/smoothed{num_envs}_return", smooth_rew, global_step)

    return smoothed_rewards, smoothed_ep_length


def log_procgen_stats(
    infos: dict[str, Any],
    writer: SummaryWriter,
    global_step: int,
    smoothed_rewards: deque[float],
    smoothed_ep_length: deque[float],
    num_envs: int,
) -> tuple[deque, deque]:
    """Log episode statistics Atari and Minigrid."""

    for item in infos:
        if "episode" in item.keys():
            ep_reward = item["episode"]["r"]
            ep_length = item["episode"]["l"]
            smoothed_rewards.append(ep_reward)
            smoothed_ep_length.append(ep_length)
            writer.add_scalar("charts/episodic_return", ep_reward, global_step)
            writer.add_scalar("charts/episodic_length", ep_length, global_step)
            writer.add_scalar(f"charts/smoothed{num_envs}_return", np.mean(smoothed_rewards), global_step)
            break

    return smoothed_rewards, smoothed_ep_length


def log_encoder_metrics(agent: nn.Module, panel_name: str) -> dict[str, float]:
    """Log layer metrics for the encoder of the agent."""
    per_layer_metrics = {}
    i = 0
    for name, layer in agent.named_modules():
        if hasattr(layer, "parametrizations"):
            i += 1
            # Has spectral normalization applied and is thus a ParametrizedLinearLayer
            weight = layer.parametrizations.weight.original
            grad = layer.parametrizations.weight.original.grad
        elif isinstance(layer, nn.Linear | nn.Conv2d):
            i += 1
            weight = layer.weight.data
            grad = layer.weight.grad
        else:
            # Skip submodules or normalization layers
            continue

        # Flatten the weights for convolutional filters
        if len(weight.data.shape) > 2:
            weight = weight.view(weight.data.shape[0], -1)
            grad = grad.view(grad.shape[0], -1)

        per_layer_metrics[f"{panel_name}/{name}_{i}_l2_weight_norm"] = torch.linalg.norm(weight, ord=2).item()
        per_layer_metrics[f"{panel_name}/{name}_{i}_l2_grad_norm"] = torch.linalg.norm(grad, ord=2).item()
        per_layer_metrics[f"{panel_name}/{name}_{i}_l1_weight_norm"] = torch.linalg.norm(weight, ord=1).item()
        per_layer_metrics[f"{panel_name}/{name}_{i}_l1_grad_norm"] = torch.linalg.norm(grad, ord=1).item()

    return per_layer_metrics


def log_actor_critic_metrics(
    layer: nn.Module, name: Literal["actor", "critic", "actor_critic", "q-function", "policy"]
) -> dict[str, float]:
    """Log detailed metrics for the actor and critic of the agent.

    Args:
        layer: A PyTorch module representing the actor or critic of the agent.
        name: The name of the layer, e.g. "actor" or "critic".
        "actor": The actor layer of a PPO agent.
        "critic": The critic layer of a PPO agent.
        "actor_critic": Shared actor and critic layer for PPO.
        "q-function": The Q-function layer for DQN.
        "policy": The policy layer for REINFORCE.
    """
    metrics = {}
    # Layer weights never live in hyperbolic space
    weight = layer.weight.data
    weight_grad = layer.weight.grad
    metrics[f"{name}/weight_l2_norm"] = torch.linalg.norm(weight, ord=2).item()
    metrics[f"{name}/weight_l1_norm"] = torch.linalg.norm(weight, ord=1).item()
    metrics[f"{name}/weight_l2_grad_norm"] = torch.linalg.norm(weight_grad, ord=2).item()
    metrics[f"{name}/weight_l1_grad_norm"] = torch.linalg.norm(weight_grad, ord=1).item()

    # Bias logging
    bias = layer.bias.data
    if hasattr(layer.bias, "manifold") and isinstance(layer.bias.manifold, PoincareBall):
        # Bias is Poincare ball parameter
        bias_grad = layer.manifold.egrad2rgrad(layer.bias.grad, layer.bias)
        bias_l2_norm = layer.bias.manifold.dist_0(bias)
        bias_l1_norm = _poincare_l1_dist0(bias, layer.bias.manifold.c)

        if isinstance(layer, HyperbolicRegressionPoincareHDRL) or isinstance(layer, HyperbolicRegressionPoincare):
            # Bias has shape (out_dim, in_dim), distances have shape (out_dim, 1) -> Must average over out_dim
            bias_l2_norm = bias_l2_norm.mean()
            bias_l1_norm = bias_l1_norm.mean()
    else:
        # Euclidean bias (Euclidean layer or Hyperbolic networks++ or Hyperboloid Regression)
        bias_grad = layer.bias.grad
        bias_l2_norm = torch.linalg.norm(bias, ord=2)
        bias_l1_norm = torch.linalg.norm(bias, ord=1)

    metrics[f"{name}/bias_l2_norm"] = bias_l2_norm.item()
    metrics[f"{name}/bias_l1_norm"] = bias_l1_norm.item()
    metrics[f"{name}/bias_l2_grad_norm"] = torch.linalg.norm(bias_grad, ord=2).item()
    metrics[f"{name}/bias_l1_grad_norm"] = torch.linalg.norm(bias_grad, ord=1).item()

    return metrics


def effective_rank(representation: torch.Tensor, srank_threshold: float = 0.99) -> torch.Tensor:
    """Approximate the effective rank of the current representation.

    Feature rank is defined as the number of singular values greater than some epsilon.
    See the paper for details: https://arxiv.org/pdf/2010.14498.pdf.

    """

    U, S, V = la.svd(representation)

    # Effective feature rank is the number of normalized singular values
    # such that their cumulative sum is greater than some epsilon.
    assert (S < 0).sum() == 0, "Singular values cannot be non-negative."
    s_sum = torch.sum(S)

    if np.isclose(s_sum.item(), 0.0):
        # Catch case where the regularizer has collapsed the network features
        # This makes the training not crash entirely when rank collapse occurs
        print("Rank collapse occurred. Consider aborting the experiments.", RuntimeWarning)
        return torch.zeros(1)
    else:
        S_normalized = S / s_sum
        S_cum = torch.cumsum(S_normalized, dim=-1)
        # Get the first index where the rank threshold is exceeded
        k = (S_cum > srank_threshold).nonzero()[0].squeeze() + 1
        return k


def _get_activation(name: str, activations: dict[str, torch.Tensor]):
    """Fetches and stores the activations of a network layer."""

    def hook(layer: nn.Linear | nn.Conv2d, input: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        Get the activations of a layer with relu nonlinearity.
        ReLU has to be called explicitly here because the hook is attached to the conv/linear layer.
        """
        activations[name] = F.relu(output)

    return hook


def _get_redo_masks(activations: dict[str, torch.Tensor], tau: float) -> torch.Tensor:
    """
    Computes the ReDo mask for a given set of activations.
    The returned mask has True where neurons are dormant and False where they are active.
    """
    masks = []

    # Last activation are the q-values, which are never reset
    for name, activation in list(activations.items())[:-1]:
        # Taking the mean here conforms to the expectation under D in the main paper's formula
        if activation.ndim == 4:
            # Conv layer
            score = activation.abs().mean(dim=(0, 2, 3))
        else:
            # Linear layer
            score = activation.abs().mean(dim=0)

        # Divide by activation mean to make the threshold independent of the layer size
        # see https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py#L314
        # https://github.com/google/dopamine/issues/209
        normalized_score = score / (score.mean() + 1e-9)

        layer_mask = torch.zeros_like(normalized_score, dtype=torch.bool)
        if tau > 0.0:
            layer_mask[normalized_score <= tau] = 1
        else:
            layer_mask[torch.isclose(normalized_score, torch.zeros_like(normalized_score))] = 1
        masks.append(layer_mask)
    return masks


def dormant_neurons(obs: torch.Tensor, encoder: nn.Module) -> tuple[float, int]:
    """
    Checks the number of dormant neurons for a given model.
    If re_initialize is True, then the dormant neurons are re-initialized according to the scheme in
    https://arxiv.org/abs/2302.12902

    Returns the number of dormant neurons.
    """

    activations = {}
    activation_getter = partial(_get_activation, activations=activations)

    # Register hooks for all Conv2d and Linear layers to calculate activations
    handles = []
    for name, module in encoder.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(activation_getter(name)))

    # Calculate activations
    _ = encoder(obs)

    # Masks for tau=0 logging
    zero_masks = _get_redo_masks(activations, 0.0)
    total_neurons = sum([torch.numel(mask) for mask in zero_masks])
    zero_count = sum([torch.sum(mask) for mask in zero_masks])
    zero_fraction = (zero_count / total_neurons) * 100

    # Remove the hooks again
    for handle in handles:
        handle.remove()

    return zero_fraction, zero_count
