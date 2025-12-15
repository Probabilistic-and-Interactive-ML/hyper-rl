from typing import Literal

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.hyperbolic_math.src.manifolds import Euclidean, Hyperboloid, PoincareBall
from src.hyperbolic_math.src.nn_layers import (
    HyperbolicLinearPoincare,
    HyperbolicLinearPoincarePP,
    HyperbolicRegressionHyperboloid,
    HyperbolicRegressionPoincare,
    HyperbolicRegressionPoincareHDRL,
    HyperbolicRegressionPoincarePP,
)
from src.rl.agents.c51 import C51Loss
from src.rl.agents.hl_gauss import HLGaussLoss
from src.rl.agents.networks import hyp_layer_init


def get_manifold_by_name(manifold_name: Literal["euclidean", "poincare", "hyperboloid"]):
    if manifold_name == "euclidean":
        manifold = Euclidean
    elif manifold_name == "poincare":
        manifold = PoincareBall
    elif manifold_name == "hyperboloid":
        manifold = Hyperboloid
    else:
        raise ValueError(f"Unknown manifold: {manifold_name}")
    return manifold


def get_hyperbolic_layer(
    cfg: DictConfig, manifold: PoincareBall | Hyperboloid, fw_method: str, embedding_dim: int, output_dim: int
) -> torch.nn.Module:
    """Returns a hyperbolic layer based on the provided hyperbolic arguments."""
    if fw_method == "HNNpp_MLR" and isinstance(manifold, Hyperboloid):
        layer = hyp_layer_init(
            HyperbolicRegressionHyperboloid(
                manifold=manifold,
                input_dim=embedding_dim + 1,
                output_dim=output_dim,
                backproject=True,
                params_dtype=cfg.manifold_params_dtype,
                input_space="tangent",
                clamping_factor=cfg.clamping_factor,
                smoothing_factor=cfg.smoothing_factor,
            ),
            cfg.small_weights,
        )
    elif fw_method == "HRL_forward" and isinstance(manifold, PoincareBall):
        layer = hyp_layer_init(
            HyperbolicRegressionPoincareHDRL(
                manifold=manifold,
                input_dim=embedding_dim,
                output_dim=output_dim,
                backproject=True,
                params_dtype=cfg.manifold_params_dtype,
                input_space="tangent",
                version="standard",
            ),
            cfg.small_weights,
        )
    elif fw_method == "HRL_forward_rs" and isinstance(manifold, PoincareBall):
        layer = hyp_layer_init(
            HyperbolicRegressionPoincareHDRL(
                manifold=manifold,
                input_dim=embedding_dim,
                output_dim=output_dim,
                backproject=True,
                params_dtype=cfg.manifold_params_dtype,
                input_space="tangent",
                version="rs",
            ),
            cfg.small_weights,
        )
    elif fw_method == "HNN_FC" and isinstance(manifold, PoincareBall):
        layer = hyp_layer_init(
            HyperbolicLinearPoincare(
                manifold=manifold,
                input_dim=embedding_dim,
                output_dim=output_dim,
                backproject=True,
                params_dtype=cfg.manifold_params_dtype,
                input_space="tangent",
                clamping_factor=cfg.clamping_factor,
                smoothing_factor=cfg.smoothing_factor,
            ),
            cfg.small_weights,
        )
    elif fw_method == "HNN_MLR" and isinstance(manifold, PoincareBall):
        layer = hyp_layer_init(
            HyperbolicRegressionPoincare(
                manifold=manifold,
                input_dim=embedding_dim,
                output_dim=output_dim,
                backproject=True,
                params_dtype=cfg.manifold_params_dtype,
                input_space="tangent",
                clamping_factor=cfg.clamping_factor,
                smoothing_factor=cfg.smoothing_factor,
            ),
            cfg.small_weights,
        )
    elif fw_method == "HNNpp_FC" and isinstance(manifold, PoincareBall):
        layer = hyp_layer_init(
            HyperbolicLinearPoincarePP(
                manifold=manifold,
                input_dim=embedding_dim,
                output_dim=output_dim,
                backproject=True,
                params_dtype=cfg.manifold_params_dtype,
                input_space="tangent",
                clamping_factor=cfg.clamping_factor,
                smoothing_factor=cfg.smoothing_factor,
            ),
            cfg.small_weights,
        )
    elif fw_method == "HNNpp_MLR" and isinstance(manifold, PoincareBall):
        layer = hyp_layer_init(
            HyperbolicRegressionPoincarePP(
                manifold=manifold,
                input_dim=embedding_dim,
                output_dim=output_dim,
                backproject=True,
                params_dtype=cfg.manifold_params_dtype,
                input_space="tangent",
                clamping_factor=cfg.clamping_factor,
                smoothing_factor=cfg.smoothing_factor,
            ),
            cfg.small_weights,
        )
    else:
        raise ValueError(f"Unknown hyperbolic layer forward method: {fw_method}")

    return layer


def get_loss_fn(value_fn_cfg: DictConfig, device: torch.device) -> nn.Module:
    """Returns the loss function based on the provided loss arguments."""
    if value_fn_cfg.loss_fn == "mse":
        return nn.MSELoss()
    elif value_fn_cfg.loss_fn == "huber":
        return nn.HuberLoss()
    elif value_fn_cfg.loss_fn == "hlgauss":
        return HLGaussLoss(
            device=device,
            min_value=value_fn_cfg.loss_min_value,
            max_value=value_fn_cfg.loss_max_value,
            num_bins=value_fn_cfg.loss_num_bins,
        )
    elif value_fn_cfg.loss_fn == "c51":
        return C51Loss(
            device=device,
            min_value=value_fn_cfg.loss_min_value,
            max_value=value_fn_cfg.loss_max_value,
            num_bins=value_fn_cfg.loss_num_bins,
        )
    else:
        raise ValueError(f"Unknown loss function: {value_fn_cfg.loss_fn}")
