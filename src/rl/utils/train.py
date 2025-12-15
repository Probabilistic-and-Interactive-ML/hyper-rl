import os
import random
from collections.abc import Iterable
from warnings import warn

import numpy as np
import torch
from omegaconf import DictConfig
from torch import optim

from src.hyperbolic_math.src.optim import RiemannianAdam, RiemannianSGD


def linear_eps_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    """Linear schedule from start_e to end_e over duration."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def set_seeds(seed: int, torch_deterministic: bool) -> None:
    """Set the seeds of all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    if torch_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)


def set_cuda_configuration(gpu: int) -> torch.device:
    """Set up the device for the desired GPU or all GPUs."""
    if gpu == -1:
        device = torch.device("cpu")
    else:
        assert gpu <= torch.cuda.device_count(), "Invalid CUDA index specified."
        device = torch.device(f"cuda:{gpu}")

    return device


def get_optimizer(
    encoder_params: Iterable[torch.nn.Parameter],
    head_params: Iterable[torch.nn.Parameter],
    cfg: DictConfig,
    use_riemannian: bool,
) -> torch.optim.Optimizer:
    """Get the optimizer depending on config specification and the manifold used."""

    if use_riemannian:
        if cfg.encoder_weight_decay > 0:
            warn("Weight decay is not implemented for Riemannian optimizers.", RuntimeWarning)
        parameters = list(encoder_params) + list(head_params)
        if cfg.algorithm == "adam":
            # Curvature parameter
            optimizer = RiemannianAdam(parameters, lr=cfg.learning_rate, expmap_update=False, eps=cfg.adam_eps)
        else:
            optimizer = RiemannianSGD(parameters, lr=cfg.learning_rate, expmap_update=False, momentum=cfg.momentum)
    else:
        # Use regular optimizer
        if cfg.algorithm == "adam":
            if cfg.encoder_weight_decay > 0:
                optimizer = optim.AdamW(
                    [
                        {"params": encoder_params, "weight_decay": cfg.encoder_weight_decay},
                        {"params": head_params, "weight_decay": 0.0},
                    ],
                    lr=cfg.learning_rate,
                    eps=cfg.adam_eps,
                )
            else:
                parameters = list(encoder_params) + list(head_params)
                optimizer = optim.Adam(parameters, lr=cfg.learning_rate, eps=cfg.adam_eps)
        else:
            parameters = list(encoder_params) + list(head_params)
            optimizer = optim.SGD(parameters, lr=cfg.learning_rate, momentum=cfg.momentum)

    return optimizer
