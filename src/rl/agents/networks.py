import math
from typing import Literal

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.utils.parametrizations import spectral_norm

from src.hyperbolic_math.src.manifolds import Euclidean, Hyperboloid, PoincareBall
from src.hyperbolic_math.src.nn_layers import (
    HyperbolicLinearPoincare,
    HyperbolicLinearPoincarePP,
    HyperbolicRegressionHyperboloid,
    HyperbolicRegressionPoincare,
    HyperbolicRegressionPoincareHDRL,
    HyperbolicRegressionPoincarePP,
)
from src.rl.utils.model import get_out_shapes
from src.types import EnvType

RegularizationType = Literal["none", "sn", "sn_penultimate", "ln", "rms"]
""" Regularization method for the Euclidean encoder of the hyperbolic agent."""

ScalerType = Literal["none", "dim", "unit_ball", "learnable"]
""" Method for scaling the features of the penultimate layer to stabilize training. """


def build_encoder(
    env_type: EnvType,
    encoder_cfg: DictConfig,
    curvature: float,
    envs,
) -> nn.Module:
    """Builds an encoder based on the environment type."""
    if env_type == "minigrid":
        return GridworldCNN(
            embedding_dim=encoder_cfg.embedding_dim,
            feature_scaling=encoder_cfg.feature_scaling,
            scaling_alpha=encoder_cfg.scaling_alpha,
            regularization=encoder_cfg.regularization,
            last_layer_tanh=encoder_cfg.last_layer_tanh,
            curvature=curvature,
            envs=envs,
        )
    elif env_type == "atari":
        return NatureCNN(
            embedding_dim=encoder_cfg.embedding_dim,
            feature_scaling=encoder_cfg.feature_scaling,
            scaling_alpha=encoder_cfg.scaling_alpha,
            regularization=encoder_cfg.regularization,
            last_layer_tanh=encoder_cfg.last_layer_tanh,
            curvature=curvature,
            envs=envs,
        )
    elif env_type == "procgen":
        return ImpalaCNN(
            embedding_dim=encoder_cfg.embedding_dim,
            feature_scaling=encoder_cfg.feature_scaling,
            scaling_alpha=encoder_cfg.scaling_alpha,
            regularization=encoder_cfg.regularization,
            last_layer_tanh=encoder_cfg.last_layer_tanh,
            curvature=curvature,
            envs=envs,
        )
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def apply_sn_fn(m):
    if isinstance(m, nn.Conv2d | nn.Linear):
        return spectral_norm(m)
    else:
        return m


def apply_sn_to_module(module: nn.Module):
    for name, submodule in module.named_children():
        if isinstance(submodule, nn.Conv2d | nn.Linear):
            submodule.apply(apply_sn_fn)
        # Recursively apply spectral norm to child modules
        apply_sn_to_module(submodule)


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialization function for linear and convolutional layers."""
    assert not isinstance(
        layer,
        HyperbolicRegressionHyperboloid
        | HyperbolicLinearPoincare
        | HyperbolicRegressionPoincare
        | HyperbolicRegressionPoincarePP
        | HyperbolicLinearPoincarePP
        | HyperbolicRegressionPoincareHDRL,
    ), "This initializer should not be used on hyperbolic layers."
    torch.nn.init.orthogonal_(layer.weight, std)

    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def hyp_layer_init(layer: nn.Module, small: bool, bias_const: float = 0.0) -> nn.Module:
    """Initialization function for hyperbolic layers."""
    assert isinstance(
        layer,
        HyperbolicRegressionHyperboloid
        | HyperbolicLinearPoincare
        | HyperbolicRegressionPoincare
        | HyperbolicRegressionPoincarePP
        | HyperbolicLinearPoincarePP
        | HyperbolicRegressionPoincareHDRL,
    ), "This initializer should only be used on hyperbolic layers"

    if isinstance(layer.manifold, Euclidean):
        torch.nn.init.orthogonal_(layer.weight, np.sqrt(2))
    elif isinstance(layer.manifold, PoincareBall):
        std = 1 / np.sqrt(layer.weight.shape[1])
        if small:
            std = std * 0.01
        nn.init.normal_(layer.weight, 0.0, std)
    elif isinstance(layer.manifold, Hyperboloid):
        std = 1 / np.sqrt(layer.weight.shape[1])
        if small:
            std = std * 0.01
        nn.init.normal_(layer.weight, 0.0, std)
    else:
        raise NotImplementedError(f"Layer initialization not implemented for manifold {layer.manifold}")

    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FeatureScaling(nn.Module):
    def __init__(
        self,
        dim: int,
        scale_type: ScalerType,
        curvature: float,
        max_norm: float,
    ):
        super().__init__()
        self.dim = dim
        self.type = scale_type

        self.scale_mean = None
        self.scale_std = None

        if self.type == "learnable":
            max_scale = math.atanh(max_norm) / math.sqrt(curvature)
            self.max_scale = max_scale
            self.scale = nn.Linear(self.dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.type == "dim":
            x = x / np.sqrt(x.shape[-1])
        elif self.type == "unit_ball":
            x = x / (torch.linalg.norm(x, dim=-1, keepdim=True).detach() + 1e-8)
        elif self.type == "learnable":
            x = x / np.sqrt(x.shape[-1])
            learned_scale = F.sigmoid(self.scale(x))
            x = learned_scale * self.max_scale * x

            # Logging
            if learned_scale.numel() > 1:
                self.scale_mean = learned_scale.mean().detach()
                self.scale_std = learned_scale.std().detach()
        elif self.type == "none":
            pass
        else:
            raise ValueError(f"Unknown feature scaling type: {self.type}")

        return x


class GridworldCNN(nn.Module):
    """CNN Encoder for MiniGrid environments."""

    def __init__(
        self,
        embedding_dim: int,
        feature_scaling: ScalerType,
        scaling_alpha: float,
        regularization: RegularizationType,
        last_layer_tanh: bool,
        curvature: float,
        envs: gymnasium.vector.SyncVectorEnv,
    ):
        super().__init__()

        in_shape = envs.single_observation_space.shape
        self.obs_max = envs.single_observation_space.high.max()
        self.embedding_dim = embedding_dim
        self.regularization = regularization

        self.conv1 = nn.Conv2d(in_channels=in_shape[0], out_channels=16, kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, 2)
        self.conv3 = nn.Conv2d(32, 64, 2)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(envs.single_observation_space.sample()[None, ...]).float()).shape[1]

        self.fc = nn.Linear(n_flatten, embedding_dim)
        self.scaling = FeatureScaling(
            dim=embedding_dim, scale_type=feature_scaling, curvature=curvature, max_norm=scaling_alpha
        )
        self.linear_act = F.tanh if last_layer_tanh else F.relu
        self.norm = nn.Identity()

        with torch.no_grad():
            dummy_input = torch.as_tensor(envs.single_observation_space.sample()[None, ...]).float()
            out_shapes = get_out_shapes(self, dummy_input)

        if regularization == "sn":
            apply_sn_to_module(self)
        elif regularization == "sn_penultimate":
            apply_sn_to_module(self.fc)
        elif regularization == "ln":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        elif regularization == "rms":
            self.norm = nn.RMSNorm(embedding_dim, elementwise_affine=False)
        elif regularization == "none":
            pass
        else:
            raise ValueError(f"Unknown regularization type: {regularization}")

    def __str__(self):
        return "GridworldCNN"

    def cnn(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x / self.obs_max - 0.5)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        # Flatten
        x = x.reshape(x.shape[0], -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.norm(self.fc(x))
        x = self.linear_act(x)
        return self.scaling(x)


class NatureCNN(nn.Module):
    """CNN Encoder for Atari environments based on the DQN paper."""

    def __init__(
        self,
        embedding_dim: int,
        feature_scaling: ScalerType,
        scaling_alpha: float,
        regularization: RegularizationType,
        last_layer_tanh: bool,
        curvature: float,
        envs: gymnasium.vector.SyncVectorEnv,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.regularization = regularization

        self.conv1 = layer_init(nn.Conv2d(4, 32, 8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, 3, stride=1))
        self.fc = layer_init(nn.Linear(64 * 7 * 7, embedding_dim))
        self.scaling = FeatureScaling(
            dim=embedding_dim, scale_type=feature_scaling, curvature=curvature, max_norm=scaling_alpha
        )
        self.linear_act = F.tanh if last_layer_tanh else F.relu
        self.norm = nn.Identity()

        with torch.no_grad():
            dummy_input = torch.as_tensor(envs.single_observation_space.sample()[None, ...]).float()
            out_shapes = get_out_shapes(self, dummy_input)

        if regularization == "sn":
            apply_sn_to_module(self)
        elif regularization == "sn_penultimate":
            apply_sn_to_module(self.fc)
        elif regularization == "ln":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        elif regularization == "rms":
            self.norm = nn.RMSNorm(embedding_dim, elementwise_affine=False)
        elif regularization == "none":
            pass
        else:
            raise ValueError(f"Unknown regularization type: {regularization}")

    def __str__(self):
        return "NatureCNN"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x / 255.0 - 0.5)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        # Flatten
        x = x.view(x.shape[0], -1)
        x = self.norm(self.fc(x))
        x = self.linear_act(x)
        return self.scaling(x)


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = x
        x = F.relu(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], out_channels: int):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)

        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self) -> tuple[int, int, int]:
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class ImpalaCNN(nn.Module):
    """IMPALA CNN Encoder for ProcGen/Atari environments."""

    def __init__(
        self,
        embedding_dim: int,
        feature_scaling: ScalerType,
        scaling_alpha: float,
        regularization: RegularizationType,
        last_layer_tanh: bool,
        curvature: float,
        envs: gymnasium.vector.SyncVectorEnv,
    ):
        super().__init__()
        h, w, c = envs.single_observation_space.shape
        shape = (c, h, w)
        self.embedding_dim = embedding_dim
        self.regularization = regularization

        self.convseq1 = ConvSequence(shape, 16)
        self.convseq2 = ConvSequence(self.convseq1.get_output_shape(), 32)
        self.convseq3 = ConvSequence(self.convseq2.get_output_shape(), 32)

        conv_out_shape = self.convseq3.get_output_shape()
        flattened_dim = conv_out_shape[0] * conv_out_shape[1] * conv_out_shape[2]
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(flattened_dim, embedding_dim)
        self.scaling = FeatureScaling(
            dim=embedding_dim, scale_type=feature_scaling, curvature=curvature, max_norm=scaling_alpha
        )

        self.linear_act = F.tanh if last_layer_tanh else F.relu

        self.norm = nn.Identity()
        if regularization == "sn":
            apply_sn_to_module(self)
        elif regularization == "sn_penultimate":
            apply_sn_to_module(self.fc)
        elif regularization == "ln":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        elif regularization == "rms":
            self.norm = nn.RMSNorm(embedding_dim, elementwise_affine=False)
        elif regularization == "none":
            pass
        else:
            raise ValueError(f"Unknown regularization type: {regularization}")

    def __str__(self):
        return "ImpalaCNN"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute((0, 3, 1, 2))
        x = x / 255.0 - 0.5
        x = self.convseq1(x)
        x = self.convseq2(x)
        x = self.convseq3(x)
        # Flatten
        x = self.flatten(x)
        x = F.relu(x)
        x = self.norm(self.fc(x))
        x = self.linear_act(x)
        return self.scaling(x)


class FeedforwardEncoder(nn.Module):
    """Encoder for a state-based agent."""

    def __init__(
        self,
        embedding_dim: int,
        feature_scaling: ScalerType,
        scaling_alpha: float,
        regularization: RegularizationType,
        last_layer_tanh: bool,
        curvature: float,
        envs: gymnasium.vector.SyncVectorEnv,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.regularization = regularization

        self.fc1 = layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 512))
        self.fc2 = layer_init(nn.Linear(512, 256))
        self.fc3 = layer_init(nn.Linear(256, embedding_dim))
        self.scaling = FeatureScaling(
            dim=embedding_dim, scale_type=feature_scaling, curvature=curvature, max_norm=scaling_alpha
        )

        with torch.no_grad():
            dummy_input = torch.as_tensor(envs.single_observation_space.sample()[None, ...]).float()
            out_shapes = get_out_shapes(self, dummy_input)

        self.linear_act = F.tanh if last_layer_tanh else F.relu

        self.norm = nn.Identity()
        if regularization == "sn":
            apply_sn_to_module(self)
        elif regularization == "sn_penultimate":
            apply_sn_to_module(self.fc)
        elif regularization == "ln":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        elif regularization == "rms":
            self.norm = nn.RMSNorm(embedding_dim, elementwise_affine=False)
        elif regularization == "none":
            pass
        else:
            raise ValueError(f"Unknown regularization type: {regularization}")

    def __str__(self):
        return "FeedforwardEncoder"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return self.scaling(x)
