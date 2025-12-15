import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.special


class HLGaussLoss(nn.Module):
    """
    Categorical loss based on the HLGauss paper, building on the 'Stop regressing' implementation.

    A scalar value is transformed into a probability distribution by:
    1. Creating a Gaussian with fixed standard deviation (sigma) arund the target value.
    2. Using the Gaussian CDF to distribute probability mass to the nearby bins.
    """

    def __init__(self, device: torch.device, min_value: float, max_value: float, num_bins: int, smoothing_ratio: float = 0.75):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.smoothing_ratio = smoothing_ratio
        # Sigma is the smoothing ratio times the bin width, implementing the paper formula 𝜎/𝜍 = 0.75.
        self.sigma = self.smoothing_ratio * (max_value - min_value) / num_bins
        self.support = torch.linspace(min_value, max_value, num_bins + 1, dtype=torch.float32, device=device)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Expects the categorical logits of a value function and a scalar target."""
        return F.cross_entropy(logits, self.value_to_probs(target))

    def value_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        """Expects a scalar target and returns a probability distribution over the bins."""
        # Clamp the target to the min/max value to prevent NaN
        target = torch.clamp(target, self.min_value, self.max_value)
        cdf_evals = torch.special.erf((self.support - target.unsqueeze(-1)) / (torch.sqrt(torch.tensor(2.0)) * self.sigma))
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)

    def probs_to_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Expects a probability distribution over the bins and returns the expected value."""
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)

    def logits_to_value(self, logits: torch.Tensor) -> torch.Tensor:
        """Expects logits over the bins and returns the expected value."""
        probs = F.softmax(logits, dim=-1)
        return self.probs_to_value(probs)
