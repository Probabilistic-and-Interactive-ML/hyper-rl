import torch
import torch.nn as nn
import torch.nn.functional as F


class C51Loss(nn.Module):
    """
    Categorical distributional loss based on the C51 algorithm (Bellemare et al., 2017).

    Unlike point estimates, C51 represents value distributions as categorical distributions
    over a discrete support of atoms. This enables learning multi-modal return distributions
    and improved stability in deep RL.

    Reference: "A Distributional Perspective on Reinforcement Learning"
               Bellemare, Dabney, Munos (ICML 2017)
    """

    def __init__(
        self,
        device: torch.device,
        min_value: float,
        max_value: float,
        num_bins: int,
    ):
        """
        Initialize C51 categorical distribution.

        Args:
            device: Device to place tensors on
            min_value: Minimum value of the support
            max_value: Maximum value of the support
            num_bins: Number of atoms in the categorical distribution (typically 51)
        """
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins

        # Create support: linearly spaced atoms from min_value to max_value
        self.support = torch.linspace(
            min_value,
            max_value,
            num_bins,
            dtype=torch.float32,
            device=device,
        )

        # Compute atom spacing (delta_z in the C51 paper)
        self.delta_z = (max_value - min_value) / (num_bins - 1)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss between predicted distribution and target distribution.

        Args:
            logits: Predicted logits over atoms, shape (..., num_bins)
            target: Scalar target values, shape (...)

        Returns:
            Cross-entropy loss (scalar)
        """
        return F.cross_entropy(logits, self.value_to_probs(target))

    def value_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        """
        Project scalar target values onto categorical distribution via linear interpolation.

        For a target value between two atoms, probability is distributed proportionally
        based on distance. This is the "categorical projection" step from C51.

        Args:
            target: Scalar target values, shape (...)

        Returns:
            Probability distribution over atoms, shape (..., num_bins)
        """
        # Clamp target to valid support range
        target = torch.clamp(target, self.min_value, self.max_value).to(torch.float32)

        # Compute continuous position in atom grid
        # If target is exactly on atom k, then b = k
        # If target is between atoms k and k+1, then b is in (k, k+1)
        b = (target - self.min_value) / self.delta_z

        # Get lower and upper atom indices
        l = b.floor().to(torch.long)
        u = l + 1

        # Compute interpolation weights BEFORE clamping indices
        # This ensures edge cases work correctly (e.g., when target = max_value)
        # When b = k exactly: l_weight = 1, u_weight = 0 (all mass on atom k)
        # When b = k + 0.3: l_weight = 0.7, u_weight = 0.3 (closer atom gets more)
        l_weight = u.float() - b  # Weight for lower atom
        u_weight = b - l.float()  # Weight for upper atom

        # Clamp indices to valid range [0, num_bins - 1]
        l = torch.clamp(l, 0, self.num_bins - 1)
        u = torch.clamp(u, 0, self.num_bins - 1)

        # Create probability distribution tensor
        probs = torch.zeros(*target.shape, self.num_bins, device=target.device)

        # Distribute probability mass to atoms
        # scatter_add_ accumulates values, handling cases where l == u
        probs.scatter_add_(-1, l.unsqueeze(-1), l_weight.unsqueeze(-1))
        probs.scatter_add_(-1, u.unsqueeze(-1), u_weight.unsqueeze(-1))

        return probs

    def probs_to_value(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Convert categorical distribution to expected scalar value.

        Args:
            probs: Probability distribution over atoms, shape (..., num_bins)

        Returns:
            Expected values, shape (...)
        """
        return torch.sum(probs * self.support, dim=-1)

    def logits_to_value(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to expected scalar value.

        Args:
            logits: Predicted logits over atoms, shape (..., num_bins)

        Returns:
            Expected values, shape (...)
        """
        probs = F.softmax(logits, dim=-1)
        return self.probs_to_value(probs)
