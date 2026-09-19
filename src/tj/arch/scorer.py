import torch
from torch import nn


class DecisionScorer(nn.Module):
    """Maps an encoded decision context to a scalar logit."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        """Return one score per context/candidate pair.

        Args:
            representation: (B, D)

        Returns:
            (B, 1)
        """
        return self.network(representation)
