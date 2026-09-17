import torch
from torch import nn

from tj.arch.encoder import CrossEncoder
from tj.arch.scorer import DecisionScorer


class SystemOneModel(nn.Module):
    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = CrossEncoder(model_name=model_name)
        self.scorer = DecisionScorer(
            input_dim=self.encoder.hidden_size,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score a batch of context/candidate pairs.

        Args:
            input_ids: (N, T)
            attention_mask: (N, T)
            token_type_ids: (N, T) or None

        Returns:
            logits: (N,)
        """
        representation = self.encoder(
            input_ids,
            attention_mask,
            token_type_ids,
        )
        return self.scorer(representation)
