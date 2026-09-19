import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from tj.arch.encoders import CandidateEncoder, CrossEncoder
from tj.arch.scorer import DecisionScorer
from tj.schema.config import SystemOneConfig


class SystemOneModel(nn.Module):
    def __init__(self, config: SystemOneConfig):
        super().__init__()
        self.cross_encoder = CrossEncoder(base_model=config.cross_encoder.base_model)
        self.candidate_encoder = CandidateEncoder(
            embedding_dim=config.candidate_encoder.embedding_dim,
            num_heads=config.candidate_encoder.num_heads,
            num_layers=config.candidate_encoder.num_layers,
        )
        self.scorer = DecisionScorer(
            input_dim=self.cross_encoder.hidden_size,
            hidden_dim=config.scorer.hidden_dim,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        group_sizes: list[int],
        *,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # (N, T) -> (N, D)
        representations = self.cross_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # (N, D) -> (B, C, D)
        representations, candidate_mask = self._pad_candidates(representations, group_sizes)

        # (B, C, D) -> (B, C, D)
        representations = self.candidate_encoder(representations, candidate_mask)

        # (B, C, D) -> (B, C)
        logits: torch.Tensor = self.scorer(representations)

        # Don't allow padding to receive probability.
        return logits.masked_fill(candidate_mask, float("-inf"))

    def _pad_candidates(
        self,
        representations: torch.Tensor,
        group_sizes: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_reprs = torch.split(representations, group_sizes, dim=0)

        padded = pad_sequence(list(group_reprs), batch_first=True)

        max_candidates = padded.shape[1]
        candidate_mask = torch.ones(
            len(group_sizes),
            max_candidates,
            dtype=torch.bool,
            device=representations.device,
        )

        for i, size in enumerate(group_sizes):
            candidate_mask[i, :size] = False

        return padded, candidate_mask
