import torch
from torch import nn
from transformers import AutoModel


class CrossEncoder(nn.Module):
    """Jointly encodes a context/question and candidate.

    Input sequences should already be tokenized as something like:

        [CLS] state/question [SEP] candidate [SEP]

    The tokenizer and batching happen upstream.
    """

    def __init__(
        self,
        *,
        base_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(base_model)
        self.hidden_size = self.transformer.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one representation per context/candidate pair.

        Args:
            input_ids: (N, T)
            attention_mask: (N, T)
            token_type_ids: (N, T)

        Returns:
            (N, D)
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        return outputs.last_hidden_state[:, 0]


class CandidateEncoder(nn.Module):
    """Encodes candidate representations jointly within each decision.

    Each candidate embedding is treated as a token, allowing self-attention to model
    relationships between candidates in the same decision. Padding candidates
    are ignored using the provided mask.

    Input:
        representations: Tensor of shape (B, C, D), where B is the number
            of decisions, C is the maximum number of candidates, and D is
            the representation dimension.
        padding_mask: Boolean tensor of shape (B, C), where True indicates
            a padded candidate.

    Output:
        Tensor of shape (B, C, D) containing contextualized candidate
        representations.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 1,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        representations: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(
            representations,
            src_key_padding_mask=padding_mask,
        )
