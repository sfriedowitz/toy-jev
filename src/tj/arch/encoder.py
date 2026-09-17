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
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
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
