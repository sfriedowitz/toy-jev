from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class NoulOutput:
    probability: torch.Tensor


@dataclass
class ChoiceOutput:
    choices: list[str]
    probability: torch.Tensor


@dataclass
class ScoreOutput:
    probability: torch.Tensor

    @property
    def num_levels(self) -> int:
        return self.probability.shape[-1]

    @property
    def score(self) -> torch.Tensor:
        """Return the total score, defined:

        ```
        S = E[L] = sum_i l_i P(l_i)
        ```
        """
        ranks = torch.arange(1, self.num_levels + 1).to(self.probability)
        return (self.probability * ranks).sum(dim=-1, keepdim=True)


class NoulHead(nn.Module):
    """Binary decision.

    ```
    logits (B x 1) -> sigmoid -> P(yes)
    ```
    """

    def forward(self, logits: torch.Tensor) -> NoulOutput:
        return NoulOutput(probability=torch.sigmoid(logits))


class ChoiceHead(nn.Module):
    """Categorical choice.

    ```
    logits (B x C) -> softmax -> P(c_i)
    ```
    """

    def forward(self, choices: list[str], logits: torch.Tensor) -> ChoiceOutput:
        p = torch.softmax(logits, dim=-1)
        return ChoiceOutput(
            choices=choices,
            probability=p,
        )


class ScoreHead(nn.Module):
    """Ordered categorical scoring.

    ```
    logits (B x L) -> softmax -> P(l_i)
    ```
    """

    def forward(self, logits: torch.Tensor) -> ScoreOutput:
        p = torch.softmax(logits, dim=-1)
        return ScoreOutput(probability=p)
