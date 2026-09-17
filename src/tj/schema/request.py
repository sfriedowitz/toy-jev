from abc import abstractmethod
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class _BaseDecisionRequest(BaseModel):
    @property
    @abstractmethod
    def candidates(self) -> list[str]:
        """Return the candidate options for this decision."""
        pass


class NoulRequest(_BaseDecisionRequest):
    type: Literal["noul"] = "noul"
    question: str

    @property
    def candidates(self) -> list[str]:
        return ["yes", "no"]


class ChoiceRequest(_BaseDecisionRequest):
    type: Literal["choice"] = "choice"
    question: str
    choices: list[str]

    @property
    def candidates(self) -> list[str]:
        return self.choices


class ScoreRequest(_BaseDecisionRequest):
    type: Literal["score"] = "score"
    question: str
    levels: list[str]

    @property
    def candidates(self) -> list[str]:
        return self.levels


DecisionRequest = Annotated[
    NoulRequest | ChoiceRequest | ScoreRequest,
    Field(discriminator="type"),
]


class BatchRequest(BaseModel):
    state: str
    decisions: list[DecisionRequest]
