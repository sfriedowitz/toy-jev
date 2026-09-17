from typing import Annotated, Literal

from pydantic import BaseModel, Field


class NoulRequest(BaseModel):
    type: Literal["noul"] = "noul"
    question: str

    def choices(self) -> list[str]:
        return ["yes", "no"]


class ChoiceRequest(BaseModel):
    type: Literal["choice"] = "choice"
    question: str
    choices: list[str]


class ScoreRequest(BaseModel):
    type: Literal["score"] = "score"
    question: str
    levels: list[str]


DecisionRequest = Annotated[
    NoulRequest | ChoiceRequest | ScoreRequest,
    Field(discriminator="type"),
]


class BatchRequest(BaseModel):
    state: str
    decisions: list[DecisionRequest]
