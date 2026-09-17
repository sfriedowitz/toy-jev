from typing import Annotated, Literal

from pydantic import BaseModel, Field


class NoulResponse(BaseModel):
    type: Literal["noul"] = "noul"
    probability: float


class ChoiceResponse(BaseModel):
    type: Literal["choice"] = "choice"
    choices: list[str]
    probability: list[float]


class ScoreResponse(BaseModel):
    type: Literal["score"] = "score"
    score: float
    probability: list[float]


DecisionResponse = Annotated[
    NoulResponse | ChoiceResponse | ScoreResponse,
    Field(discriminator="type"),
]


class BatchResponse(BaseModel):
    decisions: list[DecisionResponse]
