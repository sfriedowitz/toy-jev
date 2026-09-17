from dataclasses import dataclass
from typing import Self

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tj.arch.model import SystemOneModel
from tj.schema.config import JevConfig
from tj.schema.request import BatchRequest, ChoiceRequest, NoulRequest, ScoreRequest
from tj.schema.response import BatchResponse, ChoiceResponse, NoulResponse, ScoreResponse


@dataclass
class EngineBatch:
    contexts: list[str]
    candidates: list[str]
    groups: list[tuple[int, int]]


class ToyJev:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, engine: SystemOneModel):
        self.tokenizer = tokenizer
        self.engine = engine

    @classmethod
    def from_config(cls, config: JevConfig) -> Self:
        tokenizer = AutoTokenizer.from_pretrained(
            config.encoder.model_name,
        )
        engine = SystemOneModel(
            model_name=config.encoder.model_name,
            hidden_dim=config.scorer.hidden_dim,
        )
        return cls(tokenizer=tokenizer, engine=engine)

    @torch.no_grad()
    def predict(self, request: BatchRequest) -> BatchResponse:
        batch = self._prepare_batch(request)

        inputs = self.tokenizer(
            batch.contexts,  # first sequence in pair
            batch.candidates,  # second sequence in pair
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        logits: torch.Tensor = self.engine(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids"),
        )

        responses = []
        for decision, (start, end) in zip(request.decisions, batch.groups, strict=True):
            decision_logits = logits[start:end]
            probability = torch.softmax(decision_logits, dim=-1)
            match decision:
                case NoulRequest():
                    # Only take the "yes" from the yes/no candidates
                    responses.append(NoulResponse(probability=probability[0].item()))
                case ChoiceRequest():
                    responses.append(
                        ChoiceResponse(
                            choices=decision.choices,
                            probability=probability.tolist(),
                        )
                    )
                case ScoreRequest():
                    score = sum((i + 1) * p for i, p in enumerate(probability.tolist()))
                    responses.append(
                        ScoreResponse(
                            score=score,
                            probability=probability.tolist(),
                        )
                    )

        return BatchResponse(decisions=responses)

    def _prepare_batch(self, request: BatchRequest) -> EngineBatch:
        contexts: list[str] = []
        candidates: list[str] = []
        groups: list[tuple[int, int]] = []

        for decision in request.decisions:
            start = len(contexts)
            context = f"{request.state}\n\n{decision.question}"

            match decision:
                case NoulRequest():
                    candidates = ["yes", "no"]
                case ChoiceRequest():
                    candidates = decision.choices
                case ScoreRequest():
                    candidates = decision.levels

            for candidate in candidates:
                contexts.append(context)
                candidates.append(candidate)

            end = len(contexts)
            groups.append((start, end))

        return EngineBatch(
            contexts=contexts,
            candidates=candidates,
            groups=groups,
        )
