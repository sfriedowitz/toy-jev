from dataclasses import dataclass
from typing import Self

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tj.arch.engine import SystemOneModel
from tj.schema.config import SystemOneConfig
from tj.schema.request import BatchRequest, ChoiceRequest, NoulRequest, ScoreRequest
from tj.schema.response import (
    BatchResponse,
    ChoiceResponse,
    NoulResponse,
    ScoreResponse,
)


@dataclass
class EngineBatch:
    contexts: list[str]
    candidates: list[str]
    groups: list[tuple[int, int]]

    @property
    def group_sizes(self) -> list[int]:
        return [end - start for start, end in self.groups]


class ToyJev:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        engine: SystemOneModel,
    ):
        self.tokenizer = tokenizer
        self.engine = engine

    @classmethod
    def from_config(cls, config: SystemOneConfig) -> Self:
        tokenizer = AutoTokenizer.from_pretrained(
            config.cross_encoder.base_model,
        )
        engine = SystemOneModel(config)

        return cls(
            tokenizer=tokenizer,
            engine=engine,
        )

    @torch.no_grad()
    def predict(
        self,
        request: BatchRequest,
    ) -> BatchResponse:
        batch = self._prepare_batch(request)

        inputs = self.tokenizer(
            batch.contexts,
            batch.candidates,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        logits = self.engine(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids"),
            group_sizes=batch.group_sizes,
        )

        responses = []
        for batch_idx, decision in enumerate(request.decisions):
            start, end = batch.groups[batch_idx]
            num_candidates = end - start

            # Remove padding from candidate logits
            decision_logits = logits[batch_idx, :num_candidates]
            probability = torch.softmax(decision_logits, dim=-1)

            match decision:
                case NoulRequest():
                    responses.append(
                        NoulResponse(
                            probability=probability[0].item(),
                        )
                    )

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

    def _prepare_batch(
        self,
        request: BatchRequest,
    ) -> EngineBatch:
        contexts: list[str] = []
        candidates: list[str] = []
        groups: list[tuple[int, int]] = []

        for decision in request.decisions:
            start = len(contexts)
            context = f"{request.state}\n\n{decision.question}"

            for candidate in decision.candidates:
                contexts.append(context)
                candidates.append(candidate)

            end = len(contexts)
            groups.append((start, end))

        return EngineBatch(
            contexts=contexts,
            candidates=candidates,
            groups=groups,
        )
