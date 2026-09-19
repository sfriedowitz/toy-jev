from pydantic import BaseModel


class CrossEncoderConfig(BaseModel):
    base_model: str


class CandidateEncoderConfig(BaseModel):
    embedding_dim: int = 256
    num_heads: int = 4
    num_layers: int = 1


class ScorerConfig(BaseModel):
    hidden_dim: int = 256


class SystemOneConfig(BaseModel):
    cross_encoder: CrossEncoderConfig
    candidate_encoder: CandidateEncoderConfig
    scorer: ScorerConfig
