from pydantic import BaseModel


class EncoderConfig(BaseModel):
    model_name: str


class ScorerConfig(BaseModel):
    hidden_dim: int = 256


class JevConfig(BaseModel):
    encoder: EncoderConfig
    scorer: ScorerConfig
