from typing import Literal

from pydantic import BaseModel, Field

SchedulingPolicyName = Literal["fifo", "round_robin"]

class ServeConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000

class SchedulerConfig(BaseModel):
    policy: SchedulingPolicyName = "fifo"
    batch_size: int = Field(default=1, gt=0)

class ModelSpec(BaseModel):
    repo: str = "Qwen/Qwen3-0.6B"
    family: Literal["qwen3"] = "qwen3"
    parameter_count_b: float = 0.6


class Qwen3Config(BaseModel):
    vocab_size: int = 151936

    hidden_size: int = 1024
    intermediate_size: int = 3072

    num_hidden_layers: int = 28

    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128

    rms_norm_eps: float = 1e-6

    # satisfy huggingface config requirements
    # to use huggingface's rope implementation
    rope_parameters: dict = Field(
        default_factory=lambda: {
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
        }
    )

    max_position_embeddings: int = 40960

    attention_bias: bool = False
    tie_word_embeddings: bool = True

SupportedBackends = Literal[
    "huggingface",
    "custom",
]

class AppConfig(BaseModel):
    cors_domains: list[str] = Field(default_factory=list)

class Config(BaseModel):
    backend: SupportedBackends = "custom"

    model: ModelSpec = Field(
        default_factory=ModelSpec
    )
    app: AppConfig = Field(
        default_factory=AppConfig
    )
    architecture: Qwen3Config = Field(
        default_factory=Qwen3Config
    )

    serve: ServeConfig = Field(
        default_factory=ServeConfig
    )

    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig
    )