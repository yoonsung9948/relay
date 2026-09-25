from typing import Literal

from pydantic import BaseModel, Field

SchedulingPolicyName = Literal["fifo", "round_robin"]

class ServeConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000

class SchedulerConfig(BaseModel):
    policy: SchedulingPolicyName = "fifo"
    batch_size: int = Field(default=1, gt=0)

class ModelConfig(BaseModel):
    repo: str = "Qwen/Qwen3-0.6B"
    family: str = "qwen3"
    parameter_count_b: float = 0.6

class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)