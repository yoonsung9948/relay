from typing import Literal

from pydantic import BaseModel, Field

SchedulingPolicyName = Literal["fifo"]
ModelName = Literal["qwen3-8b"]


class ServeConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class SchedulerConfig(BaseModel):
    policy: SchedulingPolicyName = "fifo"
    batch_size: int = 1


class Config(BaseModel):
    model: ModelName = "qwen3-8b"
    serve: ServeConfig = Field(default_factory=ServeConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
