import asyncio
from dataclasses import dataclass, field
from enum import Enum

from engine.resources import Resources


class BootStatus(str, Enum):
    TOKENIZER_LOAD = "tokenizer_load"
    SCHEDULER_INIT = "scheduler_init"
    MODEL_INIT = "model_init"
    RUNNER_INIT = "runner_init"
    READY = "ready"
    ERROR = "error"


@dataclass
class AppState:
    status: BootStatus = BootStatus.TOKENIZER_LOAD
    error: str | None = None
    resources: Resources | None = None
    tasks: list[asyncio.Task] = field(default_factory=list)
