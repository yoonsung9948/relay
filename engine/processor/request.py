import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from engine.execution import ExecutionOptions, resolve_options
from engine.kv_cache import KVCache
from engine.kv_pool import KVHandle
from engine.metrics import RequestTimings
from engine.sampling import SamplingOptions
from engine.tracing import RequestTrace


# API facing request
class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=128, ge=0)
    stream: bool = True
    temperature: float = Field(default=0.0, ge=0, allow_inf_nan=False, strict=True)
    use_kv_cache: bool | None = None
    kv_cache_mode: str | None = None
    decode_batching: bool = False

    @model_validator(mode="after")
    def validate_execution(self):
        options = resolve_options(self.kv_cache_mode, self.use_kv_cache, self.decode_batching)
        self.kv_cache_mode = options.kv_cache_mode
        self.use_kv_cache = options.kv_cache_mode != "none"
        return self


# Internal request used by the engine
@dataclass
class Request:
    request_id: UUID = field(default_factory=uuid4)
    tokens: list[int] = field(default_factory=list)
    max_tokens: int = 128
    sampling: SamplingOptions = field(default_factory=SamplingOptions)
    generated_tokens: int = 0
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    cancelled: bool = False
    error: Exception | None = None
    timings: RequestTimings = field(default_factory=RequestTimings)
    finish_reason: str | None = None
    trace: RequestTrace | None = field(default=None, repr=False)
    use_kv_cache: bool = False
    kv_cache: KVCache | None = field(default=None, repr=False)

    kv_cache_mode: str | None = None
    decode_batching: bool = False
    execution: ExecutionOptions = field(init=False, repr=False)
    kv_handle: KVHandle | None = field(default=None, repr=False)
    cached_tokens: int = 0

    def __post_init__(self):
        self.execution = resolve_options(
            self.kv_cache_mode,
            self.use_kv_cache if self.kv_cache_mode is None else None,
            self.decode_batching,
        )
        self.kv_cache_mode = self.execution.kv_cache_mode
        self.use_kv_cache = self.kv_cache_mode != "none"
        self.timings.input_tokens = len(self.tokens)
