"""Versioned, process-local generation measurements (never GPU kernel timings)."""

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GenerationMetrics(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    version: Literal[1] = 1
    source: Literal["engine"] = "engine"
    prefill_ms: float | None = Field(default=None, ge=0)
    decode_steps: int = Field(default=0, ge=0)
    decode_batch_size_max: int = Field(default=0, ge=0)
    decode_batch_size_mean: float | None = Field(default=None, ge=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    ttft_ms: float | None = Field(default=None, ge=0)
    decode_duration_ms: float | None = Field(default=None, ge=0)
    decode_tokens_per_second: float | None = Field(default=None, ge=0)
    total_latency_ms: float = Field(ge=0)
    initial_queue_ms: float | None = Field(default=None, ge=0)


@dataclass
class RequestTimings:
    """All timestamps share one monotonic clock; only durations leave Python."""

    clock: Callable[[], float] = field(default=perf_counter, repr=False)
    started_at: float | None = None
    input_tokens: int = 0
    enqueued_at: float | None = None
    scheduled_at: float | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    _final: GenerationMetrics | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = self.clock()

    prefill_ms: float | None = None
    decode_batch_sizes: list[int] = field(default_factory=list)

    def enqueue(self):
        self.enqueued_at = self.clock()

    def schedule(self):
        if self.scheduled_at is None:
            self.scheduled_at = self.clock()

    def token(self):
        now = self.clock()
        if self.first_token_at is None:
            self.first_token_at = now
        self.last_token_at = now

    def finish(self, output_tokens: int) -> GenerationMetrics:
        if self._final is not None:
            return self._final
        assert self.started_at is not None
        decode = None
        rate = None
        if output_tokens > 1 and self.first_token_at is not None:
            assert self.last_token_at is not None
            decode = self.last_token_at - self.first_token_at
            if decode > 0:
                rate = (output_tokens - 1) / decode
        self._final = GenerationMetrics(
            prefill_ms=self.prefill_ms,
            decode_steps=len(self.decode_batch_sizes),
            decode_batch_size_max=max(self.decode_batch_sizes, default=0),
            decode_batch_size_mean=sum(self.decode_batch_sizes) / len(self.decode_batch_sizes)
            if self.decode_batch_sizes
            else None,
            input_tokens=self.input_tokens,
            output_tokens=output_tokens,
            ttft_ms=(self.first_token_at - self.started_at) * 1000
            if self.first_token_at is not None
            else None,
            decode_duration_ms=decode * 1000 if decode is not None else None,
            decode_tokens_per_second=rate,
            total_latency_ms=(self.clock() - self.started_at) * 1000,
            initial_queue_ms=(self.scheduled_at - self.enqueued_at) * 1000
            if self.scheduled_at is not None and self.enqueued_at is not None
            else None,
        )
        return self._final
