from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from engine.metrics import GenerationMetrics

from .tokenizer import Tokenizer

GenerationMetadata = dict[str, str | int | float | bool | None]


# Internal engine output
@dataclass
class GenerateResult:
    request_id: UUID
    token_ids: list[int]
    generated_tokens: int


# API-facing output
class GenerateResponse(BaseModel):
    request_id: UUID | None
    generated_tokens: int
    text: str
    metrics: GenerationMetrics | None = None
    finish_reason: Literal["eos", "length"] | None = None
    generation: GenerationMetadata | None = None


class StreamResponse(BaseModel):
    type: Literal["delta", "done", "error"] = "delta"
    request_id: UUID | None
    text: str = ""
    error: str | None = None
    metrics: GenerationMetrics | None = None
    finish_reason: Literal["eos", "length"] | None = None
    generation: GenerationMetadata | None = None


def build_generate_response(
    response: GenerateResult,
    tokenizer: Tokenizer,
    generated_tokens: int,
) -> GenerateResponse:
    count = response.generated_tokens
    output_ids = response.token_ids[-count:] if count > 0 else []

    return GenerateResponse(
        request_id=response.request_id,
        text=tokenizer.decode(output_ids),
        generated_tokens=count,
    )
