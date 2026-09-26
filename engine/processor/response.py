from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel
from .tokenizer import Tokenizer


# Internal engine output
@dataclass
class GenerateResult:
    request_id: UUID
    token_ids: list[int]
    generated_tokens: int

# API-facing output
class GenerateResponse(BaseModel):
    request_id: UUID | None
    text: str
    generated_tokens: int


def build_generate_response(
    response: GenerateResult,
    tokenizer: Tokenizer,
    generated_tokens: int,
) -> GenerateResponse:
    return GenerateResponse(
        request_id=response.request_id,
        text=tokenizer.decode(response.token_ids),
        generated_tokens=generated_tokens,
    )