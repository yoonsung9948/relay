from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel
from .tokenizer import Tokenizer


# Internal engine output
@dataclass
class GenerateResult:
    request_id: UUID
    token_ids: list[int]
    generated_tokens: list[int]

# API-facing output
class GenerateResponse(BaseModel):
    request_id: UUID
    text: str
    generated_tokens: list[int]


def build_generate_response(
    response: GenerateResult,
    tokenizer: Tokenizer,
    generated_tokens: list[int],
) -> GenerateResponse:
    return GenerateResponse(
        request_id=response.request_id,
        text=tokenizer.decode(response.generated_tokens),
        generated_tokens=generated_tokens,
    )