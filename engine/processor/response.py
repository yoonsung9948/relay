from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel
from .tokenizer import Tokenizer


# Internal engine output
@dataclass
class Response:
    request_id: UUID
    token_ids: list[int]


# API-facing output
class GenerateResponse(BaseModel):
    request_id: UUID
    text: str


def build_generate_response(
    response: Response,
    tokenizer: Tokenizer,
) -> GenerateResponse:
    return GenerateResponse(
        request_id=response.request_id,
        text=tokenizer.decode(response.token_ids),
    )