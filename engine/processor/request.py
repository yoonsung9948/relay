import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from engine.processor.tokenizer import Qwen3Tokenizer, Tokenizer
from pydantic import BaseModel

# API facing request
class GenerateRequest(BaseModel):
    prompt: str    # raw prompt
    max_tokens: int = 128

# Internal request used by the engine
@dataclass
class Request:
    request_id: UUID = field(default_factory=uuid4)
    tokens: list[int] = field(default_factory=list)
    max_tokens: int = 128
    generated_tokens: int = 0
    done_event: asyncio.Event = field(default_factory=asyncio.Event)