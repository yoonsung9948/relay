from dataclasses import dataclass

from engine.engine import Engine
from engine.processor import Tokenizer


@dataclass(frozen=True)
class Resources:
    engine: Engine
    tokenizer: Tokenizer
