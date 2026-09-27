from dataclasses import dataclass
import logging
from engine.engine import Engine
from engine.processor import Tokenizer


@dataclass(frozen=True)
class Resources:
    engine: Engine
    tokenizer: Tokenizer
    logger: logging.Logger
