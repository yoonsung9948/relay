from typing import Protocol
from transformers import AutoTokenizer

class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, tokens: list[int]) -> str:
        ...


class Qwen3Tokenizer:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
    def encode(
        self, 
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(
        self,
        tokens: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)