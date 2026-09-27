from typing import Protocol, Literal, TypedDict
from transformers import AutoTokenizer

class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, tokens: list[int]) -> str:
        ...

    def encode_chat(
        self,
        prompt: str,
        enable_thinking: bool = False,
    ) -> list[int]:
        ...

class Qwen3Tokenizer:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        self.eos_token_ids = frozenset([self.tokenizer.eos_token_id])
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

    def encode_chat(
        self,
        prompt: str,
        enable_thinking: bool = False,
    ) -> list[int]:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            return_dict=False,
        )