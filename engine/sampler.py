import torch
from typing import Protocol

class Sampler(Protocol):
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        ...

class GreedySampler:
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)