import torch
import torch.nn as nn


class FakeModel(nn.Module):
    def __init__(self, logits: torch.Tensor, vocab_size: int = 32):
        super().__init__()
        self.call_count = 0
        self.last_input: torch.Tensor | None = None
        self.logits = logits
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        self.last_input = input_ids.clone()
        return self.logits

    