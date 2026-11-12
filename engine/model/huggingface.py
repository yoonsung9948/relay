from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM


class HuggingFaceModel(nn.Module):
    def __init__(
        self,
        repo: str,
        dtype: torch.dtype = torch.float16,
        *,
        revision: str | None = None,
    ):
        super().__init__()

        self.model: Any = AutoModelForCausalLM.from_pretrained(
            repo,
            dtype=dtype,
            revision=revision,
        )

    @torch.inference_mode()
    def forward(self, input_ids: Tensor) -> Tensor:
        output = self.model(
            input_ids=input_ids,
            use_cache=False,
        )

        return output.logits
