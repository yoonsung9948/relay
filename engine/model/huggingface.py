import torch
from torch import Tensor, nn
from typing import Any
from transformers import AutoModelForCausalLM


class HuggingFaceModel(nn.Module):
    def __init__(
        self,
        repo: str,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.model: Any = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=dtype,
        )

    @torch.inference_mode()
    def forward(self, input_ids: Tensor) -> Tensor:
        output = self.model(
            input_ids=input_ids,
            use_cache=False,
        )

        return output.logits