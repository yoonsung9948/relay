import torch
from transformers import AutoModelForCausalLM

from engine.model.qwen3 import Qwen3ForCausalLM


def load_custom_qwen3(
    repo: str,
    config,
    dtype: torch.dtype = torch.float16,
) -> Qwen3ForCausalLM:
    reference = AutoModelForCausalLM.from_pretrained(
        repo,
        torch_dtype=dtype,
    )

    model = Qwen3ForCausalLM(config)

    model = model.to(dtype=dtype)

    result = model.load_state_dict(
        reference.state_dict(),
        strict=True,
    )

    print(result)

    return model