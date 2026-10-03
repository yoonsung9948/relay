import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from engine.model.qwen3 import Qwen3ForCausalLM


def load_custom_qwen3(
    repo: str,
    config,
    dtype: torch.dtype = torch.float16,
) -> Qwen3ForCausalLM:
    model = Qwen3ForCausalLM(config)
    model = model.to(dtype=dtype)

    checkpoint_dir = snapshot_download(
        repo,
        allow_patterns=["*.safetensors", "*.safetensors.index.json"],
    )

    state_dict = {}
    for filename in sorted(os.listdir(checkpoint_dir)):
        if filename.endswith(".safetensors"):
            state_dict.update(
                load_file(os.path.join(checkpoint_dir, filename), device="cpu")
            )

    result = model.load_state_dict(state_dict, strict=True)
    print(result)


    return model