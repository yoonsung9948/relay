from inspect import signature
from pathlib import Path

import pytest
import torch

from engine.cli.cli import serve
from engine.config import load_config
from engine.model.qwen3 import Qwen3ForCausalLM, qwen3_8b_config

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("filename", ["docker/engine.yaml", "examples/configs/config.yaml"])
def test_8b_launch_configs_match_checkpoint(filename):
    config = load_config(str(ROOT / filename))
    assert config.backend == "custom"
    assert config.model.repo == "Qwen/Qwen3-8B"
    assert config.model.parameter_count_b == 8.2
    assert config.architecture == qwen3_8b_config()
    assert config.architecture.hidden_size == 4096
    assert config.architecture.intermediate_size == 12288
    assert config.architecture.num_hidden_layers == 36
    assert config.architecture.num_attention_heads == 32
    assert config.architecture.num_key_value_heads == 8
    assert config.architecture.head_dim == 128
    assert config.architecture.tie_word_embeddings is False
    assert config.architecture.rope_parameters["rope_theta"] == 1000000
    assert config.scheduler.batch_size == 1


def test_cli_defaults_to_verified_8b_config():
    assert signature(serve).parameters["path"].default == "docker/engine.yaml"


def test_8b_parameter_shapes_without_allocating_weights():
    config = load_config(str(ROOT / "docker/engine.yaml"))
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config.architecture)
    assert len(model.model.layers) == 36
    assert model.model.embed_tokens.weight.shape == (151936, 4096)
    assert model.lm_head.weight.shape == (151936, 4096)
    assert model.lm_head.weight is not model.model.embed_tokens.weight
    attention = model.model.layers[0].self_attn
    assert attention.q_proj.weight.shape == (4096, 4096)
    assert attention.k_proj.weight.shape == (1024, 4096)
    assert model.model.layers[0].mlp.up_proj.weight.shape == (12288, 4096)
