from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from engine.config import Qwen3Config
from engine.kv_cache import KVCache
from engine.kv_pool import PoolBatch


def qwen3_06b_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        tie_word_embeddings=True,
    )


def qwen3_8b_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        tie_word_embeddings=False,
    )


class Qwen3ForCausalLM(nn.Module):
    supports_kv_cache = True

    def __init__(
        self,
        config: Qwen3Config,
    ):
        super().__init__()

        self.num_hidden_layers = config.num_hidden_layers
        self.model = Qwen3Model(config)

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        cache: KVCache | PoolBatch | None = None,
    ) -> torch.Tensor:

        x = self.model(
            input_ids,
            position_ids=position_ids,
            cache=cache,
        )

        logits = self.lm_head(x)

        return logits


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        from transformers import Qwen3Config as HFQwen3Config

        hf_config = HFQwen3Config(**config.model_dump())
        self.rotary_emb = Qwen3RotaryEmbedding(hf_config)

        self.layers = nn.ModuleList(
            [Qwen3Layer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        cache: KVCache | PoolBatch | None = None,
    ) -> torch.Tensor:
        # Capture once: layer zero's cache grows before the other layers run.
        past_length = cache.get_seq_length() if cache is not None else 0
        if cache is not None and not isinstance(cache, PoolBatch):
            if len(cache.layers) != len(self.layers):
                raise ValueError("cache layer count does not match the model")
            if any(cache.get_seq_length(i) != past_length for i in range(len(self.layers))):
                raise ValueError("cache layers have inconsistent sequence lengths")
            if past_length and input_ids.shape[1] != 1:
                raise ValueError("cached decoding currently accepts one new token per step")
        x = self.embed_tokens(input_ids)

        _, T, _ = x.shape

        if position_ids is None and isinstance(cache, PoolBatch):
            position_ids = cache.positions(T, x.device)
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + T,
                device=x.device,
            ).unsqueeze(0)

        position_embeddings = self.rotary_emb(
            x,
            position_ids,
        )

        for layer in self.layers:
            x = layer(
                x,
                position_embeddings,
                cache=cache,
            )

        return self.norm(x)


class Qwen3Layer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int = 0,
    ):
        super().__init__()

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.self_attn = Qwen3SelfAttention(config, layer_idx)

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.mlp = Qwen3MLP(
            config.hidden_size,
            config.intermediate_size,
        )

    def forward(
        self, x: torch.Tensor, positional_embeddings, cache: KVCache | PoolBatch | None = None
    ) -> torch.Tensor:
        # Attention sublayer
        residual = x

        x = self.input_layernorm(x)
        x = self.self_attn(x, positional_embeddings, cache=cache)

        x = residual + x

        # MLP sublayer
        residual = x

        x = self.post_attention_layernorm(x)
        x = self.mlp(x)

        x = residual + x

        return x


class Qwen3SelfAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int = 0,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.q_size,
            bias=config.attention_bias,
        )

        self.k_proj = nn.Linear(
            self.hidden_size,
            self.kv_size,
            bias=config.attention_bias,
        )

        self.v_proj = nn.Linear(
            self.hidden_size,
            self.kv_size,
            bias=config.attention_bias,
        )

        self.o_proj = nn.Linear(
            self.q_size,
            self.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )

        self.k_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        positional_embeddings,
        cache: KVCache | PoolBatch | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        past_length = cache.get_seq_length(self.layer_idx) if cache is not None else 0

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = positional_embeddings

        q, k = apply_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
        )

        # Cache normalized, rotated K and projected V before expanding GQA heads.
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)

        k = k.repeat_interleave(
            self.num_kv_groups,
            dim=1,
        )

        v = v.repeat_interleave(
            self.num_kv_groups,
            dim=1,
        )

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            # A single newest query may attend to every cached key. Using the
            # square-prefill mask for non-square decoding would hide past keys.
            attn_mask=cache.mask(T, q.device) if isinstance(cache, PoolBatch) else None,
            is_causal=past_length == 0
            and not (isinstance(cache, PoolBatch) and any(cache.previous)),
        )

        x = x.transpose(1, 2).contiguous().view(B, T, self.q_size)

        return self.o_proj(x)


# TODO: Implement a custom rotary positional embedding class to replace Qwen3RotaryEmbedding.
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        # Apply rotary positional embedding to q and k
        # This is a placeholder implementation; replace with actual rotary embedding logic
        return q, k


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-8):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype

        # explicitly promote to float to match HG logits
        x = x.float()

        variance = x.pow(2).mean(
            -1,
            keepdim=True,
        )

        x = x * torch.rsqrt(variance + self.eps)

        return self.weight * x.to(input_dtype)


class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)

        return self.down_proj(gate * up)
