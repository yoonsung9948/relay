"""Concatenating, per-request inference cache; KV heads stay unexpanded."""

import torch

KVPair = tuple[torch.Tensor, torch.Tensor]


class KVCache:
    def __init__(self, num_layers: int):
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.layers: list[KVPair | None] = [None] * num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        entry = self.layers[layer_idx]
        return 0 if entry is None else entry[0].shape[-2]

    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> KVPair:
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError("K/V must have matching [batch, kv_heads, sequence, head_dim] shapes")
        if key.device != value.device or key.dtype != value.dtype:
            raise ValueError("K/V must have matching devices and dtypes")
        previous = self.layers[layer_idx]
        if previous is not None:
            previous_key, previous_value = previous
            if (
                previous_key.shape[:2] != key.shape[:2]
                or previous_key.shape[-1] != key.shape[-1]
                or previous_key.device != key.device
                or previous_key.dtype != key.dtype
            ):
                raise ValueError("new K/V must match the existing request cache")
            key = torch.cat((previous_key, key), dim=-2)
            value = torch.cat((previous_value, value), dim=-2)
        self.layers[layer_idx] = (key, value)
        return key, value

    def clear(self) -> None:
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx] = None
