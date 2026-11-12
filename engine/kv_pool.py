"""Fixed GPU slots. Release returns capacity to the pool, not to the driver."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVHandle:
    slot: int
    generation: int


class KVPool:
    def __init__(self, layers, slots, heads, capacity, head_dim, *, device, dtype, max_bytes):
        required = (
            2
            * layers
            * slots
            * heads
            * capacity
            * head_dim
            * torch.empty((), dtype=dtype).element_size()
        )
        if required > max_bytes:
            raise ValueError(f"KV pool requires {required} bytes; budget is {max_bytes}")
        self.capacity = capacity
        self.allocated_bytes = required
        self.keys = [
            torch.empty((slots, heads, capacity, head_dim), device=device, dtype=dtype)
            for _ in range(layers)
        ]
        self.values = [torch.empty_like(k) for k in self.keys]
        self.owners = [None] * slots
        self.generations = [0] * slots
        self.lengths = [0] * slots

    def allocate(self, owner):
        for slot, current in enumerate(self.owners):
            if current is None:
                self.generations[slot] += 1
                self.owners[slot] = owner
                self.lengths[slot] = 0
                return KVHandle(slot, self.generations[slot])
        return None

    def validate(self, handle):
        if (
            not (0 <= handle.slot < len(self.owners))
            or self.owners[handle.slot] is None
            or self.generations[handle.slot] != handle.generation
        ):
            raise ValueError("stale KV handle")

    def release(self, handle):
        if (
            0 <= handle.slot < len(self.owners)
            and self.generations[handle.slot] == handle.generation
        ):
            self.owners[handle.slot] = None
            self.lengths[handle.slot] = 0

    def batch(self, handles):
        return PoolBatch(self, handles)


class PoolBatch:
    def __init__(self, pool, handles):
        if len({h.slot for h in handles}) != len(handles):
            raise ValueError("duplicate KV slots in batch")
        for handle in handles:
            pool.validate(handle)
        self.pool = pool
        self.handles = handles
        self.previous = [pool.lengths[h.slot] for h in handles]
        self.updated = set()
        self.added = None

    def get_seq_length(self, layer_idx=0):
        return self.previous[0]

    def positions(self, count, device):
        return (
            torch.tensor(self.previous, device=device)[:, None]
            + torch.arange(count, device=device)[None, :]
        )

    def update(self, layer_idx, key, value):
        count = key.shape[-2]
        if any(n and count != 1 for n in self.previous):
            raise ValueError("decode requires one new token")
        if len(self.handles) != key.shape[0] or any(
            n + count > self.pool.capacity for n in self.previous
        ):
            raise ValueError("KV capacity exceeded")
        if layer_idx in self.updated or (self.added is not None and self.added != count):
            raise ValueError("inconsistent layer updates")
        self.added = count
        self.updated.add(layer_idx)
        width = max(self.previous) + count
        keys, values = [], []
        for row, (handle, previous) in enumerate(zip(self.handles, self.previous)):
            self.pool.validate(handle)
            end = previous + count
            k = self.pool.keys[layer_idx][handle.slot]
            v = self.pool.values[layer_idx][handle.slot]
            k[:, previous:end].copy_(key[row])
            v[:, previous:end].copy_(value[row])
            # Never expose stale/uninitialized padding to attention.
            out_k = torch.zeros_like(k[:, :width])
            out_v = torch.zeros_like(v[:, :width])
            out_k[:, :end].copy_(k[:, :end])
            out_v[:, :end].copy_(v[:, :end])
            keys.append(out_k)
            values.append(out_v)
        return torch.stack(keys), torch.stack(values)

    def mask(self, count, device):
        if all(n == 0 for n in self.previous):
            return None
        width = max(self.previous) + count
        lengths = torch.tensor(self.previous, device=device) + count
        return (torch.arange(width, device=device)[None, :] < lengths[:, None])[:, None, None, :]

    def commit(self):
        if len(self.updated) != len(self.pool.keys):
            raise ValueError("not every layer updated the cache")
        for handle, previous in zip(self.handles, self.previous):
            self.pool.validate(handle)
            self.pool.lengths[handle.slot] = previous + self.added
