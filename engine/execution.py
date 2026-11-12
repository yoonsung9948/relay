from dataclasses import dataclass
from typing import Literal

CacheMode = Literal["none", "dynamic", "pool"]


@dataclass(frozen=True)
class ExecutionOptions:
    kv_cache_mode: CacheMode = "none"
    decode_batching: bool = False


def resolve_options(mode=None, legacy=None, batching=False) -> ExecutionOptions:
    if mode is None:
        mode = "dynamic" if legacy else "none"
    if mode not in ("none", "dynamic", "pool"):
        raise ValueError("unknown kv_cache_mode")
    if legacy is not None and legacy != (mode != "none"):
        raise ValueError("conflicting use_kv_cache and kv_cache_mode")
    if batching and mode != "pool":
        raise ValueError("decode_batching requires kv_cache_mode=pool")
    return ExecutionOptions(mode, batching)


class CapacityError(RuntimeError):
    pass
