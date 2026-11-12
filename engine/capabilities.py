"""Sanitized description of the running engine, without configuration secrets."""


def describe_engine(engine, config_path=None):
    runner = getattr(engine, "runner", None)
    scheduler = getattr(engine, "scheduler", None)
    config = getattr(scheduler, "config", None)
    pool = getattr(runner, "pool", None)
    metadata = getattr(engine, "generation_metadata", {})
    cached = getattr(runner, "supports_kv_cache", False) is True
    modes = ["none"]
    if cached:
        modes.append("dynamic")
    if pool is not None:
        modes.append("pool")
    instrumentation = dict(getattr(engine, "instrumentation", {}))
    memory = getattr(engine, "memory", None)
    instrumentation["memory_enabled"] = memory is not None and not memory.disabled
    instrumentation["memory_supported"] = bool(memory is not None and memory.probe.supported)
    return {
        "hardware": describe_hardware(runner),
        "temperature_sampling": True,
        "decode_batching": pool is not None and getattr(config, "max_decode_batch_size", 0) > 1,
        "max_seq_len": getattr(getattr(runner, "cache_config", None), "max_seq_len", None),
        "config_path": config_path,
        "identity": {
            key: metadata.get(key)
            for key in (
                "engine_revision",
                "engine_source_sha256",
                "engine_dirty",
                "model_revision",
                "config_sha256",
                "device_name",
                "torch_version",
                "transformers_version",
            )
        },
        "instrumentation": instrumentation,
        "in_flight_requests": len(getattr(engine, "_requests", {})),
        "active_requests": len(getattr(scheduler, "active", {})),
        "pending_requests": len(getattr(scheduler, "pending", [])),
        "max_pending_requests": getattr(config, "max_pending_requests", 0),
        "pool_max_seq_len": getattr(pool, "capacity", 0),
        "model": metadata.get("model"),
        "backend": metadata.get("backend"),
        "device": str(runner.device) if runner is not None and hasattr(runner, "device") else None,
        "dtype": metadata.get("dtype"),
        "kv_cache_modes": modes,
        "pool_slots": len(pool.owners) if pool is not None else 0,
        "pool_allocated_bytes": pool.allocated_bytes if pool is not None else 0,
        "max_active_requests": getattr(config, "max_active_requests", 0),
        "max_decode_batch_size": getattr(config, "max_decode_batch_size", 0),
    }


def describe_hardware(runner):
    """Report visible CUDA hardware, never infer it from a requested GPU type."""
    hardware = {
        "gpu_name": None,
        "visible_gpu_count": None,
        "vram_bytes_per_gpu": None,
        "cuda_version": None,
    }
    if runner is None or not str(getattr(runner, "device", "")).startswith("cuda"):
        return hardware
    try:
        import torch

        properties = torch.cuda.get_device_properties(runner.device)
        hardware.update(
            gpu_name=properties.name,
            visible_gpu_count=torch.cuda.device_count(),
            vram_bytes_per_gpu=properties.total_memory,
            cuda_version=torch.version.cuda,
        )
    except (ImportError, RuntimeError, AssertionError, AttributeError, IndexError):
        pass
    return hardware
