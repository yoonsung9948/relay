from types import SimpleNamespace

import torch

from engine.capabilities import describe_engine, describe_hardware


def test_cuda_hardware_is_reported_from_worker(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(name="Actual GPU", total_memory=24 * 2**30),
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    value = describe_hardware(SimpleNamespace(device="cuda:0"))
    assert value["gpu_name"] == "Actual GPU"
    assert value["vram_bytes_per_gpu"] == 24 * 2**30
    assert value["visible_gpu_count"] == 1


def test_missing_hardware_never_fabricates_specs(monkeypatch):
    assert describe_hardware(SimpleNamespace(device="cpu"))["gpu_name"] is None

    def failed(device):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(torch.cuda, "get_device_properties", failed)
    assert describe_hardware(SimpleNamespace(device="cuda:0"))["visible_gpu_count"] is None


def test_batching_capability_requires_enabled_pool():
    runner = SimpleNamespace(device="cpu", supports_kv_cache=True, pool=None)
    engine = SimpleNamespace(
        runner=runner, scheduler=SimpleNamespace(config=SimpleNamespace(max_decode_batch_size=4))
    )
    assert describe_engine(engine)["decode_batching"] is False
    runner.pool = SimpleNamespace(capacity=32, owners={}, allocated_bytes=0)
    caps = describe_engine(engine)
    assert caps["kv_cache_modes"] == ["none", "dynamic", "pool"]
    assert caps["decode_batching"] is True
