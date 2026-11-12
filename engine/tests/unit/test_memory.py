import asyncio
import json
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from engine.memory import CudaMemoryProbe, MemoryRecorder
from engine.processor import Request
from engine.tests.unit.test_streaming import make_engine


class FakeProbe:
    supported = True

    def __init__(self):
        self.begins = 0
        self.ends = 0

    def begin(self):
        self.begins += 1
        return {"allocated_bytes": 100, "reserved_bytes": 200}

    def end(self):
        self.ends += 1
        return (
            {"allocated_bytes": 100, "reserved_bytes": 400},
            {"allocated_bytes": 300, "reserved_bytes": 400},
        )


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_shared_peaks_reset_only_at_idle_and_flush_without_shutdown(tmp_path):
    path = tmp_path / "memory.jsonl"
    probe = FakeProbe()
    recorder = MemoryRecorder(path, probe, {"pool_allocated_bytes": 50})
    recorder.request_started("a")
    recorder.request_started("b")
    recorder.request_finished("a")
    recorder.request_started("c")
    assert probe.begins == 1
    assert not read(path)
    recorder.request_finished("b")
    assert probe.ends == 0
    recorder.request_finished("c")
    recorder.request_finished("c")
    window = read(path)[0]
    assert window["request_ids"] == ["a", "b", "c"]
    assert window["max_in_flight"] == 2
    assert window["request_count"] == 3
    assert window["complete"] is True
    assert window["peak"]["allocated_bytes"] == 300
    assert window["final"]["allocated_bytes"] == 100
    recorder.request_started("d")
    recorder.request_finished("d")
    assert probe.begins == probe.ends == 2
    assert len(read(path)) == 2


def test_interrupted_window_is_preserved_and_archive_appends(tmp_path):
    path = tmp_path / "memory.jsonl"
    for _ in range(2):
        recorder = MemoryRecorder(path, FakeProbe())
        recorder.request_started("request")
        recorder.close()
        recorder.close()
    windows = read(path)
    assert len(windows) == 2
    assert all(not window["complete"] for window in windows)
    assert windows[0]["instance_id"] != windows[1]["instance_id"]


def test_unsupported_device_is_null_not_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", Mock(side_effect=AssertionError))
    recorder = MemoryRecorder(tmp_path / "memory.jsonl", CudaMemoryProbe("cpu"))
    recorder.request_started("a")
    recorder.request_finished("a")
    window = read(recorder.path)[0]
    assert window["status"] == "unsupported_device"
    assert window["baseline"] is window["peak"] is window["final"] is None


def test_cuda_probe_uses_explicit_device_and_peak_counters(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name, value in [
        ("synchronize", None),
        ("reset_peak_memory_stats", None),
        ("memory_allocated", 100),
        ("memory_reserved", 200),
        ("max_memory_allocated", 300),
        ("max_memory_reserved", 400),
    ]:

        def call(device, name=name, value=value):
            assert device == torch.device("cuda:1")
            calls.append(name)
            return value

        monkeypatch.setattr(torch.cuda, name, call)
    probe = CudaMemoryProbe("cuda:1")
    assert probe.begin()["allocated_bytes"] == 100
    assert calls == [
        "synchronize",
        "memory_allocated",
        "memory_reserved",
        "reset_peak_memory_stats",
    ]
    assert probe.end()[1] == {"allocated_bytes": 300, "reserved_bytes": 400}
    assert calls.count("reset_peak_memory_stats") == 1


@pytest.mark.parametrize("failure", ["begin", "end"])
def test_probe_error_does_not_fail_generation(tmp_path, failure):
    probe = FakeProbe()
    setattr(probe, failure, Mock(side_effect=RuntimeError("CUDA unavailable")))
    recorder = MemoryRecorder(tmp_path / "memory.jsonl", probe)
    recorder.request_started("a")
    recorder.request_finished("a")
    assert read(recorder.path)[0]["status"] == "measurement_error"


def test_request_id_tracking_is_bounded_and_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.memory.MAX_WINDOW_REQUESTS", 2)
    recorder = MemoryRecorder(tmp_path / "memory.jsonl", FakeProbe())
    for request in ("a", "b", "c"):
        recorder.request_started(request)
    for request in ("a", "b", "c"):
        recorder.request_finished(request)
    window = read(recorder.path)[0]
    assert window["request_ids"] == ["a", "b"]
    assert window["request_ids_truncated"]
    assert window["request_count"] == 3


def test_disabled_by_default_and_write_failure_is_nonfatal(tmp_path, monkeypatch):
    monkeypatch.delenv("RELAY_MEMORY_PATH", raising=False)
    assert MemoryRecorder.from_env(None, {}) is None
    recorder = MemoryRecorder(tmp_path / "memory.jsonl", FakeProbe())
    recorder.request_started("a")
    recorder.path.unlink()
    recorder.path.mkdir()
    recorder.request_finished("a")
    assert recorder.disabled
    recorder.request_started("b")
    assert recorder.window is None


def test_env_setup_preserves_pool_and_engine_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAY_MEMORY_PATH", str(tmp_path / "nested" / "memory.jsonl"))
    runner = SimpleNamespace(
        device="cpu",
        cache_config=SimpleNamespace(dynamic_max_bytes=1024),
        pool=SimpleNamespace(allocated_bytes=512, owners=[None] * 4, capacity=2048),
    )
    recorder = MemoryRecorder.from_env(runner, {"model": "test"})
    recorder.request_started("a")
    recorder.request_finished("a")
    metadata = read(recorder.path)[0]["metadata"]
    assert metadata["pool_allocated_bytes"] == 512
    assert metadata["pool_slots"] == 4
    assert metadata["pool_max_seq_len"] == 2048
    assert metadata["dynamic_max_bytes"] == 1024
    assert metadata["model"] == "test"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_cuda_transient_peak_survives_tensor_release():
    probe = CudaMemoryProbe("cuda:0")
    baseline = probe.begin()
    tensor = torch.empty(4 * 1024 * 1024, device="cuda:0", dtype=torch.float32)
    allocated = probe.current()["allocated_bytes"]
    assert allocated >= baseline["allocated_bytes"] + tensor.numel() * tensor.element_size()
    del tensor
    final, peak = probe.end()
    assert peak["allocated_bytes"] >= allocated
    assert final["allocated_bytes"] < peak["allocated_bytes"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["success", "cancel", "failure", "shutdown"])
async def test_engine_lifecycle_closes_window_after_cleanup(tmp_path, stop):
    engine = make_engine(fail=stop == "failure")
    engine.memory = MemoryRecorder(tmp_path / "memory.jsonl", FakeProbe())
    loop = asyncio.create_task(engine.run_loop())
    await asyncio.sleep(0)
    request = Request(tokens=[1, 2], max_tokens=3)
    try:
        if stop == "success":
            await engine.generate(request)
        elif stop == "failure":
            with pytest.raises(RuntimeError):
                await engine.generate(request)
        else:
            engine.submit(request)
            if stop == "cancel":
                engine.cancel_request(request)
            else:
                await engine.shutdown()
        window = read(engine.memory.path)[0]
        assert window["request_ids"] == [str(request.request_id)]
        assert window["complete"] is (stop != "shutdown")
        engine.cancel_request(request)
        assert len(read(engine.memory.path)) == 1
    finally:
        loop.cancel()
        with suppress(asyncio.CancelledError, RuntimeError):
            await loop
