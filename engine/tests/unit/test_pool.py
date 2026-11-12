import asyncio
from contextlib import suppress

import pytest
import torch

from engine.config import SchedulerConfig
from engine.config.config import KVPoolConfig
from engine.engine import EngineState
from engine.execution import CapacityError, resolve_options
from engine.kv_pool import KVPool
from engine.processor import GenerateRequest, Request
from engine.scheduler import Scheduler
from engine.tests.unit.test_kv_cache import make_engine
from engine.tests.unit.test_kv_cache import model as model


def pooled(model, slots=3, active=3):
    engine = make_engine(model)
    engine.runner.configure_cache(KVPoolConfig(slots=slots, max_seq_len=32, max_bytes=1000000))
    engine.scheduler = Scheduler(
        SchedulerConfig(max_active_requests=active, max_decode_batch_size=3, max_pending_requests=4)
    )
    engine.scheduler.admit = engine.runner.admit
    return engine


@torch.inference_mode()
def test_batched_variable_lengths_match_full_context(model):
    engine = pooled(model)
    requests = [
        Request(tokens=[1, 2], kv_cache_mode="pool", decode_batching=True),
        Request(tokens=[3, 4, 5, 6], kv_cache_mode="pool", decode_batching=True),
    ]
    for req in requests:
        assert engine.runner.admit(req)
        token = engine.runner.run_requests([req])[0]
        assert token == engine.runner.run([req.tokens])[0]
        req.tokens.append(token)
    for _ in range(4):
        expected = [engine.runner.run([req.tokens])[0] for req in requests]
        actual = engine.runner.run_requests(requests)
        assert actual == expected
        for req, token in zip(requests, actual):
            req.tokens.append(token)
    assert requests[0].kv_handle.slot != requests[1].kv_handle.slot


@pytest.mark.asyncio
async def test_join_leave_and_slot_reuse(model):
    engine = pooled(model, slots=2, active=2)
    observed = []
    run = engine.runner.run_requests

    def record(batch):
        observed.append((len(batch), [req.generated_tokens == 0 for req in batch]))
        return run(batch)

    engine.runner.run_requests = record
    loop = asyncio.create_task(engine.run_loop())
    await asyncio.sleep(0)
    requests = [
        Request(tokens=[1, 2 + i], max_tokens=length, kv_cache_mode="pool", decode_batching=True)
        for i, length in enumerate([2, 5, 3])
    ]
    tasks = [asyncio.create_task(engine.generate(req)) for req in requests]
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks), 5)
        assert [r.generated_tokens for r in results] == [2, 5, 3]
        assert any(size == 2 for size, _ in observed)
        assert all(not any(prefill) or size == 1 for size, prefill in observed)
        assert engine.runner.pool.owners == [None, None]
        assert not engine.scheduler.active
        assert all(req.kv_handle is None for req in requests)
    finally:
        loop.cancel()
        with suppress(asyncio.CancelledError):
            await loop


def test_pool_budget_and_stale_release():
    with pytest.raises(ValueError, match="budget"):
        KVPool(2, 2, 2, 32, 8, device="cpu", dtype=torch.float32, max_bytes=1)
    pool = KVPool(1, 1, 1, 4, 2, device="cpu", dtype=torch.float32, max_bytes=1024)
    first = pool.allocate("one")
    assert pool.allocate("two") is None
    pool.release(first)
    second = pool.allocate("two")
    pool.release(first)
    assert pool.owners == ["two"]
    with pytest.raises(ValueError, match="stale"):
        pool.validate(first)
    pool.validate(second)


def test_capacity_and_option_validation(model):
    engine = pooled(model)
    engine.state = EngineState.READY
    for _ in range(4):
        engine.submit(Request(tokens=[1], max_tokens=1))
    with pytest.raises(CapacityError):
        engine.submit(Request(tokens=[1], max_tokens=1))
    assert len(engine._requests) == 4
    with pytest.raises(ValueError):
        resolve_options("pool", False, True)
    with pytest.raises(ValueError):
        GenerateRequest(prompt="hi", kv_cache_mode="dynamic", decode_batching=True)
    assert GenerateRequest(prompt="hi", use_kv_cache=True).kv_cache_mode == "dynamic"
    assert GenerateRequest(prompt="hi").kv_cache_mode == "none"


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["cancel", "failure", "shutdown"])
async def test_pool_release_terminal_paths(model, stop):
    engine = pooled(model)
    engine.state = EngineState.READY
    req = Request(tokens=[1, 2], max_tokens=10, kv_cache_mode="pool", decode_batching=True)
    engine.submit(req)
    batch = engine.scheduler.schedule()
    engine.runner.run_requests(batch)
    engine.scheduler.finish_step(req, finished=False)
    assert req.kv_handle is not None
    if stop == "cancel":
        engine.cancel_request(req)
    elif stop == "shutdown":
        await engine.shutdown()
    else:
        engine._fail_active("model failed")
    assert req.kv_handle is None
    assert req.done_event.is_set()
    assert engine.runner.pool.owners == [None] * 3
    assert not engine.scheduler.active


@torch.inference_mode()
def test_pool_logits_with_unequal_lengths(model):
    engine = pooled(model)
    requests = [
        Request(tokens=[1, 2], kv_cache_mode="pool"),
        Request(tokens=[3, 4, 5, 6], kv_cache_mode="pool"),
    ]
    for req in requests:
        engine.runner.admit(req)
        context = engine.runner.pool.batch([req.kv_handle])
        ids = torch.tensor([req.tokens])
        torch.testing.assert_close(model(ids, cache=context), model(ids), rtol=1e-4, atol=1e-5)
        context.commit()
        req.tokens.append(7)
    for _ in range(3):
        context = engine.runner.pool.batch([req.kv_handle for req in requests])
        actual = model(torch.tensor([[req.tokens[-1]] for req in requests]), cache=context)
        for row, req in enumerate(requests):
            expected = model(torch.tensor([req.tokens]))[:, -1:, :]
            torch.testing.assert_close(actual[row : row + 1], expected, rtol=1e-4, atol=1e-5)
        context.commit()
        for req in requests:
            req.tokens.append(8)



def test_active_limit_is_independent_of_queue_capacity(model):
    engine = pooled(model, slots=3, active=1)
    engine.state = EngineState.READY
    requests = [Request(tokens=[1], max_tokens=3, kv_cache_mode="pool") for _ in range(3)]
    for request in requests:
        engine.submit(request)
    batch = engine.scheduler.schedule()
    assert len(batch) == 1
    assert len(engine.scheduler.active) == 1
    assert len(engine.scheduler.pending) == 2
    engine._complete_request(batch[0])
    assert batch[0].kv_handle is None
    assert len(engine.scheduler.schedule()) == 1
    assert len(engine.scheduler.active) == 1


def test_http_capacity_is_reported_before_stream_headers(model):
    import logging
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from engine.api.server import create_app
    from engine.boot_state import AppState, BootStatus
    from engine.config.config import AppConfig

    engine = pooled(model)
    engine.state = EngineState.READY
    engine.scheduler.config.max_pending_requests = 1
    engine.submit(Request(tokens=[1], max_tokens=1))

    @asynccontextmanager
    async def lifespan(app):
        app.state.boot = AppState(status=BootStatus.READY, resources=SimpleNamespace(
            engine=engine, tokenizer=engine.tokenizer, logger=logging.getLogger("test"),
        ))
        yield

    with TestClient(create_app(lifespan, AppConfig())) as client:
        response = client.post("/generate", json={"prompt":"hello", "max_tokens":3,
            "kv_cache_mode":"pool", "decode_batching":True})
        assert response.status_code == 503
        assert response.headers["retry-after"] == "1"
        assert response.headers["content-type"].startswith("application/json")
        assert len(engine._requests) == 1
        invalid = client.post("/generate", json={"prompt":"hello", "max_tokens":1000})
        assert invalid.status_code == 422
