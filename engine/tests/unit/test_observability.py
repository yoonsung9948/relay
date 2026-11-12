import asyncio
from contextlib import asynccontextmanager, suppress

import pytest
from fastapi.testclient import TestClient

from engine.api.server import create_app
from engine.config.config import AppConfig
from engine.engine import EngineState
from engine.execution import CapacityError
from engine.observability import Metrics, engine_snapshot
from engine.processor import Request
from engine.tests.unit.test_kv_cache import model as model
from engine.tests.unit.test_pool import pooled
from engine.tests.unit.test_tracing import tracing as tracing


@pytest.mark.asyncio
async def test_pool_metrics_count_physical_steps_and_release_capacity(model, tracing):
    engine = pooled(model, slots=2, active=2)
    instrumentation, exporter = tracing
    requests = [
        Request(tokens=[1, 2], max_tokens=4, kv_cache_mode="pool", decode_batching=True)
        for _ in range(2)
    ]
    for request in requests:
        request.trace = instrumentation.request(request.request_id)
    calls = []
    run = engine.runner.run_requests

    def record(batch):
        calls.append(len(batch))
        return run(batch)

    engine.runner.run_requests = record
    loop = asyncio.create_task(engine.run_loop())
    await asyncio.sleep(0)
    try:
        await asyncio.wait_for(asyncio.gather(*(engine.generate(r) for r in requests)), 5)
        for request in requests:
            request.trace.finish("success", request)
        text = engine_snapshot(engine)
        assert "relay_engine_kv_pool_slots_used 0\n" in text
        assert "relay_engine_active_requests 0\n" in text
        assert "relay_engine_kv_pool_tokens_used 0\n" in text
        states = [
            v
            for (name, _), v in engine.metrics.histograms.items()
            if name == "relay_engine_step_batch_size"
        ]
        assert sum(v[0] for v in states) == len(calls)
        assert sum(v[1] for v in states) == sum(calls)
        for span in exporter.get_finished_spans():
            if span.name == "runner.step":
                assert span.attributes["input_tokens"] == (
                    2 if span.attributes["step.index"] == 0 else 1
                )
        metrics = requests[0].timings.finish(4)
        assert metrics.decode_steps == 3
        assert metrics.prefill_ms is not None
        before = engine.metrics.render()
        engine.cancel_request(requests[0])
        assert engine.metrics.render() == before
    finally:
        loop.cancel()
        with suppress(asyncio.CancelledError):
            await loop


@pytest.mark.parametrize("cancel", [False, True])
def test_failed_and_cancelled_requests_release_resources_once(model, cancel):
    engine = pooled(model, slots=1)
    engine.state = EngineState.READY
    request = Request(tokens=[1], max_tokens=2, kv_cache_mode="pool")
    engine.submit(request)
    engine.scheduler.schedule()
    assert "relay_engine_kv_pool_slots_used 1\n" in engine_snapshot(engine)
    if cancel:
        engine.cancel_request(request)
    else:
        engine._fail_active("test failure")
    snapshot = engine_snapshot(engine)
    assert "relay_engine_kv_pool_slots_used 0\n" in snapshot
    outcome = "cancelled" if cancel else "failed"
    assert f'outcome="{outcome}"' in snapshot
    assert (
        engine.metrics.counters[
            (
                "relay_engine_generations_total",
                (("batching", "false"), ("mode", "pool"), ("outcome", outcome)),
            )
        ]
        == 1
    )


def test_queue_rejection_is_counted_without_creating_generation(model):
    engine = pooled(model)
    engine.state = EngineState.READY
    for _ in range(4):
        engine.submit(Request(tokens=[1], max_tokens=1))
    with pytest.raises(CapacityError):
        engine.submit(Request(tokens=[1], max_tokens=1))
    assert 'relay_engine_rejections_total{reason="queue_full"} 1\n' in engine_snapshot(engine)


def test_metrics_available_before_model_boot_and_exclude_scrapes():
    @asynccontextmanager
    async def lifespan(app):
        yield

    with TestClient(create_app(lifespan, AppConfig())) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "relay_engine_ready 0\n" in response.text
        assert "http_requests_total" not in response.text
        client.post("/generate", json={"max_tokens": -1})
        assert 'outcome="rejected",status="422"' in client.get("/metrics").text


def test_histogram_buckets_are_cumulative_and_bounded():
    metrics = Metrics()
    for value in (0, 0.1, 1000):
        metrics.observe("duration_seconds", value)
    text = metrics.render()
    assert 'duration_seconds_bucket{le="0.1"} 2\n' in text
    assert 'duration_seconds_bucket{le="+Inf"} 3\n' in text
    assert "duration_seconds_count 3\n" in text


def test_http_200_stream_error_is_counted_as_failed():
    from unittest.mock import AsyncMock

    from engine.tests.unit.test_streaming import make_app

    class FailedEngine:
        health = AsyncMock()

        def submit(self, request):
            pass

        async def stream_generate(self, request):
            yield "partial text"
            raise RuntimeError("internal detail")

        def cancel_request(self, request):
            pass

    app = make_app(FailedEngine())
    with TestClient(app) as client:
        response = client.post("/generate", json={"prompt": "hi", "stream": True})
    assert response.status_code == 200
    assert 'relay_engine_http_requests_total{outcome="failed",status="200"} 1' in (
        app.state.metrics.render()
    )


def test_disconnected_delivery_is_not_counted_as_success():
    from types import SimpleNamespace

    from engine.observability import MetricsMiddleware

    metrics = Metrics()

    async def app(scope, receive, send):
        scope["relay.outcome"] = "success"
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"done"})

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("disconnected")

    async def run():
        with pytest.raises(OSError):
            await MetricsMiddleware(app, SimpleNamespace(metrics=metrics))(
                {"type": "http", "path": "/generate"}, None, send
            )

    asyncio.run(run())
    assert metrics.http_in_flight == 0
    assert 'outcome="cancelled",status="200"' in metrics.render()
