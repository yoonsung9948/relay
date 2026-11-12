import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from engine.api.server import create_app
from engine.boot_state import AppState, BootStatus
from engine.config.config import AppConfig
from engine.processor import Request
from engine.tests.unit.test_streaming import make_engine, wait_submitted
from engine.tracing import Tracing


@pytest.fixture
def tracing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create({"service.name": "test", "service.instance.id": "one"})
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield Tracing(provider), exporter
    provider.shutdown()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("budget", [0, 1, 3])
def test_full_trace_parentage_and_no_prompt_capture(tracing, stream, budget):
    instrument, exporter = tracing
    engine = make_engine()

    @asynccontextmanager
    async def lifespan(app):
        app.state.tracing = instrument
        task = asyncio.create_task(engine.run_loop())
        await asyncio.sleep(0)
        app.state.boot = AppState(
            status=BootStatus.READY,
            resources=SimpleNamespace(
                engine=engine, tokenizer=engine.tokenizer, logger=logging.getLogger("test")
            ),
        )
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    with TestClient(create_app(lifespan, AppConfig())) as client:
        response = client.post(
            "/generate",
            json={"prompt": "private prompt", "max_tokens": budget, "stream": stream},
            headers={"traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"},
        )
        assert response.status_code == 200
        # Health polling does not create spans.
        client.get("/health/live")
    spans = exporter.get_finished_spans()
    server = next(s for s in spans if s.name == "POST /generate")
    generation = next(s for s in spans if s.name == "generation.request")
    assert server.context.trace_id == int("a" * 32, 16)
    assert server.parent.span_id == int("b" * 16, 16)
    assert generation.parent.span_id == server.context.span_id
    assert generation.attributes["request.outcome"] == "success"
    assert generation.attributes["output_tokens"] == budget
    steps = [s for s in spans if s.name == "runner.step"]
    assert len(steps) == budget
    assert all(s.parent.span_id == generation.context.span_id for s in steps)
    assert server.end_time >= max(s.end_time for s in spans)
    assert "private prompt" not in str([dict(s.attributes) for s in spans])
    assert all(not s.events for s in spans)


@pytest.mark.asyncio
async def test_interleaved_requests_keep_their_own_parent(tracing):
    instrument, exporter = tracing
    engine = make_engine()
    task = asyncio.create_task(engine.run_loop())
    await asyncio.sleep(0)
    requests = [Request(tokens=[1], max_tokens=3) for _ in range(2)]
    for request in requests:
        request.trace = instrument.request(request.request_id)
    try:
        await asyncio.gather(*(engine.generate(r) for r in requests))
        for request in requests:
            request.trace.finish("success", request)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    spans = exporter.get_finished_spans()
    for request in requests:
        parent = next(s for s in spans if s.attributes.get("request.id") == str(request.request_id))
        assert (
            len(
                [
                    s
                    for s in spans
                    if s.name == "runner.step" and s.parent.span_id == parent.context.span_id
                ]
            )
            == 3
        )


@pytest.mark.asyncio
async def test_cancelled_queue_span_is_closed(tracing):
    instrument, exporter = tracing
    engine = make_engine()
    from engine.engine import EngineState

    engine.state = EngineState.READY
    request = Request(tokens=[1], max_tokens=3)
    request.trace = instrument.request(request.request_id)
    consumer = asyncio.create_task(engine.generate(request))
    await wait_submitted(engine, 1)
    consumer.cancel()
    with suppress(asyncio.CancelledError):
        await consumer
    request.trace.finish("cancelled", request)
    assert {s.name for s in exporter.get_finished_spans()} == {
        "generation.request",
        "scheduler.initial_wait",
    }
    assert request.trace.span.status.status_code == StatusCode.ERROR


def test_disabled_does_not_create_request_trace(monkeypatch):
    monkeypatch.delenv("RELAY_TRACING_ENABLED", raising=False)
    assert Tracing.from_env().request("request") is None
    monkeypatch.setenv("RELAY_TRACING_ENABLED", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert Tracing.from_env().provider is None


def test_stream_error_after_http_200_marks_trace_failed_without_exception_text(tracing):
    from unittest.mock import AsyncMock

    from engine.tests.unit.test_streaming import make_app

    instrument, exporter = tracing

    class FailedEngine:
        def submit(self, request):
            pass

        health = AsyncMock()

        async def stream_generate(self, request):
            yield "first text"
            raise RuntimeError("private model internals")

        def cancel_request(self, request):
            pass

    app = make_app(FailedEngine())
    app.state.tracing = instrument
    with TestClient(app) as client:
        response = client.post("/generate", json={"prompt": "secret", "stream": True})
    assert response.status_code == 200
    spans = exporter.get_finished_spans()
    generation = next(s for s in spans if s.name == "generation.request")
    server = next(s for s in spans if s.name == "POST /generate")
    assert generation.attributes["request.outcome"] == "failed"
    assert generation.status.status_code == server.status.status_code == StatusCode.ERROR
    assert "private model internals" not in str([s.to_json() for s in spans])


def test_phase_only_tracing_omits_runner_steps(tracing, monkeypatch):
    from engine.tracing import phase

    instrument, exporter = tracing
    monkeypatch.setenv("RELAY_TRACE_STEPS", "false")
    request = Request(tokens=[1])
    request.trace = instrument.request(request.request_id)
    with phase(request, "runner.step"):
        pass
    request.trace.finish("success", request)
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["generation.request"]
    assert spans[0].attributes["tracing.steps.enabled"] is False
