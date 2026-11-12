import asyncio
import json
import logging
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from engine.api.server import create_app
from engine.boot_state import AppState, BootStatus
from engine.config import SchedulerConfig
from engine.config.config import AppConfig
from engine.engine import Engine, EngineState
from engine.processor import Request
from engine.scheduler import Scheduler


class FakeTokenizer:
    eos_token_ids = frozenset()

    def __init__(self):
        self.tokenizer = self

    def encode_chat(self, text):
        return [1, 2]

    def decode(self, tokens, **kwargs):
        return "".join(f"{token} " for token in tokens)


class FakeRunner:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def run_requests(self, batch):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model exploded")
        return [7] * len(batch)


def make_engine(fail=False):
    return Engine(Scheduler(SchedulerConfig()), FakeRunner(fail), FakeTokenizer())


async def collect(engine, request):
    return [text async for text in engine.stream_generate(request)]


async def wait_submitted(engine, count):
    async with asyncio.timeout(1):
        while len(engine._requests) < count:
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_stream_and_nonstream_finish_and_release():
    engine = make_engine()
    loop = asyncio.create_task(engine.run_loop())
    await asyncio.sleep(0)
    try:
        streaming = Request(tokens=[1, 2], max_tokens=3)
        plain = Request(tokens=[1], max_tokens=2)
        chunks, result = await asyncio.wait_for(
            asyncio.gather(
                collect(engine, streaming),
                engine.generate(plain),
            ),
            1,
        )
        assert "".join(chunks) == "7 7 7 "
        assert result.generated_tokens == 2
        assert result.token_ids == [1, 7, 7]
        assert not engine._requests
        assert not engine.scheduler.pending
        assert not streaming.cancelled
    finally:
        loop.cancel()
        with suppress(asyncio.CancelledError):
            await loop


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["failure", "loop_cancel", "shutdown", "request_cancel"])
async def test_terminal_paths_wake_both_consumers(stop):
    engine = make_engine(fail=stop == "failure")
    engine.state = EngineState.READY
    requests = [Request(tokens=[1], max_tokens=20) for _ in range(2)]
    consumers = [
        asyncio.create_task(collect(engine, requests[0])),
        asyncio.create_task(engine.generate(requests[1])),
    ]
    await wait_submitted(engine, 2)
    if stop == "request_cancel":
        for req in requests:
            engine.cancel_request(req)
    elif stop == "shutdown":
        await engine.shutdown()
    else:
        loop = asyncio.create_task(engine.run_loop())
        await asyncio.sleep(0)
        if stop == "loop_cancel":
            loop.cancel()
        with suppress(RuntimeError, asyncio.CancelledError):
            await loop
    results = await asyncio.wait_for(asyncio.gather(*consumers, return_exceptions=True), 1)
    assert all(isinstance(result, RuntimeError) for result in results)
    assert not engine.scheduler.pending
    assert not engine.scheduler.running
    assert not engine._requests
    assert all(req.done_event.is_set() for req in requests)
    if stop == "failure":
        assert engine.state is EngineState.ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_consumer_cancellation_removes_work(stream):
    engine = make_engine()
    engine.state = EngineState.READY
    req = Request(tokens=[1], max_tokens=30)
    consumer = asyncio.create_task(collect(engine, req) if stream else engine.generate(req))
    await wait_submitted(engine, 1)
    consumer.cancel()
    with suppress(asyncio.CancelledError):
        await consumer
    assert req.cancelled
    assert req.done_event.is_set()
    assert not engine.scheduler.pending
    assert not engine._requests


@pytest.mark.asyncio
async def test_zero_budget_never_calls_runner():
    engine = make_engine()
    engine.state = EngineState.READY
    assert await collect(engine, Request(tokens=[1], max_tokens=0)) == []
    assert engine.runner.calls == 0


@pytest.mark.asyncio
async def test_decoder_failure_does_not_hang():
    engine = make_engine()
    engine.tokenizer.decode = lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError("decode failed")
    )
    loop = asyncio.create_task(engine.run_loop())
    await asyncio.sleep(0)
    try:
        with pytest.raises(ValueError, match="decode failed"):
            await asyncio.wait_for(collect(engine, Request(tokens=[1], max_tokens=3)), 1)
        assert not engine._requests
    finally:
        loop.cancel()
        with suppress(asyncio.CancelledError):
            await loop


def make_app(engine):
    @asynccontextmanager
    async def lifespan(app):
        app.state.boot = AppState(
            status=BootStatus.READY,
            resources=SimpleNamespace(
                engine=engine,
                tokenizer=FakeTokenizer(),
                logger=logging.getLogger("test"),
            ),
        )
        yield

    return create_app(lifespan, AppConfig())


def test_api_emits_done_and_structured_error():
    class FakeEngine:
        def submit(self, request):
            pass

        health = AsyncMock()

        def __init__(self, fail):
            self.fail = fail
            self.cancelled = False

        async def stream_generate(self, request):
            yield "hello "
            if self.fail:
                raise RuntimeError("private model details")

        def cancel_request(self, request):
            self.cancelled = True

    for fail in [False, True]:
        engine = FakeEngine(fail)
        with TestClient(make_app(engine)) as client:
            response = client.post("/generate", json={"prompt": "hello", "stream": True})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[0]["type"] == "delta"
        assert events[0]["text"] == "hello "
        assert events[-1]["type"] == ("error" if fail else "done")
        assert events[0]["request_id"] == events[-1]["request_id"]
        assert "private model details" not in response.text
        assert engine.cancelled


def test_api_rejects_unavailable_before_stream_headers():
    engine = make_engine()
    engine.state = EngineState.ERROR
    with TestClient(make_app(engine)) as client:
        assert client.get("/health/ready").status_code == 503
        response = client.post("/generate", json={"prompt": "hello", "stream": True})
        assert response.status_code == 503
        assert response.headers["content-type"].startswith("application/json")


@pytest.mark.asyncio
async def test_http_disconnect_closes_stream_and_producer():
    from starlette.requests import ClientDisconnect

    engine = make_engine()
    engine.state = EngineState.READY
    app = make_app(engine)
    request = None

    async def receive():
        return {
            "type": "http.request",
            "body": b'{"prompt":"hello","stream":true}',
            "more_body": False,
        }

    # Use a real engine stream, but supply a token without completing generation.
    async def send(message):
        nonlocal request
        if message["type"] == "http.response.start":

            async def seed():
                nonlocal request
                await wait_submitted(engine, 1)
                request = next(iter(engine._requests.values()))
                request.output_queue.put_nowait(7)

            asyncio.create_task(seed())
        elif message["type"] == "http.response.body" and message.get("body"):
            raise OSError("client disconnected")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/generate",
        "raw_path": b"/generate",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "server": ("testserver", 80),
        "client": ("testclient", 1),
    }
    async with app.router.lifespan_context(app):
        with pytest.raises(ClientDisconnect):
            await asyncio.wait_for(app(scope, receive, send), 1)
    assert request is not None and request.cancelled
    assert not engine.scheduler.pending
    assert not engine._requests
