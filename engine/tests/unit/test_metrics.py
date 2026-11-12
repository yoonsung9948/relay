import asyncio
import json
import logging
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from engine.api.server import create_app
from engine.boot_state import AppState, BootStatus
from engine.config.config import AppConfig
from engine.metrics import RequestTimings
from engine.tests.unit.test_streaming import make_engine


class Clock:
    now = 10.0

    def __call__(self):
        return self.now


def test_measurements_use_first_and_last_token_and_freeze_once():
    clock = Clock()
    timings = RequestTimings(clock=clock, input_tokens=7)
    clock.now = 10.01
    timings.enqueue()
    clock.now = 10.03
    timings.schedule()
    clock.now = 10.1
    timings.token()
    clock.now = 10.4
    timings.token()
    clock.now = 10.6
    timings.token()
    clock.now = 10.7
    result = timings.finish(3)
    assert result.ttft_ms == pytest.approx(100)
    assert result.initial_queue_ms == pytest.approx(20)
    assert result.decode_duration_ms == pytest.approx(500)
    assert result.decode_tokens_per_second == pytest.approx(4)
    assert result.total_latency_ms == pytest.approx(700)
    clock.now = 20
    assert timings.finish(3) is result


@pytest.mark.parametrize("tokens", [0, 1])
def test_short_outputs_do_not_invent_decode_rates(tokens):
    timings = RequestTimings(clock=Clock())
    if tokens:
        timings.token()
    result = timings.finish(tokens)
    assert result.output_tokens == tokens
    assert result.ttft_ms == (0 if tokens else None)
    assert result.decode_duration_ms is None
    assert result.decode_tokens_per_second is None


def test_zero_decode_interval_never_produces_infinity():
    timings = RequestTimings(clock=Clock())
    timings.token()
    timings.token()
    assert timings.finish(2).decode_tokens_per_second is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("budget", [0, 1, 3])
def test_api_attaches_final_measurements_after_text(stream, budget):
    engine = make_engine()

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(engine.run_loop())
        await asyncio.sleep(0)
        app.state.boot = AppState(
            status=BootStatus.READY,
            resources=SimpleNamespace(
                engine=engine,
                tokenizer=engine.tokenizer,
                logger=logging.getLogger("test"),
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
            "/generate", json={"prompt": "hi", "max_tokens": budget, "stream": stream}
        )
    assert response.status_code == 200
    if stream:
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]["type"] == "done"
        assert sum(event["type"] == "done" for event in events) == 1
        terminal = events[-1]
    else:
        terminal = response.json()
        assert terminal["generated_tokens"] == budget
    assert terminal["metrics"]["input_tokens"] == 2
    assert terminal["metrics"]["output_tokens"] == budget
    assert terminal["metrics"]["source"] == "engine"
    assert terminal["generation"] == {
        "max_tokens": budget,
        "sampling": "greedy",
        "temperature": 0.0,
        "use_kv_cache": False,
        "kv_cache_mode": "none",
        "decode_batching": False,
        "implementation_version": "pool-v1",
    }
    assert terminal["finish_reason"] == "length"
    assert terminal["metrics"]["total_latency_ms"] >= (terminal["metrics"]["ttft_ms"] or 0)


def test_timing_records_are_request_local():
    clock = Clock()
    first, second = RequestTimings(clock=clock), RequestTimings(clock=clock)
    clock.now = 11
    first.token()
    clock.now = 12
    second.token()
    assert first.finish(1).ttft_ms == 1000
    assert second.finish(1).ttft_ms == 2000


@pytest.mark.asyncio
async def test_eos_counts_the_stop_token_and_reports_its_finish_reason():
    from engine.processor import Request

    engine = make_engine()
    engine.stop_token_ids = frozenset({7})
    task = asyncio.create_task(engine.run_loop())
    await asyncio.sleep(0)
    try:
        request = Request(tokens=[1, 2], max_tokens=10)
        result = await asyncio.wait_for(engine.generate(request), 1)
        assert result.generated_tokens == 1
        assert request.finish_reason == "eos"
        assert request.timings.finish(1).output_tokens == 1
        assert request.timings.finish(1).decode_tokens_per_second is None
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
