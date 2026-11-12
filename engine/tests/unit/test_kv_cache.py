import asyncio
import json
import logging
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from fastapi.testclient import TestClient

from engine.api.server import create_app
from engine.boot_state import AppState, BootStatus
from engine.config import Qwen3Config, SchedulerConfig
from engine.config.config import AppConfig
from engine.engine import Engine, EngineState
from engine.kv_cache import KVCache
from engine.model.qwen3 import Qwen3ForCausalLM
from engine.processor import Request
from engine.runner import Runner
from engine.sampler import GreedySampler
from engine.scheduler import Scheduler


class Tokenizer:
    eos_token_ids = frozenset()

    def __init__(self):
        self.tokenizer = self

    def encode_chat(self, text):
        return [1, 2, 3]

    def decode(self, tokens, **kwargs):
        return "".join(f"{token} " for token in tokens)


@pytest.fixture
def model():
    torch.manual_seed(42)
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=48,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
    ).eval()


def make_engine(model):
    return Engine(
        Scheduler(SchedulerConfig()),
        Runner(model, torch.device("cpu"), GreedySampler()),
        Tokenizer(),
    )


@pytest.mark.parametrize("prefill_length", [1, 4])
@pytest.mark.parametrize("tied", [False, True])
def test_cached_logits_match_full_context(model, prefill_length, tied):
    if tied:
        model.lm_head.weight = model.model.embed_tokens.weight
    tokens = torch.tensor([[1, 4, 2, 7, 9, 3, 8]])
    cache = KVCache(2)
    with torch.inference_mode():
        prefill = model(tokens[:, :prefill_length], cache=cache)
        expected = model(tokens[:, :prefill_length])
        torch.testing.assert_close(prefill, expected, rtol=1e-4, atol=1e-5)
        for layer in cache.layers:
            assert layer[0].shape == (1, 2, prefill_length, 8)
            assert not layer[0].requires_grad
        for position in range(prefill_length, tokens.shape[1]):
            actual = model(tokens[:, position : position + 1], cache=cache)
            expected = model(tokens[:, : position + 1])[:, -1:, :]
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            assert all(cache.get_seq_length(i) == position + 1 for i in range(2))
    cache.clear()
    assert cache.layers == [None, None]


def test_cache_is_not_checkpoint_state_and_rejects_unsupported_calls(model):
    keys = set(model.state_dict())
    cache = KVCache(2)
    with torch.inference_mode():
        model(torch.tensor([[1, 2]]), cache=cache)
        with pytest.raises(ValueError, match="one new token"):
            model(torch.tensor([[3, 4]]), cache=cache)
        with pytest.raises(ValueError, match="layer count"):
            model(torch.tensor([[1]]), cache=KVCache(1))
    assert set(model.state_dict()) == keys
    assert not any("cache" in key for key in keys)


def test_interleaved_requests_have_independent_caches_and_single_token_decode(model):
    runner = Runner(model, torch.device("cpu"), GreedySampler())
    cached = [
        Request(tokens=[1, 2], use_kv_cache=True),
        Request(tokens=[5, 6, 7, 8], use_kv_cache=True),
    ]
    plain = [Request(tokens=list(req.tokens)) for req in cached]
    shapes = []
    hook = model.register_forward_pre_hook(lambda module, args: shapes.append(args[0].shape[1]))
    try:
        for _ in range(3):
            for a, b in zip(cached, plain):
                actual = runner.run_requests([a])[0]
                expected = runner.run_requests([b])[0]
                assert actual == expected
                a.tokens.append(actual)
                b.tokens.append(expected)
    finally:
        hook.remove()
    # Cached and uncached calls alternate. Each request prefills once.
    assert shapes[::2] == [2, 4, 1, 1, 1, 1]
    assert shapes[1::2] == [2, 4, 3, 5, 4, 6]
    assert cached[0].kv_cache.layers[0][0].data_ptr() != cached[1].kv_cache.layers[0][0].data_ptr()
    assert all(req.kv_cache is None for req in plain)


async def collect(engine, request):
    return "".join([chunk async for chunk in engine.stream_generate(request)])


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_cached_engine_matches_uncached_and_releases_at_completion(model, stream):
    engine = make_engine(model)
    caches = []
    original = engine.runner.run_requests

    def record(batch):
        result = original(batch)
        caches.extend(req.kv_cache for req in batch if req.kv_cache is not None)
        return result

    engine.runner.run_requests = record
    loop = asyncio.create_task(engine.run_loop())
    await asyncio.sleep(0)
    cached = Request(tokens=[1, 2, 3], max_tokens=4, use_kv_cache=True)
    plain = Request(tokens=[1, 2, 3], max_tokens=4)
    try:
        await asyncio.wait_for(
            asyncio.gather(
                collect(engine, cached) if stream else engine.generate(cached),
                engine.generate(plain),
            ),
            5,
        )
        assert cached.tokens == plain.tokens
        assert cached.generated_tokens == 4
        assert cached.kv_cache is None
        assert caches and all(cache.layers == [None, None] for cache in caches)
        assert not engine._requests
    finally:
        loop.cancel()
        with suppress(asyncio.CancelledError):
            await loop


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stop", ["cancel_request", "cancel_consumer", "shutdown", "loop_cancel", "failure"]
)
async def test_terminal_paths_release_allocated_cache(model, stop):
    engine = make_engine(model)
    engine.state = EngineState.READY
    cache = KVCache(2)
    with torch.inference_mode():
        model(torch.tensor([[1, 2]]), cache=cache)
    request = Request(tokens=[1, 2, 3], max_tokens=30, use_kv_cache=True, kv_cache=cache)
    consumer = asyncio.create_task(collect(engine, request))
    async with asyncio.timeout(2):
        while request.request_id not in engine._requests:
            await asyncio.sleep(0)
    if stop == "cancel_request":
        engine.cancel_request(request)
    elif stop == "cancel_consumer":
        consumer.cancel()
    elif stop == "shutdown":
        await engine.shutdown()
    else:
        if stop == "failure":
            engine.runner._run = Mock(side_effect=RuntimeError("model failure"))
        loop = asyncio.create_task(engine.run_loop())
        await asyncio.sleep(0)
        if stop == "loop_cancel":
            loop.cancel()
        with suppress(RuntimeError, asyncio.CancelledError):
            await loop
    with suppress(RuntimeError, asyncio.CancelledError):
        await asyncio.wait_for(consumer, 2)
    assert request.kv_cache is None
    assert cache.layers == [None, None]
    assert not engine._requests


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("budget", [0, 3])
def test_http_cache_toggle_and_effective_metadata(model, stream, use_cache, budget):
    engine = make_engine(model)

    @asynccontextmanager
    async def lifespan(app):
        loop = asyncio.create_task(engine.run_loop())
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
            loop.cancel()
            with suppress(asyncio.CancelledError):
                await loop

    with TestClient(create_app(lifespan, AppConfig())) as client:
        response = client.post(
            "/generate",
            json={
                "prompt": "hello",
                "max_tokens": budget,
                "stream": stream,
                "use_kv_cache": use_cache,
            },
        )
        assert response.status_code == 200, response.text
        result = json.loads(response.text.splitlines()[-1]) if stream else response.json()
        assert result["generation"]["use_kv_cache"] is use_cache
        assert result["metrics"]["output_tokens"] == budget
        assert not engine._requests


def test_unsupported_backend_rejected_before_generation():
    engine = make_engine(torch.nn.Linear(1, 1))
    engine.state = EngineState.READY

    @asynccontextmanager
    async def lifespan(app):
        app.state.boot = AppState(
            status=BootStatus.READY,
            resources=SimpleNamespace(
                engine=engine,
                tokenizer=engine.tokenizer,
                logger=logging.getLogger("test"),
            ),
        )
        yield

    with TestClient(create_app(lifespan, AppConfig())) as client:
        response = client.post("/generate", json={"prompt": "hello", "use_kv_cache": True})
        assert response.status_code == 422
        assert "custom Qwen3" in response.text
    assert not engine._requests


@pytest.mark.asyncio
async def test_eos_releases_cache_without_extra_decode_steps(model):
    engine = make_engine(model)
    prompt = [1, 2, 3]
    first = engine.runner.run([prompt])[0]
    engine.stop_token_ids = frozenset([first])
    request = Request(tokens=list(prompt), max_tokens=10, use_kv_cache=True)
    loop = asyncio.create_task(engine.run_loop())
    await asyncio.sleep(0)
    try:
        result = await asyncio.wait_for(engine.generate(request), 2)
        assert result.generated_tokens == 1
        assert request.finish_reason == "eos"
        assert request.kv_cache is None
        assert not engine.scheduler.pending
    finally:
        loop.cancel()
        with suppress(asyncio.CancelledError):
            await loop
