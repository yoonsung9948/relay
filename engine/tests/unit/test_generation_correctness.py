import asyncio
import json
import logging
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from engine.api.server import create_app
from engine.boot_state import AppState, BootStatus
from engine.config import Config, Qwen3Config, SchedulerConfig
from engine.config.config import AppConfig
from engine.engine import Engine, EngineState
from engine.model.qwen3 import Qwen3ForCausalLM
from engine.processor import Qwen3Tokenizer, Request
from engine.runner import Runner
from engine.sampler import GreedySampler
from engine.scheduler import Scheduler


class Tokenizer:
    eos_token_ids = frozenset()

    def __init__(self):
        self.tokenizer = self

    def decode(self, tokens, **kwargs):
        return "".join(f"{token} " for token in tokens)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_custom_model_generates_multiple_integer_tokens(stream):
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    )
    engine = Engine(
        Scheduler(SchedulerConfig()),
        Runner(model, torch.device("cpu"), GreedySampler()),
        Tokenizer(),
    )
    request = Request(tokens=[1, 2], max_tokens=3)
    loop = asyncio.create_task(engine.run_loop())
    await asyncio.sleep(0)
    try:
        if stream:

            async def collect():
                return [chunk async for chunk in engine.stream_generate(request)]

            chunks = await asyncio.wait_for(collect(), 5)
            assert "".join(chunks) == engine.tokenizer.decode(request.tokens[2:])
        else:
            result = await asyncio.wait_for(engine.generate(request), 5)
            assert result.generated_tokens == 3
            assert result.token_ids == request.tokens
        assert request.generated_tokens == 3
        assert len(request.tokens) == 5
        assert all(type(token) is int and 0 <= token < 32 for token in request.tokens)
        assert not engine._requests
        assert not engine.scheduler.pending
        assert not engine.scheduler.running
    finally:
        loop.cancel()
        with suppress(asyncio.CancelledError):
            await loop


@pytest.mark.parametrize("results", [[[7]], [True], [7.0], [], [7, 8]])
@pytest.mark.asyncio
async def test_invalid_runner_output_does_not_mutate_request(results):
    runner = SimpleNamespace(run_requests=lambda batch: results)
    engine = Engine(Scheduler(SchedulerConfig()), runner, Tokenizer())
    engine.state = EngineState.READY
    request = Request(tokens=[1, 2], max_tokens=3)
    consumer = asyncio.create_task(engine.generate(request))
    await asyncio.sleep(0)
    loop = asyncio.create_task(engine.run_loop())
    with pytest.raises(RuntimeError, match="runner"):
        await asyncio.wait_for(loop, 5)
    with pytest.raises(RuntimeError, match="runner"):
        await asyncio.wait_for(consumer, 5)
    assert request.tokens == [1, 2]
    assert request.generated_tokens == 0
    assert not engine._requests
    assert not engine.scheduler.pending
    assert not engine.scheduler.running


@pytest.mark.parametrize("batch_size", [0, 2, 8])
def test_config_rejects_unsupported_batch_sizes(batch_size):
    with pytest.raises(ValidationError):
        Config.model_validate({"scheduler": {"batch_size": batch_size}})
    assert SchedulerConfig().batch_size == 1


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_chat_prompt_starts_an_assistant_turn(monkeypatch, enable_thinking):
    backend = Mock()
    backend.apply_chat_template.return_value = [1, 2, 3]
    factory = Mock(return_value=backend)
    monkeypatch.setattr("engine.processor.tokenizer.AutoTokenizer.from_pretrained", factory)
    tokenizer = Qwen3Tokenizer("test-model")

    assert tokenizer.encode_chat("hello", enable_thinking=enable_thinking) == [1, 2, 3]
    backend.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "hello"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_dict=False,
    )


@pytest.mark.parametrize("stream", [False, True])
def test_api_validates_negative_budget_and_accepts_zero(stream):
    tokenizer = Tokenizer()
    tokenizer.encode_chat = Mock(return_value=[1, 2])
    engine = Engine(Scheduler(SchedulerConfig()), Mock(), tokenizer)
    engine.state = EngineState.READY

    @asynccontextmanager
    async def lifespan(app):
        app.state.boot = AppState(
            status=BootStatus.READY,
            resources=SimpleNamespace(
                engine=engine,
                tokenizer=tokenizer,
                logger=logging.getLogger("test"),
            ),
        )
        yield

    with TestClient(create_app(lifespan, AppConfig())) as client:
        invalid = client.post(
            "/generate",
            json={
                "prompt": "hello",
                "max_tokens": -1,
                "stream": stream,
            },
        )
        assert invalid.status_code == 422
        tokenizer.encode_chat.assert_not_called()
        response = client.post(
            "/generate",
            json={
                "prompt": "hello",
                "max_tokens": 0,
                "stream": stream,
            },
        )
        assert response.status_code == 200
        if stream:
            events = [json.loads(line) for line in response.text.splitlines()]
            assert [event["type"] for event in events] == ["done"]
        else:
            assert response.json()["generated_tokens"] == 0
            assert response.json()["text"] == ""
    engine.runner.run.assert_not_called()
    assert not engine._requests
