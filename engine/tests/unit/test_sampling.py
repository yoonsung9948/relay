import asyncio
import json
import logging
import math
from contextlib import asynccontextmanager, suppress
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from engine.api.server import create_app
from engine.boot_state import AppState, BootStatus
from engine.config.config import AppConfig
from engine.processor import GenerateRequest, Request
from engine.runner import Runner
from engine.sampler import GreedySampler, TemperatureSampler
from engine.sampling import SamplingOptions
from engine.tests.fakes.fake_model import FakeModel
from engine.tests.unit.test_kv_cache import model as model
from engine.tests.unit.test_pool import pooled
from engine.tests.unit.test_tracing import tracing as tracing


@pytest.mark.parametrize("temperature", [-1, math.nan, math.inf, -math.inf, None, "0.7", True])
def test_temperature_validation_matches_api_and_internal_options(temperature):
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hi", temperature=temperature)
    with pytest.raises(ValueError):
        SamplingOptions(temperature=temperature)


def test_sampling_defaults_are_immutable_and_preserve_greedy():
    assert GenerateRequest(prompt="hi").temperature == 0
    assert Request().sampling == SamplingOptions()
    assert SamplingOptions().strategy == "greedy"
    assert SamplingOptions(0.7).strategy == "temperature"
    with pytest.raises(FrozenInstanceError):
        SamplingOptions().temperature = 1
    logits = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 1.0]])
    generator = torch.Generator().manual_seed(7)
    state = generator.get_state().clone()
    result = TemperatureSampler().sample(logits, [SamplingOptions()] * 2, generator=generator)
    torch.testing.assert_close(result, GreedySampler().sample(logits))
    torch.testing.assert_close(generator.get_state(), state)
    assert result.dtype == torch.long and result.device == logits.device


def test_mixed_temperatures_preserve_row_order_and_greedy_rows():
    logits = torch.tensor([[0.0, 8.0, 1.0], [1.0, 0.0, -1.0], [5.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    options = [SamplingOptions(t) for t in [0, 0.5, 0, 2.0]]
    seed = 9
    actual = TemperatureSampler().sample(
        logits, options, generator=torch.Generator().manual_seed(seed)
    )
    sampled_rows = logits[[1, 3]]
    probabilities = torch.softmax(sampled_rows / torch.tensor([[0.5], [2.0]]), dim=-1)
    expected = torch.multinomial(
        probabilities, 1, generator=torch.Generator().manual_seed(seed)
    ).flatten()
    assert actual[[0, 2]].tolist() == [1, 0]
    torch.testing.assert_close(actual[[1, 3]], expected)
    assert actual.shape == (4,)


def test_temperature_changes_the_sampling_distribution():
    logits = torch.tensor([[math.log(3), 0.0]]).repeat(4096, 1)
    sampler = TemperatureSampler()
    for temperature in [0.5, 2.0]:
        actual = sampler.sample(
            logits,
            [SamplingOptions(temperature)] * len(logits),
            generator=torch.Generator().manual_seed(3),
        )
        expected_probability = 3 ** (1 / temperature) / (1 + 3 ** (1 / temperature))
        assert (actual == 0).float().mean().item() == pytest.approx(expected_probability, abs=0.025)


@pytest.mark.parametrize("temperature", [1e-300, 1e300])
def test_extreme_finite_temperatures_do_not_produce_nan(temperature):
    logits = torch.tensor([[1000.0, 1001.0, 999.0]], dtype=torch.float16)
    sampled = TemperatureSampler().sample(
        logits, [SamplingOptions(temperature)], generator=torch.Generator().manual_seed(5)
    )
    assert 0 <= sampled.item() < 3
    if temperature < 1:
        assert sampled.item() == 1


def test_invalid_probability_distributions_raise_without_greedy_fallback():
    with pytest.raises(RuntimeError, match="temperature sampling failed"):
        TemperatureSampler().sample(torch.tensor([[math.nan, 1.0]]), [SamplingOptions(1)])
    with pytest.raises(ValueError, match="match the batch size"):
        TemperatureSampler().sample(torch.zeros(2, 3), [SamplingOptions()])


def test_uncached_runner_uses_each_requests_sampling_options():
    class RecordingSampler(TemperatureSampler):
        def sample(self, logits, options=None):
            self.options = options
            return super().sample(logits, options, generator=torch.Generator().manual_seed(5))

    sampler = RecordingSampler()
    # The high-scoring first position must not be used to predict the next token.
    logits = torch.tensor([[[100.0, -100.0], [0.0, 0.0]], [[100.0, -100.0], [0.0, 0.0]]])
    runner = Runner(FakeModel(logits), torch.device("cpu"), sampler)
    requests = [Request(tokens=[1, 2], sampling=SamplingOptions(t)) for t in [0, 1]]
    results = runner.run_requests(requests)
    assert sampler.options == [request.sampling for request in requests]
    assert results[0] == 0
    assert all(type(token) is int for token in results)


@pytest.mark.parametrize(
    ("mode", "batching"),
    [
        ("none", False),
        ("dynamic", False),
        ("pool", False),
        ("pool", True),
    ],
)
@pytest.mark.asyncio
async def test_engine_passes_sampling_options_through_every_cache_mode(model, mode, batching):
    engine = pooled(model)
    seen = []

    class RecordingSampler(TemperatureSampler):
        def sample(self, logits, options=None):
            seen.append(tuple(option.temperature for option in options))
            return super().sample(logits, options)

    engine.runner.sampler = RecordingSampler()
    requests = [
        Request(
            tokens=[1, 2],
            max_tokens=3,
            kv_cache_mode=mode,
            decode_batching=batching,
            sampling=SamplingOptions(t),
        )
        for t in [0, 0.7]
    ]
    loop = asyncio.create_task(engine.run_loop())
    await asyncio.sleep(0)
    try:
        await asyncio.wait_for(asyncio.gather(*(engine.generate(r) for r in requests)), 5)
        assert all(r.generated_tokens == 3 for r in requests)
        assert sum(len(options) for options in seen) == 6
        assert any(0.7 in options for options in seen)
        if batching:
            assert (0, 0.7) in seen or (0.7, 0) in seen
        assert not engine.runner.reservations
        assert engine.runner.pool.owners == [None] * 3
    finally:
        loop.cancel()
        with suppress(asyncio.CancelledError):
            await loop


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_api_reports_effective_temperature_in_response_and_trace(
    model, tracing, stream, temperature
):
    engine = pooled(model)
    engine.runner.sampler = TemperatureSampler()
    instrumentation, exporter = tracing

    @asynccontextmanager
    async def lifespan(app):
        app.state.tracing = instrumentation
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
            json={"prompt": "hi", "max_tokens": 3, "stream": stream, "temperature": temperature},
        )
    assert response.status_code == 200
    terminal = json.loads(response.text.splitlines()[-1]) if stream else response.json()
    assert isinstance(terminal["generation"]["temperature"], float)
    assert terminal["generation"]["temperature"] == temperature
    assert terminal["generation"]["sampling"] == ("temperature" if temperature else "greedy")
    assert terminal["metrics"]["output_tokens"] == 3
    request_span = next(s for s in exporter.get_finished_spans() if s.name == "generation.request")
    assert request_span.attributes["sampling.temperature"] == temperature
