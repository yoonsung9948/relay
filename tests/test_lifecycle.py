import time
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from engine import bootstrap
from engine.cli import cli as cli_module
from engine.config import Config
from engine.processor import GenerateResult


@pytest.fixture
def factories(monkeypatch):
    tokenizer_factory = Mock(
        side_effect=lambda *a, **k: Mock(
            encode_chat=Mock(return_value=[1, 2]),
            decode=Mock(return_value="generated text"),
        )
    )

    def generate(request):
        return GenerateResult(
            request_id=request.request_id,
            token_ids=[*request.tokens, 3, 4, 5],
            generated_tokens=3,
        )

    engine_factory = Mock(
        side_effect=lambda **kwargs: Mock(
            generate=AsyncMock(side_effect=generate),
            health=AsyncMock(),
            shutdown=AsyncMock(),
            run_loop=AsyncMock(),
            generation_metadata=kwargs["generation_metadata"],
            runner=kwargs["runner"],
            scheduler=kwargs["scheduler"],
            _requests={},
        )
    )
    monkeypatch.setattr(bootstrap, "resolve_revision", lambda *args: "a" * 40)
    monkeypatch.setattr(bootstrap, "Qwen3Tokenizer", tokenizer_factory)
    monkeypatch.setattr(bootstrap, "Engine", engine_factory)
    monkeypatch.setattr(bootstrap.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(bootstrap.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(
        "engine.model.load_custom_qwen3",
        Mock(side_effect=lambda *a, **k: torch.nn.Linear(2, 2)),
    )
    return tokenizer_factory, engine_factory


def wait_for_ready(client):
    # Model loading now runs in a background task rather than blocking lifespan startup.
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if client.get("/health/ready").status_code == 200:
            return
        if client.app.state.boot.error is not None:
            pytest.fail(client.app.state.boot.error)
        time.sleep(0.01)
    pytest.fail("Engine did not become ready")


def test_shared_resources_are_created_at_startup_and_closed(factories):
    tokenizer_factory, engine_factory = factories
    app = bootstrap.build_app(Config())
    tokenizer_factory.assert_not_called()
    engine_factory.assert_not_called()

    with TestClient(app) as client:
        wait_for_ready(client)
        resources = app.state.boot.resources
        for prompt in ("first", "second"):
            response = client.post(
                "/generate", json={"prompt": prompt, "max_tokens": 3, "stream": False}
            )
            assert response.status_code == 200
            assert response.json()["text"] == "generated text"
            assert response.json()["generated_tokens"] == 3
        tokenizer_factory.assert_called_once()
        assert tokenizer_factory.call_args.kwargs["revision"] == "a" * 40
        assert resources.engine.generation_metadata["model_revision"] == "a" * 40
        engine_factory.assert_called_once()
        assert resources.tokenizer.encode_chat.call_count == 2
        assert resources.engine.generate.await_count == 2
        internal = resources.engine.generate.await_args.args[0]
        assert internal.tokens == [1, 2]
        assert internal.max_tokens == 3
        resources.engine.shutdown.assert_not_awaited()

    resources.engine.shutdown.assert_awaited_once()
    assert app.state.boot.resources is not None  # shutdown does not clear it; see bootstrap.py


def test_apps_have_independent_resources(factories):
    first = bootstrap.build_app(Config())
    second = bootstrap.build_app(Config())
    with TestClient(first) as first_client, TestClient(second) as second_client:
        wait_for_ready(first_client)
        wait_for_ready(second_client)
        assert first.state.boot.resources.engine is not second.state.boot.resources.engine
        assert first.state.boot.resources.tokenizer is not second.state.boot.resources.tokenizer


def test_shutdown_runs_when_serving_raises(factories):
    app = bootstrap.build_app(Config())
    with pytest.raises(RuntimeError, match="generation failed"):
        with TestClient(app) as client:
            wait_for_ready(client)
            engine = app.state.boot.resources.engine
            engine.generate.side_effect = RuntimeError("generation failed")
            client.post("/generate", json={"prompt": "hello", "stream": False})
    engine.shutdown.assert_awaited_once()


def test_cli_passes_configured_host_and_port(monkeypatch, tmp_path):
    config_path = tmp_path / "engine.yaml"
    config_path.write_text("serve:\n  host: 0.0.0.0\n  port: 9000\n")
    build_app = Mock()
    run = Mock()
    monkeypatch.setattr(cli_module, "build_app", build_app)
    monkeypatch.setattr(cli_module.uvicorn, "run", run)
    result = CliRunner().invoke(cli_module.app, ["serve", "--path", str(config_path)])
    assert result.exit_code == 0, result.output
    config = build_app.call_args.args[0]
    assert config.serve.host == "0.0.0.0"
    assert config.serve.port == 9000
    run.assert_called_once_with(build_app.return_value, host="0.0.0.0", port=9000)
