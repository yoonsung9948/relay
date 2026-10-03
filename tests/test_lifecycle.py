from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from engine import bootstrap
from engine.cli import cli as cli_module
from engine.config import Config


@pytest.fixture
def factories(monkeypatch):
    tokenizer_factory = Mock(side_effect=lambda *a, **k: Mock(encode=Mock(return_value=[1, 2])))
    engine_factory = Mock(
        side_effect=lambda **kwargs: Mock(
            generate=AsyncMock(return_value={"ok": True}),
            shutdown=AsyncMock(),
            run_loop=AsyncMock(),
        )
    )
    monkeypatch.setattr(bootstrap, "Qwen3Tokenizer", tokenizer_factory)
    monkeypatch.setattr(bootstrap, "Engine", engine_factory)
    return tokenizer_factory, engine_factory


def test_shared_resources_are_created_at_startup_and_closed(factories):
    tokenizer_factory, engine_factory = factories
    app = bootstrap.build_app(Config())
    tokenizer_factory.assert_not_called()
    engine_factory.assert_not_called()

    with TestClient(app) as client:
        resources = app.state.boot.resources
        for prompt in ("first", "second"):
            response = client.post("/generate", json={"prompt": prompt, "max_tokens": 3})
            assert response.status_code == 200
            assert response.json() == {"ok": True}
        tokenizer_factory.assert_called_once()
        engine_factory.assert_called_once()
        assert resources.tokenizer.encode.call_count == 2
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
    with TestClient(first), TestClient(second):
        assert first.state.boot.resources.engine is not second.state.boot.resources.engine
        assert first.state.boot.resources.tokenizer is not second.state.boot.resources.tokenizer


def test_shutdown_runs_when_serving_raises(factories):
    app = bootstrap.build_app(Config())
    with pytest.raises(RuntimeError, match="generation failed"):
        with TestClient(app) as client:
            engine = app.state.boot.resources.engine
            engine.generate.side_effect = RuntimeError("generation failed")
            client.post("/generate", json={"prompt": "hello"})
    engine.shutdown.assert_awaited_once()


def test_cli_passes_host_and_port(monkeypatch):
    build_app = Mock()
    run = Mock()
    monkeypatch.setattr(cli_module, "build_app", build_app)
    monkeypatch.setattr(cli_module.uvicorn, "run", run)
    result = CliRunner().invoke(
        cli_module.app, ["serve", "--host", "0.0.0.0", "--port", "9000"]
    )
    assert result.exit_code == 0, result.output
    config = build_app.call_args.args[0]
    assert config.serve.host == "0.0.0.0"
    assert config.serve.port == 9000
    run.assert_called_once_with(build_app.return_value, host="0.0.0.0", port=9000)
