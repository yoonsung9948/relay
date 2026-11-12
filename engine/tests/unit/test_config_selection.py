import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from engine.api.server import create_app
from engine.boot_state import AppState, BootStatus
from engine.capabilities import describe_engine
from engine.cli import cli
from engine.config import Config


def test_config_precedence_and_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docker").mkdir()
    for name, port in (("docker/engine.yaml", 8101), ("env.yaml", 8102), ("explicit.yaml", 8103)):
        (tmp_path / name).write_text(json.dumps({"serve": {"port": port}}))
    build = Mock()
    run = Mock()
    monkeypatch.setattr(cli, "build_app", build)
    monkeypatch.setattr(cli.uvicorn, "run", run)
    monkeypatch.delenv("RELAY_CONFIG_PATH", raising=False)
    runner = CliRunner()
    assert runner.invoke(cli.app, ["serve"]).exit_code == 0
    assert run.call_args.kwargs["port"] == 8101
    monkeypatch.setenv("RELAY_CONFIG_PATH", "env.yaml")
    assert runner.invoke(cli.app, ["serve"]).exit_code == 0
    assert run.call_args.kwargs["port"] == 8102
    assert runner.invoke(cli.app, ["serve", "--path", "explicit.yaml"]).exit_code == 0
    assert run.call_args.kwargs["port"] == 8103
    assert build.call_args.kwargs["config_path"] == str(tmp_path / "explicit.yaml")
    run.reset_mock()
    result = runner.invoke(cli.app, ["serve", "--path", "missing.yaml"])
    assert result.exit_code != 0
    run.assert_not_called()


def test_ready_reports_live_pool_not_file_contents():
    config = Config()
    pool = SimpleNamespace(owners=[None] * 4, allocated_bytes=100)
    engine = SimpleNamespace(
        runner=SimpleNamespace(pool=pool, supports_kv_cache=True, device="cuda:0"),
        scheduler=SimpleNamespace(config=config.scheduler),
        generation_metadata={"model": "test", "backend": "custom", "HF_TOKEN": "secret"},
    )

    async def health():
        pass

    engine.health = health

    @asynccontextmanager
    async def lifespan(app):
        app.state.boot = AppState(status=BootStatus.READY, resources=SimpleNamespace(engine=engine))
        app.state.config_path = "/app/docker/pool.yaml"
        yield

    with TestClient(create_app(lifespan, config.app)) as client:
        response = client.get("/health/ready")
        assert response.json()["capabilities"]["pool_slots"] == 4
        assert response.json()["capabilities"]["config_path"] == "/app/docker/pool.yaml"
        assert "secret" not in response.text
        engine.runner.pool = None
        data = client.get("/health/ready").json()
        assert data["status"] == "ok"  # Non-pool execution is still a healthy service.
        assert data["capabilities"]["pool_slots"] == 0
        assert "pool" not in data["capabilities"]["kv_cache_modes"]
    assert describe_engine(engine)["config_path"] is None


def test_image_allows_environment_config_selection():
    dockerfile = Path(__file__).resolve().parents[3] / "Dockerfile"
    lines = dockerfile.read_text().splitlines()
    assert 'CMD ["serve"]' in lines
    assert 'ENTRYPOINT ["relay"]' in lines
    assert "RELAY_CONFIG_PATH=/app/config.yaml" in dockerfile.read_text()
