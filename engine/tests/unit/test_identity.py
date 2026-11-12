import json
from types import SimpleNamespace
from unittest.mock import Mock

from engine.config import Config
from engine.identity import engine_identity, resolve_revision, source_sha256
from engine.model import loader
from engine.model.huggingface import HuggingFaceModel
from engine.processor.tokenizer import Qwen3Tokenizer

SHA = "a" * 40


def test_full_commit_is_offline_and_configuration_precedes_environment(monkeypatch):
    monkeypatch.setenv("RELAY_MODEL_REVISION", "b" * 40)
    monkeypatch.setattr(
        "huggingface_hub.HfApi", lambda: (_ for _ in ()).throw(AssertionError("network call"))
    )
    assert resolve_revision("test", SHA) == SHA
    assert resolve_revision("test", None) == "b" * 40


def test_branch_resolves_to_one_immutable_commit(monkeypatch):
    api = Mock()
    api.model_info.return_value = SimpleNamespace(sha=SHA)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: api)
    assert resolve_revision("test", "main") == SHA
    api.model_info.assert_called_once_with("test", revision="main")


def test_all_loaders_receive_the_checkpoint_revision(monkeypatch, tmp_path):
    snapshot = Mock(return_value=str(tmp_path))
    model = Mock()
    model.to.return_value = model
    monkeypatch.setattr(loader, "Qwen3ForCausalLM", Mock(return_value=model))
    monkeypatch.setattr(loader, "snapshot_download", snapshot)
    assert loader.load_custom_qwen3("test", object(), revision=SHA) is model
    assert snapshot.call_args.kwargs["revision"] == SHA
    hf = Mock(return_value=model)
    monkeypatch.setattr("engine.model.huggingface.AutoModelForCausalLM.from_pretrained", hf)
    HuggingFaceModel("test", revision=SHA)
    assert hf.call_args.kwargs["revision"] == SHA
    tokenizer = Mock(return_value=SimpleNamespace(eos_token_id=1))
    monkeypatch.setattr("engine.processor.tokenizer.AutoTokenizer.from_pretrained", tokenizer)
    Qwen3Tokenizer("test", revision=SHA)
    assert tokenizer.call_args.kwargs["revision"] == SHA


def test_fingerprint_tracks_code_and_ignores_cache_and_tests(tmp_path):
    code = tmp_path / "code.py"
    code.write_text("x=1")
    original = source_sha256(tmp_path)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/code.pyc").write_bytes(b"cache")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_code.py").write_text("assert True")
    assert source_sha256(tmp_path) == original
    code.write_text("x=2")
    assert source_sha256(tmp_path) != original


def test_build_stamp_is_checked_against_running_source(monkeypatch, tmp_path):
    build = tmp_path / "build.json"
    build.write_text(json.dumps({"revision": SHA, "dirty": False, "source_sha256": "stale"}))
    monkeypatch.setenv("RELAY_BUILD_INFO", str(build))
    result = engine_identity(Config())
    assert result["engine_revision"] == SHA and result["engine_dirty"] is True
    assert len(result["engine_source_sha256"]) == len(result["config_sha256"]) == 64
    other = Config()
    other.scheduler.max_active_requests += 1
    assert engine_identity(other)["config_sha256"] != result["config_sha256"]


def test_unknown_dirty_state_survives_streaming_response():
    from engine.processor.response import StreamResponse

    response = StreamResponse(type="done", request_id=None, generation={"engine_dirty": None})
    assert response.model_dump()["generation"]["engine_dirty"] is None


def test_instrumentation_reports_effective_sampler_and_disabled_state(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ParentBased

    from engine.tracing import Tracing

    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "parentbased_always_off")
    provider = TracerProvider(sampler=ParentBased(ALWAYS_OFF))
    try:
        settings = Tracing(provider).settings()
        assert settings["tracing_enabled"] is True
        assert settings["sampler"] == provider.sampler.get_description()
        assert settings["sampler_name"] == "parentbased_always_off"
        assert Tracing().settings()["tracing_enabled"] is False
    finally:
        provider.shutdown()
