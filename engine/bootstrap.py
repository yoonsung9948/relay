import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from importlib.metadata import version

import torch
from fastapi import FastAPI

from engine.api import create_app
from engine.boot_state import AppState, BootStatus
from engine.config import Config
from engine.engine import Engine
from engine.identity import engine_identity, resolve_revision
from engine.memory import MemoryRecorder
from engine.model.huggingface import HuggingFaceModel
from engine.processor import Qwen3Tokenizer
from engine.resources import Resources
from engine.runner import Runner
from engine.sampler import TemperatureSampler
from engine.scheduler import Scheduler
from engine.tracing import Tracing


def build_app(config: Config, *, config_path: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        logger = logging.getLogger("inference")
        logger.setLevel(logging.INFO)
        started = time.perf_counter()

        app.state.tracing = Tracing.from_env()
        app.state.boot = AppState()
        try:
            app.state.boot.tasks.append(
                asyncio.create_task(boot_engine(app, logger, config, started))
            )
        except Exception:
            logger.exception(
                "startup_failed stage=%s backend=%s", app.state.boot.status, config.backend
            )
            raise

        # shutdown sequence
        try:
            yield
        finally:
            logger.info("shutdown_begin")
            for task in app.state.boot.tasks:
                task.cancel()
            try:
                with suppress(asyncio.CancelledError):
                    results = await asyncio.gather(*app.state.boot.tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, Exception) and not isinstance(
                            result, asyncio.CancelledError
                        ):
                            logger.error("boot_engine_task_error: %s", str(result))
            finally:
                try:
                    resources = app.state.boot.resources
                    if resources is not None:
                        await resources.engine.shutdown()
                finally:
                    await app.state.tracing.shutdown()
                    logger.info("shutdown_complete")

    app = create_app(lifespan, config.app)
    app.state.config_path = config_path
    return app


async def boot_engine(
    app: FastAPI,
    logger: logging.Logger,
    config: Config,
    started: float,
):
    try:
        logger.info("boot_engine_start")
        logger.info("startup_begin backend=%s repo=%s", config.backend, config.model.repo)
        app.state.boot.status = BootStatus.TOKENIZER_LOAD
        logger.info("tokenizer_load_begin repo=%s", config.model.repo)
        revision = resolve_revision(config.model.repo, config.model.revision)
        tokenizer = Qwen3Tokenizer(config.model.repo, revision=revision)
        app.state.boot.status = BootStatus.SCHEDULER_INIT
        logger.info("tokenizer_load_complete")

        scheduler = Scheduler(config.scheduler)
        app.state.boot.status = BootStatus.MODEL_INIT
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        device = torch.device(device)

        logger.info("model_init_begin backend=%s device=%s", config.backend, device)
        if config.backend == "huggingface":
            model = HuggingFaceModel(config.model.repo, revision=revision)
        elif config.backend == "custom":
            from engine.model import load_custom_qwen3

            model = load_custom_qwen3(config.model.repo, config.architecture, revision=revision)
        else:
            raise ValueError(f"Unsupported backend: {config.backend}")

        logger.info("model_init_complete backend=%s", config.backend)

        app.state.boot.status = BootStatus.RUNNER_INIT
        logger.info("runner_init_begin device=%s", device)
        runner = Runner(model, device=device, sampler=TemperatureSampler())
        logger.info("runner_init_complete device=%s", device)
        runner.configure_cache(config.kv_cache)
        logger.info(
            "engine_config_loaded path=%s pool_slots=%d "
            "max_active_requests=%d max_decode_batch_size=%d",
            getattr(app.state, "config_path", None),
            len(runner.pool.owners) if runner.pool else 0,
            config.scheduler.max_active_requests,
            config.scheduler.max_decode_batch_size,
        )
        engine = Engine(
            scheduler=scheduler,
            runner=runner,
            tokenizer=tokenizer,
            generation_metadata={
                "model": config.model.repo,
                "backend": config.backend,
                "device": str(device),
                "dtype": str(next(model.parameters()).dtype),
                "device_name": (
                    torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
                ),
                "torch_version": torch.__version__,
                "transformers_version": version("transformers"),
                **engine_identity(config),
                "model_revision": revision,
            },
        )
        engine.memory = MemoryRecorder.from_env(
            runner,
            {
                **engine.generation_metadata,
                "max_active_requests": config.scheduler.max_active_requests,
                "max_decode_batch_size": config.scheduler.max_decode_batch_size,
                "tracing_enabled": app.state.tracing.provider is not None,
            },
        )
        engine.instrumentation = app.state.tracing.settings()
        engine.instrumentation["memory_enabled"] = engine.memory is not None
        engine.instrumentation["memory_supported"] = bool(
            engine.memory is not None and engine.memory.probe.supported
        )
        loop_task = asyncio.create_task(engine.run_loop(), name="inference-engine")
        app.state.boot.tasks.append(loop_task)

        app.state.boot.resources = Resources(engine=engine, tokenizer=tokenizer, logger=logger)
        logger.info("startup_ready elapsed_s=%.3f", time.perf_counter() - started)
        app.state.boot.status = BootStatus.READY
        logger.info("boot_engine_complete")
    except Exception as e:
        logger.error("boot_engine_error: %s", str(e))
        app.state.boot.error = str(e)
        app.state.boot.status = BootStatus.ERROR
        raise
