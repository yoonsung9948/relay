import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from engine.api import create_app
from engine.config import Config
from engine.engine import Engine
from engine.model.huggingface import HuggingFaceModel
import torch
from engine.processor import Qwen3Tokenizer
from engine.resources import Resources
from engine.runner import Runner
from engine.scheduler import Scheduler
import logging
import time


def build_app(config: Config) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        logger = logging.getLogger("inference")
        logger.setLevel(logging.INFO)
        started = time.perf_counter()

        stage = "startup"
        try:
            logger.info("startup_begin backend=%s repo=%s", config.backend, config.model.repo)
            stage = "tokenizer_load"
            logger.info("tokenizer_load_begin repo=%s", config.model.repo)
            tokenizer = Qwen3Tokenizer(config.model.repo)
            logger.info("tokenizer_load_complete")
            stage = "scheduler_init"
            scheduler = Scheduler(config.scheduler)
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            device = torch.device(device)

            stage = "model_init"
            logger.info("model_init_begin backend=%s device=%s", config.backend, device)
            if config.backend == "huggingface":
                model = HuggingFaceModel(config.model.repo)
            elif config.backend == "custom":
                from engine.model import load_custom_qwen3
                model = load_custom_qwen3(config.model.repo, config.architecture)
            else:
                raise ValueError(f"Unsupported backend: {config.backend}")

            logger.info("model_init_complete backend=%s", config.backend)
            stage = "runner_init"
            logger.info("runner_init_begin device=%s", device)
            runner = Runner(model, device=device)
            logger.info("runner_init_complete device=%s", device)
            engine = Engine(scheduler=scheduler, runner=runner, stop_ids=tokenizer.eos_token_ids)
            loop_task = asyncio.create_task(engine.run_loop(), name="inference-engine")

            app.state.resources = Resources(engine=engine, tokenizer=tokenizer, logger=logger)
        except Exception:
            logger.exception("startup_failed stage=%s backend=%s", stage, config.backend)
            raise
        logger.info("startup_ready elapsed_s=%.3f", time.perf_counter() - started)
        try:
            yield
        finally:
            logger.info("shutdown_begin")
            loop_task.cancel()
            try:
                with suppress(asyncio.CancelledError):
                    await loop_task
            finally:
                try:
                    await engine.shutdown()
                finally:
                    del app.state.resources
                    logger.info("shutdown_complete")

    return create_app(lifespan)
