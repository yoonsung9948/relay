from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from engine.api import create_app
from engine.config import Config
from engine.engine import Engine
from engine.processor import Qwen3Tokenizer
from engine.resources import Resources
from engine.runner import Runner
from engine.scheduler import FifoPolicy, Scheduler


def build_app(config: Config) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        tokenizer = Qwen3Tokenizer()
        scheduler = Scheduler(config.scheduler, FifoPolicy())
        runner = Runner()
        engine = Engine(scheduler=scheduler, runner=runner)

        try:
            app.state.resources = Resources(engine=engine, tokenizer=tokenizer)
            yield
        finally:
            try:
                await engine.shutdown()
            finally:
                del app.state.resources

    return create_app(lifespan)
