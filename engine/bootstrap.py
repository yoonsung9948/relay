from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from engine.api import create_app
from engine.config import Config
from engine.engine import Engine
from engine.model.huggingface import HuggingFaceModel
import torch
from engine.processor import Qwen3Tokenizer
from engine.resources import Resources
from engine.runner import Runner
from engine.scheduler import FifoPolicy, Scheduler


def build_app(config: Config) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        tokenizer = Qwen3Tokenizer(config.model.repo)
        scheduler = Scheduler(config.scheduler, FifoPolicy())

        device = "cpu"
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        device = torch.device(device)
        model = HuggingFaceModel(config.model.repo, device=device)
        eos = model.model.generation_config.eos_token_id

        runner = Runner(model, device=device)
        engine = Engine(scheduler=scheduler, runner=runner, eos=eos)

        try:
            app.state.resources = Resources(engine=engine, tokenizer=tokenizer)
            yield
        finally:
            try:
                await engine.shutdown()
            finally:
                del app.state.resources

    return create_app(lifespan)
