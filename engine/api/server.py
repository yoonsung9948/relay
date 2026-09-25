from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Annotated, cast

from fastapi import APIRouter, Depends, FastAPI, Request

from engine.processor import GenerateRequest, build_request
from engine.resources import Resources

router = APIRouter()


async def get_resources(request: Request) -> Resources:
    return cast(Resources, request.app.state.resources)


ResourcesDep = Annotated[Resources, Depends(get_resources)]


@router.post("/generate")
async def generate(body: GenerateRequest, resources: ResourcesDep):
    internal = build_request(body, resources.tokenizer)
    return await resources.engine.generate(internal)


def create_app(
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]],
) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app
