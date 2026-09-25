from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Annotated, cast

from fastapi import APIRouter, Depends, FastAPI, Request

from engine.processor import GenerateRequest, build_request, GenerateResponse, build_generate_response
from engine.resources import Resources

router = APIRouter()


async def get_resources(request: Request) -> Resources:
    return cast(Resources, request.app.state.resources)


ResourcesDep = Annotated[Resources, Depends(get_resources)]


@router.post(
    "/generate",
    response_model=GenerateResponse,
)
async def generate(body: GenerateRequest, request: Request):
    resources = await get_resources(request)
    tokenizer = resources.tokenizer
    engine = resources.engine

    internal = build_request(body, tokenizer)

    result = await engine.generate(internal)

    return build_generate_response(result, tokenizer, result.generated_tokens)

def create_app(
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]],
) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app
