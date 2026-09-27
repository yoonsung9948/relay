import time
from uuid import uuid4

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Annotated, cast

from fastapi import APIRouter, Depends, FastAPI, Request

from engine.processor import GenerateRequest, GenerateResponse, build_generate_response, Request as InternalRequest
from engine.resources import Resources

router = APIRouter()


async def get_resources(request: Request) -> Resources:
    return cast(Resources, request.app.state.resources)


ResourcesDep = Annotated[Resources, Depends(get_resources)]


@router.post(
    "/generate",
    response_model=GenerateResponse,
)
async def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
    if len(body.prompt) == 0:
        return GenerateResponse(
            request_id=None,
            text="prompt cannot be empty",
            generated_tokens=0,
        )
    resources = await get_resources(request)
    tokenizer = resources.tokenizer
    engine = resources.engine

    logger = resources.logger
    request_id = uuid4()
    started = time.perf_counter()
    stage = "tokenize"
    logger.info(
        "request_received request_id=%s prompt_chars=%d max_tokens=%d",
        request_id, len(body.prompt), body.max_tokens,
    )
    try:
        internal = InternalRequest(
            request_id=request_id,
            tokens=tokenizer.encode_chat(body.prompt),
            max_tokens=body.max_tokens,
        )
        logger.info(
            "request_tokenized request_id=%s prompt_tokens=%d",
            request_id, len(internal.tokens),
        )
        stage = "generate"
        result = await engine.generate(internal)
        stage = "decode"
        response = build_generate_response(result, tokenizer, result.generated_tokens)
    except Exception:
        logger.exception("request_failed request_id=%s stage=%s", request_id, stage)
        raise
    logger.info(
        "request_complete request_id=%s generated_tokens=%d elapsed_s=%.3f",
        request_id, result.generated_tokens, time.perf_counter() - started,
    )
    return response


def create_app(
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]],
) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app
