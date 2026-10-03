import time
from uuid import uuid4

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Annotated, cast
from engine.boot_state import AppState

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from engine.config.config import AppConfig
from engine.boot_state import AppState
from engine.processor import GenerateRequest, GenerateResponse, build_generate_response, Request as InternalRequest
from engine.boot_state import BootStatus
from engine.resources import Resources

router = APIRouter()


async def get_resources(request: Request) -> Resources:
    return cast(Resources, request.app.state.boot.resources)


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

async def get_boot(request: Request) -> AppState:
    return cast(AppState, request.app.state.boot)

@router.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "ok"}

@router.get("/health/ready")
async def health_ready(request: Request) -> dict[str, str]:
    boot = await get_boot(request)
    if boot.status == BootStatus.READY:
        return {"status": "ok"}
    if boot.status == BootStatus.ERROR:
        return {"status": "error", "stage": boot.status, "detail": boot.error or ""}
    return {"status": "not_ready", "stage": boot.status}

@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    boot = await get_boot(request)
    if boot.status == BootStatus.READY:
        return {"status": "ok", "stage": boot.status}
    if boot.status == BootStatus.ERROR:
        return {"status": "error", "stage": boot.status, "detail": "model failed to load"}
    return {"status": "not_ready", "stage": boot.status}

def create_app(
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]],
    config: AppConfig,
) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    if config.cors_domains:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_domains,
            allow_methods=["GET"],
            allow_headers=["*"],
        )
    app.include_router(router)
    return app
