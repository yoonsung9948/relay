import asyncio
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, aclosing, nullcontext
from typing import Annotated, cast
from uuid import uuid4

import anyio
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from engine.boot_state import AppState, BootStatus
from engine.capabilities import describe_engine
from engine.config.config import AppConfig
from engine.execution import CapacityError
from engine.metrics import RequestTimings
from engine.observability import (
    CONTENT_TYPE,
    Metrics,
    MetricsMiddleware,
    engine_snapshot,
)
from engine.processor import GenerateRequest, GenerateResponse, build_generate_response
from engine.processor import Request as InternalRequest
from engine.processor.response import GenerationMetadata, StreamResponse
from engine.resources import Resources
from engine.sampling import SamplingOptions
from engine.tracing import TraceMiddleware, Tracing, phase


class EngineStreamingResponse(StreamingResponse):
    def __init__(self, content, *, on_close=None, **kwargs):
        super().__init__(content, **kwargs)
        self.on_close = on_close

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            if self.on_close is not None:
                self.on_close()
            # Close the generator even when sending a body chunk fails on disconnect.
            if isinstance(self.body_iterator, AsyncGenerator):
                with anyio.CancelScope(shield=True):
                    await self.body_iterator.aclose()


router = APIRouter()


def _generation_metadata(engine, request: InternalRequest) -> GenerationMetadata:
    return {
        **getattr(engine, "generation_metadata", {}),
        "max_tokens": request.max_tokens,
        "sampling": request.sampling.strategy,
        "temperature": request.sampling.temperature,
        "use_kv_cache": request.use_kv_cache,
        "kv_cache_mode": request.kv_cache_mode,
        "decode_batching": request.decode_batching,
        "implementation_version": "pool-v1",
    }


async def get_resources(request: Request) -> Resources:
    boot = request.app.state.boot
    if boot.resources is None or boot.status != BootStatus.READY:
        raise HTTPException(status_code=503, detail="Engine not ready")
    try:
        await boot.resources.engine.health()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Engine unavailable") from exc
    return cast(Resources, boot.resources)


ResourcesDep = Annotated[Resources, Depends(get_resources)]


@router.post(
    "/generate",
    response_model=GenerateResponse,
)
async def generate(body: GenerateRequest, request: Request) -> StreamingResponse | GenerateResponse:
    request.scope["relay.outcome"] = "cancelled"
    started = time.perf_counter()
    resources = await get_resources(request)
    tokenizer = resources.tokenizer
    engine = resources.engine
    if body.use_kv_cache and not getattr(engine.runner, "supports_kv_cache", False):
        raise HTTPException(
            status_code=422,
            detail="KV cache is supported only by the custom Qwen3 backend",
        )

    logger = resources.logger
    request_id = uuid4()
    request_trace = request.app.state.tracing.request(request_id)
    request.scope["relay.request_trace"] = request_trace
    if request_trace:
        request_trace.span.set_attribute("cache.enabled", body.use_kv_cache)
        request_trace.span.set_attribute("cache.mode", body.kv_cache_mode)
        request_trace.span.set_attribute("decode.batching", body.decode_batching)
        request_trace.span.set_attribute("sampling.temperature", body.temperature)
        request_trace.span.set_attribute(
            "sampling.strategy", "greedy" if body.temperature == 0 else "temperature"
        )
        for key in ("model", "backend", "device", "dtype", "device_name"):
            value = getattr(engine, "generation_metadata", {}).get(key)
            if value is not None:
                request_trace.span.set_attribute(f"inference.{key}", value)
    stage = "tokenize"
    logger.info(
        "request_received request_id=%s prompt_chars=%d max_tokens=%d",
        request_id,
        len(body.prompt),
        body.max_tokens,
    )
    internal = None
    try:
        with request_trace.phase("tokenization") if request_trace else nullcontext():
            tokens = tokenizer.encode_chat(body.prompt)
        internal = InternalRequest(
            request_id=request_id,
            tokens=tokens,
            trace=request_trace,
            max_tokens=body.max_tokens,
            sampling=SamplingOptions(temperature=body.temperature),
            use_kv_cache=body.use_kv_cache,
            kv_cache_mode=body.kv_cache_mode,
            decode_batching=body.decode_batching,
            timings=RequestTimings(started_at=started),
        )
        logger.info(
            "request_tokenized request_id=%s prompt_tokens=%d",
            request_id,
            len(internal.tokens),
        )
        stage = "generate"
        try:
            engine.submit(internal)
        except CapacityError as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc),
                headers={"Retry-After": "1", "X-Relay-Rejection": "engine_capacity"},
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if body.stream:

            async def stream():
                try:
                    async with aclosing(engine.stream_generate(internal)) as chunks:
                        async for text in chunks:
                            yield (
                                StreamResponse(
                                    request_id=request_id,
                                    text=text,
                                ).model_dump_json()
                                + "\n"
                            )
                    terminal = (
                        StreamResponse(
                            type="done",
                            request_id=request_id,
                            metrics=internal.timings.finish(internal.generated_tokens),
                            finish_reason=internal.finish_reason,
                            generation=_generation_metadata(engine, internal),
                        ).model_dump_json()
                        + "\n"
                    )
                    if request_trace:
                        request_trace.finish("success", internal)
                    request.scope["relay.outcome"] = "success"
                    yield terminal
                except asyncio.CancelledError:
                    raise
                except Exception:
                    request.scope["relay.outcome"] = "failed"
                    if request_trace:
                        request_trace.finish("failed", internal)
                    logger.exception("stream_failed request_id=%s", request_id)
                    yield (
                        StreamResponse(
                            type="error",
                            request_id=request_id,
                            error="Generation failed",
                        ).model_dump_json()
                        + "\n"
                    )
                finally:
                    if request_trace:
                        request_trace.finish("cancelled", internal)
                    engine.cancel_request(internal)
                    logger.info(
                        "stream_closed request_id=%s elapsed_s=%.3f",
                        request_id,
                        time.perf_counter() - started,
                    )

            return EngineStreamingResponse(
                stream(),
                on_close=lambda: engine.cancel_request(internal),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        result = await engine.generate(internal)
        stage = "decode"
        with phase(internal, "text.finalize"):
            response = build_generate_response(result, tokenizer, result.generated_tokens)
        response.metrics = internal.timings.finish(result.generated_tokens)
        response.finish_reason = internal.finish_reason
        response.generation = _generation_metadata(engine, internal)
        if request_trace:
            request_trace.finish("success", internal)
    except BaseException as exc:
        request.scope["relay.outcome"] = (
            "cancelled"
            if isinstance(exc, asyncio.CancelledError)
            else "rejected"
            if isinstance(exc, HTTPException)
            else "failed"
        )
        if internal is not None:
            engine.cancel_request(internal)
        if request_trace:
            request_trace.finish(
                "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed", internal
            )
        logger.exception("request_failed request_id=%s stage=%s", request_id, stage)
        raise
    logger.info(
        "request_complete request_id=%s generated_tokens=%d elapsed_s=%.3f",
        request_id,
        result.generated_tokens,
        time.perf_counter() - started,
    )
    request.scope["relay.outcome"] = "success"
    return response


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request):
    registry = request.app.state.metrics
    output = registry.render()
    output += (
        "# TYPE relay_engine_http_in_flight gauge\n"
        f"relay_engine_http_in_flight {registry.http_in_flight}\n"
    )
    boot = getattr(request.app.state, "boot", None)
    engine = boot.resources.engine if boot and boot.resources else None
    if engine is not None:
        output += engine_snapshot(engine)
    else:
        output += "# TYPE relay_engine_ready gauge\nrelay_engine_ready 0\n"
    return Response(output, headers={"Content-Type": CONTENT_TYPE})


async def get_boot(request: Request) -> AppState:
    return cast(AppState, request.app.state.boot)


@router.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def health_ready(request: Request):
    boot = await get_boot(request)
    if boot.status == BootStatus.READY and boot.resources is not None:
        try:
            await boot.resources.engine.health()
        except RuntimeError:
            return JSONResponse(status_code=503, content={"status": "error"})
        return {
            "status": "ok",
            "stage": boot.status,
            "capabilities": describe_engine(
                boot.resources.engine, getattr(request.app.state, "config_path", None)
            ),
        }
    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "stage": boot.status},
    )


@router.get("/health")
async def health(request: Request):
    return await health_ready(request)


def create_app(
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]],
    config: AppConfig,
) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.state.metrics = Metrics()
    app.add_middleware(MetricsMiddleware, state=app.state)
    app.state.tracing = Tracing()
    app.add_middleware(TraceMiddleware, state=app.state)
    if config.cors_domains:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_domains,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    app.include_router(router)
    return app
