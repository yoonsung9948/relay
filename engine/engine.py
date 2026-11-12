import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import ExitStack, suppress
from enum import Enum, auto
from time import perf_counter

import torch
from transformers import AsyncTextIteratorStreamer

from engine.execution import CapacityError
from engine.memory import MemoryRecorder
from engine.observability import EngineMetrics
from engine.processor import GenerateResult, Qwen3Tokenizer, Request
from engine.runner import Runner
from engine.scheduler import Scheduler
from engine.tracing import phase

logger = logging.getLogger("inference.engine")


class EngineState(Enum):
    OFFLINE = auto()
    STARTING = auto()
    LOADING = auto()
    READY = auto()
    ERROR = auto()


DONE = object()


class Engine:
    def __init__(
        self,
        scheduler: Scheduler,
        runner: Runner,
        tokenizer: Qwen3Tokenizer,
        generation_metadata: dict[str, str | int] | None = None,
        memory: MemoryRecorder | None = None,
    ):
        self.metrics = EngineMetrics()
        self.scheduler = scheduler
        self.runner = runner
        self.scheduler.admit = getattr(runner, "admit", lambda request: True)
        self.tokenizer = tokenizer
        self.generation_metadata = generation_metadata or {}
        self.memory = memory
        self.stop_token_ids = tokenizer.eos_token_ids
        self.state = EngineState.OFFLINE
        # Includes queued, running, and output-consuming requests.
        self._requests: dict = {}

    async def generate(self, request: Request) -> GenerateResult:
        self._ensure_submitted(request)
        try:
            await request.done_event.wait()
            if request.error is not None:
                raise request.error
            return GenerateResult(
                request_id=request.request_id,
                token_ids=request.tokens,
                generated_tokens=request.generated_tokens,
            )
        finally:
            self.cancel_request(request)

    async def _stream_task(self, request: Request, streamer: AsyncTextIteratorStreamer):
        decode_seconds = 0.0
        try:
            while True:
                token_id = await request.output_queue.get()
                if token_id is DONE:
                    if request.error is not None:
                        raise request.error
                    break
                started = perf_counter() if request.trace else 0
                streamer.put(torch.tensor([token_id]))
                if request.trace:
                    decode_seconds += perf_counter() - started
        finally:
            try:
                with phase(request, "text.finalize"):
                    streamer.end()
                if request.trace:
                    request.trace.span.set_attribute("text.decode_total_ms", decode_seconds * 1000)
            except Exception:
                # A decoding failure during end() must also wake the text consumer.
                streamer.on_finalized_text("", stream_end=True)
                raise

    async def stream_generate(self, request: Request) -> AsyncGenerator[str, None]:
        # Set up decoding before submission so setup failures leave no scheduled work.
        streamer = AsyncTextIteratorStreamer(
            tokenizer=self.tokenizer.tokenizer,
            skip_prompt=False,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self._ensure_submitted(request)
        producer = asyncio.create_task(self._stream_task(request, streamer))
        try:
            async for text in streamer:
                if text:
                    yield text
            await producer
        finally:
            self.cancel_request(request)
            producer.cancel()
            with suppress(asyncio.CancelledError):
                await producer

    def submit(self, request: Request):
        self._submit(request)

    def _ensure_submitted(self, request):
        if self._requests.get(request.request_id) is not request:
            self._submit(request)

    def _submit(self, request: Request):
        if self.state != EngineState.READY:
            raise RuntimeError("engine not ready")
        if request.cancelled or request.done_event.is_set():
            raise RuntimeError("request already terminated")
        if request.request_id in self._requests:
            raise RuntimeError("request already submitted")
        if request.max_tokens < 0:
            raise ValueError("max_tokens must not be negative")
        if request.use_kv_cache and not getattr(self.runner, "supports_kv_cache", False):
            raise ValueError("KV cache is supported only by the custom Qwen3 backend")
        if hasattr(self.runner, "validate_request"):
            self.runner.validate_request(request)
        if (
            request.max_tokens
            and len(self.scheduler.pending) >= self.scheduler.config.max_pending_requests
        ):
            self.metrics.increment("relay_engine_rejections_total", reason="queue_full")
            self.scheduler.rejected_requests += 1
            raise CapacityError("generation waiting queue is full")
        if self.memory is not None:
            self.memory.request_started(request.request_id)
        self._requests[request.request_id] = request
        if request.max_tokens == 0:
            request.finish_reason = "length"
            self._complete_request(request)
        else:
            request.timings.enqueue()
            if request.trace:
                request.trace.enqueue()
            self.scheduler.add_request(request)

    async def health(self) -> None:
        if self.state != EngineState.READY:
            raise RuntimeError("engine not ready")

    async def run_loop(self):
        self.state = EngineState.READY
        logger.info("engine_loop_started")
        try:
            while True:
                batch = self.scheduler.schedule()
                batch = [req for req in batch if not req.cancelled]
                if not batch:
                    await asyncio.sleep(0.01)
                    continue
                for request in batch:
                    request.timings.schedule()
                    if request.trace:
                        request.trace.scheduled()
                step_started = perf_counter()
                with ExitStack() as spans:
                    for req in batch:
                        if not req.trace or not req.trace.steps:
                            continue
                        cached_tokens = req.cached_tokens if req.kv_cache_mode == "pool" else 0
                        if req.kv_cache is not None:
                            cached_tokens = req.kv_cache.get_seq_length()
                        spans.enter_context(
                            phase(
                                req,
                                "runner.step",
                                **{
                                    "step.index": req.generated_tokens,
                                    "step.phase": "first"
                                    if req.generated_tokens == 0
                                    else "decode",
                                    "input_tokens": len(req.tokens) - cached_tokens,
                                    "batch_size": len(batch),
                                    "scheduler.wait_ms": req.trace.wait_ms if req.trace else 0.0,
                                    "cache.enabled": req.use_kv_cache,
                                    "cache.mode": req.kv_cache_mode,
                                },
                            )
                        )
                    # Runner returns host token IDs: includes device-to-host completion.
                    results = self.runner.run_requests(batch)
                step_seconds = perf_counter() - step_started
                self.metrics.model_step(batch, step_seconds)
                if len(results) != len(batch):
                    raise RuntimeError("runner returned an incorrect number of tokens")
                if any(type(token) is not int for token in results):
                    raise RuntimeError("runner must return one integer token ID per request")
                for request, result in zip(batch, results):
                    if request.cancelled:
                        continue
                    if request.generated_tokens == 0:
                        request.timings.prefill_ms = step_seconds * 1000
                    else:
                        request.timings.decode_batch_sizes.append(len(batch))
                    request.timings.token()
                    request.tokens.append(result)
                    request.generated_tokens += 1
                    finished = (
                        result in self.stop_token_ids
                        or request.generated_tokens >= request.max_tokens
                    )
                    try:
                        request.output_queue.put_nowait(result)
                    except asyncio.QueueFull:
                        self._fail_request(request, RuntimeError("request output queue is full"))
                        self.scheduler.cancel_request(request)
                        continue
                    self.scheduler.finish_step(request, finished=finished)
                    if not finished and request.trace:
                        request.trace.requeued()
                    if finished:
                        request.finish_reason = "eos" if result in self.stop_token_ids else "length"
                        self._complete_request(request)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.state = EngineState.OFFLINE
            self._fail_active("engine stopped")
            logger.info("engine_loop_stopped")
            raise
        except Exception as exc:
            self.metrics.increment("relay_engine_loop_failures_total")
            self.state = EngineState.ERROR
            logger.exception("engine_loop_failed")
            self._fail_active(f"engine generation failed: {exc}")
            raise

    def _complete_request(self, request: Request, error: Exception | None = None):
        # A terminal outcome is recorded only once; cancellation can't rewrite success.
        if request.done_event.is_set():
            return
        if request.output_queue.full() and error is None:
            error = RuntimeError("request output queue is full")
        self.metrics.generation_finished(request, error)
        request.error = error
        self.scheduler.finish_step(request, finished=True)
        self._release_cache(request)
        if error is not None:
            while not request.output_queue.empty():
                request.output_queue.get_nowait()
        request.output_queue.put_nowait(DONE)
        request.done_event.set()

    def _fail_request(self, request: Request, exc: Exception):
        self._complete_request(request, error=exc)

    def _fail_active(self, message: str):
        for request in list(self._requests.values()):
            self._fail_request(request, RuntimeError(message))
        self.scheduler.clear()

    def _release_cache(self, request: Request):
        if hasattr(self.runner, "release"):
            self.runner.release(request)
        if request.kv_cache is not None:
            request.kv_cache.clear()
            request.kv_cache = None

    def cancel_request(self, request: Request):
        if not request.done_event.is_set():
            self.scheduler.cancel_request(request)
            self._fail_request(request, RuntimeError("request cancelled"))
        self._release_cache(request)
        self._requests.pop(request.request_id, None)
        if self.memory is not None:
            self.memory.request_finished(request.request_id)

    async def shutdown(self):
        self.state = EngineState.OFFLINE
        self._fail_active("engine shut down")
        if self.memory is not None:
            self.memory.close()
        logger.info("engine_shutdown")
