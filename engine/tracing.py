"""Opt-in phase tracing. No prompts, completions, or benchmark-specific state."""

import asyncio
import os
from contextlib import contextmanager, nullcontext
from time import perf_counter
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


def enabled() -> bool:
    return (
        os.getenv("RELAY_TRACING_ENABLED", "false").lower() == "true"
        and os.getenv("OTEL_SDK_DISABLED", "false").lower() != "true"
    )


class Tracing:
    def __init__(self, provider=None):
        self.provider = provider
        self.tracer = (provider or trace.NoOpTracerProvider()).get_tracer("relay.engine")

    @classmethod
    def from_env(cls):
        if not enabled():
            return cls()
        # SDK and network exporter are optional when tracing is disabled.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": os.getenv("OTEL_SERVICE_NAME", "relay-engine"),
                    "service.version": os.getenv("RELAY_REVISION", "unknown"),
                    "service.instance.id": str(uuid4()),
                }
            )
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(timeout=2),
                max_queue_size=1024,
                max_export_batch_size=128,
                schedule_delay_millis=1000,
                export_timeout_millis=3000,
            )
        )
        return cls(provider)

    def settings(self):
        return {
            "tracing_enabled": self.provider is not None,
            "sampler": (self.provider.sampler.get_description() if self.provider else "disabled"),
            "sampler_name": (
                os.getenv("OTEL_TRACES_SAMPLER", "parentbased_always_on")
                if self.provider
                else "disabled"
            ),
            "trace_steps": os.getenv("RELAY_TRACE_STEPS", "true").lower() == "true",
        }

    async def shutdown(self):
        if self.provider:
            await asyncio.to_thread(self.provider.shutdown)

    def request(self, request_id):
        if not self.provider:
            return None
        steps = os.getenv("RELAY_TRACE_STEPS", "true").lower() == "true"
        span = self.tracer.start_span(
            "generation.request",
            attributes={
                "request.id": str(request_id),
                "tracing.steps.enabled": steps,
                "cache.enabled": False,
            },
        )
        return RequestTrace(self.tracer, span, steps=steps) if span.is_recording() else None


class RequestTrace:
    """Explicit parent context follows a request into the scheduler background task."""

    def __init__(self, tracer, span, *, steps=True):
        self.tracer = tracer
        self.span = span
        self.context = trace.set_span_in_context(span)
        self.steps = steps
        self.queue_span = None
        self.waiting_since = None
        self.wait_ms = 0.0
        self.finished = False
        self.outcome = None

    @contextmanager
    def phase(self, name, **attributes):
        with self.tracer.start_as_current_span(
            name,
            context=self.context,
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield span
            except BaseException:
                span.set_status(Status(StatusCode.ERROR, "phase_interrupted"))
                raise

    def requeued(self):
        self.waiting_since = perf_counter()

    def enqueue(self):
        self.requeued()
        self.queue_span = self.tracer.start_span("scheduler.initial_wait", context=self.context)

    def scheduled(self):
        if self.waiting_since is not None:
            self.wait_ms = (perf_counter() - self.waiting_since) * 1000
            self.waiting_since = None
        if self.queue_span:
            self.queue_span.end()
            self.queue_span = None

    def finish(self, outcome, request=None):
        if self.finished:
            return
        self.finished = True
        self.outcome = outcome
        self.scheduled()
        self.span.set_attribute("request.outcome", outcome)
        if request is not None:
            self.span.set_attributes(
                {
                    "input_tokens": request.timings.input_tokens,
                    "output_tokens": request.generated_tokens,
                }
            )
            if request.finish_reason:
                self.span.set_attribute("finish_reason", request.finish_reason)
        if outcome != "success":
            self.span.set_status(Status(StatusCode.ERROR, outcome))
        self.span.end()


def phase(request, name, **attributes):
    if not request.trace or (name == "runner.step" and not request.trace.steps):
        return nullcontext()
    return request.trace.phase(name, **attributes)


class TraceMiddleware:
    """ASGI server span stays open through streaming and disconnect cleanup."""

    def __init__(self, app, state):
        self.app, self.state = app, state

    async def __call__(self, scope, receive, send):
        tracing = self.state.tracing
        if scope["type"] != "http" or scope["path"] != "/generate" or not tracing.provider:
            return await self.app(scope, receive, send)
        headers = {k.decode("latin1"): v.decode("latin1") for k, v in scope["headers"]}
        parent = TraceContextTextMapPropagator().extract(headers)
        with tracing.tracer.start_as_current_span(
            "POST /generate",
            context=parent,
            kind=SpanKind.SERVER,
            attributes={"http.request.method": "POST", "http.route": "/generate"},
            record_exception=False,
            set_status_on_exception=False,
        ) as span:

            async def traced_send(message):
                if message["type"] == "http.response.start":
                    span.set_attribute("http.response.status_code", message["status"])
                    if message["status"] >= 400:
                        span.set_status(Status(StatusCode.ERROR, "http_error"))
                await send(message)

            try:
                await self.app(scope, receive, traced_send)
            except BaseException:
                span.set_status(Status(StatusCode.ERROR, "request_interrupted"))
                raise
            finally:
                request_trace = scope.get("relay.request_trace")
                if request_trace:
                    request_trace.finish("cancelled")
                    if request_trace.outcome != "success":
                        span.set_status(Status(StatusCode.ERROR, "generation_failed"))
