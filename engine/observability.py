"""Bounded Prometheus metrics. No network exporter or per-request labels."""

import asyncio
import json
from collections import Counter
from time import perf_counter

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 15, 60, 180)


def sample(name, value, labels=()):
    suffix = ",".join(f"{k}={json.dumps(str(v))}" for k, v in labels)
    return f"{name}" + (f"{{{suffix}}}" if suffix else "") + f" {value}\n"


class Metrics:
    """Mutated and scraped on the serving event loop, without worker threads."""

    def __init__(self):
        self.counters = Counter()
        self.histograms = {}
        self.http_in_flight = 0

    def increment(self, name, *, amount=1, **labels):
        self.counters[(name, tuple(sorted(labels.items())))] += amount

    def observe(self, name, value, **labels):
        key = (name, tuple(sorted(labels.items())))
        # One fixed-size accumulator per bounded label combination.
        bounds = (1, 2, 4, 8, 16, 32, 64, 128) if name.endswith("batch_size") else LATENCY_BUCKETS
        state = self.histograms.setdefault(key, [0, 0.0, [0] * len(bounds), bounds])
        state[0] += 1
        state[1] += value
        for index, bound in enumerate(bounds):
            state[2][index] += value <= bound

    def render(self):
        lines = []
        for name in sorted({name for name, _ in self.counters}):
            lines.append(f"# TYPE {name} counter\n")
            for (key, labels), value in sorted(self.counters.items()):
                if key == name:
                    lines.append(sample(name, value, labels))
        for name in sorted({name for name, _ in self.histograms}):
            lines.append(f"# TYPE {name} histogram\n")
            for (key, labels), (count, total, buckets, bounds) in sorted(self.histograms.items()):
                if key != name:
                    continue
                for bound, value in zip(bounds, buckets):
                    lines.append(sample(name + "_bucket", value, (*labels, ("le", bound))))
                lines.append(sample(name + "_bucket", count, (*labels, ("le", "+Inf"))))
                lines.append(sample(name + "_count", count, labels))
                lines.append(sample(name + "_sum", total, labels))
        return "".join(lines)


class EngineMetrics(Metrics):
    def model_step(self, batch, seconds):
        request = batch[0]
        labels = {
            "phase": "prefill" if request.generated_tokens == 0 else "decode",
            "mode": request.kv_cache_mode,
            "batching": str(request.decode_batching).lower(),
        }
        self.observe("relay_engine_step_duration_seconds", seconds, **labels)
        # Count each physical model call once, regardless of participating requests.
        self.observe("relay_engine_step_batch_size", len(batch), **labels)

    def generation_finished(self, request, error):
        outcome = "cancelled" if request.cancelled else "failed" if error else "success"
        labels = {
            "mode": request.kv_cache_mode,
            "batching": str(request.decode_batching).lower(),
            "outcome": outcome,
        }
        self.increment("relay_engine_generations_total", **labels)
        self.observe(
            "relay_engine_generation_duration_seconds",
            request.timings.clock() - request.timings.started_at,
            **labels,
        )
        self.increment(
            "relay_engine_generated_tokens_total", amount=request.generated_tokens, **labels
        )
        timings = request.timings
        for name, value in (
            (
                "ttft",
                timings.first_token_at - timings.started_at
                if timings.first_token_at is not None
                else None,
            ),
            (
                "initial_queue",
                timings.scheduled_at - timings.enqueued_at
                if timings.scheduled_at is not None and timings.enqueued_at is not None
                else None,
            ),
            ("prefill", timings.prefill_ms / 1000 if timings.prefill_ms is not None else None),
        ):
            if value is not None:
                self.observe(f"relay_engine_{name}_seconds", value, **labels)


def engine_snapshot(engine):
    scheduler, runner = engine.scheduler, engine.runner
    config, cache = scheduler.config, runner.cache_config
    pool = runner.pool
    gauges = {
        "ready": int(engine.state.name == "READY"),
        "pending_requests": len(scheduler.pending),
        "ready_requests": len(scheduler.ready),
        "active_requests": len(scheduler.active),
        "active_requests_limit": config.max_active_requests,
        "pending_requests_limit": config.max_pending_requests,
        "decode_batch_size_limit": config.max_decode_batch_size,
        "kv_pool_slots": len(pool.owners) if pool else 0,
        "kv_pool_slots_used": sum(owner is not None for owner in pool.owners) if pool else 0,
        "kv_pool_tokens_used": sum(pool.lengths) if pool else 0,
        "kv_pool_tokens_capacity": len(pool.owners) * pool.capacity if pool else 0,
        "kv_pool_allocated_bytes": pool.allocated_bytes if pool else 0,
        "kv_pool_budget_bytes": cache.max_bytes,
        "kv_dynamic_reserved_bytes": sum(runner.reservations.values()),
        "kv_dynamic_budget_bytes": cache.dynamic_max_bytes,
    }
    if runner.device.type == "cuda":
        import torch

        # Allocator statistics are host-side; do not synchronize CUDA for telemetry.
        gauges.update(
            cuda_allocated_bytes=torch.cuda.memory_allocated(runner.device),
            cuda_reserved_bytes=torch.cuda.memory_reserved(runner.device),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(runner.device),
        )
    return (
        "".join(
            f"# TYPE relay_engine_{name} gauge\n" + sample(f"relay_engine_{name}", value)
            for name, value in gauges.items()
        )
        + engine.metrics.render()
    )


class MetricsMiddleware:
    """Full HTTP lifetime, including stream delivery and errors after HTTP 200."""

    def __init__(self, app, state):
        self.app, self.state = app, state

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/generate":
            return await self.app(scope, receive, send)
        metrics = self.state.metrics
        started, status, interrupted, complete = perf_counter(), 500, False, False
        metrics.http_in_flight += 1

        async def measured_send(message):
            nonlocal status, complete
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                complete = True

        try:
            await self.app(scope, receive, measured_send)
        except (asyncio.CancelledError, OSError):
            interrupted = True
            raise
        finally:
            outcome = scope.get("relay.outcome", "rejected" if 400 <= status < 500 else "failed")
            if status >= 400:
                outcome = "rejected" if status < 500 or outcome == "rejected" else "failed"
            elif interrupted or not complete:
                outcome = "cancelled"
            metrics.http_in_flight -= 1
            metrics.increment("relay_engine_http_requests_total", status=status, outcome=outcome)
            metrics.observe(
                "relay_engine_http_duration_seconds", perf_counter() - started, outcome=outcome
            )
