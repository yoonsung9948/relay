# Lightweight diagnostic tracing

The platform exports standard OTLP traces. Loadgen needs neither an OTel SDK nor
access to the Collector or Jaeger. Loadgen's offline analyzer only reads saved files and imports no engine code. Tracing is off by default.

## Start the local trace stack

From this directory (requires Docker Compose):

```sh
mkdir -p traces
export OTEL_COLLECTOR_UID="$(id -u)"
export OTEL_COLLECTOR_GID="$(id -g)"
docker compose up -d
```

Collector receives OTLP/HTTP at `http://127.0.0.1:4318`. Jaeger's UI is
`http://127.0.0.1:16686`. Collector appends OTLP JSON **batches**, one per line,
to `traces/spans.jsonl`; each line may contain many spans. Jaeger's default
in-memory store disappears on restart; the host file is the offline archive.

`docker compose down` stops both services. Stop the Collector before copying its
archive for analysis, so the last batch is flushed. The file is append-only with
no rotation: archive/rename it between diagnostic sessions while the Collector
is stopped. Its output directory must exist and be writable by the configured
UID/GID. The memory limiter bounds the Collector pipeline, and SDK/export queues
are bounded; congestion or outages can drop spans rather than block generation.
This is best-effort diagnostic telemetry, not lossless storage.

For allocator peaks independent of tracing, see [memory measurements](MEMORY.md).
The engine saves its own memory archive; the offline analyzer joins it to each
trial using engine request IDs already present in client results.

If `docker compose` is unavailable, install the Compose plugin for your Docker
runtime before using these commands.

## Enable each service

Install Python tracing support from the inference repository:

```sh
.venv/bin/python -m pip install '.[tracing]'
```

The engine container image includes these dependencies. Rebuild and publish your
engine image before provisioning a worker with this change.

Set these variables in **each** service process, then start it normally:

```sh
export RELAY_TRACING_ENABLED=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
export OTEL_TRACES_SAMPLER=parentbased_always_on
export RELAY_REVISION="$(git rev-parse HEAD)"
```

Service names default to `relay-control-plane` and `relay-engine`; override each
separately with `OTEL_SERVICE_NAME`. `OTEL_SDK_DISABLED=true` also disables tracing.
Export uses OTLP/HTTP protobuf; an explicit `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`
must include `/v1/traces`. Standard OTLP headers/TLS settings are supported by the
exporters. Use `RELAY_TRACE_STEPS=false` on Python for phase-only tracing, without
one span per generated token. No prompts, generated text, authentication headers,
or raw exception messages are added by the custom phase instrumentation.

For the remote GPU, `127.0.0.1` refers to the worker itself, not your Mac.
Use a reachable Collector on the same private network, or an SSH reverse tunnel
from your Mac to the worker (use the worker's actual SSH host/port):

```sh
ssh -N -R 4318:127.0.0.1:4318 WORKER_SSH_HOST
```

Then the engine can export to `http://127.0.0.1:4318` on the worker. For an engine
inside another container, ensure that address reaches the tunnel in its network
namespace. This stack binds the receiver to localhost on your Mac; it does not
create a public ingestion endpoint.

Go and Python configuration are independent. Go's environment is not copied to
the remote worker. For future RunPod provisioning, the optional `engine_env` map
under `control_plane_config.gpu_provider_config.runpod` supplies worker variables:

```yaml
engine_env:
  RELAY_TRACING_ENABLED: "true"
  OTEL_EXPORTER_OTLP_ENDPOINT: "http://REACHABLE_COLLECTOR:4318"
  OTEL_TRACES_SAMPLER: "parentbased_always_on"
  RELAY_TRACE_STEPS: "true"
```

This only affects newly created pods; configure/restart an existing worker
separately. `HF_TOKEN` retains its existing dedicated configuration.

## Run a diagnostic benchmark

From the loadgen repository, with the control plane and engine ready:

```sh
export RELAY_DEMO_KEY=YOUR_EXISTING_DEMO_KEY
.venv/bin/loadgen run configs/scenarios/relay-diagnostic.yaml
```

The only new loadgen option is:

```yaml
request:
  trace_context: true
```

Each outgoing request receives a fresh W3C `traceparent` with sampled flag `01`.
Loadgen saves `trace_id` and `client_span_id` in its own request records. It does
not export a client span; the analyzer recognizes that external parent. No run ID
or benchmark-specific header is sent to the platform. A configured static
`traceparent` conflicts with this option and is rejected. Warmups have independent
IDs and remain excluded from measured-request analysis.

The sampled flag asks a parent-based sampler to retain diagnostic requests; a
parent-based ratio sampler will therefore still retain these requests. Use
tracing disabled for baseline numbers, or deliberately configure a non-parent-
based sampler at the entry service if you want to sample these requests.

## Analyze saved artifacts

Offline analysis is now bundled with loadgen. From the loadgen checkout:

```sh
.venv/bin/loadgen benchmark analyze \
  --run results/RUN_ID \
  --traces /path/to/inference/observability/traces/spans.jsonl \
  --out /tmp/benchmark-RUN_ID \
  --jaeger-url http://127.0.0.1:16686
```

Open `/tmp/benchmark-RUN_ID/report.html`. `analysis.json` preserves the client
summary, experiment metadata, trace coverage, per-operation statistics, and the
request-to-trace mapping. `spans.jsonl` contains normalized spans belonging only
to the selected run, including incomplete traces for inspection. Inputs remain
unchanged. Repeat `--traces` for multiple archive files and deduplicates identical
spans. Old successful Relay records can fall back to the engine's `request.id`.
Malformed input fails clearly; an unfinished final archive line is reported as a
warning and ignored. Analyze a stopped/copied archive for reproducible results.

See loadgen's `ANALYZER.md` for definitions and limits. Analysis requires no
running engine or GPU.

## What the spans mean

Typical streaming request:

```text
Go POST /generate                 # full proxy HTTP lifetime
  request.parse                   # local request parsing only
  engine.request                  # HTTP client span, includes streamed body
    Python POST /generate         # full ASGI lifetime including response stream
      generation.request          # tokenization through text completion
        tokenization
        scheduler.initial_wait
        runner.step [phase=first, index=0]
        runner.step [phase=decode, index=1...]
        text.finalize
```

`runner.step` includes model invocation, sampling, and transfer of the resulting
token ID to the host. Its `input_tokens` records the actual prefix length and
`scheduler.wait_ms` records time awaiting that scheduling turn. Streaming text
conversion is accumulated as `text.decode_total_ms` on `generation.request`.
Errors/cancellation close outstanding spans; streamed application errors mark
spans as failures even after HTTP 200 has been sent. Health polling is excluded.

**Execution mode controls prefix work.** Uncached requests recompute the prefix;
dynamic and pooled caches process the uncached suffix. Pooled decode uses one new
input token per request, with optional batching. `phase=first` is the
first token-producing runner call, not a separately measured GPU-only prefill.
`phase=decode` means subsequent calls; it does not imply a KV cache exists.
Wall spans include host work and synchronization, and cannot diagnose individual
GPU kernels. Use CUDA events/Nsight for that later.

For KV-cache comparisons, save baseline results with tracing disabled, plus a
separate small diagnostic run. Repeat with the cache using the same model,
hardware, workload, warmups, token budget, and tracing settings. Quantify tracing
overhead separately before relying on instrumented performance numbers.


## Capacity and performance metrics

Both services expose Prometheus metrics without enabling tracing or adding exporter
dependencies. Keep the engine endpoint on your service network. The Go endpoint
requires `X-Demo-Key`, including during collection.

Before starting this stack, export `RELAY_DEMO_KEY`. Set
`ENGINE_METRICS_TARGET` and `CONTROL_PLANE_METRICS_TARGET` to host:port addresses
reachable from the Collector container; defaults use `host.docker.internal:8000`
and `host.docker.internal:8080`. Services bound only to the host loopback may need a
reachable interface or a local proxy. These targets are separate from OTLP endpoints.
For a remote worker, forward a local port with SSH (`-L 18000:127.0.0.1:8000`) and
set `ENGINE_METRICS_TARGET=host.docker.internal:18000`; ensure the forwarded listener
is reachable from Docker (on Linux, use an appropriate private bind address).

```sh
export RELAY_DEMO_KEY=YOUR_EXISTING_DEMO_KEY
cd observability
docker compose up -d
curl -fsS http://127.0.0.1:8000/metrics
curl -fsS -H "X-Demo-Key: $RELAY_DEMO_KEY" http://127.0.0.1:8080/metrics
```

Open Prometheus at `http://127.0.0.1:9090`. The Collector scrapes both services every
five seconds and exports the metrics to Prometheus, which retains at most seven
days / 1 GB. Inspect `up{job="relay-engine"}` and
`up{job="relay-control-plane"}` first: zero means the Collector cannot scrape that
service (address, authentication, or readiness of the HTTP listener). The Prometheus scrape uses `honor_labels: true` to preserve the original service
job and instance labels exported by the Collector.

Useful expressions:

```promql
relay_engine_active_requests / relay_engine_active_requests_limit
relay_engine_kv_pool_slots_used / relay_engine_kv_pool_slots
rate(relay_engine_rejections_total[5m])
rate(relay_control_plane_rejections_total[5m])
sum by (mode, phase, batching) (rate(relay_engine_step_batch_size_sum[5m]))
  / sum by (mode, phase, batching) (rate(relay_engine_step_batch_size_count[5m]))
histogram_quantile(0.95, sum by (le, mode, outcome) (rate(relay_engine_ttft_seconds_bucket[5m])))
```

Resource gauges are snapshots between synchronous model steps; they cannot expose
every brief transition. Step histograms and terminal counters retain short events
between scrapes. Tracing can remain off for timing comparisons. CUDA allocator
memory is included on CUDA devices; detailed GPU utilization and kernel/KV-copy
profiling are separate tools, as described in [METRICS.md](../METRICS.md).
