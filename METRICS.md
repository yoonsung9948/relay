# Generation measurements, version 1

Python owns generation measurements. Go forwards them through SSE; the website and loadgen never infer model tokens from text chunks.

## Timing boundaries

All internal timestamps use one process-local monotonic clock. Only durations leave Python.

| Field | Definition |
| --- | --- |
| `prefill_ms` | Wall time for the first token-producing model step, including sampling and host transfer |
| `decode_steps` | Number of subsequent token-producing steps observed by this request |
| `decode_batch_size_mean` / `decode_batch_size_max` | Mean / maximum batch participation on subsequent steps; null mean and zero maximum for no decode steps |
| `input_tokens` | Input IDs after chat formatting, before generation mutates the history |
| `output_tokens` | Generated IDs, including an emitted EOS |
| `ttft_ms` | Python handler entry to first generated token available on the CPU |
| `decode_duration_ms` | First-to-last generated token interval, available for at least two tokens |
| `decode_tokens_per_second` | `(output_tokens - 1) / decode_duration_seconds`, including scheduler waits |
| `total_latency_ms` | Handler entry to finalized output, immediately before success payload construction |
| `initial_queue_ms` | First enqueue to first scheduled execution; excludes later scheduler waits |

These are serving timings, not isolated GPU kernel timings. Total latency includes tokenization, scheduling, text finalization, and any local output backpressure. It excludes Go/edge/browser transit and final response serialization. Loading/provisioning happens separately before generation readiness.

Zero-token requests have null TTFT and decode measurements. One-token requests have null decode measurements. An unmeasurably short decode interval has a null rate. Real zeros remain zero. Completed results are frozen once; failures and cancellations do not emit successful final metrics.

## Wire contract

`POST /generate` with `stream: true` returns NDJSON from Python. Each text event is `{type:"delta", request_id, text}`. After all decoded text is flushed, one `{type:"done", request_id, finish_reason, generation, metrics}` event terminates the stream. `finish_reason` is `eos` or `length`. Errors use `{type:"error", request_id, error}` and never a subsequent `done`.

`metrics` has `version: 1`, `source: "engine"`, and the fields above. The nonstreaming JSON response carries the same object and retains `generated_tokens`. Consumers may tolerate missing/unsupported metrics while marking their absence explicitly. EOF without a terminal event is never completion.

`generation` describes the effective model, backend, device, dtype, runtime versions, output cap, effective sampling strategy, and temperature. Set `RELAY_REVISION` and `RELAY_MODEL_REVISION` in immutable deployments; absent revision information is reported as `unreported`, never guessed.

Go adds an optional typed metrics object to response structs and converts NDJSON into SSE. Nullable floats use pointers so zero is not lost. The public demo allows at most 64 generated tokens, accepts zero, rejects negative limits, and uses the single configured model. Temperature defaults to zero (greedy); positive values enable sampling.

## Website integration and budgets

The browser explicitly requests SSE using streamed fetch and requires `done` before displaying completion. It buffers incomplete SSE/UTF-8 frames, preserves partial text on failure/Stop, and populates all four cards only from final server metrics. Sample playback has no measurements.

Default generation budgets are Go 180s, edge proxy 185s, browser 190s. Custom Go budgets must preserve the outer margins in the proxy and browser. The proxy propagates cancellation, keeps the demo key server-side, forwards the body incrementally, and disables caching/transformation. Nonstreaming Go responses receive the same handler budget plus a write margin.

## Verification

Run engine unit tests and Go tests separately:

```sh
python -m pytest engine/tests/unit
cd control-plane
go test -race ./...
```

Tests cover exact formulas with an injected clock, zero/one-token outputs, final metrics in both response modes, null versus zero, effective limits, fragmented upstream events, and cancellation. Full checkpoint/GPU performance must be measured separately with the actual deployment.


## Operational metrics

The engine serves Prometheus text at `GET /metrics`, including before model readiness.
The control plane serves `GET /metrics` with its existing `X-Demo-Key` authentication.
Scrapes and health polling do not count as generation traffic. Metrics have bounded
labels (mode, batching, phase, outcome, status, rejection reason); they never label
by request ID, prompt, completion, or exception text. Counters reset on restart.

Engine gauges cover waiting/ready/active requests and configured limits, used/total
KV slots and tokens, preallocated pool bytes, and conservative dynamic reservations
and budget. Released slots return to the pool; allocated pool bytes stay constant.
Dynamic reservations include torch.cat headroom and are not actual tensor allocation.
CUDA gauges report PyTorch allocator allocated/reserved/peak bytes when on CUDA,
without synchronization; they exclude allocations made outside PyTorch.

Engine histograms cover model-step wall duration and physical batch sizes,
generation completion duration, TTFT, initial queue, and prefill. Physical-step
histograms count each model call once even when several requests share it.
Generation counters include success/failure/cancellation exactly once at model
termination. HTTP counters measure delivery separately, including stream failures
and disconnects after HTTP 200. Backend generation success does not imply delivery.
The generation-duration histogram ends at engine completion (before text finalization),
whereas the response's `total_latency_ms` ends at finalized text.

The control plane exports full HTTP lifetime histograms and outcome/status counters,
current HTTP requests, generation concurrency usage/limit, and rejection counters.
`X-Relay-Rejection` identifies engine or Go capacity rejections end to end; 503 without
an explicit capacity reason remains an availability failure in loadgen. `Retry-After`
is preserved in raw records. HTTP 429 is also counted as rejection for older servers.

Loadgen preserves all version-1 fields in raw records and reports prefill percentiles,
request batch mean/max/coverage, actual execution mode counts, and rejection reasons.
Its request-step-weighted mean counts shared steps per participant; use the engine
batch histogram sum/count for a physical-step-weighted mean. Older servers' missing
fields remain unavailable, never invented as zero.

See [the observability setup](observability/README.md) for collection and queries.
GPU utilization, temperature, cross-process device memory, and individual KV-copy
or kernel costs require a device exporter or profiler; allocator gauges and serving
wall timings do not measure them. No CUDA synchronization is added for telemetry.


## Temperature sampling

`temperature` is optional and defaults to `0.0` in Go and Python. Zero selects
argmax; positive values sample one token from `softmax(logits / temperature)`.
Both endpoints reject negative, non-finite, null, boolean, and string values.
Go preserves the supplied temperature while retaining the existing output cap.

Sampling options belong to each request, independently of cache mode. A shared
sampler handles greedy and sampled rows in the same decode batch. Probabilities
are computed in float32 on the model device, and only generated token IDs are
transferred to the host. Invalid probability distributions raise an error rather
than silently producing greedy output. The model-step duration includes sampling.

Both response modes report `generation.sampling` (`greedy` or `temperature`) and
`generation.temperature`. Request traces record `sampling.strategy` and
`sampling.temperature`; arbitrary temperatures are not Prometheus labels.
Loadgen saves these effective settings and flags differences in comparison reports.
Keep temperature at zero for cache/batching performance comparisons, since sampled
outputs may hit EOS at different lengths. There is no public seed option yet:
positive-temperature requests share the device RNG and output depends on scheduling.

Example, through the control plane:

```sh
curl --fail-with-body -N http://127.0.0.1:8080/generate \
  -H "X-Demo-Key: $RELAY_DEMO_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write two sentences about Seoul.","max_tokens":64,"stream":true,"temperature":0.7}'
```
