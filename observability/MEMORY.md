# Lightweight per-trial memory measurements

Memory recording is opt-in, independent of OTel, and needs no new dependencies,
service, HTTP endpoint or loadgen changes. The engine writes a JSONL archive.
Loadgen's bundled offline analyzer joins it to existing `requests.jsonl` by engine request
ID, then saves a memory summary for each trial. Neither side imports the other.
Existing benchmarks cannot acquire memory measurements retroactively.

## Enable on the GPU worker

Set this in the **Python engine process environment**, before starting it:

```sh
export RELAY_MEMORY_PATH=/tmp/relay-memory.jsonl
```

An unset variable disables recording completely. The parent directory is created
if necessary and must be writable by the engine user. CPU/MPS runs explicitly
record `unsupported_device` with null measurements, never zero GPU usage.
Initialization, probe and disk failures are logged; memory telemetry does not
stop generation. Check for `memory_recording_enabled` in the worker logs and
inspect analyzer coverage after running.

For future RunPod provisioning, add to the existing Go config's RunPod
`engine_env` map:

```yaml
RELAY_MEMORY_PATH: "/tmp/relay-memory.jsonl"
```

This requires an engine image containing the new recorder. After that initial
image update, measurements work for all request execution modes without another
image change. An existing worker needs the variable in its engine process and
that process restarted; changing local Go environment variables does not update
a running remote process. There is no need to reprovision the GPU merely to
change a process environment variable when you can restart that process directly.

## Run the existing benchmarks

Use loadgen's normal run/sweep commands. No memory option, measurement request,
benchmark run ID or platform dependency is added to loadgen. Its existing Relay
request records already save the engine UUID needed for the offline join.
Tracing can stay disabled on both services.

Compare matching model, device, precision, token lengths, concurrency, warmup and
instrumentation settings. To isolate batching, use pool mode with batching off
versus the same pool configuration with batching on. Comparing dynamic-cache
execution to pooled batched execution changes two things; label that experiment
accordingly. Start with the existing short concurrency 1/2/4 sweep.

Run only the benchmark against this engine while collecting clean measurements.
A busy period containing warmup or unrelated requests is detected by its IDs
and excluded, rather than attributing shared memory to a single trial.

## Save and compare

Wait for requests to drain. Each finished busy period is appended and flushed
immediately; normal engine shutdown is not needed to get completed records.
Download/copy `/tmp/relay-memory.jsonl` from the GPU worker **before terminating
the pod**. This archive is on the worker, not in the local Collector or loadgen
results. `/tmp` is not durable across pod replacement. Use a mounted persistent
path instead if desired. Archive files append across process restarts, with
distinct engine instance and window IDs. They have no automatic rotation.

Store the downloaded archive locally, then run from the loadgen checkout:

```sh
.venv/bin/loadgen benchmark analyze \
  --run results/sweep-BEFORE --run results/sweep-AFTER \
  --memory /path/to/before.jsonl --memory /path/to/after.jsonl \
  --out results/batching-memory-comparison
```

Sweep paths expand using their saved manifests; individual run directories also
work. The analyzer is included with loadgen and needs no running service, engine
imports, GPU or separate installation. Optionally supply `--traces PATH` for the
existing trace analysis. For benchmark sessions, `loadgen benchmark collect`
imports the archive and builds all per-trial memory reports automatically.

Open `batching-memory-comparison/report.html`. It compares execution mode,
concurrency, successful token throughput, completion p50 and memory. Each linked
trial directory contains:

- `memory.json`: status, coverage, raw byte measurements, pool/model metadata and
  measurement definitions.
- `memory-windows.jsonl`: matching engine records, including excluded records so
  incomplete/shared measurements can be inspected.
- `analysis.json` and `report.html`: client results and memory together, with
  traces when requested.

The top-level `comparison.json` preserves the combined reports. A single trial
writes its files directly under `--out`. Source benchmark files remain unchanged.
Keep the original engine archive and these reports alongside your raw results.

## What is measured

The engine detects a busy period when its first accepted request arrives. It
synchronizes CUDA, reads idle allocated/reserved memory, then resets allocator
peak counters. Additional concurrent requests share that window. After every
request has completed/cancelled, released its request cache or pool slot, and
left the engine, it synchronizes again and reads final and peak counters.
No CUDA polling, per-token instrumentation, `empty_cache()` or background
exporter is introduced. At concurrency one, each request may be a separate busy
period; at higher concurrency, many requests can share a period. The analyzer
takes the **maximum** of their peaks, never a sum or average of shared peaks.

Reported metrics:

- Idle allocated and reserved bytes before the first usable measured window.
- Peak allocated and reserved bytes across the trial's usable windows.
- Peak allocated bytes above that initial idle baseline.
- Allocated/reserved bytes after the last usable window drains.
- Change in allocated memory after drain, relative to initial idle.
- Maximum engine requests in flight, plus preallocated KV pool bytes, slots,
  sequence capacity, model/device/precision, scheduler limits and tracing status.

These are **PyTorch allocator measurements for the engine process and its runner
device**, not total GPU memory, CPU RSS, or a KV-only allocation estimate. Model
weights and preallocated pools remain in the idle baseline. Reserved memory
includes reusable blocks and can remain high after drain. Peak reset does not
free reserved blocks; use equivalent warmup and a fresh engine process per
variant when comparing reserved memory. Keep the GPU provisioned while doing so.

Synchronization/reset and one small file append at idle boundaries have overhead;
enable them equally for both variants. They cannot identify GPU kernels. This
recorder owns the process/device peak counters while enabled; do not reset those
counters from another profiler simultaneously.
If the Prometheus endpoint is also used, its `relay_engine_cuda_peak_allocated_bytes`
gauge reflects the allocator peak since the latest busy-window reset while this
recorder is enabled. Use the archive/analyzer for trial peaks; scrapes can miss a
completed window between resets. Current allocated/reserved gauges are unaffected.

Missing records, missing engine IDs, unsupported devices, unfinished work,
duplicate ID mappings, mixed instances and shared/truncated windows are reported.
Partial peaks are lower bounds; missing values are never filled with zero.
Request IDs per window are capped at 4096 and overflow is flagged. Hard process
death/OOM termination can lose an active window; missing coverage exposes that
loss. Completed records are flushed through normal file I/O, not fsync durability
against host failure.

For an optional whole-device timeline on RunPod, run this separately:

```sh
nvidia-smi -i 0 \
  --query-gpu=timestamp,memory.used,utilization.gpu \
  --format=csv --loop-ms=200 > /tmp/gpu-memory.csv
```

That sampled CSV is supplemental and is not consumed by this analyzer. It can
include other processes and miss brief peaks; it does not replace allocator
peak counters. Stop the sampler and download its file before terminating the pod.
