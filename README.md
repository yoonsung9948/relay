# relay

## CI

[CI](.github/workflows/ci.yml) runs on pull requests and pushes to `main` with
two CPU-only jobs:

- **Go:** formatting, `go vet`, and tests with race detection.
- **Python:** Ruff and the lifecycle/unit tests, using fakes and tiny local models.

Jobs have timeouts, read-only repository permissions, and dependency caches.
New commits cancel older runs for the same PR or branch. Hugging Face offline
mode prevents checkpoint downloads during tests. GPU/checkpoint integration tests
in `engine/tests/integration/` are excluded.

### Run locally

Use the Go version declared in `control-plane/go.mod` and Python 3.12. From
`control-plane/`:

```sh
gofmt -l . # must print no filenames; use gofmt -w on any listed files
go vet ./...
go test -race -timeout 5m ./...
```

From the repository root, in a virtual environment:

```sh
# Linux CPU-only Torch; on macOS, install torch==2.14.0 from PyPI instead.
python -m pip install 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r ci/requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -m ruff check engine tests
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests engine/tests/unit -q
```

CI reuses the Docker runtime pins and pins its test tools in
`ci/requirements.txt`; transitive Python dependencies are not fully locked.
Torch is installed from its CPU index first to avoid downloading CUDA packages.
When changing its version, update both `docker/requirements.txt` and the CPU
installation command in the workflow and this guide.

## Docker

See [the Docker guide](docker/README.md) for local and NVIDIA GPU startup.

## Request metrics

See [the measurement contract](METRICS.md) for timing boundaries, the final stream event, effective demo limits, and verification.


## Diagnostic tracing

See [observability/README.md](observability/README.md) for optional Go/Python
tracing, the local Collector/Jaeger stack, and the standalone offline benchmark
analyzer. Tracing is disabled by default and requires no loadgen dependency.


## Selecting service configurations

Go selects its local YAML explicitly:

```sh
cd control-plane
go run ./cmd/server --config examples/config.yaml
```

Without `--config`, Go still loads `examples/config.yaml`. Unknown flags and
positional arguments fail before configuration loading or provisioning. Go's
`--config` does not select the Python worker configuration.

Python selects its config in this order: explicit `--path`, then
`RELAY_CONFIG_PATH`, then `docker/engine.yaml`. Docker sets the environment default
to `/app/config.yaml` and starts `relay serve` without a hardcoded path. To enable
the bundled pool configuration, set this in the existing RunPod environment map:

```yaml
engine_env:
  RELAY_CONFIG_PATH: "/app/docker/pool.yaml"
```

The normal image build includes this behavior; no special benchmark image command
is needed. Rebuild/publish the image once to introduce the selector, then point Go
at that image tag. Environment/config edits apply on worker startup, not to an
already-running Python process. They do not require rebuilding the image when the
chosen config is already bundled.

Python `/health/ready` includes sanitized live capabilities while preserving
ordinary readiness semantics: a worker without a pool can still be healthy.
Authenticated Go `GET /engine/info` fetches current worker readiness and reports
its admission limit, without caching or generating tokens. It returns 503 if the
worker is unavailable and omits configuration secrets. For example:

```sh
curl -fsS http://127.0.0.1:8080/engine/info -H "X-Demo-Key: $RELAY_DEMO_KEY"
```

The loadgen command `loadgen benchmark batching` checks those capabilities before
its smoke test. A wrong active config now reports its path and pool size before
any benchmark generation requests are sent. This requires updated Go and Python
code, not just a modified config file.

## Continuous decode batching

Execution is selected per request with `kv_cache_mode: none|dynamic|pool` and
`decode_batching: true|false`. Batching requires `pool`. The legacy
`use_kv_cache` boolean maps to none/dynamic; conflicting fields are rejected.
Pool requests prefill separately, then join variable-length single-token decode
batches. Prefill, decode, and baseline work receive rotating scheduling turns.

The default deployment allocates no pool. To enable four Qwen3-8B FP16 slots of
2048 tokens, start with `relay serve --path docker/pool.yaml`. On a GPU, the pool
reserves 1.125 GiB; the example separately allows up to 1.125 GiB of conservative
dynamic-cache reservations. These budgets exclude model weights and temporary
attention/gather/logit tensors. Slots are reusable on completion, not returned to
the CUDA driver. Do not configure these budgets to consume all free VRAM.

Scheduler settings are `max_active_requests`, `max_decode_batch_size`, and
`max_pending_requests`. Requests exceeding their context or per-request KV limit
receive 422. A full waiting queue receives 503 with Retry-After. Go preserves
these upstream status codes. Go's public concurrency/rate limits remain separate.

Use the separate loadgen project for performance comparisons.
Record GPU/model/dtype/config alongside results; disable tracing for primary
benchmarks. For batching comparisons keep the same pool enabled in both modes. To compare
pooled versus non-pooled memory, use separate engine starts: idle pool storage
still consumes VRAM. Application throughput does not establish numerical correctness or measure GPU kernel performance.


Benchmark orchestration now lives in the separate loadgen project. Use
`loadgen benchmark batching` to run comparisons and `loadgen benchmark collect`
to import telemetry, including sessions created by the former driver. See
loadgen's `BENCHMARK_BATCHING.md` for the workflow. Offline trace and memory analysis is bundled in loadgen. Collection runs it
automatically; no separate analyzer installation or executable path is required.

### Reproducible batching experiments

The experiment planner, runner and offline analyzer live in the sibling loadgen repository. See its `EXPERIMENTS.md` for workload preparation, paired repeats, open-loop demand, budgets and instrumentation controls. This engine only exposes deployment identity and serving capabilities. Build an identified image with `python3 tools/build_image.py --tag YOUR_IMAGE_TAG`; use `--print-only` to inspect the build command without building. `model.revision` (or `RELAY_MODEL_REVISION`) pins weights and tokenizer to the same resolved checkpoint commit.
