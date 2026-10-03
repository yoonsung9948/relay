# Run the engine in Docker

Run these commands from the repository root. Docker Engine/Desktop with the
Compose plugin is required. The image runs the custom Qwen3 engine only; it does
not build or start the Go control plane.

## Local smoke test

```sh
docker compose up --build
```

Wait for `startup_ready` and `engine_loop_started`. First startup downloads the
checkpoint; later containers reuse the `hf-cache` volume. On macOS the Linux
container uses CPU, not Apple's MPS accelerator. Use the native Python setup for
MPS performance testing. Allocate sufficient Docker memory: the current loader
briefly holds both the reference model and custom model during weight loading.

In another terminal:

```sh
curl --fail-with-body --max-time 180 -sS http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is the capital of South Korea?","max_tokens":16}'
```

## NVIDIA GPU server

Use a Linux x86-64 host with an NVIDIA GPU, a compatible NVIDIA driver, and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
configured for Docker. PyTorch is installed from PyPI; its Linux x86-64 wheel
provides its CUDA runtime dependencies. Host driver compatibility still matters.

```sh
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Check that startup logs report `device=cuda`. To check container GPU visibility:

```sh
docker compose -f compose.yaml -f compose.gpu.yaml run --rm --no-deps \
  --entrypoint python engine -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The override reserves one GPU using Docker's
[GPU device reservations](https://docs.docker.com/compose/how-tos/gpu-support/).

## Configuration and operations

- Edit `docker/engine.yaml`, then recreate the service with
  `docker compose up --force-recreate`. No image rebuild is needed for config edits.
- The engine listens on `0.0.0.0` inside the container. The published port is bound
  to the host's loopback interface; change that mapping when exposing a remote
  worker to your control plane.
- Keep one server process per container so each container loads one model.
- Logs: `docker compose logs -f engine`.
- Stop: `docker compose down`. This preserves downloaded weights; `down -v`
  also deletes the named cache volume.
- For GPU commands, consistently include both Compose files.
- The HTTP healthcheck probes `/openapi.json` after startup. It does not prove
  generation works or detect every background engine-loop failure.
- Runtime versions in `docker/requirements.txt` match the development environment
  when this setup was created. Transitive dependencies are not fully locked;
  use immutable image digests for repeatable benchmark deployments.

## Standalone image

```sh
docker build -t relay-engine:local .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v relay-hf-cache:/home/engine/.cache/huggingface relay-engine:local
```

The image includes the default Docker config. Mount a replacement at
`/app/config.yaml` to override it. Add `--gpus all` for NVIDIA execution on a
configured Linux host. For a different config path, append
`serve --path /path/in/container.yaml` after the image name.

## Validation status

The config and package entry point were checked locally. Docker was unavailable
in the authoring environment, so image build, container startup, and GPU execution
still need verification on a Docker host.
