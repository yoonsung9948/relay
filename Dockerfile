# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/engine/.cache/huggingface

WORKDIR /app

# PyTorch's Linux runtime needs OpenMP; HTTPS is used for model downloads.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /tmp/requirements.txt

COPY pyproject.toml README.md ./
COPY engine/ ./engine/
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps . && python -m pip check

COPY docker/engine.yaml /app/config.yaml
RUN useradd --create-home --uid 10001 engine \
    && mkdir -p /home/engine/.cache/huggingface \
    && chown -R engine:engine /home/engine/.cache
USER engine

EXPOSE 8000
# Confirms HTTP startup, not generation-loop health. Allow time for first download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/openapi.json', timeout=3).close()"

ENTRYPOINT ["relay"]
CMD ["serve", "--path", "/app/config.yaml"]
