# One image, four entrypoints (train / serve / monitor / controller). The command is set
# per workload by compose/k8s — this image ships a neutral default, not a hardcoded server.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep the virtualenv outside /app so a state volume mounted at /app never shadows it.
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# make drives the documented command API; curl backs the compose healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends make curl \
    && rm -rf /var/lib/apt/lists/*

# uv for reproducible, locked installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
# Stage 1: install only third-party deps (cached unless the lockfile changes).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Stage 2: add the source + README (hatchling needs it) and install the project itself.
COPY src ./src
COPY configs ./configs
COPY Makefile README.md ./
RUN uv sync --frozen --no-dev

# Pristine copy for k8s: a PVC mounted at /app shadows the baked code (k8s does not
# auto-populate volumes like Docker does), so the seed Job's initContainer copies this
# into the empty PVC once. Compose doesn't use it (Docker seeds named volumes from /app).
RUN mkdir -p /opt/app && cp -a /app/. /opt/app/

# Drift/realized-F1 (controller :9100) and serving counters (:8000) are the two scrape targets.
EXPOSE 8000 9100

# Neutral default: print the targets. Real workloads override `command:`/`args:`:
#   serve      -> make up HOST=0.0.0.0
#   train/seed -> make train
#   controller -> make loop
#   monitor    -> make monitor
CMD ["make", "help"]
