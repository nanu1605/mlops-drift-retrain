# Serving image — DELIVERABLE, NOT BUILT/DEPLOYED in this environment (pure-local mode,
# spec §0.3). Documents how serving would be containerized for a kind/K8s deployment.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# uv for reproducible, locked installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY configs ./configs

EXPOSE 8000
# @champion is resolved from the MLflow registry the container is configured to reach
# (MLFLOW_TRACKING_URI). In local mode this is the sqlite file + mlartifacts on the host.
CMD ["uv", "run", "uvicorn", "mlops_drift.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
