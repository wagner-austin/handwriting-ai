# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg62-turbo-dev \
    zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Copy lockfiles first for better layer caching
COPY pyproject.toml poetry.lock /app/

# Copy the rest of the project (app src, configs, docs as needed)
COPY src /app/src
COPY config /app/config
COPY README.md /app/README.md

# Install deps (no dev) into venv managed by poetry
RUN poetry config virtualenvs.create true \
 && poetry config virtualenvs.in-project true \
 && poetry install --only main --no-interaction --no-ansi

EXPOSE 8081

# Default command (the app module will be implemented later)
CMD ["/app/.venv/bin/uvicorn", "handwriting_ai.api.app:app", "--host", "0.0.0.0", "--port", "8081"]
