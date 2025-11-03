# Handwriting AI (Digits Service) – Foundation

This repo hosts a standalone HTTP service for MNIST digit inference that the Discord bot calls. The service is implemented with FastAPI and includes strict typing, standardized error handling, and robust preprocessing/inference.

Quality Gates
- Make targets:
  - `make install` / `make install-dev`
  - `make serve` – runs uvicorn for the FastAPI app
  - `make lint` – ruff fix + format + mypy (strict)
  - `make test` – pytest with coverage (`--cov=src`)
  - `make check` – lint + test
  - `make guards` – fail build if `typing.Any`, `typing.cast`, or `type: ignore` are present
  - Docker: `make start` / `make stop` / `make clean`
- Tooling config lives in `pyproject.toml`, `pytest.ini`.

Configuration Policy
- Tunables belong in `config/handai.toml` and/or a mounted config path.
- Only secrets go in `.env` (e.g., `SECURITY__API_KEY`).
- Environment variables use nested keys (e.g., `APP__PORT`, `DIGITS__MODEL_DIR`) following the documented plan.

Containerization
- Multi-stage Dockerfile installs dependencies via Poetry and runs `uvicorn`.
- `docker-compose.yml` exposes port `8081` and mounts `config/`, `artifacts/`, and `logs/`.

API Summary
- `GET /healthz` – liveness
- `GET /readyz` – readiness (refuses incompatible artifacts)
- `GET /version` – version/build info
- `GET /v1/models/active` – active manifest summary
- `POST /v1/read` (alias `/v1/predict`) – image → digit

Security
- Optional API key via header `X-Api-Key`. When configured, unauthorized requests return a standardized error body: `{code, message, request_id}` with `code="unauthorized"`.

Implementation Notes
- Preprocessing: grayscale → Otsu → largest component → deskew (angle+confidence gated) → center → 28×28 normalize.
- Inference: 1‑channel CIFAR‑style ResNet‑18, temperature scaling, small-translation TTA (configurable).
- Error responses: standardized `{code,message,request_id}` schema for all errors.
