# Handwriting AI (Digits Service) – Foundation

This repo hosts a standalone HTTP service for MNIST digit inference that the Discord bot calls. This commit sets up quality gates and ops scaffolding only (no runtime app code yet).

Quality Gates
- Make targets:
  - `make install` / `make install-dev`
  - `make serve` – runs uvicorn (app module to be added next)
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

Next Steps (when implementing app code)
- Add `src/handwriting_ai/api/app.py` (FastAPI app) with `/healthz`, `/readyz`, and `/v1/predict`.
- Implement strict, typed preprocessing and inference as planned (no fallbacks, fail-closed).
- Wire settings loader to prefer config file for tunables and only read secrets from `.env`.

