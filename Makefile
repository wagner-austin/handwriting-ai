.PHONY: help install install-dev lock serve test lint check start stop clean

help:
	@echo "Targets:"
	@echo "  make install     - Install dependencies with Poetry"
	@echo "  make install-dev - Install with dev dependencies"
	@echo "  make serve       - Run the API locally with uvicorn"
	@echo "  make test        - Run pytest with coverage"
	@echo "  make lint        - Ruff fix+format, then mypy (strict)"
	@echo "  make check       - Lint + Test"
	@echo "  make start       - Docker compose up (build)"
	@echo "  make stop        - Docker compose down"
	@echo "  make clean       - Prune and rebuild compose stack"
	@echo "  (guards run inside make lint)"

install:
	poetry lock
	poetry install

install-dev:
	poetry lock
	poetry install --with dev

serve: install-dev
	# Note: this expects an app module when implemented.
	poetry run uvicorn handwriting_ai.api.app:app --host 0.0.0.0 --port $${APP__PORT:-8081}

test: install-dev
	poetry run pytest --cov=src --cov-report=term-missing

lint: install-dev
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy
	poetry run python scripts/guard_checks.py

check: lint | test


start:
	docker compose up -d --build

stop:
	docker compose down

clean:
	docker compose down -v --remove-orphans || true
	docker compose up -d --build
