.PHONY: help install install-dev lock serve test lint check start stop clean train seed-model

# Port configuration: use APP__PORT if set, otherwise default to 8081
PORT := $(if $(APP__PORT),$(APP__PORT),8081)

help:
	@echo "Targets:"
	@echo "  make install     - Install dependencies with Poetry"
	@echo "  make install-dev - Install with dev dependencies"
	@echo "  make serve       - Run the API locally with uvicorn"
	@echo "  make test        - Run pytest with coverage"
	@echo "  make lint        - Ruff fix+format, then mypy (strict) + YAML lint"
	@echo "  make check       - Lint + Test"
	@echo "  make train       - Train MNIST model (PowerShell-friendly)"
	@echo "  make seed-model  - Copy trained model from artifacts/ to seed/ for Docker image seeding"
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
# Run API locally with uvicorn (expects handwriting_ai.api.app:app)
	poetry run uvicorn handwriting_ai.api.app:app --host 0.0.0.0 --port $(PORT)

test: install-dev
	poetry run pytest --cov=src --cov-report=term-missing

lint: install-dev
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy
	poetry run python scripts/guard_checks.py
	poetry run yamllint -c .yamllint .

check: lint | test


# Training configuration (override via: make train VAR=value)
MODEL_ID ?= mnist_resnet18_v1
DATA_ROOT ?= ./data/mnist
OUT_DIR ?= ./artifacts/digits/models
EPOCHS ?= 4
BATCH_SIZE ?= 128
LR ?= 0.001
SEED ?= 42
DEVICE ?= cpu

# Train a service-compatible MNIST model (PowerShell-friendly)
train: install-dev
	poetry run python scripts/train_mnist_resnet18.py \
		--data-root "$(DATA_ROOT)" \
		--out-dir "$(OUT_DIR)" \
		--model-id "$(MODEL_ID)" \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--lr $(LR) \
		--seed $(SEED) \
		--device $(DEVICE)

# Copy a trained artifact from artifacts/ to seed/ so Dockerfile can bake it into /seed.
# Usage (PowerShell): make seed-model MODEL_ID=mnist_resnet18_v1
SEED_SRC ?= ./artifacts/digits/models
SEED_DST ?= ./seed/digits/models

seed-model:
	poetry run python scripts/seed_model.py --model-id "$(MODEL_ID)" --from-dir "$(SEED_SRC)" --to-dir "$(SEED_DST)"


start:
	docker compose up -d --build

stop:
	docker compose down

clean:
	docker compose down -v --remove-orphans || true
	docker compose up -d --build
