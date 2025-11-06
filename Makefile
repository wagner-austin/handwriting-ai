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
	@echo "  make train       - Train MNIST model (pretty color logs)"
	@echo "  make train-long  - Long run (JSON logs; uses config/trainer.toml)"
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
EPOCHS ?= 1
BATCH_SIZE ?= 512
LR ?= 0.0015
SEED ?= 42
DEVICE ?= cpu
# New training knobs (optional)
OPTIM ?= adamw
SCHED ?= cosine
WD ?= 0.01
STEP_SIZE ?= 10
GAMMA ?= 0.5
MIN_LR ?= 1e-5
PATIENCE ?= 0
MIN_DELTA ?= 0.0005
THREADS ?= 16
AUGMENT ?= 1
AUG_ROTATE ?= 12
AUG_TRANSLATE ?= 0.1
NOISE_PROB ?= 0
NOISE_SALT_FRAC ?= 0.5
DOTS_PROB ?= 0
DOTS_COUNT ?= 0
DOTS_SIZE ?= 1
BLUR_SIGMA ?= 0
MORPH ?= none
MORPH_KERNEL ?= 1

# Train a service-compatible MNIST model (PowerShell-friendly)
train: install-dev
	poetry run python scripts/train_mnist_resnet18.py \
		--data-root "$(DATA_ROOT)" \
		--out-dir "$(OUT_DIR)" \
		--model-id "$(MODEL_ID)" \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--lr $(LR) \
		--weight-decay $(WD) \
		--seed $(SEED) \
		--device $(DEVICE) \
		--optim "$(OPTIM)" \
		--scheduler "$(SCHED)" \
		--step-size $(STEP_SIZE) \
		--gamma $(GAMMA) \
		--min-lr $(MIN_LR) \
		--patience $(PATIENCE) \
		--min-delta $(MIN_DELTA) \
		--threads $(THREADS) \
		$(if $(filter $(AUGMENT),1 true yes),--augment,) \
		--aug-rotate $(AUG_ROTATE) \
		--aug-translate $(AUG_TRANSLATE) \
		--noise-prob $(NOISE_PROB) \
		--noise-salt-vs-pepper $(NOISE_SALT_FRAC) \
		--dots-prob $(DOTS_PROB) \
		--dots-count $(DOTS_COUNT) \
		--dots-size $(DOTS_SIZE) \
		--blur-sigma $(BLUR_SIGMA) \
		--morph $(MORPH) \
		--morph-kernel $(MORPH_KERNEL) \

		--log-style pretty

train-long: install-dev
	poetry run python scripts/train_mnist_resnet18.py --config ./config/trainer.toml --log-style pretty

# Copy a trained artifact from artifacts/ to seed/ so Dockerfile can bake it into /seed
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

