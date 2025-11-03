# Handwriting AI - Digits Service Plan (MNIST)

Status: planning document (no code in this commit)
Audience: bot + service maintainers
Goal: production-ready, typed, modular service for single-digit OCR via an HTTP API that the Discord `/read` command calls. Training and evaluation are optional Phase 2.

**Decisions**
- Use a standalone service in `../handwriting-ai/` and keep DiscordBot thin (HTTP client only).
- Use Torch CPU for MVP with a ResNet-18 (1-channel) baseline; ONNX export is an optional stretch goal.
- Align config style with Model-Trainer (nested envs), but do not couple repos.

**Principles**
- Strict typing end-to-end: mypy --strict, no Any, no typing.cast; Protocols, TypedDicts, frozen dataclasses, Pydantic v2 models.
- DRY, modular, standardized: shared patterns for config, logging, artifacts, and (later) jobs/events.
- Reliability first: structured logs with request_id, health/ready probes, explicit timeouts.
- Anti-drift: manifest-driven artifacts, stable preprocess signature hashing, narrow HTTP contract.

**Reuse From Existing Repos**
- DiscordBot (`../DiscordBot`)
  - Request-scoped logging, friendly error taxonomy, rate-limiting patterns.
  - Client pattern using httpx with timeouts and correlation headers.
- Model-Trainer (`../Model-Trainer`)
  - Settings pattern (pydantic-settings) with nested env keys (APP__, RQ__), TOML override hook.
  - Artifacts layout principles, logs.jsonl (for Phase 2 training).

**High-Level Architecture**
- Components
  - API (FastAPI): inference endpoint `/v1/read` (alias `/v1/predict` optional), model info `/v1/models/active`, health `/healthz`, ready `/readyz`, version `/version`.
  - Inference Engine (CPU/GPU): ensemble-first loader (multiple models), calibrated probability averaging, advanced TTA by default; bounded thread pool for request concurrency.
  - Test-Time Augmentation (optional): when enabled via `DIGITS__TTA=true`, the engine performs light-weight spatial TTA (small pixel shifts) and averages probabilities for improved robustness.
  - Optional RQ Worker(s) (Phase 2): background training/eval on a `digits` queue.
  - Storage: artifacts dir for models + manifests; logs dir for structured logs.
  - Redis (Phase 2): for jobs and events channel `digits:events`.
- Service boundaries
  - DiscordBot <-> handwriting-ai: HTTP POST multipart; no shared runtime deps.
  - Optional: Redis events for training notifications in Phase 2.

**API Specification (Competition Defaults)**
- POST `/v1/read`
  - Request: multipart/form-data
    - `file`: PNG or JPEG image
    - `invert`: optional bool
    - `center`: optional bool
    - `visualize`: optional bool
  - Headers:
    - `X-Request-ID`: correlation id from DiscordBot
    - Optional `X-Api-Key`: shared secret if publicly exposed
  - Response 200 JSON:
    - `digit: int`
    - `confidence: float`
    - `probs: [float; 10]`
    - `model_id: str`
    - `visual_png_b64: str | null` (if `visualize=true`)
     - `uncertain: bool` (true if `confidence < DIGITS__UNCERTAIN_THRESHOLD`)
  - Errors:
    - 400: invalid request
    - 413: payload or dimension too large
    - 415: unsupported media type
    - 422: malformed form
    - 500: system error
- GET `/healthz` -> `{ "status": "ok" }` (process up)
- GET `/readyz` -> `{ "status": "ready" }` when ensemble is loaded; include `{ "model_loaded": bool, "model_id": string | null, "manifest_schema_version": string | null, "build": string | null, "ensemble_size": int | null, "model_ids": list[str] | null }` otherwise.
- GET `/version` -> `{ "service": "handwriting-ai", "version": string, "build": string, "commit": string }`

- GET `/v1/models/active`
  - Returns information about the currently loaded model and readiness.
  - Response 200 JSON when model is loaded:
    - `model_loaded: true`
    - `model_id: string`
    - `arch: string` (e.g., `resnet18`)
    - `n_classes: int` (e.g., 10)
    - `version: string` (model artifact version)
    - `created_at: string` (ISO 8601)
    - `schema_version: string` (manifest schema)
    - `val_acc: float` (validation accuracy from training manifest)
    - `temperature: float` (applied to logits before softmax)
  - Response 200 JSON when not loaded:
    - `model_loaded: false`
    - `model_id: null`

Notes:
- TTA: Enable by setting environment variable `DIGITS__TTA=true` (or `1`). This performs a small set of 1-pixel shifts around the input and averages probabilities before returning results. Disabled by default for latency predictability.

Error response schema (stable):
- Body: `{ "code": string, "message": string, "request_id": string }`
- Codes and statuses:
  - `invalid_image` -> 400
  - `unsupported_media_type` -> 415
  - `bad_dimensions` -> 400
  - `too_large` -> 413
  - `preprocessing_failed` -> 400
  - `timeout` -> 408 or 504 (mapped to 500 if internal timeout)
  - `internal_error` -> 500
- Examples:
  - 400 invalid image
    ```json
    {"code": "invalid_image", "message": "Failed to decode image.", "request_id": "abc-123"}
    ```
  - 413 too large
    ```json
    {"code": "too_large", "message": "File exceeds 2 MB limit.", "request_id": "abc-123"}
    ```

**Preprocessing (train/infer parity, no fallbacks)**
- Load via PIL; strip EXIF orientation; grayscale; composite alpha onto white.
- Auto-invert if background is dark (heuristic) unless `invert` is explicitly provided.
- Threshold (Otsu) -> largest connected component -> bounding box crop.
- Deskew: compute principal-axis orientation; apply rotation only if magnitude/angle confidence exceeds a small threshold and cap rotation to +/- 10 deg.
- Center by center-of-mass on a square canvas with margin; aspect-preserving.
- Resize to 28x28; normalize with MNIST mean/std (0.1307, 0.3081).
- No fallback branches: if segmentation/normalization fails (e.g., no component found, deskew invalid, angle unreliable), return 400 with a clear error code and message (`preprocessing_failed`).

**Inference Engine (Competition-First)**
- Ensemble-first: load multiple models and average calibrated probabilities across them; `model_id` in responses becomes `ensemble_<N>_models`.
- Backbones: support ResNet-18 (CIFAR-style, 1-channel) and allow additional SOTA variants (e.g., WideResNet-28-10, DenseNet) adapted for 1-channel.
- Calibration: temperature scaling per model (logits / temperature → softmax). Optional ensemble override.
- Advanced TTA: enabled by default (strong profile: multi-shift + small-rotation + light-scale). Profiles configurable.
- Concurrency model:
  - Use a small, bounded `ThreadPoolExecutor` (size = `APP__THREADS` or CPU core count, see policy below) rather than `asyncio.to_thread`.
  - Rationale: bounded concurrency provides backpressure and predictable latency; one place to instrument queue depth and timings; avoids unbounded task spawning; enables better CPU utilization and easier capping with container limits.
- Threading policy and oversubscription avoidance (simplified):
  - Default: set `torch.set_num_threads(1)` and use a bounded `ThreadPoolExecutor`.
  - If `APP__THREADS=0` (auto): executor size `P = min(8, CPU_cores)`.
  - If `APP__THREADS>0`: executor size `P = APP__THREADS`.
  - Uvicorn workers: recommend `--workers 1` (max 2) to avoid workers x pool fan-out.
  - If enabling TTA, consider reducing `P` by 1–2 to maintain headroom.

**Configuration**
- Base (nested env style):
  - `APP__DATA_ROOT=/data`
  - `APP__ARTIFACTS_ROOT=/data/artifacts`
  - `APP__LOGS_ROOT=/data/logs`
  - `APP__THREADS=0` (0 = auto)
  - `APP__PORT=8081`
- Inference:
  - `DIGITS__MODEL_DIR=/data/digits/models`
  - `DIGITS__ACTIVE_MODEL=resnet18_h1,wideresnet_h1,...` (comma-separated, required; each is a subfolder under MODEL_DIR)
  - `DIGITS__TTA_PROFILE=strong` (`strong`|`none`)
  - `DIGITS__ENSEMBLE_TEMPERATURE_OVERRIDE=` (optional float)
  - `DIGITS__UNCERTAIN_THRESHOLD=0.70`
  - `DIGITS__MAX_IMAGE_MB=2`
  - `DIGITS__MAX_IMAGE_SIDE_PX=1024`
  - `DIGITS__PREDICT_TIMEOUT_SECONDS=5`
  - `DIGITS__VISUALIZE_MAX_KB=16`
  - `DEVICE=cpu|cuda` (optional; default cpu)
- Security (optional):
  - `SECURITY__API_KEY=` (empty disables check)
- RQ/Redis (Phase 2):
  - `REDIS_URL=redis://redis:6379/0`
  - `RQ__QUEUE_NAME=digits`
  - `RQ__JOB_TIMEOUT_SEC=900`, `RQ__RESULT_TTL_SEC=86400`, `RQ__FAILURE_TTL_SEC=604800`
  - `RQ__RETRY_MAX=2`, `RQ__RETRY_INTERVALS_SEC=60,300`
  - `DIGITS__EVENTS_CHANNEL=digits:events`, `DIGITS__RESULT_KEY_PREFIX=digits:result:`

**Typing & Contracts**
- Enforce `mypy --strict`, ban `typing.cast`; rely on Protocols/TypedDicts.
- Pydantic v2 for API I/O models; frozen dataclasses for internal configs/results.
- Stable preprocess signature hashing stored in manifest; inference validates compatibility.

**Storage & Manifests (v1.1 required)**
- Layout (under `APP__ARTIFACTS_ROOT`):
  - `digits/<model_id>/model.pt` (primary; ONNX optional later)
  - `digits/<model_id>/manifest.json`
  - `digits/<model_id>/metrics.json`
- Manifest fields (v1.1, required):
  - Core: `schema_version`, `model_id`, `arch`, `n_classes`, `version`, `created_at`, `preprocess_hash`, `val_acc`, `temperature`
  - Training provenance: `training_recipe`, `training_seed`, `training_epochs`, `test_acc`, `recipe_hash`, `parent_model_id` (optional)
- Service refuses to load manifests not matching v1.1 or with incompatible `preprocess_hash`.
- Active model selection: `DIGITS__ACTIVE_MODEL` is a comma-separated list; service loads all valid entries.
  - Hot-reload note: safe swap on change to the active set (optional future enhancement).
- Retention policy:
  - Keep N=3 recent models and archives; prune older artifacts/logs on deploy or via a scheduled cleanup job.
  - Preserve models referenced by a named tag or in active rotation.

**Observability & Ops**
- Logs: JSON with `request_id`; include predict latency ms, image size, confidence, model_id, preprocess branch flags.
- Metrics: request counts, latencies, error rates; model load time.
- Limits (explicit):
  - Accept only `image/png` and `image/jpeg` content types.
  - Reject files larger than `DIGITS__MAX_IMAGE_MB`.
  - After EXIF orientation, reject if either dimension exceeds `DIGITS__MAX_IMAGE_SIDE_PX` (prevents CPU/memory blowups).
  - Guard against PIL `DecompressionBombError`; map to HTTP 413 and log with a specific error code.
  - Hard per-request timeout via `DIGITS__PREDICT_TIMEOUT_SECONDS`.
  - Bounded `ThreadPoolExecutor` limits concurrent inference; overflow requests queue, and the global request timeout governs overall behavior.
  - Server-level upload limit: enforce request body limits at the edge (e.g., reverse proxy `client_max_body_size`) to drop oversize uploads before application code executes.
  - Strict multipart parser limits: cap number of fields and per-part size to prevent parser abuse.
  - Readiness metadata includes `ensemble_size` and `model_ids` for quick debugging.
  - Railway deployment: provision memory for multiple models; uvicorn `--workers 1` recommended; tune `APP__THREADS` to hardware.

**Security**
- Validate MIME and decode safely via Pillow; strip metadata; enforce size and dimension caps.
- Optional `X-Api-Key` header check when exposed publicly; DiscordBot provides the key if enabled.

**Deployment**
- Docker (multi-stage, Poetry). Expose `APP__PORT` (default 8081). Run `uvicorn handwriting_ai.api.app:app --host 0.0.0.0 --port 8081`.
- Volumes: mount `/data/artifacts` and `/data/logs`.
- Local dev: run on `http://localhost:8081`; DiscordBot env uses `HANDWRITING_API_URL=http://127.0.0.1:8081`.
- Managed: deploy to Render/Fly/Cloud Run; keep service private or protect with API key.

**DiscordBot Integration (defaults and UX)**
- Single slash command: `/read image:<attachment>`
  - Accepts exactly one image attachment; no other options.
  - The bot calls `POST /v1/read` with:
    - No `invert` param (server auto-detection handles background inversion).
    - `center=true`.
    - `visualize=false`.
  - Response to user (short, detailed):
    - Example: `Digit: 7 (98.7% confidence). Top-3: 7=0.987, 1=0.011, 9=0.002. Model: mnist_resnet18_v1.`
    - The bot derives Top-3 from `probs` in the JSON response; the API returns full probability distribution.
  - Client timeouts/retries:
    - HTTP timeout: 5 seconds; retry once on 5xx/timeout with 1-second backoff. Do not retry on 4xx.
  - Low-confidence UX:
    - If `uncertain=true`, append a hint like: `Low confidence; try larger digits or darker ink.`
  - Guardrails: validate file type, size, and dimensions prior to upload when possible; apply per-user rate limits; include `X-Request-ID` for correlation.

**Testing Strategy**
- Unit: preprocessing branches; inference determinism (fixed seed); manifest I/O; error paths; limits enforcement (size/dimensions/types).
- Contract: strict Pydantic models; invalid multipart; size/type/dimension limits; timeouts.
- E2E (compose): DiscordBot <-> service with golden images; health/ready checks.
- Types: mypy strict across `src/` and `tests/`; no Any/casts.

**Phased Implementation**
- Competition-first rollout
  - Implement ensemble-first inference and strong TTA defaults; expose configuration to tune profiles and ensemble temperature.
  - Extend manifest parsing to v1.1 (required) with training provenance; update `/v1/models/active` and readiness to include ensemble details.
  - Add optional training endpoints and workers to produce v1.1 artifacts with recipes and seeds.
  - Events schema (digits:events):
    - completed
      ```json
      {"type": "completed", "model_id": "mnist_resnet18_v1", "val_acc": 0.9972, "duration_sec": 4321, "request_id": "abc-123"}
      ```
    - failed
      ```json
      {"type": "failed", "model_id": "mnist_resnet18_v1", "error": "oom during epoch 3", "request_id": "abc-123"}
      ```

**Decision Records (ADR)**
- ADR-001: Service split vs bot cog — choose standalone service for isolation and speed of iteration; bot remains thin.
- ADR-002: Model/inference — Ensemble-first with strong TTA default for competition.
- ADR-003: Manifests — Require v1.1 with training provenance; refuse incompatible artifacts.
- ADR-004: Config style — pydantic-settings with nested envs; optional TOML override.
- ADR-005: Concurrency — bounded `ThreadPoolExecutor`, `torch.set_num_threads(1)`, uvicorn `--workers 1`.

**Acceptance Criteria**
- Strict typing (no Any/cast), Ruff clean; tests for core paths; reproducible inference.
- `/healthz` ok; `/readyz` ready only when an ensemble (>=1 model) loads; readiness returns `ensemble_size` and `model_ids`.
- `/v1/read` handles PNG/JPEG, limits, timeouts; returns calibrated, ensemble-averaged probabilities; `model_id=ensemble_<N>_models`.
- `/v1/models/active` exposes v1.1 manifest fields and model list; non‑v1.1 manifests are rejected.
- Discord `/read` returns: `Digit: X (YY.Y% confidence). Top-3: ...` with top‑3 probabilities and ensemble id.

**Performance Targets (on laptop i9 CPU, TTA off)**
- P50 latency: < 20 ms per request
- P95 latency: < 60 ms per request
- Notes: ResNet-18 trades some latency for accuracy; if stricter latency is required later, a compact CNN can be introduced as an alternative architecture.

**CI Gates**
- Require ruff format + lint, mypy --strict, and tests to pass before deploy.
