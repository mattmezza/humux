.PHONY: help setup setup-hooks install install-dev sync lock lint format test run cli dev dev-agent dev-css dev-wa kokoro clean release css sidecars sidecar-embeddings sidecar-browser sidecar-whisper sidecar-kokoro sidecar-searxng sidecar-stop

PORT := 8001
PYTHON := uv run python
UV := uv
TAILWIND := ./tailwindcss
CSS_IN := api/static/input.css
CSS_OUT := api/static/style.css

# Pinned upstream wacli (github.com/openclaw/wacli). Keep in sync with Dockerfile.
WACLI_VERSION := v0.11.0

# Kokoro offline TTS model (github.com/thewh1teagle/kokoro-onnx releases).
# Keep the version + paths in sync with the Dockerfile and KokoroConfig.
KOKORO_DIR := models/kokoro
KOKORO_BASE_URL := https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0

# Show available targets
help:
	@echo ""
	@echo "humux — Human Multiplexer"
	@echo ""
	@echo "  Setup & Dependencies:"
	@echo "    make setup        First-time setup (install, hooks, copy example configs)"
	@echo "    make setup-hooks  Install git pre-commit hooks (lint + format)"
	@echo "    make install      Install production dependencies"
	@echo "    make install-dev  Install all dependencies (including dev tools)"
	@echo "    make sync         Re-sync venv with lockfile"
	@echo "    make lock         Update lockfile after changing pyproject.toml"
	@echo "    make kokoro       Install Kokoro offline TTS + download its model (local runs)"
	@echo ""
	@echo "  Development:"
	@echo "    make cli          Chat with the agent from the terminal (no Telegram)"
	@echo "                      (make cli AGENT=<slug> to test a specific agent)"
	@echo "                      (make cli YOLO=1 to auto-approve all permissions)"
	@echo "    make dev          Show instructions for running dev services"
	@echo "    make dev-agent    Run agent with auto-reload"
	@echo "    make dev-css      Run Tailwind CSS watcher"
	@echo "    make dev-wa       Install WhatsApp CLI (wacli) from upstream"

	@echo ""
	@echo "  Quality:"
	@echo "    make lint         Lint with ruff"
	@echo "    make format       Format with ruff"
	@echo "    make test         Run tests"
	@echo ""
	@echo "  Build & Deploy:"
	@echo "    make run          Run the agent (production)"
	@echo "    make css          Build minified CSS"
	@echo ""
	@echo "  Docker Sidecars (agent on host, deps in containers):"
	@echo "    make sidecars         Start all dep containers"
	@echo "    make sidecar-stop     Stop and remove all sidecar containers"
	@echo ""
	@echo "    make sidecar-embeddings   Infinity embedding API (port 7997)"
	@echo "    make sidecar-browser       Browserless/chrome (port 3000)"
	@echo "    make sidecar-whisper       Faster-Whisper STT (port 8000)"
	@echo "    make sidecar-kokoro        Kokoro TTS (port 8880)"
	@echo "    make sidecar-searxng       SearXNG search (port 8080)"
	@echo ""
	@echo "  Clean:"
	@echo "    make clean        Remove venv and caches"
	@echo ""
	@echo "  Release:"
	@echo "    make release name=v0.1  Push and create a GitHub release"
	@echo ""

# First-time setup: create venv, install deps, copy example configs
setup: install-dev setup-hooks
	@test -f .env || cp .env.example .env
	@test -f config.yml || cp config.yml.example config.yml
	@echo "Done. Edit .env and config.yml with your secrets, then run: make run"

# Install git hooks (lint + format on commit).
# The path is relative to the repo/worktree root (git runs hooks from there),
# NOT to this humux/ dir — so it must be ".githooks", not "../.githooks". A
# repo-root-relative path also makes the hooks fire from every `git worktree`
# checkout (each has its own .githooks), which "../.githooks" resolved outside.
setup-hooks:
	git config core.hooksPath .githooks

# Install production dependencies
install:
	$(UV) sync --no-dev

# Install all dependencies (including dev tools)
install-dev:
	$(UV) sync

# Update lockfile after changing pyproject.toml
lock:
	$(UV) lock

# Re-sync venv with lockfile
sync:
	$(UV) sync

# Lint
lint:
	$(UV) run ruff check .

# Format
format:
	$(UV) run ruff format .

# Run tests
test:
	$(UV) run pytest

# Run the agent (production)
run:
	$(UV) run python -m core.main

# Local CLI — chat with the agent from the terminal (no Telegram)
cli:
	$(PYTHON) -m core.cli $(if $(AGENT),--agent $(AGENT),) $(if $(YOLO),--yolo,)

# Run in dev mode: instructions for running services in separate shells
dev:
	@echo ""
	@echo "Run each service in its own terminal (Ctrl-C stops each one cleanly):"
	@echo ""
	@echo "  1. Agent (auto-restart on code changes):"
	@echo "     make dev-agent"
	@echo ""
	@echo "  2. Tailwind CSS watcher:"
	@echo "     make dev-css"
	@echo ""
	@echo "  3. WhatsApp (wacli install):"
	@echo "     make dev-wa"
	@echo ""


# Dev: admin API with auto-reload on code changes (agent managed via UI)
dev-agent:
	PYTHONWARNINGS="ignore::UserWarning:multiprocessing.resource_tracker" \
	$(UV) run uvicorn core.main:app --reload --host 0.0.0.0 --port $(PORT) --log-level info \
		--reload-dir api --reload-dir core --reload-dir channels --reload-dir schema \
		--reload-dir skills --reload-dir tools --reload-dir voice

# Dev: Tailwind CSS watcher
dev-css:
	$(TAILWIND) --input $(CSS_IN) --output $(CSS_OUT) --watch

# ──────────────────────────────────────────────
# Docker sidecars — run deps in Docker, agent locally
# Simulates prod: deps in containers, agent on bare metal
# ──────────────────────────────────────────────

# Container network shared by all sidecars
SIDECAR_NET := humux-sidecars

# Port mappings (host:container)
EMBED_PORT   := 7997
BROWSER_PORT := 3000
WHISPER_PORT := 8000
KOKORO_PORT  := 8880
SEARXNG_PORT := 8080

# Start all sidecars
sidecars: $(SIDECAR_NET) sidecar-embeddings sidecar-browser sidecar-whisper sidecar-kokoro sidecar-searxng
	@echo ""
	@echo "All sidecars running. Update config.yml to point localhost:"
	@echo "  memory.embedding.base_url: http://localhost:$(EMBED_PORT)"
	@echo "  search.searxng_url:        http://localhost:$(SEARXNG_PORT)"
	@echo "  voice.stt_api_base_url:    http://localhost:$(WHISPER_PORT)/v1"
	@echo "  voice.tts_api_base_url:    http://localhost:$(KOKORO_PORT)/v1"
	@echo "  tools.browser.cdp_url:     ws://localhost:$(BROWSER_PORT)"
	@echo ""
	@echo "Stop all:  make sidecar-stop"

$(SIDECAR_NET):
	docker network create $(SIDECAR_NET) 2>/dev/null || true

define _SIDECAR_RUN
docker rm -f $(1) 2>/dev/null; docker run -d --name $(1) \
	--network $(SIDECAR_NET) $(2) --restart unless-stopped $(3)
endef

sidecar-embeddings: $(SIDECAR_NET)
	$(call _SIDECAR_RUN,humux-embeddings,\
		-p $(EMBED_PORT):7997,\
		michaelf34/infinity:latest v2 --model-id BAAI/bge-small-en-v1.5 --port 7997)
	@echo "embeddings → http://localhost:$(EMBED_PORT)/v1"

sidecar-browser: $(SIDECAR_NET)
	$(call _SIDECAR_RUN,humux-browser,\
		-p $(BROWSER_PORT):3000 -e MAX_CONCURRENT_SESSIONS=2,\
		browserless/chrome:latest)
	@echo "browser → ws://localhost:$(BROWSER_PORT)"

sidecar-whisper: $(SIDECAR_NET)
	$(eval _GPU := $(shell docker info 2>/dev/null | grep -q 'Runtimes.*nvidia' && echo '--gpus all' || echo ''))
	$(call _SIDECAR_RUN,humux-whisper,\
		-p $(WHISPER_PORT):8000 $(_GPU),\
		fedirz/faster-whisper-server:latest-cpu)
	@echo "whisper → http://localhost:$(WHISPER_PORT)/v1"

sidecar-kokoro: $(SIDECAR_NET)
	$(call _SIDECAR_RUN,humux-kokoro,\
		-p $(KOKORO_PORT):8880,\
		ghcr.io/remsky/kokoro-fastapi-cpu:latest)
	@echo "kokoro → http://localhost:$(KOKORO_PORT)/v1"

sidecar-searxng: $(SIDECAR_NET)
	$(call _SIDECAR_RUN,humux-searxng,\
		-p $(SEARXNG_PORT):8080 -v $(PWD)/searxng:/etc/searxng:rw -e SEARXNG_BASE_URL=http://localhost:$(SEARXNG_PORT)/,\
		searxng/searxng:latest)
	@echo "searxng → http://localhost:$(SEARXNG_PORT)"

sidecar-stop:
	@echo "Stopping all sidecars..."
	-docker stop humux-embeddings humux-browser humux-whisper humux-kokoro humux-searxng 2>/dev/null
	-docker rm humux-embeddings humux-browser humux-whisper humux-kokoro humux-searxng 2>/dev/null
	-docker network rm $(SIDECAR_NET) 2>/dev/null || true
	@echo "Done."

# Dev: WhatsApp (wacli) — install the pinned upstream binary into $GOBIN/~/go/bin.
# Needs Go + a C toolchain (CGO, sqlite_fts5). Override WACLI_BIN to point the
# agent elsewhere; otherwise core/wacli.py resolves it from PATH or ~/go/bin.
dev-wa:
	@command -v wacli >/dev/null 2>&1 || test -x "$(HOME)/go/bin/wacli" || { \
		echo "Installing wacli $(WACLI_VERSION) from github.com/openclaw/wacli..."; \
		CGO_ENABLED=1 CGO_CFLAGS="-Wno-error=missing-braces" \
			go install -tags sqlite_fts5 github.com/openclaw/wacli/cmd/wacli@$(WACLI_VERSION); \
	}
	@wacli version 2>/dev/null || "$(HOME)/go/bin/wacli" version

# Install the Kokoro offline TTS extra + download its model for local (non-Docker)
# runs. The Docker image bundles these via INSTALL_KOKORO. After this, set
# voice.backend: kokoro and restart. Idempotent — existing files are kept.
kokoro:
	$(UV) sync --extra kokoro
	@mkdir -p $(KOKORO_DIR)
	@test -f $(KOKORO_DIR)/kokoro-v1.0.onnx || \
		curl -fL $(KOKORO_BASE_URL)/kokoro-v1.0.onnx -o $(KOKORO_DIR)/kokoro-v1.0.onnx
	@test -f $(KOKORO_DIR)/voices-v1.0.bin || \
		curl -fL $(KOKORO_BASE_URL)/voices-v1.0.bin -o $(KOKORO_DIR)/voices-v1.0.bin
	@echo "Kokoro ready in $(KOKORO_DIR). Set voice.backend: kokoro and restart the agent."

# Remove venv and caches
clean:
	rm -rf .venv __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Create a GitHub release and tag (bumps pyproject.toml version to match)
release:
	@test -n "$(name)" || (echo "Usage: make release name=v0.x" && exit 1)
	@ver=$$(echo "$(name)" | sed 's/^v//'); \
	case "$$ver" in *.*.*) ;; *.*) ver="$$ver.0" ;; *) ver="$$ver.0.0" ;; esac; \
	sed -i.bak -E "s/^version = \".*\"/version = \"$$ver\"/" pyproject.toml && rm -f pyproject.toml.bak; \
	echo "Set pyproject.toml version to $$ver"
	$(UV) lock
	@git diff --quiet pyproject.toml || (git add pyproject.toml uv.lock && git commit -m "chore: bump version to $(name)")
	git push
	gh release create "$(name)" --generate-notes --latest

# Build minified CSS (production)
css:
	$(TAILWIND) --input $(CSS_IN) --output $(CSS_OUT) --minify


