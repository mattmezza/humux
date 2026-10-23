# syntax=docker/dockerfile:1

# Shared switch: the builder always needs Go, but runtime installation is optional.
ARG INSTALL_GO=true

# ---------- builder: compile wacli (Go + CGO) ----------
FROM golang:1.25-bookworm AS wacli-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*
ARG WACLI_VERSION=v0.11.0
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=1 CGO_CFLAGS="-Wno-error=missing-braces" \
    GOBIN=/out \
    go install -tags sqlite_fts5 github.com/openclaw/wacli/cmd/wacli@${WACLI_VERSION}

# Export the exact toolchain used by wacli, or an empty directory for lean builds.
# COPY from this output keeps INSTALL_GO=false free of runtime Go layers.
ARG INSTALL_GO
RUN mkdir -p /out/toolchain && \
    if [ "$INSTALL_GO" = "true" ]; then \
      cp -a /usr/local/go /out/toolchain/go; \
    fi

# Export only third-party dependencies: a release version bump does not change
# requirements.txt, so BuildKit can reuse the installed environment layer.
FROM python:3.14-slim AS requirements
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /build
COPY pyproject.toml uv.lock ./
ARG INSTALL_WHISPER=false
ARG INSTALL_FASTEMBED=false
ARG INSTALL_KOKORO=false
RUN set -eu; extras=""; \
    if [ "$INSTALL_WHISPER" = "true" ]; then extras="$extras --extra whisper"; fi; \
    if [ "$INSTALL_FASTEMBED" = "true" ]; then extras="$extras --extra fastembed"; fi; \
    if [ "$INSTALL_KOKORO" = "true" ]; then extras="$extras --extra kokoro"; fi; \
    uv export --locked --no-dev --no-emit-project --no-header $extras -o requirements.txt

# Compile CSS independently of Python/application changes. Cache the CLI download.
FROM python:3.14-slim AS css-builder
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "arm64" ]; then TW_ARCH="linux-arm64"; else TW_ARCH="linux-x64"; fi && \
    curl -fsSLo /usr/local/bin/tailwindcss "https://github.com/tailwindlabs/tailwindcss/releases/download/v4.1.18/tailwindcss-$TW_ARCH" && \
    chmod +x /usr/local/bin/tailwindcss
WORKDIR /build
COPY api/static/input.css api/static/input.css
COPY api/templates/ api/templates/
RUN tailwindcss --input api/static/input.css --output style.css --minify

# ---------- runtime ----------
FROM python:3.14-slim

WORKDIR /app

# Go builds use the coding harness, with the same permissions as Node builds.
# Both caches live in the existing writable /app/data volume (non-root friendly).
ARG INSTALL_GO
COPY --from=wacli-builder /out/toolchain/ /usr/local/
ENV PATH="/usr/local/go/bin:/app/data/go/bin:${PATH}" \
    GOPATH=/app/data/go \
    GOCACHE=/app/data/go-build-cache \
    GOTOOLCHAIN=local
# Include the C compiler and pkg-config for CGO repos when Go is installed.
RUN if [ "$INSTALL_GO" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config \
      && rm -rf /var/lib/apt/lists/* \
      && go version; \
    fi

# System deps (ffmpeg for voice pipeline, curl for health checks, sqlite3 for memory)
# The bash-tool allowlist (core/executor.py ALLOWED_PREFIXES) promises these CLIs
# exist — keep this list in sync with it: git (coding harness clone/commit),
# ripgrep (rg), w3m, pandoc, poppler-utils (pdftotext, pdftoppm), yt-dlp, ncal (cal).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl ca-certificates jq sqlite3 \
    bash tar gzip xz-utils \
    git ripgrep w3m pandoc poppler-utils yt-dlp ncal openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install Himalaya CLI (pre-built Rust binary for email management)
RUN curl -sSL https://raw.githubusercontent.com/pimalaya/himalaya/master/install.sh | sh

# Install GitHub CLI (gh) from the official apt repository (Tools tab: gh)
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# Node.js LTS (node/npm/npx) for JS/TS builds in the coding workspace — these
# run through the coding-harness bash (permissions.py rules); they are NOT in
# the run_command prefix allowlist (`npm run` = arbitrary execution).
# Opt out with: docker build --build-arg INSTALL_NODE=false
ARG INSTALL_NODE=true
RUN if [ "$INSTALL_NODE" = "true" ]; then \
      curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
      && apt-get install -y --no-install-recommends nodejs \
      && rm -rf /var/lib/apt/lists/* \
      && node --version && npm --version; \
    fi

# wacli is installed even when the optional runtime Go toolchain is disabled.
COPY --from=wacli-builder /out/wacli /usr/local/bin/wacli
RUN wacli version

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create ownership before adding files; avoid a late recursive chown that copies
# the entire environment into a new image layer on every application change.
RUN groupadd --gid 10001 humux && \
    useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash humux && \
    mkdir -p /app/data && chown humux:humux /app /app/data

# Download/build caches stay outside the image. Install all selected extras in
# one transaction so a later uv sync cannot remove an earlier optional extra.
ENV UV_LINK_MODE=copy
COPY --from=requirements /build/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv && \
    uv pip sync --python /app/.venv/bin/python --require-hashes /tmp/requirements.txt && \
    chown -R humux:humux /app/.venv
ARG INSTALL_WHISPER=false
ARG INSTALL_FASTEMBED=false
ARG INSTALL_KOKORO=false
# Use --no-sync below to preserve the selected optional extras.

# Bundle full Chromium for the browser tool (Tools tab: browser, issue #16).
# OFF by default: use the browserless/chrome sidecar instead (see docker-compose.yml,
# `--profile browser`).  Set tools.browser.cdp_url to ws://browser:3000.
# Opt IN for a self-contained image:  docker build --build-arg INSTALL_BROWSER=true .
# We install the FULL chromium (not chromium-headless-shell): its "new headless"
# mode renders more faithfully and is far less bot-detectable than the old
# headless shell — important since the browser/vision tooling targets exactly the
# janky, JS-heavy, sometimes bot-protected sites where the shell gets served
# degraded content.
ARG INSTALL_BROWSER=false
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN if [ "$INSTALL_BROWSER" = "true" ]; then \
      uv run --no-sync playwright install --with-deps chromium && \
      chmod -R a+rx /ms-playwright; \
    fi

# Copy application code and manifests after expensive dependency layers.
COPY --chown=10001:10001 pyproject.toml uv.lock ./
COPY --chown=10001:10001 core/ core/
COPY --chown=10001:10001 channels/ channels/
COPY --chown=10001:10001 schema/ schema/
COPY --chown=10001:10001 skills/ skills/
COPY --chown=10001:10001 tools/ tools/
COPY --chown=10001:10001 voice/ voice/
COPY --chown=10001:10001 api/ api/
COPY --from=css-builder --chown=10001:10001 /build/style.css api/static/style.css

# Prefetch the local embedding model (semantic memory, Tier 2) so it is bundled
# in the image — no runtime download, works offline. Stored in /app/models,
# OUTSIDE the /app/data volume so the mounted volume cannot shadow it. Keep the
# default in sync with EmbeddingConfig (core/config.py).
ARG EMBED_MODEL=intfloat/multilingual-e5-small
RUN if [ "$INSTALL_FASTEMBED" = "true" ]; then \
      uv run --no-sync python -m core.embeddings prefetch "${EMBED_MODEL}" /app/models && \
      chown -R humux:humux /app/models; \
    fi

# Prefetch Kokoro TTS model + voices when the kokoro backend is enabled (#84),
# mirroring the Whisper/embedding prefetch so the container runs fully offline.
# Stored in /app/models (OUTSIDE the /app/data volume) — keep paths in sync with
# KokoroConfig (core/config.py). Gated on the same INSTALL_KOKORO arg as above.
RUN if [ "$INSTALL_KOKORO" = "true" ]; then \
      mkdir -p /app/models/kokoro && \
      curl -sSL https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
        -o /app/models/kokoro/kokoro-v1.0.onnx && \
      curl -sSL https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin \
        -o /app/models/kokoro/voices-v1.0.bin && \
      chown -R humux:humux /app/models/kokoro; \
    fi

USER humux

# Identify the linked WhatsApp device as "humux" (native since wacli 0.2.0).
ENV WACLI_DEVICE_LABEL=humux

EXPOSE 8000

# Health check — orchestrators (Docker Compose, Swarm, K8s) can detect a hung
# or failed process.  Curl is pre-installed (see system-deps apt-get above).
# The start period is generous: the first boot runs the setup wizard and may
# take a while before the /health route responds 200.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f "http://localhost:${PORT:-8000}/health" || exit 1

CMD ["uv", "run", "--no-sync", "python", "-m", "core.main"]
