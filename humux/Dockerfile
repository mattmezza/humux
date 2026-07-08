# ---------- builder: compile wacli (Go + CGO) ----------
FROM golang:1.24-bookworm AS wacli-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*
ARG WACLI_VERSION=v0.11.0
RUN CGO_ENABLED=1 CGO_CFLAGS="-Wno-error=missing-braces" \
    GOBIN=/out \
    go install -tags sqlite_fts5 github.com/openclaw/wacli/cmd/wacli@${WACLI_VERSION}

# ---------- runtime ----------
FROM python:3.14-slim

WORKDIR /app

# System deps (ffmpeg for voice pipeline, curl for health checks, sqlite3 for memory)
# The bash-tool allowlist (core/executor.py ALLOWED_PREFIXES) promises these CLIs
# exist — keep this list in sync with it: git (coding harness clone/commit),
# ripgrep (rg), w3m, pandoc, poppler-utils (pdftotext, pdftoppm), yt-dlp, ncal (cal).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl ca-certificates jq sqlite3 \
    bash tar gzip xz-utils \
    git ripgrep w3m pandoc poppler-utils yt-dlp ncal \
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

# wacli binary from the builder stage — no Go toolchain in the final image.
COPY --from=wacli-builder /out/wacli /usr/local/bin/wacli
RUN wacli version

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies (cached layer — only re-runs when lockfile changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

# Local on-device STT (faster-whisper + ctranslate2, ~350MB). ON by default;
# opt out with: docker build --build-arg INSTALL_WHISPER=false
# When off, set voice.stt_api_base_url to a sidecar (see docker-compose.yml).
ARG INSTALL_WHISPER=true
RUN if [ "$INSTALL_WHISPER" = "true" ]; then \
      uv sync --no-dev --no-install-project --extra whisper; \
    fi

# Local on-device embeddings (fastembed + onnxruntime, ~200MB). ON by default;
# opt out with: docker build --build-arg INSTALL_FASTEMBED=false
# When off, set embedding.provider + base_url to a sidecar (see docker-compose.yml).
ARG INSTALL_FASTEMBED=true
RUN if [ "$INSTALL_FASTEMBED" = "true" ]; then \
      uv sync --no-dev --no-install-project --extra fastembed; \
    fi

# Offline on-device TTS (Kokoro, issue #84). ON by default so switching the TTS
# backend to "kokoro" works out of the box (edge-tts stays the default backend).
# The extra pulls onnxruntime (~200MB) plus the model later (~325MB); opt out for
# a leaner edge-tts-only image with: docker build --build-arg INSTALL_KOKORO=false .
ARG INSTALL_KOKORO=true
RUN if [ "$INSTALL_KOKORO" = "true" ]; then \
      uv sync --no-dev --no-install-project --extra kokoro; \
    fi

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
      uv run playwright install --with-deps chromium && \
      chmod -R a+rx /ms-playwright; \
    fi

# Create non-root user
RUN groupadd --gid 10001 humux && \
    useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash humux

# Copy application code
COPY core/ core/
COPY channels/ channels/
COPY schema/ schema/
COPY skills/ skills/
COPY tools/ tools/
COPY voice/ voice/
COPY api/ api/

# Prefetch the local embedding model (semantic memory, Tier 2) so it is bundled
# in the image — no runtime download, works offline. Stored in /app/models,
# OUTSIDE the /app/data volume so the mounted volume cannot shadow it. Keep the
# default in sync with EmbeddingConfig (core/config.py).
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
RUN if [ "$INSTALL_FASTEMBED" = "true" ]; then \
      uv run python -m core.embeddings prefetch "${EMBED_MODEL}" /app/models; \
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
        -o /app/models/kokoro/voices-v1.0.bin; \
    fi

# Build CSS with Tailwind CSS v4 standalone CLI
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "arm64" ]; then TW_ARCH="linux-arm64"; else TW_ARCH="linux-x64"; fi && \
    curl -sSLo /tmp/tailwindcss "https://github.com/tailwindlabs/tailwindcss/releases/download/v4.1.18/tailwindcss-$TW_ARCH" && \
    chmod +x /tmp/tailwindcss && \
    /tmp/tailwindcss --input api/static/input.css --output api/static/style.css --minify && \
    rm /tmp/tailwindcss

# Data directory
RUN mkdir -p /app/data \
    && chown -R humux:humux /home/humux /app

USER humux

# Identify the linked WhatsApp device as "humux" (native since wacli 0.2.0).
ENV WACLI_DEVICE_LABEL=humux

EXPOSE 8000

# Health check — orchestrators (Docker Compose, Swarm, K8s) can detect a hung
# or failed process.  Curl is pre-installed (see system-deps apt-get above).
# The start period is generous: the first boot runs the setup wizard and may
# take a while before the /health route responds 200.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uv", "run", "python", "-m", "core.main"]
