FROM python:3.14-slim

WORKDIR /app

# System deps (ffmpeg for voice pipeline, curl for health checks, sqlite3 for memory)
# golang + build-essential build wacli (CGO, sqlite_fts5) from the pinned upstream tag.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl ca-certificates jq sqlite3 \
    bash tar gzip xz-utils \
    golang build-essential pkg-config \
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

# Install wacli from pinned upstream tag (github.com/openclaw/wacli).
# Bump WACLI_VERSION to cross WhatsApp protocol breaks (e.g. 405 Client Outdated).
ARG WACLI_VERSION=v0.11.0
RUN CGO_ENABLED=1 CGO_CFLAGS="-Wno-error=missing-braces" \
    GOBIN=/usr/local/bin \
    go install -tags sqlite_fts5 github.com/openclaw/wacli/cmd/wacli@${WACLI_VERSION} \
    && wacli version

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies (cached layer — only re-runs when lockfile changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

# Create non-root user
RUN groupadd --gid 10001 mpa && \
    useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash mpa

# Copy application code
COPY core/ core/
COPY channels/ channels/
COPY schema/ schema/
COPY skills/ skills/
COPY personae/ personae/
COPY tools/ tools/
COPY voice/ voice/
COPY api/ api/

# Prefetch the local embedding model (semantic memory, Tier 2) so it is bundled
# in the image — no runtime download, works offline. Stored in /app/models,
# OUTSIDE the /app/data volume so the mounted volume cannot shadow it. Keep the
# default in sync with EmbeddingConfig (core/config.py).
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
RUN uv run python -m core.embeddings prefetch "${EMBED_MODEL}" /app/models

# Build CSS with Tailwind CSS v4 standalone CLI
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "arm64" ]; then TW_ARCH="linux-arm64"; else TW_ARCH="linux-x64"; fi && \
    curl -sSLo /tmp/tailwindcss "https://github.com/tailwindlabs/tailwindcss/releases/download/v4.1.18/tailwindcss-$TW_ARCH" && \
    chmod +x /tmp/tailwindcss && \
    /tmp/tailwindcss --input api/static/input.css --output api/static/style.css --minify && \
    rm /tmp/tailwindcss

# Data directory
RUN mkdir -p /app/data \
    && chown -R mpa:mpa /home/mpa /app

USER mpa

# Identify the linked WhatsApp device as "MPA" (native since wacli 0.2.0).
ENV WACLI_DEVICE_LABEL=MPA

EXPOSE 8000
CMD ["uv", "run", "python", "-m", "core.main"]
