FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEEDLE_TELEMETRY=0 \
    DO_NOT_TRACK=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home whistle
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
USER whistle
# Fetch and load the native engine + model at build time, so startup is offline.
# Remove download caches in the same layer: Needle keeps its own runtime copies.
RUN python -c 'from needle import Whistle; Whistle()' \
    && rm -rf /home/whistle/.cache/huggingface
ENV HF_HUB_OFFLINE=1
COPY server.py ./
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c 'import urllib.request; urllib.request.urlopen("http://localhost:8000/health", timeout=4)'
# One worker: the native runtime is process-global; requests serialize inference.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
