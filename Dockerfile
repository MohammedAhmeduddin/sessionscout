# Dockerfile — SessionScout API
#
# Multi-stage build:
#   Stage 1 (builder): install dependencies
#   Stage 2 (runtime): copy only what is needed — smaller final image
#
# Why multi-stage?
#   The builder stage needs compilers and build tools to install
#   packages like numpy and torch. The runtime stage does not.
#   Multi-stage keeps the final image lean (~2GB instead of ~4GB).

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (Docker layer caching)
# If pyproject.toml has not changed, this layer is cached
COPY pyproject.toml .
COPY src/ src/

# Install all dependencies into /app/.venv
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e .


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY src/ src/
COPY pyproject.toml .

# Copy model weights (required for inference)
# In production these would come from S3/MLflow, not baked into the image
COPY models/ models/

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Expose API port
EXPOSE 8000

# Health check — Docker will mark container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the API
CMD ["uvicorn", "sessionscout.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
