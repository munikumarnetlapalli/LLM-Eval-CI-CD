# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]" --prefix=/install

# ─── Stage 2: Production ──────────────────────────────────────────────────────
FROM python:3.11-slim AS production

# Non-root user for security
RUN groupadd -r evaluser && useradd -r -g evaluser evaluser

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code only — no dev files, no .git, no large datasets
COPY evaluator/ ./evaluator/
COPY backend/  ./backend/
COPY frontend/ ./frontend/
COPY demo_rag/ ./demo_rag/
COPY configs/  ./configs/
COPY datasets/ ./datasets/
COPY pyproject.toml .

# Install the package itself (already deps installed above)
RUN pip install --no-cache-dir --no-deps -e . && \
    pip cache purge

# Runtime environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STORAGE_BACKEND=local \
    LLM_PROVIDER=mock \
    API_HOST=0.0.0.0 \
    API_PORT=8000

RUN chown -R evaluser:evaluser /app
USER evaluser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
