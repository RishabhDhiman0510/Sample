FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml ./

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install -e .

FROM python:3.10-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY medical_rag/ ./medical_rag/
COPY README.md ./

RUN mkdir -p ./medical_rag_library ./.web_cache && \
    useradd -m -u 1000 appuser && chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    PATH="/opt/venv/bin:$PATH"

EXPOSE 8000 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "medical_rag.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

