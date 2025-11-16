# Production Medical RAG System

A production-quality medical retrieval-augmented generation system with learning and web enhancement.

## Quick Start

```bash
git clone <repo>
cd medical-rag
python3.10 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
cp example.env .env
./run_local.sh
```

## API Endpoints

- `GET /health` - Health check
- `POST /query` - Process medical query
- `POST /feedback` - Submit feedback
- `GET /metrics` - Prometheus metrics

## Docker

```bash
docker build -t medical-rag:latest .
docker run -p 8000:8000 medical-rag:latest
```

## Testing

```bash
pytest tests/ -v --cov=medical_rag
```

## Features

- Embedding-based correction retrieval
- Async web search (PubMed + DuckDuckGo)
- Confidence calibration
- PII redaction
- Structured logging
- Prometheus metrics
- Type-safe FastAPI server
