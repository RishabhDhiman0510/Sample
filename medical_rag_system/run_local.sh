#!/bin/bash
set -e
echo "🚀 Starting Medical RAG Server..."
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found"
    exit 1
fi
source venv/bin/activate
if [ ! -f ".env" ]; then
    cp example.env .env
fi
export $(cat .env | grep -v '^#' | xargs)
echo "✅ Starting server on http://localhost:${API_PORT:-8000}"
uvicorn medical_rag.api.server:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" --reload
