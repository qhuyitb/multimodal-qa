#!/bin/bash
# Start Multimodal QA API Server

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 Starting Multimodal QA API"
echo ""

# Check dependencies
echo "Checking dependencies..."
if ! python -c "import fastapi" 2>/dev/null; then
    echo "FastAPI not found. Syncing dependencies with uv..."
    uv sync
fi

# Start server
echo ""
echo "🌐 Starting server on http://localhost:8000"
echo "📖 API docs: http://localhost:8000/docs"
echo "🏥 Health check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop"
echo ""

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
