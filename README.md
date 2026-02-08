# Multimodal QA System

**Production-ready multilingual Question Answering system with hybrid retrieval, conversational AI, and cross-lingual support.**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-ffcc00.svg)](https://huggingface.co/docs/transformers)
[![RAG](https://img.shields.io/badge/RAG-Hybrid_Retrieval-success.svg)]()
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-3b82f6.svg)](https://www.trychroma.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

### Core Capabilities
- **Multilingual QA**: Vietnamese & English with automatic language detection
- **Hybrid Retrieval**: BM25 + semantic search + Reciprocal Rank Fusion (RRF)
- **Conversational AI**: Multi-turn conversations with context awareness
- **Cross-lingual Support**: Automatic EN↔VI translation with Helsinki-NLP models
- **Smart Chunking**: Context-aware text segmentation for documents and videos

### Model Performance
- **XLM-RoBERTa Stage 2** fine-tuned on 55K augmented ViQuAD samples
- **F1 Score**: 78.49% | **Exact Match**: 60.50%
- **Languages**: Vietnamese (primary), English (with translation)

### API Features
- Production REST API with FastAPI
- OpenAPI/Swagger documentation
- Session management for conversations
- Health checks and performance stats
- Proper error handling and validation

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/qhuyitb/multimodal-qa.git
cd multimodal-qa

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Start API Server

```bash
# Start server
.venv/bin/python -m uvicorn src.api.main:app --port 9010 --host 0.0.0.0

# Access documentation
open http://localhost:9010/docs
```

## API Usage

### 1. Single-Shot QA

Ask a question and get an answer with source references.

**Endpoint**: `POST /api/v1/qa/ask`

```bash
# Vietnamese question
curl -X POST http://localhost:9010/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "XLM-RoBERTa là gì?",
    "top_k": 3
  }'
```

**Python Client:**

```python
import requests

response = requests.post(
    "http://localhost:9010/api/v1/qa/ask",
    json={
        "question": "What is machine learning?",
        "top_k": 5,
        "target_language": "vi"  # Translate answer to Vietnamese
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Sources: {len(result['sources'])}")
```

**Response:**

```json
{
  "answer": "XLM-RoBERTa là mô hình ngôn ngữ đa ngôn ngữ...",
  "sources": [
    {
      "text": "XLM-RoBERTa (Cross-lingual Language Model)...",
      "score": 0.89,
      "metadata": {"document": "ml_guide.pdf"}
    }
  ],
  "confidence": 0.89,
  "query_language": "vi",
  "source_language": "vi",
  "target_language": null,
  "translated": false
}
```

### 2. Conversational QA

Multi-turn conversations with context awareness and follow-up handling.

**Endpoint**: `POST /api/v1/chat/`

```bash
# Create session
curl -X POST http://localhost:9010/api/v1/chat/sessions \
  -H "Content-Type: application/json" \
  -d '{}'

# First question
curl -X POST http://localhost:9010/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "question": "Transformer architecture hoạt động thế nào?",
    "top_k": 3
  }'

# Follow-up question
curl -X POST http://localhost:9010/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "question": "Attention mechanism trong đó là gì?",
    "use_context": true
  }'
```

**Python Client:**

```python
import requests

BASE_URL = "http://localhost:9010"

# Create session
session = requests.post(f"{BASE_URL}/api/v1/chat/sessions", json={}).json()
session_id = session["session_id"]

# Chat
def chat(question):
    response = requests.post(
        f"{BASE_URL}/api/v1/chat/",
        json={
            "session_id": session_id,
            "question": question,
            "use_context": True
        }
    )
    return response.json()

# Conversation flow
r1 = chat("Deep learning là gì?")
print(f"Answer 1: {r1['answer']}")

r2 = chat("Nó khác gì machine learning?")  # Follow-up with context
print(f"Answer 2: {r2['answer']}")
print(f"Reformulated: {r2['reformulated_question']}")
print(f"Used context: {r2['used_context']}")

# Get conversation history
history = requests.get(
    f"{BASE_URL}/api/v1/chat/sessions/{session_id}/history"
).json()
print(f"Total messages: {len(history['messages'])}")
```

**Response:**

```json
{
  "session_id": "abc-123",
  "answer": "Attention mechanism cho phép mô hình...",
  "sources": [...],
  "confidence": 0.85,
  "reformulated_question": "Context: User: Transformer...\n\nQuestion: Attention mechanism trong đó là gì?",
  "used_context": true,
  "conversation_length": 4
}
```

### 3. Health & Stats

Check service status and performance metrics.

```bash
# Health check
curl http://localhost:9010/api/v1/qa/health

# Performance stats
curl http://localhost:9010/api/v1/qa/stats
```

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "hybrid_retrieval_ready": true,
  "total_queries": 127,
  "languages_detected": ["vi", "en"]
}
```

```json
{
  "total_queries": 127,
  "translations": 34,
  "by_language": {
    "vi": 89,
    "en": 38
  },
  "avg_translation_time_ms": 145.2,
  "avg_inference_time_ms": 312.5
}
```

### 4. Session Management

Manage conversational sessions.

```bash
# List all sessions
curl http://localhost:9010/api/v1/chat/sessions

# Get conversation history
curl http://localhost:9010/api/v1/chat/sessions/{session_id}/history

# Clear session history (keep session)
curl -X POST http://localhost:9010/api/v1/chat/sessions/{session_id}/clear

# Delete session
curl -X DELETE http://localhost:9010/api/v1/chat/sessions/{session_id}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI REST API                        │
│  /api/v1/qa/ask  |  /api/v1/chat/  |  /health  |  /stats  │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
┌───────▼──────────┐           ┌─────────▼─────────┐
│  Adaptive QA     │           │  Conversational   │
│  Service         │           │  QA Service       │
│                  │           │                   │
│ • Language       │           │ • Session Mgmt    │
│   Detection      │           │ • History Track   │
│ • Translation    │           │ • Reformulation   │
│ • Answer Gen     │           │ • Context Aware   │
└───────┬──────────┘           └─────────┬─────────┘
        │                                 │
        └────────────────┬────────────────┘
                         │
                ┌────────▼─────────┐
                │ Hybrid Retrieval │
                │                  │
                │ • BM25 Search    │
                │ • Semantic Search│
                │ • RRF Fusion     │
                └────────┬─────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
┌───────▼──────────┐           ┌─────────▼─────────┐
│  XLM-RoBERTa     │           │  Vector DB        │
│  QA Model        │           │  (ChromaDB)       │
│                  │           │                   │
│ • F1: 78.49%     │           │ • Embeddings      │
│ • EM: 60.50%     │           │ • BM25 Index      │
│ • 278M params    │           │ • Metadata        │
└──────────────────┘           └───────────────────┘
```

## Project Structure

```
multimodal-qa/
├── src/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── qa_v2.py          # Production QA endpoints
│   │   │   └── chat.py           # Conversational endpoints
│   │   └── main.py               # FastAPI app
│   ├── services/
│   │   ├── adaptive_qa.py        # Adaptive QA with translation
│   │   ├── conversational_qa.py  # Conversational AI
│   │   └── hybrid_retrieval.py   # BM25 + semantic + RRF
│   ├── models/
│   │   ├── qa_model.py           # XLM-RoBERTa QA
│   │   └── translation.py        # EN↔VI translation
│   └── core/
│       ├── document_chunking.py  # Smart text chunking
│       └── video_chunking.py     # Video-specific chunking
├── models/
│   └── xlm_roberta_qa/
│       └── stage2_best/          # Fine-tuned model
├── scripts/
│   ├── test_chat.py              # Test conversational flow
│   └── benchmark_cross_lingual.py # Cross-lingual evaluation
└── configs/
    ├── config.yaml               # Main configuration
    └── models.yaml               # Model settings
```

## Configuration

Edit `configs/config.yaml`:

```yaml
# Model settings
model:
  path: "models/xlm_roberta_qa/stage2_best"
  device: "cpu"  # or "cuda"
  max_length: 512

# Vector database
paths:
  vector_db: "data/vector_db"

# Retrieval settings
cross_language_qa:
  embedding_model: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  top_k: 5
  chunk_size: 512
  chunk_overlap: 50

# Translation
translation:
  enabled: true
  en_to_vi_model: "Helsinki-NLP/opus-mt-en-vi"
  vi_to_en_model: "Helsinki-NLP/opus-mt-vi-en"
```

## Performance Benchmarks

### Model Metrics
- **Dataset**: ViQuAD (augmented 55K samples)
- **F1 Score**: 78.49%
- **Exact Match**: 60.50%
- **Trainable Parameters**: 886K (0.32% of total)

### Cross-lingual Consistency
- **Test Set**: XQuAD parallel EN-VI (1190 samples)
- **Consistency Rate**: Measured on 50 samples
- **Avg Translation Time**: 519ms
- **Avg Inference Time**: 303ms

### API Performance
- **Health Check**: <5ms
- **Single QA**: ~300-500ms (CPU)
- **Conversational QA**: ~350-600ms (with context)
- **Translation Overhead**: ~150ms per translation

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_services/test_adaptive_qa.py

# Test conversational flow
python scripts/test_chat.py
```

### Adding New Features

1. **New Service**: Add to `src/services/`
2. **New Endpoint**: Add route to `src/api/routes/`
3. **Update Schemas**: Add Pydantic models in `src/api/schemas/`
4. **Register Route**: Import in `src/api/main.py`

## Troubleshooting

### Common Issues

**Model not loading:**
```bash
# Check model path
ls -la models/xlm_roberta_qa/stage2_best/

# Verify config
cat configs/models.yaml
```

**API not starting:**
```bash
# Check imports
python -c "from src.api.main import app; print('OK')"

# View logs
tail -f /tmp/api.log
```

**No search results:**
```bash
# Check if vector DB has data
ls -la data/vector_db/

# Re-index documents if needed
python scripts/setup_vectordb.py
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details

## Citation

```bibtex
@software{multimodal_qa_2026,
  title = {Multimodal QA: Production Multilingual Question Answering},
  author = {To Huy},
  year = {2026},
  url = {https://github.com/qhuyitb/multimodal-qa}
}
```

## Acknowledgments

- XLM-RoBERTa model by Facebook AI
- Helsinki-NLP translation models
- ViQuAD dataset by UIT-NLP
- SQuAD dataset by Stanford NLP

---

**Built with ❤️ for multilingual QA research and production deployment**





