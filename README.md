[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%9A%96%EF%B8%8F-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-%230db7ed?logo=docker&logoColor=white)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-%23FF7A00?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![Whisper](https://img.shields.io/badge/Whisper-OpenAI-lightgrey)](https://github.com/openai/whisper)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)](https://www.trychroma.com/)

# Multimodal-QA

**Hệ thống xử lý đa phương thức: Video/Document → Subtitle/Translation → Question Answering**

## 🎯 Mục tiêu

Hệ thống cho phép:
- **Pipeline 1: Video → Vietsub + QA** - Xử lý video, tạo subtitle tiếng Việt và hệ thống hỏi đáp
- **Pipeline 2: Document → Dịch + QA** - Dịch tài liệu và tích hợp hệ thống hỏi đáp
- Sử dụng RAG (Retrieval-Augmented Generation) để trả lời câu hỏi dựa trên nội dung đã index

## ✨ Tính năng chính

### Pipeline 1: Video Processing
1. **Trích xuất audio** từ video (ffmpeg)
2. **Speech-to-Text** với timestamps (Whisper/AssemblyAI)
3. **Dịch transcript** Anh-Việt
4. **Tạo subtitle files** (.srt/.vtt)
5. **Burn subtitle** vào video (optional)
6. **Index vào vector DB** cho QA

### Pipeline 2: Document Processing
1. **Trích xuất text** từ PDF/DOCX/TXT
2. **Smart chunking** với overlap để giữ ngữ cảnh
3. **Dịch từng chunk** với context preservation
4. **Tạo dual-language document** (song ngữ)
5. **Index vào vector DB** cho QA

### QA System
- **Retrieval-based QA** từ vector database (ChromaDB)
- **Multilingual embeddings** (EN-VI)
- **Timestamp references** cho video
- **Source citations** trong câu trả lời

## 📁 Cấu trúc dự án

```
multimodal-qa/
├── configs/
│   ├── config.yaml          # Main config
│   └── models.yaml          # Model configs
├── data/
│   ├── input/
|   |   ├── videos          # video gốc (mp4, avi, mkv,..)
|   |   ├── documents       # tài liệu gốc (pdf, docx, txt…)
|   |   └── optional
|   |   
│   ├── output/
│   │   ├── transcripts/    # Chứa file .txt hoặc .json do whisper extract từ video
│   │   ├── subtitles/      # Chứa các file subtitle từ transcript
│   │   ├── translations/   # Chứa transcript hoặc subtitle đã được dịch sang ngôn ngữ khác
│   │   ├── videos/         # Chứa video đầu ra sau khi xử lý
|   |   └── documents/      # Chứa các extract từ pdf/docx/txt
│   └── vector_db/
|
├── src/
|
|	├── api/                # FastAPI REST API
│   │   ├── routes/         #  video.py, document.py, qa.py
│   │   ├── schemas/        # Pydantic models
│   │   └── main.py
|	|
│   ├── extractors/
│   │   ├── video.py         # Video extraction
│   │   └── document.py      # PDF/DOCX extraction
│   ├── models/
│   │   ├── stt.py           # Whisper STT
│   │   ├── translation.py   # EN-VI translation
│   │   └── embedding.py     # Text embeddings
│   ├── pipelines/
│   │   ├── video_pipeline.py    # Video → Vietsub + QA
│   │   ├── document_pipeline.py # Doc → Dịch + QA
│   │   └── qa_pipeline.py       # Q&A system
│   ├── services/
│   │   ├── subtitle.py      # SRT generation
│   │   ├── vector_store.py  # ChromaDB ops
│   │   └── qa_engine.py     # Q&A engine
│   ├── core/
│   │   ├── chunking.py      # Text chunking
│   │   └── config.py        # Config loader
│   └── utils/
│       ├── logger.py        # Logging
│       └── helpers.py       # Helpers
├── tests/                   # Test files
├── scripts/                 # Setup scripts
└── [root files]            # pyproject.toml, Dockerfile, etc.
```

## 🚀 Installation

### Yêu cầu hệ thống
- Python 3.9+
- ffmpeg (cho xử lý video/audio)
- CUDA (optional, cho GPU acceleration)



## 🐳 Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License

## 👤 Author

- **GitHub**: [@qhuyitb](https://github.com/qhuyitb)
- **Project**: [multimodal-qa](https://github.com/qhuyitb/multimodal-qa)

---





