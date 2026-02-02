[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%9A%96%EF%B8%8F-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-%230db7ed?logo=docker&logoColor=white)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-%23FF7A00?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![Whisper](https://img.shields.io/badge/Whisper-OpenAI-lightgrey)](https://github.com/openai/whisper)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)](https://www.trychroma.com/)

# Multimodal-QAuv add

**Hệ thống xử lý đa phương thức: Video/Document → Subtitle/Translation → Question Answering**

## 🎯 Mục tiêu

Hệ thống cho phép:
- **Pipeline 1: Video → Vietsub + QA** - Xử lý video, tạo subtitle tiếng Việt(en) và hệ thống hỏi đáp
- **Pipeline 2: Document → Dịch + QA** - Dịch tài liệu và tích hợp hệ thống hỏi đáp
- Sử dụng RAG (Retrieval-Augmented Generation) để trả lời câu hỏi dựa trên nội dung đã index

## ✨ Tính năng chính

### Pipeline 1: Video Processing
1. **Trích xuất audio** từ video (ffmpeg)
2. **Speech-to-Text** với timestamps & language detection (Whisper / AssemblyAI)
3. **Lưu transcript gốc** (no translation by default)
4. **(Optional) Translate transcript based on user request**
5. **Tạo subtitle files** (.srt / .vtt) theo ngôn ngữ được yêu cầu
6. **Burn subtitle** vào video (optional)
7. **Index transcript gốc vào vector DB** cho QA

### Pipeline 2: Document Processing
1. **Trích xuất text** từ PDF / DOCX / TXT
2. **Language detection**
3. **Smart chunking** với overlap để giữ ngữ cảnh
4. **Lưu nội dung gốc để indexing**
5. **(Optional) Translate chunks based on user request**
6. **Tạo translated hoặc dual-language document (on-demand)**
7. **Index nội dung gốc vào vector DB** cho QA

### QA System
- **Retrieval-based QA** từ vector database (ChromaDB)
- **Cross-lingual / multilingual embeddings** (EN–VI)
- **Answer language adapts to user request**
- **Timestamp references** cho video
- **Source citations** trong câu trả lời


### 🌐 Language Handling Strategy
- Hệ thống tự động phát hiện ngôn ngữ gốc của từng video hoặc tài liệu
- Toàn bộ nội dung được index theo ngôn ngữ gốc để đảm bảo truy xuất chính xác
- Mặc định **không thực hiện dịch**
- Việc dịch chỉ được áp dụng **khi người dùng yêu cầu rõ ràng**
- Người dùng có thể yêu cầu:
  - Trả lời bằng ngôn ngữ mong muốn (ví dụ: tiếng Việt hoặc tiếng Anh)
  - Tạo subtitle đã dịch cho video
  - Tạo phiên bản tài liệu đã dịch hoặc song ngữ
- Quá trình dịch chỉ diễn ra ở **output layer** và **không ảnh hưởng tới vector index**


## 📁 Cấu trúc dự án

```
multimodal-qa/
├── configs/
│   ├── config.yaml              # Main config
│   └── models.yaml              # Model configs
│
├── data/
│   ├── input/
│   │   ├── documents/           # Tài liệu gốc
│   │   │   ├── docx/
│   │   │   ├── pdf/
│   │   │   └── txt/
│   │   ├── videos/              # Video gốc
│   │   │   ├── avi/
│   │   │   ├── mkv/
│   │   │   └── mp4/
│   │   └── optional/            # Tài liệu bổ sung
│   │
│   ├── output/
│   │   ├── documents/           # Documents đã extract
│   │   │   ├── docx/
│   │   │   ├── pdf/
│   │   │   └── txt/
│   │   ├── transcripts/         # Transcript từ video
│   │   │   ├── json/
│   │   │   └── txt/
│   │   ├── subtitles/           # Subtitle files
│   │   │   ├── srt/
│   │   │   └── vtt/
│   │   ├── translations/        # Nội dung đã dịch
│   │   │   ├── docx/
│   │   │   ├── pdf/
│   │   │   └── txt/
│   │   └── videos/              # Video đã xử lý
│   │
│   └── vector_db/               # ChromaDB storage
│
├── datasets/
│   ├── raw/                     # Raw datasets
│   │   ├── squad/
│   │   ├── viquad/
│   │   ├── xquad_en/
│   │   └── xquad_vi/
│   ├── processed/               # Processed datasets
│   └── splits/                  # Train/val/test splits
│
├── examples/
│   ├── documents/               # Example documents
│   └── videos/                  # Example videos
│
├── src/
│   ├── api/                     # FastAPI REST API
│   │   ├── routes/
│   │   │   ├── document.py      # Document endpoints
│   │   │   ├── video.py         # Video endpoints
│   │   │   └── qa.py            # QA endpoints
│   │   ├── schemas/
│   │   │   ├── document.py      # Document schemas
│   │   │   ├── video.py         # Video schemas
│   │   │   └── qa.py            # QA schemas
│   │   └── main.py              # FastAPI app
│   │
│   ├── extractors/
│   │   ├── video.py             # Video extraction
│   │   └── document.py          # PDF/DOCX extraction
│   │
│   ├── models/
│   │   ├── stt.py               # Whisper STT
│   │   ├── translation.py       # EN-VI translation
│   │   └── embedding.py         # Text embeddings
│   │
│   ├── pipelines/
│   │   ├── video_pipeline.py    # Video → Transcript + QA
│   │   ├── document_pipeline.py # Doc → Extract + QA
│   │   └── qa_pipeline.py       # Q&A system
│   │
│   ├── services/
│   │   ├── subtitle.py          # SRT/VTT generation
│   │   ├── vector_store.py      # ChromaDB operations
│   │   └── qa_engine.py         # Q&A engine
│   │
│   ├── core/
│   │   ├── chunking.py          # Text chunking
│   │   └── config.py            # Config loader
│   │
│   └── utils/
│       ├── logger.py            # Logging utilities
│       ├── helpers.py           # Helper functions
│       └── metrics.py           # Performance metrics
│
├── tests/
│   ├── test_extractors/         # Extractor tests
│   ├── test_pipelines/          # Pipeline tests
│   ├── test_services/           # Service tests
│   ├── conftest.py              # Pytest config
│   └── run_all_tests.py         # Test runner
│
├── scripts/
│   ├── check_datasets.py        # Dataset verification
│   ├── download_models.py       # Model downloader
│   ├── prepare_dataset.py       # Dataset preparation
│   └── setup_vectordb.py        # Vector DB setup
│
├── docker-compose.yml           # Docker compose config
├── Dockerfile                   # Docker image
├── main.py                      # Entry point
├── pyproject.toml               # Project dependencies
├── pytest.ini                   # Pytest configuration
└── README.md                    # This file
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





