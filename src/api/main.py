"""
Multimodal QA API - FastAPI Application
Endpoints:
- /health - Health check
- /qa/* - Question answering endpoints
- /video/* - Video processing endpoints  
- /document/* - Document processing endpoints
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import time
from pathlib import Path

from api.routes import qa, video, document
from core.config import get_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    print(f"API started - Vector DB: {config.get('paths', {}).get('vector_db', 'data/vector_db')}")
    yield
    print("API stopped")

app = FastAPI(
    title="Multimodal QA API",
    description="API for multilingual Question Answering on videos and documents",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    if response.status_code >= 400:
        print(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s")
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error: {exc.errors()}")
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                       content={"detail": exc.errors(), "body": exc.body})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"Unhandled error: {str(exc)}")
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                       content={"detail": "Internal server error", "error": str(exc)})

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "qa": "available",
            "video_processing": "available",
            "document_processing": "available"
        }
    }


@app.get("/", tags=["System"])
async def root():
    """Endpoint gốc của API"""
    return {
        "message": "Multimodal QA API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Include routers
app.include_router(qa.router)
app.include_router(video.router)
app.include_router(document.router)
