from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from src.api.schemas.qa import (
    QARequest,
    QAResponse,
    BatchQARequest,
    BatchQAResponse,
    SourceInfo
)
from src.services.qa_engine import QAEngine, create_qa_engine
from src.core.config import get_config

router = APIRouter(prefix="/qa", tags=["Question Answering"])

_qa_engine: Optional[QAEngine] = None


def get_qa_engine() -> QAEngine:
    global _qa_engine
    if _qa_engine is None:
        config = get_config()
        vector_store_path = config.get("paths", {}).get("vector_db", "data/vector_db")
        embedding_model = config.get("cross_language_qa", {}).get(
            "embedding_model",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        enable_translation = config.get("translation", {}).get("enabled", True)
        
        _qa_engine = create_qa_engine(
            vector_store_path=vector_store_path,
            embedding_model_name=embedding_model,
            enable_translation=enable_translation
        )
    return _qa_engine


@router.post("/ask", response_model=QAResponse)
async def ask_question(request: QARequest, qa_engine: QAEngine = Depends(get_qa_engine)):
    result = qa_engine.query(
        question=request.question,
        top_k=request.top_k,
        source_language=request.source_language,
        target_language=request.target_language,
        translate_answer=request.translate_answer,
        filter_metadata=request.filters
    )
    
    sources = [
        SourceInfo(
            text=src["text"],
            score=src["score"],
            metadata=src["metadata"],
            original_text=src.get("original_text")
        )
        for src in result["sources"]
    ]
    
    return QAResponse(
        answer=result["answer"],
        sources=sources,
        confidence=result["confidence"],
        query_language=result.get("query_language"),
        source_language=result.get("source_language"),
        target_language=result.get("target_language"),
        translated=result["translated"]
    )


@router.post("/batch", response_model=BatchQAResponse)
async def batch_ask_questions(request: BatchQARequest, qa_engine: QAEngine = Depends(get_qa_engine)):
    results = qa_engine.batch_query(
        questions=request.questions,
        top_k=request.top_k,
        source_language=request.source_language,
        target_language=request.target_language,
        translate_answer=request.translate_answers
    )
    
    qa_responses = []
    for result in results:
        sources = [
            SourceInfo(
                text=src["text"],
                score=src["score"],
                metadata=src["metadata"],
                original_text=src.get("original_text")
            )
            for src in result["sources"]
        ]
        
        qa_response = QAResponse(
            answer=result["answer"],
            sources=sources,
            confidence=result["confidence"],
            query_language=result.get("query_language"),
            source_language=result.get("source_language"),
            target_language=result.get("target_language"),
            translated=result["translated"]
        )
        qa_responses.append(qa_response)
    
    return BatchQAResponse(
        results=qa_responses,
        total_questions=len(request.questions)
    )


@router.get("/supported-languages")
async def get_supported_languages(qa_engine: QAEngine = Depends(get_qa_engine)):
    if qa_engine.translation_service:
        languages = qa_engine.translation_service.get_supported_languages()
        return {
            "supported_languages": languages,
            "translation_enabled": True
        }
    else:
        return {
            "supported_languages": [],
            "translation_enabled": False
        }
