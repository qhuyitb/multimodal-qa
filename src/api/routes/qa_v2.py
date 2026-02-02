"""
Production QA API with hybrid retrieval and adaptive language handling
"""
from fastapi import APIRouter, HTTPException, Depends, status
from typing import Optional
from pathlib import Path

from ..schemas.qa import QARequest, QAResponse, SourceInfo
from ...services.adaptive_qa import create_adaptive_qa_service
from ...services.hybrid_retrieval import HybridRetriever
from ...core.config import get_config


router = APIRouter(prefix="/api/v1/qa", tags=["Question Answering V2"])

_adaptive_qa = None
_hybrid_retrieval = None


def get_adaptive_qa():
    """Get or create adaptive QA service"""
    global _adaptive_qa
    if _adaptive_qa is None:
        config = get_config()
        model_path = config.get("model", {}).get("path", "models/xlm_roberta_qa/stage2_best")
        _adaptive_qa = create_adaptive_qa_service(
            model_path=str(model_path),
            device="cpu",
            default_language="vi",
            auto_translate=True
        )
    return _adaptive_qa


def get_hybrid_retrieval():
    """Get or create hybrid retrieval service"""
    global _hybrid_retrieval
    if _hybrid_retrieval is None:
        config = get_config()
        vector_db_path = config.get("paths", {}).get("vector_db", "data/vector_db")
        embedding_model = config.get("cross_language_qa", {}).get(
            "embedding_model",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        _hybrid_retrieval = HybridRetriever(
            embedding_model=embedding_model,
            chroma_persist_dir=str(vector_db_path)
        )
    return _hybrid_retrieval


@router.post("/ask", response_model=QAResponse, status_code=status.HTTP_200_OK)
async def ask_question(
    request: QARequest,
    adaptive_qa = Depends(get_adaptive_qa),
    hybrid_retrieval = Depends(get_hybrid_retrieval)
):
    """
    Answer a question using hybrid retrieval and adaptive QA
    
    - **question**: Question in any language (EN/VI)
    - **top_k**: Number of context chunks to retrieve (default: 5)
    - **source_language**: Override language detection
    - **target_language**: Translate answer to this language
    """
    try:
        # Hybrid retrieval to get relevant contexts
        search_results = hybrid_retrieval.hybrid_search(
            query=request.question,
            top_k=request.top_k or 5,
            alpha=0.5  # Balance BM25 and semantic
        )
        
        if not search_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant information found in the knowledge base"
            )
        
        # Combine top contexts
        context = " ".join([r["text"] for r in search_results[:3]])
        
        # Get answer with adaptive language handling
        result = adaptive_qa.answer(
            question=request.question,
            context=context,
            target_language=request.target_language,
            source_language=request.source_language
        )
        
        # Format sources
        sources = [
            SourceInfo(
                text=src["text"],
                score=src["score"],
                metadata=src.get("metadata", {})
            )
            for src in search_results
        ]
        
        return QAResponse(
            answer=result.answer,
            sources=sources,
            confidence=result.score,
            query_language=result.source_language,
            source_language=result.source_language,
            target_language=result.target_language,
            translated=result.translated
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check(
    adaptive_qa = Depends(get_adaptive_qa),
    hybrid_retrieval = Depends(get_hybrid_retrieval)
):
    """Health check for QA service"""
    try:
        stats = adaptive_qa.get_stats()
        return {
            "status": "healthy",
            "model_loaded": True,
            "hybrid_retrieval_ready": hybrid_retrieval is not None,
            "total_queries": stats.get("total_queries", 0),
            "languages_detected": list(stats.get("by_language", {}).keys())
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@router.get("/stats", status_code=status.HTTP_200_OK)
async def get_stats(adaptive_qa = Depends(get_adaptive_qa)):
    """Get QA service statistics"""
    try:
        stats = adaptive_qa.get_stats()
        return {
            "total_queries": stats.get("total_queries", 0),
            "translations": stats.get("translations", 0),
            "by_language": stats.get("by_language", {}),
            "avg_translation_time_ms": stats.get("avg_translation_time", 0),
            "avg_inference_time_ms": stats.get("avg_inference_time", 0)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stats: {str(e)}"
        )
