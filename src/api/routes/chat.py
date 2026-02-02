"""
Conversational QA API endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, status
from typing import Optional, List
from pydantic import BaseModel, Field

from ...services.conversational_qa import create_conversational_qa
from ...services.adaptive_qa import create_adaptive_qa_service
from ...services.hybrid_retrieval import HybridRetriever
from ...core.config import get_config


router = APIRouter(prefix="/api/v1/chat", tags=["Conversational QA"])

_conversational_qa = None


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID (auto-create if not provided)")
    question: str = Field(..., description="User question")
    top_k: Optional[int] = Field(5, description="Number of context chunks")
    use_context: bool = Field(True, description="Use conversation history")
    source_language: Optional[str] = Field(None, description="Source language override")
    target_language: Optional[str] = Field(None, description="Target language for answer")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[dict]
    confidence: float
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    translated: bool = False
    reformulated_question: str
    used_context: bool
    conversation_length: int


class SessionCreateRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Custom session ID")
    metadata: Optional[dict] = Field(None, description="Session metadata")


class SessionResponse(BaseModel):
    session_id: str
    message_count: int
    created_at: str
    updated_at: str
    metadata: dict


def get_conversational_qa():
    """Get or create conversational QA service"""
    global _conversational_qa
    if _conversational_qa is None:
        config = get_config()
        
        # Create adaptive QA
        model_path = config.get("model", {}).get("path", "models/xlm_roberta_qa/stage2_best")
        adaptive_qa = create_adaptive_qa_service(
            model_path=str(model_path),
            device="cpu",
            default_language="vi",
            auto_translate=True
        )
        
        # Create hybrid retrieval
        vector_db_path = config.get("paths", {}).get("vector_db", "data/vector_db")
        embedding_model = config.get("cross_language_qa", {}).get(
            "embedding_model",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        hybrid_retrieval = HybridRetriever(
            embedding_model=embedding_model,
            chroma_persist_dir=str(vector_db_path)
        )
        
        _conversational_qa = create_conversational_qa(adaptive_qa, hybrid_retrieval)
    
    return _conversational_qa


@router.post("/", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    request: ChatRequest,
    conversational_qa = Depends(get_conversational_qa)
):
    """
    Chat with conversational context
    
    Supports follow-up questions with conversation history
    """
    try:
        result = conversational_qa.chat(
            session_id=request.session_id or "",
            question=request.question,
            top_k=request.top_k or 5,
            use_context=request.use_context,
            source_language=request.source_language,
            target_language=request.target_language
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat: {str(e)}"
        )


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: SessionCreateRequest,
    conversational_qa = Depends(get_conversational_qa)
):
    """Create new conversation session"""
    try:
        session_id = conversational_qa.create_session(
            session_id=request.session_id,
            metadata=request.metadata
        )
        session = conversational_qa.get_session(session_id)
        
        return SessionResponse(
            session_id=session.session_id,
            message_count=len(session.messages),
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            metadata=session.metadata
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating session: {str(e)}"
        )


@router.get("/sessions", response_model=List[SessionResponse])
async def list_sessions(conversational_qa = Depends(get_conversational_qa)):
    """List all active sessions"""
    try:
        sessions = conversational_qa.list_sessions()
        return [SessionResponse(**s) for s in sessions]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing sessions: {str(e)}"
        )


@router.get("/sessions/{session_id}/history")
async def get_history(
    session_id: str,
    last_n: Optional[int] = None,
    conversational_qa = Depends(get_conversational_qa)
):
    """Get conversation history for session"""
    try:
        history = conversational_qa.get_conversation_history(session_id, last_n)
        if history is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        return {"session_id": session_id, "messages": history}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving history: {str(e)}"
        )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    conversational_qa = Depends(get_conversational_qa)
):
    """Delete conversation session"""
    try:
        success = conversational_qa.delete_session(session_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting session: {str(e)}"
        )


@router.post("/sessions/{session_id}/clear", status_code=status.HTTP_200_OK)
async def clear_session(
    session_id: str,
    conversational_qa = Depends(get_conversational_qa)
):
    """Clear conversation history but keep session"""
    try:
        success = conversational_qa.clear_session(session_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        return {"session_id": session_id, "status": "cleared"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing session: {str(e)}"
        )
