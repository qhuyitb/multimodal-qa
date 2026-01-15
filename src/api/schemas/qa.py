from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class QARequest(BaseModel):
    
    question: str = Field(..., description="Question to answer")
    top_k: Optional[int] = Field(5, description="Number of results to retrieve")
    source_language: Optional[str] = Field(None, description="Language of indexed content")
    target_language: Optional[str] = Field(None, description="Target language for answer translation")
    translate_answer: bool = Field(False, description="Whether to translate the answer")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters for search")


class SourceInfo(BaseModel):
    
    text: str = Field(..., description="Source text excerpt")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source metadata")
    original_text: Optional[str] = Field(None, description="Original text before translation")


class QAResponse(BaseModel):
    
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceInfo] = Field(default_factory=list, description="Source documents")
    confidence: float = Field(..., description="Confidence score")
    query_language: Optional[str] = Field(None, description="Detected query language")
    source_language: Optional[str] = Field(None, description="Language of source content")
    target_language: Optional[str] = Field(None, description="Target language (if translated)")
    translated: bool = Field(False, description="Whether answer was translated")


class BatchQARequest(BaseModel):
    
    questions: List[str] = Field(..., description="List of questions")
    top_k: Optional[int] = Field(5, description="Number of results per question")
    source_language: Optional[str] = Field(None, description="Language of indexed content")
    target_language: Optional[str] = Field(None, description="Target language for translations")
    translate_answers: bool = Field(False, description="Whether to translate answers")


class BatchQAResponse(BaseModel):
    
    results: List[QAResponse] = Field(..., description="List of QA results")
    total_questions: int = Field(..., description="Total number of questions processed")
