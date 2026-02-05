"""
Generative QA Service - Combines RAG + LLM
Hybrid approach: Extractive (fast) + Generative (comprehensive)
"""
from typing import Dict, Optional, Any
from dataclasses import dataclass
import time

from ..models.llm import get_llm_service, LLMService


@dataclass
class GenerativeQAResult:
    """Result from generative QA"""
    answer: str
    score: float
    context: str
    sources: list
    generation_time_ms: float
    method: str  # 'extractive', 'generative', 'hybrid'
    source_language: str
    target_language: str
    translated: bool


class GenerativeQAService:
    """
    Generative QA combining extractive and LLM-based approaches
    """
    
    def __init__(
        self,
        extractive_qa_service,
        llm_service: Optional[LLMService] = None,
        enable_llm: bool = True,
        llm_threshold: float = 0.5,
        model_name: str = "qwen2.5:7b",
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize generative QA service
        
        Args:
            extractive_qa_service: AdaptiveQAService for extractive QA
            llm_service: LLM service instance (will create if None)
            enable_llm: Use LLM for generation
            llm_threshold: Confidence threshold to use LLM (if extractive score < threshold)
            model_name: LLM model name
        """
        self.extractive_qa = extractive_qa_service
        self.enable_llm = enable_llm
        self.llm_threshold = llm_threshold
        
        # Initialize LLM service
        if enable_llm:
            if llm_service is None:
                print(f"Initializing Ollama LLM: {model_name}")
                self.llm = get_llm_service(
                    model_name=model_name,
                    base_url=ollama_base_url
                )
            else:
                self.llm = llm_service
        else:
            self.llm = None
        
        self.stats = {
            "total_queries": 0,
            "extractive_only": 0,
            "generative_only": 0,
            "hybrid": 0
        }
    
    def answer(
        self,
        question: str,
        context: str,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        force_generative: bool = True,  # Always use generative by default
        max_new_tokens: int = 256,
        temperature: float = 0.3
    ) -> GenerativeQAResult:
        """
        Answer question with generative LLM
        
        Args:
            question: User question
            context: Retrieved context
            source_language: Source language
            target_language: Target language  
            force_generative: Always use LLM (default True)
            max_new_tokens: Max tokens for generation
            temperature: LLM temperature
            
        Returns:
            GenerativeQAResult with answer and metadata
        """
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        # Use LLM for generation (integrated directly into RAG)
        if self.enable_llm and self.llm:
            try:
                llm_answer = self.llm.generate_with_context(
                    question=question,
                    context=context,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                
                self.stats["generative_only"] += 1
                
                return GenerativeQAResult(
                    answer=llm_answer,
                    score=0.85,  # Fixed high score for LLM answers
                    context=context,
                    sources=[],
                    generation_time_ms=(time.time() - start_time) * 1000,
                    method="generative",
                    source_language=source_language or "vi",
                    target_language=target_language or "vi",
                    translated=False
                )
                
            except Exception as e:
                print(f"LLM generation error: {e}")
                # Fallback to extractive if LLM fails
                extractive_result = self.extractive_qa.answer(
                    question=question,
                    context=context,
                    source_language=source_language,
                    target_language=target_language
                )
                self.stats["extractive_only"] += 1
                return GenerativeQAResult(
                    answer=extractive_result.answer,
                    score=extractive_result.score,
                    context=context,
                    sources=[],
                    generation_time_ms=(time.time() - start_time) * 1000,
                    method="extractive_fallback",
                    source_language=extractive_result.source_language,
                    target_language=extractive_result.target_language,
                    translated=extractive_result.translated
                )
        
        # LLM disabled - use extractive fallback
        extractive_result = self.extractive_qa.answer(
            question=question,
            context=context,
            source_language=source_language,
            target_language=target_language
        )
        self.stats["extractive_only"] += 1
        return GenerativeQAResult(
            answer=extractive_result.answer,
            score=extractive_result.score,
            context=context,
            sources=[],
            generation_time_ms=(time.time() - start_time) * 1000,
            method="extractive_fallback",
            source_language=extractive_result.source_language,
            target_language=extractive_result.target_language,
            translated=extractive_result.translated
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            **self.stats,
            "llm_enabled": self.enable_llm,
            "llm_threshold": self.llm_threshold
        }


def create_generative_qa_service(
    extractive_qa_service,
    enable_llm: bool = True,
    model_name: str = "qwen2.5:7b",
    ollama_base_url: str = "http://localhost:11434",
    **kwargs
) -> GenerativeQAService:
    """Factory function to create generative QA service"""
    return GenerativeQAService(
        extractive_qa_service=extractive_qa_service,
        enable_llm=enable_llm,
        model_name=model_name,
        ollama_base_url=ollama_base_url,
        **kwargs
    )
