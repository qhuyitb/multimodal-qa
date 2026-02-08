from typing import Dict, List, Optional, Any
from pathlib import Path


class QAEngine:
    """Engine QA chính kết hợp vector search, translation và language detection"""
    
    def __init__(
        self,
        vector_store,
        embedding_model=None,
        translation_service=None,
        language_detector=None,
        default_top_k: int = 5
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.translation_service = translation_service
        self.language_detector = language_detector
        self.default_top_k = default_top_k
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        translate_answer: bool = False,
        filter_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        if top_k is None:
            top_k = self.default_top_k
        
        query_language = None
        if self.language_detector:
            detection = self.language_detector.detect_from_text(question)
            query_language = detection.get("language", "unknown")
        
        search_results = self.vector_store.search(
            query=question,
            top_k=top_k,
            filters=filter_metadata
        )
        
        if not search_results:
            return {
                "answer": "No relevant information found.",
                "sources": [],
                "confidence": 0.0,
                "query_language": query_language,
                "source_language": source_language,
                "translated": False
            }
        
        top_result = search_results[0]
        answer_text = top_result.get("text", "")
        confidence = top_result.get("score", 0.0)
        
        sources = []
        for result in search_results:
            source_info = {
                "text": result.get("text", ""),
                "score": result.get("score", 0.0),
                "metadata": result.get("metadata", {})
            }
            sources.append(source_info)
        
        translated = False
        if translate_answer and target_language and self.translation_service:
            if not source_language and self.language_detector:
                detection = self.language_detector.detect_from_text(answer_text)
                source_language = detection.get("language", "en")
            
            if source_language and source_language != target_language:
                answer_text = self.translation_service.translate(
                    answer_text,
                    source_lang=source_language,
                    target_lang=target_language
                )
                translated = True
                
                for source in sources:
                    source["original_text"] = source["text"]
                    source["text"] = self.translation_service.translate(
                        source["text"],
                        source_lang=source_language,
                        target_lang=target_language
                    )
        
        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": confidence,
            "query_language": query_language,
            "source_language": source_language,
            "target_language": target_language if translated else None,
            "translated": translated
        }
    
    def batch_query(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        results = []
        for question in questions:
            result = self.query(question, **kwargs)
            results.append(result)
        return results
    
    def query_with_context(
        self,
        question: str,
        context: str,
        target_language: Optional[str] = None
    ) -> Dict[str, Any]:
        answer_text = context
        
        translated = False
        if target_language and self.translation_service and self.language_detector:
            detection = self.language_detector.detect_from_text(answer_text)
            source_language = detection.get("language", "en")
            
            if source_language != target_language:
                answer_text = self.translation_service.translate(
                    answer_text,
                    source_lang=source_language,
                    target_lang=target_language
                )
                translated = True
        
        return {
            "answer": answer_text,
            "context": context,
            "translated": translated,
            "target_language": target_language if translated else None
        }
    
    def get_similar_questions(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.vector_store.search(
            query=question,
            top_k=top_k
        )
        
        return [
            {
                "question": result.get("text", ""),
                "score": result.get("score", 0.0),
                "metadata": result.get("metadata", {})
            }
            for result in results
        ]


def create_qa_engine(
    vector_store_path: Path,
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    enable_translation: bool = True,
    **kwargs
) -> QAEngine:
    from services.vector_store import VectorStore
    from models.embedding import EmbeddingModel
    from models.translation import get_translation_service
    from services.language_detector import LanguageDetector
    
    vector_store = VectorStore(vector_store_path)
    embedding_model = EmbeddingModel(model_name=embedding_model_name)
    
    translation_service = None
    if enable_translation:
        translation_service = get_translation_service()
    
    language_detector = LanguageDetector()
    
    return QAEngine(
        vector_store=vector_store,
        embedding_model=embedding_model,
        translation_service=translation_service,
        language_detector=language_detector,
        **kwargs
    )
