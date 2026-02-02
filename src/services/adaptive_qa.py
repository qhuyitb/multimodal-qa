"""
Adaptive QA Service with Language Detection and On-demand Translation
Automatically detects language and routes questions appropriately
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
import time

from ..services.language_detector import LanguageDetector
from ..models.translation import TranslationService, get_translation_service
from ..models.qa_model import XLMRobertaQA


@dataclass
class QAResult:
    """QA result with language metadata"""
    answer: str
    score: float
    context: str
    start_char: int
    end_char: int
    source_language: str
    target_language: str
    translated: bool
    translation_time_ms: float = 0.0
    inference_time_ms: float = 0.0


class AdaptiveQAService:
    """
    Adaptive QA service that handles multilingual questions
    
    Features:
    - Automatic language detection
    - On-demand translation (question/answer)
    - Cross-lingual consistency
    - Performance tracking
    """
    
    def __init__(
        self,
        qa_model: XLMRobertaQA,
        translation_service: Optional[TranslationService] = None,
        language_detector: Optional[LanguageDetector] = None,
        default_language: str = "vi",
        auto_translate: bool = True
    ):
        self.qa_model = qa_model
        self.translation_service = translation_service or get_translation_service()
        self.language_detector = language_detector or LanguageDetector()
        self.default_language = default_language
        self.auto_translate = auto_translate
        
        # Performance tracking
        self.stats = {
            "total_queries": 0,
            "by_language": {},
            "translations": 0,
            "avg_translation_time": 0.0,
            "avg_inference_time": 0.0
        }
    
    def answer(
        self,
        question: str,
        context: str,
        target_language: Optional[str] = None,
        source_language: Optional[str] = None,
        return_original: bool = False
    ) -> QAResult:
        """
        Answer question with automatic language adaptation
        
        Args:
            question: Question text (any language)
            context: Context text (assumed to be in default_language)
            target_language: Desired answer language (auto-detect if None)
            source_language: Source language of question (skip detection if provided)
            return_original: Return answer in original language before translation
        
        Returns:
            QAResult with answer and metadata
        """
        start_time = time.time()
        
        # Detect question language
        question_lang = source_language if source_language else self._detect_language(question)
        
        # Determine target language
        if target_language is None:
            target_language = question_lang
        
        # Update stats
        self.stats["total_queries"] += 1
        self.stats["by_language"][question_lang] = self.stats["by_language"].get(question_lang, 0) + 1
        
        # Translate question if needed
        translated_question = question
        translation_time = 0.0
        
        if self.auto_translate and question_lang != self.default_language:
            trans_start = time.time()
            translated_question = self.translation_service.translate(
                question,
                source_lang=question_lang,
                target_lang=self.default_language
            )
            translation_time = (time.time() - trans_start) * 1000
            self.stats["translations"] += 1
        
        # Get answer from QA model
        inference_start = time.time()
        qa_outputs = self.qa_model.predict(translated_question, context)
        inference_time = (time.time() - inference_start) * 1000
        
        # Get best answer
        qa_output = qa_outputs[0] if qa_outputs else {"text": "", "score": 0.0, "start": 0, "end": 0}
        answer = qa_output["text"]
        
        # Translate answer back if needed
        if target_language != self.default_language:
            trans_start = time.time()
            answer = self.translation_service.translate(
                answer,
                source_lang=self.default_language,
                target_lang=target_language
            )
            translation_time += (time.time() - trans_start) * 1000
            self.stats["translations"] += 1
        
        # Update average times
        total_time = (time.time() - start_time) * 1000
        self._update_avg_time(translation_time, inference_time)
        
        return QAResult(
            answer=answer,
            score=qa_output["score"],
            context=context,
            start_char=qa_output.get("start", 0),
            end_char=qa_output.get("end", len(answer)),
            source_language=question_lang,
            target_language=target_language,
            translated=(question_lang != self.default_language or target_language != self.default_language),
            translation_time_ms=translation_time,
            inference_time_ms=inference_time
        )
    
    def batch_answer(
        self,
        questions: List[str],
        contexts: List[str],
        target_language: Optional[str] = None
    ) -> List[QAResult]:
        """
        Answer multiple questions in batch
        
        More efficient than calling answer() multiple times
        """
        # Detect all languages
        languages = [self._detect_language(q) for q in questions]
        
        # Group by language for batch translation
        lang_groups = {}
        for i, (q, lang) in enumerate(zip(questions, languages)):
            if lang not in lang_groups:
                lang_groups[lang] = []
            lang_groups[lang].append((i, q))
        
        # Translate questions if needed
        translated_questions = [None] * len(questions)
        for lang, group in lang_groups.items():
            indices, qs = zip(*group)
            
            if self.auto_translate and lang != self.default_language:
                trans_qs = self.translation_service.translate(
                    list(qs),
                    source_lang=lang,
                    target_lang=self.default_language
                )
            else:
                trans_qs = qs
            
            for idx, trans_q in zip(indices, trans_qs):
                translated_questions[idx] = trans_q
        
        # Run QA inference
        results = []
        for i, (question, context) in enumerate(zip(translated_questions, contexts)):
            result = self.answer(question, context, target_language)
            results.append(result)
        
        return results
    
    def cross_lingual_consistency_check(
        self,
        question_en: str,
        question_vi: str,
        context: str,
        threshold: float = 0.8
    ) -> Dict:
        """
        Check consistency between English and Vietnamese questions
        
        Args:
            question_en: Question in English
            question_vi: Question in Vietnamese
            context: Context (in Vietnamese)
            threshold: Similarity threshold for consistency
        
        Returns:
            Dictionary with consistency metrics
        """
        # Get answers for both languages
        result_en = self.answer(question_en, context, target_language="en", source_language="en")
        result_vi = self.answer(question_vi, context, target_language="vi", source_language="vi")
        
        # Translate EN answer to VI for comparison
        answer_en_in_vi = self.translation_service.translate(
            result_en.answer,
            source_lang="en",
            target_lang="vi"
        )
        
        # Calculate similarity (simple word overlap for now)
        similarity = self._calculate_similarity(answer_en_in_vi, result_vi.answer)
        
        return {
            "consistent": similarity >= threshold,
            "similarity": similarity,
            "answer_en": result_en.answer,
            "answer_vi": result_vi.answer,
            "answer_en_translated": answer_en_in_vi,
            "score_en": result_en.score,
            "score_vi": result_vi.score,
            "score_gap": abs(result_en.score - result_vi.score)
        }
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            **self.stats,
            "languages_supported": list(self.stats["by_language"].keys()),
            "most_common_language": max(
                self.stats["by_language"],
                key=self.stats["by_language"].get
            ) if self.stats["by_language"] else None
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_queries": 0,
            "by_language": {},
            "translations": 0,
            "avg_translation_time": 0.0,
            "avg_inference_time": 0.0
        }
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text"""
        result = self.language_detector.detect_from_text(text)
        
        # Map common language codes
        lang = result.get("language", "unknown")
        if lang in ["vie", "vi"]:
            return "vi"
        elif lang in ["eng", "en", "tl"]:  # tl (Tagalog) often misclassified English
            return "en"
        
        return lang
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate word-level similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _update_avg_time(self, translation_time: float, inference_time: float):
        """Update running average times"""
        n = self.stats["total_queries"]
        
        # Running average formula: avg_new = (avg_old * (n-1) + new_value) / n
        self.stats["avg_translation_time"] = (
            self.stats["avg_translation_time"] * (n - 1) + translation_time
        ) / n
        
        self.stats["avg_inference_time"] = (
            self.stats["avg_inference_time"] * (n - 1) + inference_time
        ) / n


def create_adaptive_qa_service(
    model_path: str,
    device: str = "cpu",
    **kwargs
) -> AdaptiveQAService:
    """
    Factory function to create AdaptiveQAService
    
    Args:
        model_path: Path to QA model
        device: Device for inference
        **kwargs: Additional arguments for AdaptiveQAService
    
    Returns:
        Configured AdaptiveQAService instance
    """
    from ..models.qa_model import create_qa_model
    
    qa_model = create_qa_model(model_path, device=device)
    translation_service = get_translation_service(device=device)
    language_detector = LanguageDetector()
    
    return AdaptiveQAService(
        qa_model=qa_model,
        translation_service=translation_service,
        language_detector=language_detector,
        **kwargs
    )
