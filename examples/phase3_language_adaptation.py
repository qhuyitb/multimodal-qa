"""
Example usage of Phase 3: Language Adaptation & UX Logic features.
Demonstrates language detection, translation, cross-language QA, and multilingual content generation.
"""

from pathlib import Path
import json

# Language Detection Examples
def example_language_detection():
    """Demonstrate language detection for text, audio, and documents."""
    print("\n=== Language Detection Examples ===\n")
    
    from src.services.language_detector import LanguageDetector
    
    detector = LanguageDetector()
    
    # Detect from text
    vietnamese_text = "Đây là một câu tiếng Việt. Xin chào các bạn."
    english_text = "This is an English sentence. Hello everyone."
    
    vi_result = detector.detect_from_text(vietnamese_text)
    print(f"Vietnamese text detection: {vi_result}")
    
    en_result = detector.detect_from_text(english_text)
    print(f"English text detection: {en_result}")
    
    # Detect from audio/video (requires Whisper)
    video_path = Path("data/input/videos/mp4/video_demo.mp4")
    if video_path.exists():
        audio_result = detector.detect_from_audio(video_path)
        print(f"\nVideo language detection: {audio_result}")
    
    # Detect from document
    doc_path = Path("data/input/documents/txt/sample.txt")
    if doc_path.exists():
        doc_result = detector.detect_from_document(doc_path)
        print(f"\nDocument language detection: {doc_result}")


# Translation Examples
def example_translation():
    """Demonstrate text translation between multiple language pairs."""
    print("\n=== Translation Examples ===\n")
    
    from src.models.translation import TranslationService
    
    service = TranslationService(device="cpu")
    
    # Single text translation
    english_text = "Hello, how are you today?"
    vietnamese = service.translate(english_text, "en", "vi")
    print(f"EN -> VI: {english_text}")
    print(f"Result: {vietnamese}\n")
    
    # Batch translation
    texts = [
        "Good morning",
        "Thank you very much",
        "See you later"
    ]
    translations = service.translate(texts, "en", "vi")
    print("Batch translation:")
    for orig, trans in zip(texts, translations):
        print(f"  {orig} -> {trans}")
    
    # Translation with timing
    result = service.translate_with_timing(
        "This is a longer text that needs to be translated for performance testing.",
        "en", "vi"
    )
    print(f"\nTranslation timing:")
    print(f"  Time: {result['elapsed_time']:.3f}s")
    print(f"  Speed: {result['chars_per_second']:.1f} chars/sec")
    
    # Check supported languages
    languages = service.get_supported_languages()
    print(f"\nSupported languages: {languages}")


# Cross-Language QA Examples
def example_cross_language_qa():
    """Demonstrate cross-language question answering."""
    print("\n=== Cross-Language QA Examples ===\n")
    
    from src.services.qa_engine import create_qa_engine
    from pathlib import Path
    
    # Create QA engine with multilingual support
    qa_engine = create_qa_engine(
        vector_store_path=Path("data/vector_db"),
        embedding_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        enable_translation=True
    )
    
    # Query in English about content (any language)
    print("Query 1: English question")
    result1 = qa_engine.query(
        question="What is the main topic of the document?",
        top_k=3
    )
    print(f"Answer: {result1['answer']}")
    print(f"Query language: {result1['query_language']}")
    print(f"Confidence: {result1['confidence']:.2f}\n")
    
    # Query in Vietnamese about English content with translation
    print("Query 2: Vietnamese question with English answer translation")
    result2 = qa_engine.query(
        question="Nội dung chính của tài liệu là gì?",
        source_language="en",
        target_language="vi",
        translate_answer=True,
        top_k=3
    )
    print(f"Answer: {result2['answer']}")
    print(f"Translated: {result2['translated']}")
    print(f"Query language: {result2['query_language']}\n")
    
    # Batch queries
    print("Query 3: Batch processing")
    questions = [
        "What is the document about?",
        "Who is the author?",
        "When was it written?"
    ]
    results = qa_engine.batch_query(questions, top_k=2)
    for q, r in zip(questions, results):
        print(f"Q: {q}")
        print(f"A: {r['answer'][:100]}...\n")


# Multilingual Subtitle Examples
def example_multilingual_subtitles():
    """Demonstrate multilingual subtitle generation."""
    print("\n=== Multilingual Subtitle Examples ===\n")
    
    from src.services.subtitle import SubtitleGenerator
    from src.models.translation import get_translation_service
    
    # Sample transcript
    transcript = [
        {"start": 0.0, "end": 3.5, "text": "Welcome to this video tutorial."},
        {"start": 3.5, "end": 7.2, "text": "Today we will learn about machine learning."},
        {"start": 7.2, "end": 11.0, "text": "Let's start with the basics."}
    ]
    
    # Create subtitle generator
    translation_service = get_translation_service()
    generator = SubtitleGenerator(translation_service=translation_service)
    
    # Generate original subtitles
    subtitles = generator.from_transcript(transcript)
    print(f"Generated {len(subtitles)} subtitle entries")
    
    # Save in SRT format
    output_dir = Path("data/output/subtitles")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    srt_path = output_dir / "example.en.srt"
    generator.save_srt(subtitles, srt_path)
    print(f"Saved SRT: {srt_path}")
    
    # Generate translated subtitles
    vi_subtitles = generator.translate_subtitles(
        subtitles,
        source_lang="en",
        target_lang="vi"
    )
    
    vi_srt_path = output_dir / "example.vi.srt"
    generator.save_srt(vi_subtitles, vi_srt_path)
    print(f"Saved translated SRT: {vi_srt_path}")
    
    # Generate dual-language subtitles
    dual_subtitles = generator.create_dual_language_subtitles(
        subtitles,
        vi_subtitles
    )
    
    dual_srt_path = output_dir / "example.en_vi.srt"
    generator.save_srt(dual_subtitles, dual_srt_path)
    print(f"Saved dual-language SRT: {dual_srt_path}")
    
    # Generate all formats at once
    subtitle_files = generator.generate_multilingual_subtitles(
        transcript=transcript,
        source_lang="en",
        target_langs=["vi", "zh"],
        output_dir=output_dir,
        base_name="example_multi",
        formats=["srt", "vtt"],
        include_dual=True
    )
    
    print(f"\nGenerated subtitle files:")
    for lang, paths in subtitle_files.items():
        print(f"  {lang}: {[str(p) for p in paths]}")


# Document Translation Examples
def example_document_translation():
    """Demonstrate document translation and processing."""
    print("\n=== Document Translation Examples ===\n")
    
    from src.pipelines.document_pipeline import create_document_pipeline
    
    # Create pipeline with translation
    pipeline = create_document_pipeline(
        vector_store_path=Path("data/vector_db"),
        enable_translation=True
    )
    
    # Process document with translation
    doc_path = Path("data/input/documents/txt/sample.txt")
    if doc_path.exists():
        result = pipeline.process(
            document_path=doc_path,
            target_language="vi",
            generate_translation=True,
            generate_dual_language=True,
            index_content=True,
            index_translation=False
        )
        
        print(f"Processing result:")
        print(f"  Success: {result['success']}")
        print(f"  Source language: {result['source_language']}")
        print(f"  Output files: {result['output_files']}")
        print(f"  Indexed: {result['indexed']}")


# Evaluation Metrics Examples
def example_evaluation_metrics():
    """Demonstrate evaluation metrics for translation and QA."""
    print("\n=== Evaluation Metrics Examples ===\n")
    
    from src.utils.metrics import (
        TranslationMetrics,
        CrossLanguageQAMetrics,
        print_translation_metrics
    )
    
    # Translation quality metrics
    print("Translation Quality Evaluation:")
    reference = "Hello, how are you today?"
    candidate = "Hi, how are you doing today?"
    
    bleu = TranslationMetrics.bleu_score(reference, candidate)
    cer = TranslationMetrics.character_error_rate(reference, candidate)
    
    print(f"  BLEU Score: {bleu:.4f}")
    print(f"  Character Error Rate: {cer:.4f}\n")
    
    # Comprehensive evaluation
    eval_result = TranslationMetrics.evaluate_translation_quality(
        source_text="This is a test sentence.",
        translated_text="Đây là một câu thử nghiệm.",
        reference_translation="Đây là câu kiểm tra.",
        elapsed_time=0.5
    )
    
    print_translation_metrics(eval_result, "Translation Evaluation")
    
    # QA correctness metrics
    print("\n\nQA Correctness Evaluation:")
    qa_result = CrossLanguageQAMetrics.evaluate_qa_correctness(
        question="What is AI?",
        answer="Artificial Intelligence is the simulation of human intelligence.",
        ground_truth="AI is simulation of human intelligence in machines.",
        language="en"
    )
    
    print(f"  Exact Match: {qa_result['exact_match']:.2f}")
    print(f"  Token F1: {qa_result['token_f1']:.4f}")
    print(f"  Char Similarity: {qa_result['char_similarity']:.4f}")
    
    # Translation overhead analysis
    print("\n\nTranslation Overhead Analysis:")
    overhead = CrossLanguageQAMetrics.translation_overhead_analysis(
        baseline_latency=1.0,
        with_translation_latency=2.5,
        translation_only_latency=1.3
    )
    
    print(f"  Baseline: {overhead['baseline_latency']:.2f}s")
    print(f"  With Translation: {overhead['with_translation_latency']:.2f}s")
    print(f"  Overhead: {overhead['overhead_percentage']:.1f}%")


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3: Language Adaptation & UX Logic - Examples")
    print("=" * 60)
    
    # Run all examples
    try:
        example_language_detection()
    except Exception as e:
        print(f"Error in language detection: {e}")
    
    try:
        example_translation()
    except Exception as e:
        print(f"Error in translation: {e}")
    
    try:
        example_cross_language_qa()
    except Exception as e:
        print(f"Error in cross-language QA: {e}")
    
    try:
        example_multilingual_subtitles()
    except Exception as e:
        print(f"Error in multilingual subtitles: {e}")
    
    try:
        example_document_translation()
    except Exception as e:
        print(f"Error in document translation: {e}")
    
    try:
        example_evaluation_metrics()
    except Exception as e:
        print(f"Error in evaluation metrics: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
