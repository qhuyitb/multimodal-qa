"""
Cross-lingual QA Benchmark
Tests consistency between English and Vietnamese questions
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_from_disk
from src.services.adaptive_qa import create_adaptive_qa_service
from typing import Dict, List
import numpy as np
import time


def load_parallel_dataset():
    """Load XQuAD parallel dataset (EN-VI)"""
    xquad_en = load_from_disk(project_root / "datasets/processed/xquad_en_normalized")
    xquad_vi = load_from_disk(project_root / "datasets/processed/xquad_vi_normalized")
    
    # Create parallel pairs
    parallel_data = []
    for i, (en_item, vi_item) in enumerate(zip(xquad_en["validation"], xquad_vi["validation"])):
        parallel_data.append({
            "id": en_item["id"],
            "question_en": en_item["question"],
            "question_vi": vi_item["question"],
            "context_en": en_item["context"],
            "context_vi": vi_item["context"],
            "answers_en": en_item["answers"],
            "answers_vi": vi_item["answers"]
        })
    
    return parallel_data


def evaluate_cross_lingual_consistency(
    adaptive_qa,
    parallel_data: List[Dict],
    max_samples: int = None
) -> Dict:
    """
    Evaluate cross-lingual consistency
    
    Metrics:
    - Consistency rate: % of questions with consistent answers
    - Similarity score: Average similarity between EN and VI answers
    - Score gap: Difference in confidence scores
    """
    if max_samples:
        parallel_data = parallel_data[:max_samples]
    
    print(f"\n🔍 Evaluating cross-lingual consistency on {len(parallel_data)} samples...\n")
    
    results = {
        "consistent": 0,
        "inconsistent": 0,
        "similarities": [],
        "score_gaps": [],
        "translation_times": [],
        "inference_times": []
    }
    
    for i, item in enumerate(parallel_data, 1):
        print(f"  Sample {i}/{len(parallel_data)}...", end="\r")
        
        # Check consistency using Vietnamese context
        consistency = adaptive_qa.cross_lingual_consistency_check(
            question_en=item["question_en"],
            question_vi=item["question_vi"],
            context=item["context_vi"],
            threshold=0.7
        )
        
        if consistency["consistent"]:
            results["consistent"] += 1
        else:
            results["inconsistent"] += 1
        
        results["similarities"].append(consistency["similarity"])
        results["score_gaps"].append(consistency["score_gap"])
    
    print()  # New line after progress
    
    # Calculate metrics
    total = len(parallel_data)
    consistency_rate = results["consistent"] / total
    avg_similarity = np.mean(results["similarities"])
    avg_score_gap = np.mean(results["score_gaps"])
    
    # Get service stats
    stats = adaptive_qa.get_stats()
    
    return {
        "consistency_rate": consistency_rate,
        "avg_similarity": avg_similarity,
        "avg_score_gap": avg_score_gap,
        "consistent_count": results["consistent"],
        "inconsistent_count": results["inconsistent"],
        "total_samples": total,
        "avg_translation_time_ms": stats["avg_translation_time"],
        "avg_inference_time_ms": stats["avg_inference_time"],
        "total_translations": stats["translations"]
    }


def print_results(results: Dict):
    """Pretty print results"""
    print("\n" + "="*80)
    print("  CROSS-LINGUAL QA BENCHMARK RESULTS")
    print("="*80 + "\n")
    
    print(f"📊 Consistency Metrics:")
    print(f"  Consistency Rate:     {results['consistency_rate']:.2%}")
    print(f"  Avg Similarity:       {results['avg_similarity']:.4f}")
    print(f"  Avg Score Gap:        {results['avg_score_gap']:.4f}")
    print()
    
    print(f"📈 Counts:")
    print(f"  Consistent:           {results['consistent_count']}/{results['total_samples']}")
    print(f"  Inconsistent:         {results['inconsistent_count']}/{results['total_samples']}")
    print()
    
    print(f"⚡ Performance:")
    print(f"  Avg Translation:      {results['avg_translation_time_ms']:.2f} ms")
    print(f"  Avg Inference:        {results['avg_inference_time_ms']:.2f} ms")
    print(f"  Total Translations:   {results['total_translations']}")
    print()
    
    # Interpretation
    print(f"💡 Interpretation:")
    if results['consistency_rate'] >= 0.85:
        print(f"  ✅ Excellent cross-lingual consistency!")
    elif results['consistency_rate'] >= 0.75:
        print(f"  ✅ Good cross-lingual consistency")
    elif results['consistency_rate'] >= 0.65:
        print(f"  ⚠️  Moderate consistency - consider improving translation")
    else:
        print(f"  ❌ Low consistency - investigate translation quality")
    
    if results['avg_score_gap'] < 0.05:
        print(f"  ✅ Very stable confidence scores across languages")
    elif results['avg_score_gap'] < 0.15:
        print(f"  ✅ Acceptable score variance")
    else:
        print(f"  ⚠️  High score variance - model may be language-sensitive")


def main():
    print("="*80)
    print("  CROSS-LINGUAL QA BENCHMARK")
    print("="*80)
    
    # Load model
    print("\n📦 Loading adaptive QA service...")
    model_path = str(project_root / "models/xlm_roberta_qa/stage2_best")
    adaptive_qa = create_adaptive_qa_service(
        model_path=model_path,
        device="cpu",
        default_language="vi",
        auto_translate=True
    )
    print("✅ Model loaded")
    
    # Load dataset
    print("\n📚 Loading XQuAD parallel dataset...")
    parallel_data = load_parallel_dataset()
    print(f"✅ Loaded {len(parallel_data)} parallel samples")
    
    # Run benchmark
    results = evaluate_cross_lingual_consistency(
        adaptive_qa,
        parallel_data,
        max_samples=50  # Reasonable sample size
    )
    
    # Print results
    print_results(results)
    
    # Example queries
    print("\n" + "="*80)
    print("  EXAMPLE QUERIES")
    print("="*80 + "\n")
    
    sample = parallel_data[0]
    
    print("📝 Question (EN):", sample["question_en"])
    print("📝 Question (VI):", sample["question_vi"])
    print()
    
    # Answer in English
    result_en = adaptive_qa.answer(
        sample["question_en"],
        sample["context_vi"],
        target_language="en"
    )
    print(f"🇬🇧 Answer (EN): {result_en.answer}")
    print(f"   Score: {result_en.score:.4f}, Translated: {result_en.translated}")
    print()
    
    # Answer in Vietnamese
    result_vi = adaptive_qa.answer(
        sample["question_vi"],
        sample["context_vi"],
        target_language="vi"
    )
    print(f"🇻🇳 Answer (VI): {result_vi.answer}")
    print(f"   Score: {result_vi.score:.4f}, Translated: {result_vi.translated}")
    print()
    
    print("="*80)
    print("✅ Benchmark complete!")


if __name__ == "__main__":
    main()
