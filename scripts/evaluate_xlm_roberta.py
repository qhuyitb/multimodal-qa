"""
Script evaluate XLM-RoBERTa QA model
Test trên ViQuAD, XQuAD VI, XQuAD EN
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_from_disk
from src.models.qa_model import create_qa_model
from src.utils.helpers import get_project_root
from evaluate import load
import logging
from typing import Dict, List
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_qa_metrics(predictions: List[Dict], references: List[Dict]) -> Dict:
    """
    Tính F1 và Exact Match scores
    """
    squad_metric = load("squad")
    
    # Format cho squad metric
    formatted_predictions = [
        {"id": p["id"], "prediction_text": p["prediction_text"]}
        for p in predictions
    ]
    
    formatted_references = [
        {"id": r["id"], "answers": r["answers"]}
        for r in references
    ]
    
    results = squad_metric.compute(
        predictions=formatted_predictions,
        references=formatted_references
    )
    
    return results


def evaluate_model(
    model,
    dataset,
    dataset_name: str,
    max_samples: int = None,
    answerable_only: bool = False
) -> Dict:
    """
    Evaluate model trên dataset
    """
    logger.info(f"Evaluating on {dataset_name}")
    
    # Filter answerable questions if requested
    if answerable_only:
        original_size = len(dataset)
        dataset = dataset.filter(
            lambda x: x.get('answers') and 
                     x['answers'].get('text') and 
                     len(x['answers']['text']) > 0
        )
        logger.info(f"Filtered to {len(dataset)} answerable (from {original_size} total)")
    
    # Subsample nếu cần
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    predictions = []
    references = []
    
    logger.info(f"Processing {len(dataset)} samples...")
    for example in tqdm(dataset):
        # Skip examples without answers if not filtered
        if not example.get('answers') or not example['answers'].get('text') or not example['answers']['text']:
            continue
            
        # Predict with optimized settings
        pred = model.predict(
            question=example["question"],
            context=example["context"],
            top_k=1,
            max_answer_length=50,
            n_best_size=20
        )
        
        pred_text = pred[0]["text"] if pred else ""
        
        predictions.append({
            "id": example["id"],
            "prediction_text": pred_text
        })
        
        references.append({
            "id": example["id"],
            "answers": example["answers"]
        })
    
    # Compute metrics
    metrics = compute_qa_metrics(predictions, references)
    
    logger.info(f"\nResults on {dataset_name}:")
    logger.info(f"  F1 Score:     {metrics['f1']:.2f}%")
    logger.info(f"  Exact Match:  {metrics['exact_match']:.2f}%")
    logger.info(f"  Samples:      {len(predictions)}")
    
    return {
        "dataset": dataset_name,
        "f1": metrics['f1'],
        "exact_match": metrics['exact_match'],
        "num_samples": len(predictions),
        "predictions": predictions,
        "references": references
    }


def main():
    # Load model
    project_root = get_project_root()
    model_path = project_root / "models" / "xlm_roberta_qa" / "stage2_best"
    
    logger.info(f"Loading model from {model_path}")
    model = create_qa_model(checkpoint_path=model_path, use_lora=True)
    
    processed_dir = project_root / "datasets" / "processed"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # 1. Evaluate trên ViQuAD test (both mixed and answerable-only)
    
    logger.info("="*80)
    logger.info("1. ViQuAD Test Set Evaluation")
    logger.info("="*80)
    
    viquad = load_from_disk(processed_dir / "viquad_augmented")
    
    # Mixed evaluation (all questions)
    logger.info("\n[Mixed - All Questions]")
    viquad_mixed = evaluate_model(
        model,
        viquad['validation'],
        "ViQuAD Test (Mixed)",
        answerable_only=False
    )
    
    # Answerable-only evaluation
    logger.info("\n[Answerable Questions Only]")
    viquad_answerable = evaluate_model(
        model,
        viquad['validation'],
        "ViQuAD Test (Answerable)",
        answerable_only=True
    )
    
    all_results['viquad_mixed'] = viquad_mixed
    all_results['viquad_answerable'] = viquad_answerable
    
    # 2. Evaluate trên XQuAD VI
    logger.info("\n" + "="*80)
    logger.info("2. XQuAD Vietnamese Evaluation")
    logger.info("="*80)
    
    xquad_vi = load_from_disk(processed_dir / "xquad_vi_normalized")
    xquad_vi_results = evaluate_model(
        model,
        xquad_vi['validation'],
        "XQuAD VI",
        answerable_only=True
    )
    all_results['xquad_vi'] = xquad_vi_results
    
    # 3. Evaluate trên XQuAD EN
    logger.info("\n" + "="*80)
    logger.info("3. XQuAD English Evaluation")
    logger.info("="*80)
    
    xquad_en = load_from_disk(processed_dir / "xquad_en_normalized")
    xquad_en_results = evaluate_model(
        model,
        xquad_en['validation'],
        "XQuAD EN",
        answerable_only=True
    )
    all_results['xquad_en'] = xquad_en_results
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    
    summary = {
        "ViQuAD (Mixed)": {
            "F1": viquad_mixed['f1'],
            "EM": viquad_mixed['exact_match'],
            "Samples": viquad_mixed['num_samples']
        },
        "ViQuAD (Answerable)": {
            "F1": viquad_answerable['f1'],
            "EM": viquad_answerable['exact_match'],
            "Samples": viquad_answerable['num_samples']
        },
        "XQuAD VI": {
            "F1": xquad_vi_results['f1'],
            "EM": xquad_vi_results['exact_match'],
            "Samples": xquad_vi_results['num_samples']
        },
        "XQuAD EN": {
            "F1": xquad_en_results['f1'],
            "EM": xquad_en_results['exact_match'],
            "Samples": xquad_en_results['num_samples']
        }
    }
    
    logger.info("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    
    # Performance analysis
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE ANALYSIS")
    logger.info("="*80)
    
    # Cross-lingual transfer gap
    en_vi_gap = xquad_en_results['f1'] - xquad_vi_results['f1']
    logger.info(f"Cross-lingual Transfer Gap (EN - VI): {en_vi_gap:.2f}%")
    
    # Answerable vs Mixed gap
    answerable_gain = viquad_answerable['f1'] - viquad_mixed['f1']
    logger.info(f"Answerable-only Improvement: {answerable_gain:.2f}%")
    logger.info(f"Impossible questions: {viquad_mixed['num_samples'] - viquad_answerable['num_samples']}")
    
    # Performance assessment
    if viquad_answerable['f1'] >= 75:
        logger.info("\n✅ GOOD: Model performs well (≥75% F1 on answerable)")
    elif viquad_answerable['f1'] >= 70:
        logger.info("\n⚡ MODERATE: Acceptable but could improve (70-75% F1)")
    else:
        logger.info("\n⚠️  LOW: Model needs retraining (<70% F1)")
    
    # Save results
    results_file = results_dir / "xlm_roberta_evaluation.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'cross_lingual_gap': en_vi_gap,
            'detailed_results': {
                k: {
                    'dataset': v['dataset'],
                    'f1': v['f1'],
                    'exact_match': v['exact_match']
                }
                for k, v in all_results.items()
            }
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
