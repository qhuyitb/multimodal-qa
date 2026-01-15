from typing import List, Set, Dict, Any, Tuple, Optional
import numpy as np
from collections import defaultdict
import time


class RetrievalMetrics:
    
    @staticmethod
    def recall_at_k(
        retrieved: List[str], 
        relevant: Set[str], 
        k: int
    ) -> float:
        if not relevant:
            return 0.0
        
        top_k = set(retrieved[:k])
        hits = len(top_k & relevant)
        
        return hits / len(relevant)
    
    @staticmethod
    def precision_at_k(
        retrieved: List[str], 
        relevant: Set[str], 
        k: int
    ) -> float:
        if k == 0:
            return 0.0
        
        top_k = retrieved[:k]
        hits = sum(1 for doc_id in top_k if doc_id in relevant)
        
        return hits / k
    
    @staticmethod
    def mean_reciprocal_rank(
        retrieved_lists: List[List[str]], 
        relevant_lists: List[Set[str]]
    ) -> float:
        if len(retrieved_lists) != len(relevant_lists):
            raise ValueError("Retrieved and relevant lists must have same length")
        
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            rank = None
            for i, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant:
                    rank = i
                    break
            
            if rank is not None:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    @staticmethod
    def average_precision(
        retrieved: List[str], 
        relevant: Set[str]
    ) -> float:
        if not relevant:
            return 0.0
        
        precision_sum = 0.0
        hits = 0
        
        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                hits += 1
                precision_at_i = hits / i
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant) if relevant else 0.0
    
    @staticmethod
    def mean_average_precision(
        retrieved_lists: List[List[str]], 
        relevant_lists: List[Set[str]]
    ) -> float:
        if not retrieved_lists or not relevant_lists:
            return 0.0
        
        ap_scores = []
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            ap = RetrievalMetrics.average_precision(retrieved, relevant)
            ap_scores.append(ap)
        
        return np.mean(ap_scores)
    
    @staticmethod
    def ndcg_at_k(
        retrieved: List[str], 
        relevant: Dict[str, float], 
        k: int
    ) -> float:
        if not relevant or k == 0:
            return 0.0
        
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], start=1):
            rel = relevant.get(doc_id, 0.0)
            dcg += rel / np.log2(i + 1)
        
        ideal_relevance = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 1) for i, rel in enumerate(ideal_relevance, start=1))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def f1_score_at_k(
        retrieved: List[str], 
        relevant: Set[str], 
        k: int
    ) -> float:
        precision = RetrievalMetrics.precision_at_k(retrieved, relevant, k)
        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def compute_all_metrics(
        retrieved_lists: List[List[str]], 
        relevant_lists: List[Set[str]],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        metrics = {}
        
        for k in k_values:
            recalls = []
            precisions = []
            f1_scores = []
            
            for retrieved, relevant in zip(retrieved_lists, relevant_lists):
                recalls.append(RetrievalMetrics.recall_at_k(retrieved, relevant, k))
                precisions.append(RetrievalMetrics.precision_at_k(retrieved, relevant, k))
                f1_scores.append(RetrievalMetrics.f1_score_at_k(retrieved, relevant, k))
            
            metrics[f'recall@{k}'] = np.mean(recalls)
            metrics[f'precision@{k}'] = np.mean(precisions)
            metrics[f'f1@{k}'] = np.mean(f1_scores)
        
        metrics['mrr'] = RetrievalMetrics.mean_reciprocal_rank(retrieved_lists, relevant_lists)
        metrics['map'] = RetrievalMetrics.mean_average_precision(retrieved_lists, relevant_lists)
        
        return metrics


class ChunkingMetrics:
    
    @staticmethod
    def chunk_statistics(chunks: List[Any]) -> Dict[str, float]:
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_length': 0.0,
                'min_length': 0,
                'max_length': 0,
                'std_length': 0.0
            }
        
        lengths = [len(chunk.text) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_length': np.mean(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'std_length': np.std(lengths)
        }
    
    @staticmethod
    def overlap_statistics(chunks: List[Any]) -> Dict[str, float]:
        if len(chunks) < 2:
            return {
                'avg_overlap': 0.0,
                'max_overlap': 0,
                'overlap_ratio': 0.0
            }
        
        overlaps = []
        total_text_length = sum(len(chunk.text) for chunk in chunks)
        
        for i in range(len(chunks) - 1):
            current = chunks[i].text
            next_chunk = chunks[i + 1].text
            
            max_possible = min(len(current), len(next_chunk))
            overlap_len = 0
            
            for length in range(max_possible, 0, -1):
                if current[-length:] == next_chunk[:length]:
                    overlap_len = length
                    break
            
            overlaps.append(overlap_len)
        
        total_overlap = sum(overlaps)
        
        return {
            'avg_overlap': np.mean(overlaps),
            'max_overlap': max(overlaps),
            'overlap_ratio': total_overlap / total_text_length if total_text_length > 0 else 0.0
        }
    
    @staticmethod
    def compare_chunking_strategies(
        chunks_a: List[Any],
        chunks_b: List[Any],
        name_a: str = "Strategy A",
        name_b: str = "Strategy B"
    ) -> Dict[str, Any]:
        stats_a = ChunkingMetrics.chunk_statistics(chunks_a)
        stats_b = ChunkingMetrics.chunk_statistics(chunks_b)
        
        overlap_a = ChunkingMetrics.overlap_statistics(chunks_a)
        overlap_b = ChunkingMetrics.overlap_statistics(chunks_b)
        
        return {
            name_a: {
                'chunk_stats': stats_a,
                'overlap_stats': overlap_a
            },
            name_b: {
                'chunk_stats': stats_b,
                'overlap_stats': overlap_b
            },
            'comparison': {
                'chunk_count_diff': stats_b['total_chunks'] - stats_a['total_chunks'],
                'avg_length_diff': stats_b['avg_length'] - stats_a['avg_length'],
                'overlap_ratio_diff': overlap_b['overlap_ratio'] - overlap_a['overlap_ratio']
            }
        }


def print_metrics_report(metrics: Dict[str, float], title: str = "Retrieval Metrics") -> None:
    print(f"\n{title}")
    
    
    for metric_name, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"{metric_name:20s}: {value:.4f}")
        else:
            print(f"{metric_name:20s}: {value}")


class TranslationMetrics:
    """
    Metrics for evaluating translation quality and performance.
    """
    
    @staticmethod
    def bleu_score(
        reference: str,
        candidate: str,
        max_n: int = 4
    ) -> float:
        """
        Calculate BLEU score between reference and candidate translations.
        Simplified implementation for basic evaluation.
        
        Args:
            reference: Reference translation
            candidate: Candidate translation
            max_n: Maximum n-gram size
            
        Returns:
            BLEU score (0-1)
        """
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if not cand_tokens:
            return 0.0
        
        # Calculate n-gram precisions
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = TranslationMetrics._get_ngrams(ref_tokens, n)
            cand_ngrams = TranslationMetrics._get_ngrams(cand_tokens, n)
            
            if not cand_ngrams:
                precisions.append(0.0)
                continue
            
            matches = sum(min(ref_ngrams.get(ng, 0), cand_ngrams.get(ng, 0)) 
                         for ng in cand_ngrams)
            precision = matches / len(cand_tokens) if cand_tokens else 0.0
            precisions.append(precision)
        
        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            bleu = np.exp(np.mean([np.log(p) for p in precisions]))
        else:
            bleu = 0.0
        
        # Brevity penalty
        bp = min(1.0, np.exp(1 - len(ref_tokens) / len(cand_tokens))) if cand_tokens else 0.0
        
        return bp * bleu
    
    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        """Get n-gram counts from token list."""
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return dict(ngrams)
    
    @staticmethod
    def character_error_rate(
        reference: str,
        candidate: str
    ) -> float:
        """
        Calculate character-level edit distance ratio.
        
        Args:
            reference: Reference text
            candidate: Candidate text
            
        Returns:
            CER score (0-1, lower is better)
        """
        if not reference:
            return 1.0 if candidate else 0.0
        
        distance = TranslationMetrics._levenshtein_distance(reference, candidate)
        return distance / len(reference)
    
    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return TranslationMetrics._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def translation_latency_metrics(
        char_count: int,
        elapsed_time: float
    ) -> Dict[str, float]:
        """
        Calculate translation latency metrics.
        
        Args:
            char_count: Number of characters translated
            elapsed_time: Time taken in seconds
            
        Returns:
            Dictionary with latency metrics
        """
        return {
            "elapsed_time": elapsed_time,
            "characters": char_count,
            "chars_per_second": char_count / elapsed_time if elapsed_time > 0 else 0.0,
            "seconds_per_1k_chars": (elapsed_time / char_count * 1000) if char_count > 0 else 0.0
        }
    
    @staticmethod
    def evaluate_translation_quality(
        source_text: str,
        translated_text: str,
        reference_translation: Optional[str] = None,
        elapsed_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive translation quality evaluation.
        
        Args:
            source_text: Original source text
            translated_text: Generated translation
            reference_translation: Optional reference for comparison
            elapsed_time: Optional timing information
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            "source_length": len(source_text),
            "translation_length": len(translated_text),
            "length_ratio": len(translated_text) / len(source_text) if source_text else 0.0
        }
        
        # Add reference-based metrics if available
        if reference_translation:
            metrics["bleu_score"] = TranslationMetrics.bleu_score(
                reference_translation,
                translated_text
            )
            metrics["character_error_rate"] = TranslationMetrics.character_error_rate(
                reference_translation,
                translated_text
            )
        
        # Add latency metrics if available
        if elapsed_time is not None:
            latency = TranslationMetrics.translation_latency_metrics(
                len(source_text),
                elapsed_time
            )
            metrics.update(latency)
        
        return metrics
    
    @staticmethod
    def batch_evaluate(
        source_texts: List[str],
        translated_texts: List[str],
        reference_translations: Optional[List[str]] = None,
        elapsed_times: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple translations in batch.
        
        Args:
            source_texts: List of source texts
            translated_texts: List of translations
            reference_translations: Optional reference translations
            elapsed_times: Optional list of elapsed times
            
        Returns:
            Aggregated metrics
        """
        if not source_texts or not translated_texts:
            return {}
        
        # Evaluate each translation
        individual_metrics = []
        for i in range(len(source_texts)):
            ref = reference_translations[i] if reference_translations else None
            time_elapsed = elapsed_times[i] if elapsed_times else None
            
            metrics = TranslationMetrics.evaluate_translation_quality(
                source_texts[i],
                translated_texts[i],
                ref,
                time_elapsed
            )
            individual_metrics.append(metrics)
        
        # Aggregate results
        aggregated = {
            "count": len(individual_metrics),
            "total_source_chars": sum(m["source_length"] for m in individual_metrics),
            "total_translation_chars": sum(m["translation_length"] for m in individual_metrics),
            "avg_length_ratio": np.mean([m["length_ratio"] for m in individual_metrics])
        }
        
        # Add reference-based aggregates
        if reference_translations:
            aggregated["avg_bleu_score"] = np.mean([m.get("bleu_score", 0) for m in individual_metrics])
            aggregated["avg_cer"] = np.mean([m.get("character_error_rate", 0) for m in individual_metrics])
        
        # Add latency aggregates
        if elapsed_times:
            aggregated["total_time"] = sum(elapsed_times)
            aggregated["avg_time"] = np.mean(elapsed_times)
            aggregated["avg_chars_per_second"] = np.mean([m.get("chars_per_second", 0) for m in individual_metrics])
        
        return aggregated


class CrossLanguageQAMetrics:
    """
    Metrics for evaluating cross-language QA performance.
    """
    
    @staticmethod
    def evaluate_qa_correctness(
        question: str,
        answer: str,
        ground_truth: str,
        language: str = "en"
    ) -> Dict[str, float]:
        """
        Evaluate QA answer correctness.
        
        Args:
            question: Question text
            answer: Generated answer
            ground_truth: Correct answer
            language: Language code
            
        Returns:
            Correctness metrics
        """
        # Normalize texts
        answer_norm = answer.lower().strip()
        truth_norm = ground_truth.lower().strip()
        
        # Exact match
        exact_match = 1.0 if answer_norm == truth_norm else 0.0
        
        # Token overlap (F1)
        answer_tokens = set(answer_norm.split())
        truth_tokens = set(truth_norm.split())
        
        if not answer_tokens or not truth_tokens:
            token_f1 = 0.0
        else:
            common = answer_tokens & truth_tokens
            precision = len(common) / len(answer_tokens)
            recall = len(common) / len(truth_tokens)
            token_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Character-level similarity
        char_similarity = 1.0 - TranslationMetrics.character_error_rate(truth_norm, answer_norm)
        
        return {
            "exact_match": exact_match,
            "token_f1": token_f1,
            "char_similarity": char_similarity,
            "answer_length": len(answer)
        }
    
    @staticmethod
    def compare_monolingual_vs_crosslingual(
        monolingual_results: List[Dict[str, Any]],
        crosslingual_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare QA performance between monolingual and cross-lingual queries.
        
        Args:
            monolingual_results: Results from same-language QA
            crosslingual_results: Results from cross-language QA
            
        Returns:
            Comparison metrics
        """
        def aggregate_results(results):
            return {
                "avg_exact_match": np.mean([r.get("exact_match", 0) for r in results]),
                "avg_token_f1": np.mean([r.get("token_f1", 0) for r in results]),
                "avg_char_similarity": np.mean([r.get("char_similarity", 0) for r in results]),
                "count": len(results)
            }
        
        mono_agg = aggregate_results(monolingual_results)
        cross_agg = aggregate_results(crosslingual_results)
        
        return {
            "monolingual": mono_agg,
            "crosslingual": cross_agg,
            "degradation": {
                "exact_match": mono_agg["avg_exact_match"] - cross_agg["avg_exact_match"],
                "token_f1": mono_agg["avg_token_f1"] - cross_agg["avg_token_f1"],
                "char_similarity": mono_agg["avg_char_similarity"] - cross_agg["avg_char_similarity"]
            }
        }
    
    @staticmethod
    def translation_overhead_analysis(
        baseline_latency: float,
        with_translation_latency: float,
        translation_only_latency: float
    ) -> Dict[str, float]:
        """
        Analyze latency overhead from translation.
        
        Args:
            baseline_latency: Latency without translation
            with_translation_latency: Total latency with translation
            translation_only_latency: Time spent on translation alone
            
        Returns:
            Overhead metrics
        """
        overhead = with_translation_latency - baseline_latency
        overhead_percentage = (overhead / baseline_latency * 100) if baseline_latency > 0 else 0.0
        
        return {
            "baseline_latency": baseline_latency,
            "with_translation_latency": with_translation_latency,
            "translation_only_latency": translation_only_latency,
            "total_overhead": overhead,
            "overhead_percentage": overhead_percentage,
            "translation_proportion": (translation_only_latency / with_translation_latency * 100) 
                                     if with_translation_latency > 0 else 0.0
        }


def print_translation_metrics(
    metrics: Dict[str, Any],
    title: str = "Translation Evaluation Metrics"
) -> None:
    """
    Print translation metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        title: Report title
    """
    print(f"\n{title}")
    print("=" * 60)
    
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        elif isinstance(value, int):
            print(f"{key:30s}: {value}")
        elif isinstance(value, dict):
            print(f"\n{key}:")
            for subkey, subval in value.items():
                if isinstance(subval, float):
                    print(f"  {subkey:28s}: {subval:.4f}")
                else:
                    print(f"  {subkey:28s}: {subval}")
        else:
            print(f"{key:30s}: {value}")
    
    print("=" * 60)
