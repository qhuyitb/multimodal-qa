from typing import List, Set, Dict, Any, Tuple
import numpy as np
from collections import defaultdict


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
    
    
