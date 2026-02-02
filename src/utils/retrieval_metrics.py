"""
Evaluation metrics for retrieval quality
"""

from typing import List, Dict, Set
import numpy as np


def recall_at_k(
    relevant_docs: Set[str],
    retrieved_docs: List[str],
    k: int
) -> float:
    """
    Recall@K: Proportion of relevant documents in top-K
    
    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs (ranked)
        k: Number of top results to consider
    
    Returns:
        Recall score [0, 1]
    """
    if not relevant_docs:
        return 0.0
    
    top_k = set(retrieved_docs[:k])
    return len(relevant_docs & top_k) / len(relevant_docs)


def precision_at_k(
    relevant_docs: Set[str],
    retrieved_docs: List[str],
    k: int
) -> float:
    """
    Precision@K: Proportion of relevant documents among top-K
    
    Returns:
        Precision score [0, 1]
    """
    if k == 0:
        return 0.0
    
    top_k = set(retrieved_docs[:k])
    return len(relevant_docs & top_k) / k


def mean_reciprocal_rank(
    relevant_docs: Set[str],
    retrieved_docs: List[str]
) -> float:
    """
    MRR: Average of reciprocal ranks of first relevant document
    
    Returns:
        MRR score [0, 1]
    """
    for i, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            return 1.0 / i
    return 0.0


def average_precision(
    relevant_docs: Set[str],
    retrieved_docs: List[str]
) -> float:
    """
    Average Precision: Average of precision values at each relevant document
    
    Returns:
        AP score [0, 1]
    """
    if not relevant_docs:
        return 0.0
    
    precisions = []
    num_relevant = 0
    
    for i, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            num_relevant += 1
            precisions.append(num_relevant / i)
    
    if not precisions:
        return 0.0
    
    return sum(precisions) / len(relevant_docs)


def ndcg_at_k(
    relevant_docs: Set[str],
    retrieved_docs: List[str],
    k: int,
    relevance_scores: Dict[str, float] = None
) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain
    
    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs
        k: Number of top results
        relevance_scores: Optional relevance scores for each doc
    
    Returns:
        NDCG score [0, 1]
    """
    # If no relevance scores provided, use binary (1 if relevant, 0 otherwise)
    if relevance_scores is None:
        relevance_scores = {doc_id: 1.0 for doc_id in relevant_docs}
    
    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs[:k], start=1):
        rel = relevance_scores.get(doc_id, 0.0)
        dcg += rel / np.log2(i + 1)
    
    # Calculate ideal DCG
    ideal_scores = sorted(
        [relevance_scores.get(doc_id, 0.0) for doc_id in relevant_docs],
        reverse=True
    )[:k]
    
    idcg = 0.0
    for i, rel in enumerate(ideal_scores, start=1):
        idcg += rel / np.log2(i + 1)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def evaluate_retrieval(
    test_set: List[Dict],
    retriever,
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Comprehensive retrieval evaluation
    
    Args:
        test_set: List of test items with 'query' and 'relevant_docs'
        retriever: Retriever object with search() method
        k_values: List of K values to evaluate
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        f"recall@{k}": [] for k in k_values
    }
    metrics.update({
        f"precision@{k}": [] for k in k_values
    })
    metrics.update({
        f"ndcg@{k}": [] for k in k_values
    })
    metrics["mrr"] = []
    metrics["map"] = []
    
    for item in test_set:
        query = item["query"]
        relevant_docs = set(item["relevant_docs"])
        
        # Retrieve documents
        results = retriever.search(query, top_k=max(k_values))
        retrieved_docs = [r.get('id', r.get('doc_id', '')) for r in results]
        
        # Calculate metrics
        for k in k_values:
            metrics[f"recall@{k}"].append(
                recall_at_k(relevant_docs, retrieved_docs, k)
            )
            metrics[f"precision@{k}"].append(
                precision_at_k(relevant_docs, retrieved_docs, k)
            )
            metrics[f"ndcg@{k}"].append(
                ndcg_at_k(relevant_docs, retrieved_docs, k)
            )
        
        metrics["mrr"].append(
            mean_reciprocal_rank(relevant_docs, retrieved_docs)
        )
        metrics["map"].append(
            average_precision(relevant_docs, retrieved_docs)
        )
    
    # Average all metrics
    return {k: np.mean(v) for k, v in metrics.items()}


def compare_retrievers(
    test_set: List[Dict],
    retrievers: Dict[str, any],
    k_values: List[int] = [5, 10]
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple retrievers
    
    Args:
        test_set: Test dataset
        retrievers: Dict of {name: retriever_object}
        k_values: K values to evaluate
    
    Returns:
        Dict of {retriever_name: metrics}
    """
    results = {}
    
    for name, retriever in retrievers.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_retrieval(test_set, retriever, k_values)
        results[name] = metrics
    
    return results


def print_comparison_table(results: Dict[str, Dict[str, float]]):
    """Print comparison table of retriever results"""
    
    # Get all metric names
    metric_names = list(next(iter(results.values())).keys())
    
    # Print header
    header = f"{'Metric':<15}"
    for name in results.keys():
        header += f"{name:<20}"
    print(header)
    print("=" * len(header))
    
    # Print metrics
    for metric in metric_names:
        row = f"{metric:<15}"
        for name in results.keys():
            value = results[name][metric]
            row += f"{value:>18.4f}  "
        print(row)


if __name__ == "__main__":
    # Test
    print("=== Testing Retrieval Metrics ===\n")
    
    # Sample data
    relevant_docs = {"doc1", "doc3", "doc5"}
    retrieved_docs = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]
    
    print(f"Relevant: {relevant_docs}")
    print(f"Retrieved: {retrieved_docs[:5]}\n")
    
    # Calculate metrics
    print(f"Recall@5: {recall_at_k(relevant_docs, retrieved_docs, 5):.4f}")
    print(f"Precision@5: {precision_at_k(relevant_docs, retrieved_docs, 5):.4f}")
    print(f"MRR: {mean_reciprocal_rank(relevant_docs, retrieved_docs):.4f}")
    print(f"Average Precision: {average_precision(relevant_docs, retrieved_docs):.4f}")
    print(f"NDCG@5: {ndcg_at_k(relevant_docs, retrieved_docs, 5):.4f}")
