"""
Cross-Encoder Reranker for improved retrieval accuracy
"""

from typing import List, Dict
from sentence_transformers import CrossEncoder
import numpy as np


class CrossEncoderReranker:
    """
    Rerank retrieval results using cross-encoder model
    More accurate but slower than bi-encoder retrieval
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = None
    ):
        """
        Initialize cross-encoder reranker
        
        Args:
            model_name: HuggingFace model name
            device: cuda/cpu (auto-detected if None)
        """
        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Rerank candidate documents
        
        Args:
            query: Search query
            candidates: List of candidate documents with 'document' field
            top_k: Number of top results to return
            return_scores: Whether to include reranking scores
        
        Returns:
            Reranked list of candidates
        """
        if not candidates:
            return []
        
        # Prepare pairs
        pairs = [(query, c['document']) for c in candidates]
        
        # Score with cross-encoder
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Add scores to candidates
        for candidate, score in zip(candidates, scores):
            if return_scores:
                candidate['rerank_score'] = float(score)
                candidate['original_score'] = candidate.get('score', 0.0)
        
        # Sort by rerank score
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [c for c, _ in ranked[:top_k]]
    
    def rerank_with_confidence(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        confidence_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Rerank and filter by confidence threshold
        
        Args:
            query: Search query
            candidates: List of candidates
            top_k: Maximum number of results
            confidence_threshold: Minimum score to include
        
        Returns:
            Filtered and reranked results
        """
        reranked = self.rerank(query, candidates, top_k=len(candidates))
        
        # Filter by threshold
        filtered = [
            c for c in reranked 
            if c.get('rerank_score', 0) >= confidence_threshold
        ]
        
        return filtered[:top_k]


if __name__ == "__main__":
    # Test
    print("=== Testing Cross-Encoder Reranker ===\n")
    
    # Sample candidates (from retrieval)
    query = "how does deep learning work?"
    candidates = [
        {
            "document": "Deep learning uses neural networks with multiple layers to learn representations.",
            "score": 0.75,
            "metadata": {"source": "doc1"}
        },
        {
            "document": "Machine learning is a field of artificial intelligence.",
            "score": 0.68,
            "metadata": {"source": "doc2"}
        },
        {
            "document": "Neural networks are inspired by biological neurons in the brain.",
            "score": 0.72,
            "metadata": {"source": "doc3"}
        },
    ]
    
    # Initialize reranker
    print("Loading cross-encoder model...")
    reranker = CrossEncoderReranker()
    
    print(f"\nQuery: '{query}'\n")
    print("Original ranking:")
    for i, c in enumerate(candidates, 1):
        print(f"{i}. [{c['score']:.4f}] {c['document'][:60]}...")
    
    # Rerank
    print("\nAfter reranking:")
    reranked = reranker.rerank(query, candidates, top_k=3)
    for i, c in enumerate(reranked, 1):
        print(f"{i}. [Rerank: {c['rerank_score']:.4f}, Original: {c['original_score']:.4f}]")
        print(f"   {c['document'][:60]}...")
    
    # With confidence threshold
    print("\nWith confidence threshold (0.5):")
    filtered = reranker.rerank_with_confidence(query, candidates, top_k=3, confidence_threshold=0.5)
    for i, c in enumerate(filtered, 1):
        print(f"{i}. [{c['rerank_score']:.4f}] {c['document'][:60]}...")
