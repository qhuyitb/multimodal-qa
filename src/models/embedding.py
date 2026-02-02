from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import re

class MultilingualEmbedding:
    
    def __init__(
        self, 
        model_name: str = "keepitreal/vietnamese-sbert",
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        emb1, emb2 = self.encode([text1, text2], normalize=True)
        similarity = np.dot(emb1, emb2)
        return float(similarity)
    
    def batch_similarity(
        self, 
        query: str, 
        candidates: List[str],
        top_k: Optional[int] = None
    ) -> List[tuple[int, float]]:
        query_emb = self.encode(query, normalize=True)
        candidate_embs = self.encode(candidates, normalize=True)
        
        similarities = np.dot(candidate_embs, query_emb)
        
        results = [(idx, float(score)) for idx, score in enumerate(similarities)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def get_model_info(self) -> dict:
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'batch_size': self.batch_size,
            'max_seq_length': self.model.max_seq_length
        }


class HybridEmbedding:
    
    def __init__(
        self,
        semantic_model: MultilingualEmbedding,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        self.semantic_model = semantic_model
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
    
    def _extract_keywords(self, text: str) -> set:
        
        words = re.findall(r'\w+', text.lower())
        return set(words)
    
    def _keyword_similarity(self, text1: str, text2: str) -> float:
        keywords1 = self._extract_keywords(text1)
        keywords2 = self._extract_keywords(text2)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.0
    
    def similarity(self, text1: str, text2: str) -> float:
        semantic_sim = self.semantic_model.similarity(text1, text2)
        keyword_sim = self._keyword_similarity(text1, text2)
        
        hybrid_score = (
            self.semantic_weight * semantic_sim + 
            self.keyword_weight * keyword_sim
        )
        
        return hybrid_score
    
    def batch_similarity(
        self, 
        query: str, 
        candidates: List[str],
        top_k: Optional[int] = None
    ) -> List[tuple[int, float]]:
        semantic_results = self.semantic_model.batch_similarity(query, candidates, top_k=None)
        
        keyword_scores = []
        for candidate in candidates:
            keyword_scores.append(self._keyword_similarity(query, candidate))
        
        hybrid_results = []
        for idx, semantic_score in semantic_results:
            keyword_score = keyword_scores[idx]
            hybrid_score = (
                self.semantic_weight * semantic_score + 
                self.keyword_weight * keyword_score
            )
            hybrid_results.append((idx, hybrid_score))
        
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            hybrid_results = hybrid_results[:top_k]
        
        return hybrid_results


# Alias để dễ import
EmbeddingModel = MultilingualEmbedding
