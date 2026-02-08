"""
Hybrid Retrieval System
Combines BM25 keyword search + Semantic search
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb


class HybridRetriever:
    """Hybrid retrieval kết hợp BM25 và semantic search"""
    
    def __init__(
        self,
        embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
        collection_name: str = "multimodal_qa",
        chroma_persist_dir: str | None = None
    ):
        self.encoder = SentenceTransformer(embedding_model, device="cpu")

        self.chroma_client = chromadb.Client()

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Multimodal QA documents (in-memory)"}
        )
        
        self.bm25 = None
        self.documents: List[str] = []
        self.doc_metadata: List[Dict] = []
        self.doc_ids: List[str] = []
        self.id_to_index: Dict[str, int] = {}
    
    def index_documents(
        self,
        documents: List[str],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 32,
        append: bool = False
    ):
        """Index documents vào cả BM25 và vector store"""
        if not documents:
            return

        new_docs = documents
        new_meta = metadata or [{} for _ in new_docs]
        new_ids = ids

        if new_ids is None:
            offset = len(self.documents)
            new_ids = [f"doc_{offset + i}" for i in range(len(new_docs))]

        if append:
            existing_docs = self.documents
            existing_meta = self.doc_metadata
            existing_ids = self.doc_ids

            self.documents = existing_docs + new_docs
            self.doc_metadata = existing_meta + new_meta
            self.doc_ids = existing_ids + new_ids
        else:
            self.documents = new_docs
            self.doc_metadata = new_meta
            self.doc_ids = new_ids

        self.id_to_index = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}

        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        for i in range(0, len(new_docs), batch_size):
            batch_docs = new_docs[i:i + batch_size]
            batch_meta = new_meta[i:i + batch_size]
            batch_ids = new_ids[i:i + batch_size]

            embeddings = self.encoder.encode(
                batch_docs,
                show_progress_bar=False,
                batch_size=batch_size
            )

            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )

        print(f"Indexed {len(new_docs)} documents (total: {len(self.documents)})")
    
    def load_existing_documents(self):
        """No-op for in-memory client (kept for compatibility)."""
        if not self.documents:
            print("Warning: No documents found in memory")
        else:
            print(f"Documents already in memory: {len(self.documents)}")
    
    def bm25_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        BM25 keyword search
        
        Returns:
            List of (doc_index, score) tuples
        """
        if self.bm25 is None:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        
        return results
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        document_filter: str = None
    ) -> List[Dict]:
        """
        Semantic search using embeddings
        
        Args:
            query: Search query
            top_k: Number of results
            document_filter: Filter by document filename or source
        
        Returns:
            List of results with documents and scores
        """
        query_embedding = self.encoder.encode([query])[0]
        
        where_filter = None
        if document_filter:
            where_filter = {
                "$or": [
                    {"filename": {"$contains": document_filter}},
                    {"source": {"$contains": document_filter}}
                ]
            }
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter
        )
        
        formatted = []
        for i in range(len(results['documents'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'score': 1.0 - results['distances'][0][i],
                'metadata': results['metadatas'][0][i]
            })
        
        return formatted

    def clear(self):
        """Clear all indexed documents (in-memory)."""
        try:
            # Drop and recreate collection to clear vector data
            name = self.collection.name
            self.chroma_client.delete_collection(name)
            self.collection = self.chroma_client.create_collection(
                name=name,
                metadata=self.collection.metadata
            )
        except Exception:
            # Fallback: recreate client/collection
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.create_collection(name=name)

        # Clear in-memory structures
        self.bm25 = None
        self.documents = []
        self.doc_metadata = []
        self.doc_ids = []
        self.id_to_index = {}
        print("In-memory retrieval cleared")
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        bm25_weight: float = None,
        semantic_weight: float = None,
        document_filter: str = None
    ) -> List[Dict]:
        """
        Hybrid search combining BM25 and semantic search
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Balance between semantic (1.0) and BM25 (0.0)
            bm25_weight: Override alpha for BM25
            semantic_weight: Override alpha for semantic
            document_filter: Filter by document filename
        
        Returns:
            List of ranked results
        """
        if bm25_weight is None:
            bm25_weight = 1.0 - alpha
        if semantic_weight is None:
            semantic_weight = alpha
        
        bm25_results = self.bm25_search(query, top_k=top_k * 2)
        semantic_results = self.semantic_search(query, top_k=top_k * 2, document_filter=document_filter)
        
        bm25_scores = self._normalize_scores([s for _, s in bm25_results])
        semantic_scores_dict = {r['id']: r['score'] for r in semantic_results}
        
        combined_scores = {}
        
        for (idx, _), norm_score in zip(bm25_results, bm25_scores):
            if idx < len(self.doc_ids):
                doc_id = self.doc_ids[idx]
                
                if document_filter:
                    metadata = self.doc_metadata[idx] if idx < len(self.doc_metadata) else {}
                    filename = metadata.get('filename', '')
                    source = metadata.get('source', '')
                    if document_filter not in filename and document_filter not in source:
                        continue
                
                combined_scores[doc_id] = {
                    'bm25_score': norm_score * bm25_weight,
                    'semantic_score': 0.0,
                    'index': idx
                }
        
        # Add semantic scores
        for result in semantic_results:
            doc_id = result['id']
            
            # Skip if ID not in our mapping
            if doc_id not in self.id_to_index:
                continue
                
            idx = self.id_to_index[doc_id]
                
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    'bm25_score': 0.0,
                    'semantic_score': result['score'] * semantic_weight,
                    'index': idx
                }
            else:
                combined_scores[doc_id]['semantic_score'] = result['score'] * semantic_weight
        
        # Calculate final scores and rank
        ranked_results = []
        for doc_id, scores in combined_scores.items():
            final_score = scores['bm25_score'] + scores['semantic_score']
            idx = scores['index']
            
            ranked_results.append({
                'id': doc_id,
                'document': self.documents[idx],
                'metadata': self.doc_metadata[idx],
                'score': final_score,
                'bm25_score': scores['bm25_score'],
                'semantic_score': scores['semantic_score']
            })
        
        # Sort by final score
        ranked_results.sort(key=lambda x: x['score'], reverse=True)
        
        return ranked_results[:top_k]
    
    def reciprocal_rank_fusion(
        self,
        query: str,
        top_k: int = 10,
        k: int = 60
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) for combining rankings
        
        RRF score = sum(1 / (k + rank_i))
        
        Args:
            query: Search query
            top_k: Number of results
            k: RRF constant (default 60)
        """
        # Get rankings from both methods
        bm25_results = self.bm25_search(query, top_k=top_k * 2)
        semantic_results = self.semantic_search(query, top_k=top_k * 2)
        
        # Calculate RRF scores
        rrf_scores = {}
        
        # BM25 rankings
        for rank, (idx, _) in enumerate(bm25_results, start=1):
            doc_id = f"doc_{idx}"
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
        
        # Semantic rankings
        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
        
        # Rank by RRF score
        ranked_results = []
        for doc_id, score in rrf_scores.items():
            # Skip if ID not in our mapping
            if doc_id not in self.id_to_index:
                continue
                
            idx = self.id_to_index[doc_id]
            ranked_results.append({
                'id': doc_id,
                'document': self.documents[idx],
                'metadata': self.doc_metadata[idx],
                'score': score
            })
        
        ranked_results.sort(key=lambda x: x['score'], reverse=True)
        return ranked_results[:top_k]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-max normalization"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
