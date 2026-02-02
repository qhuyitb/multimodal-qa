from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pathlib import Path
import uuid
from utils.helpers import get_data_dir

class ChromaVectorStore:
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "multimodal_qa",
        embedding_function: Optional[Any] = None
    ):
        self.persist_directory = Path(persist_directory) if persist_directory else get_data_dir("vector_db")
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        if embedding_function is None:
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="keepitreal/vietnamese-sbert"
            )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.collection_name = collection_name
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        if not texts:
            return []
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        
        for i, metadata in enumerate(metadatas):
            if 'text_length' not in metadata:
                metadata['text_length'] = len(texts[i])
        
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        return {
            'ids': results['ids'][0] if results['ids'] else [],
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else []
        }
    
    def hybrid_search(
        self,
        query: str,
        n_results: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        final_top_k: int = 5
    ) -> Dict[str, List]:
        semantic_results = self.semantic_search(query, n_results=n_results)
        
        if not semantic_results['documents']:
            return semantic_results
        
        reranked = []
        for idx in range(len(semantic_results['documents'])):
            doc = semantic_results['documents'][idx]
            semantic_score = 1 - semantic_results['distances'][idx]
            
            keyword_score = self._keyword_match_score(query, doc)
            
            hybrid_score = (
                semantic_weight * semantic_score + 
                keyword_weight * keyword_score
            )
            
            reranked.append({
                'index': idx,
                'score': hybrid_score,
                'semantic_score': semantic_score,
                'keyword_score': keyword_score
            })
        
        reranked.sort(key=lambda x: x['score'], reverse=True)
        reranked = reranked[:final_top_k]
        
        final_results = {
            'ids': [],
            'documents': [],
            'metadatas': [],
            'distances': [],
            'scores': []
        }
        
        for item in reranked:
            idx = item['index']
            final_results['ids'].append(semantic_results['ids'][idx])
            final_results['documents'].append(semantic_results['documents'][idx])
            final_results['metadatas'].append(semantic_results['metadatas'][idx])
            final_results['distances'].append(semantic_results['distances'][idx])
            final_results['scores'].append({
                'hybrid': item['score'],
                'semantic': item['semantic_score'],
                'keyword': item['keyword_score']
            })
        
        return final_results
    
    def _keyword_match_score(self, query: str, document: str) -> float:
        import re
        
        query_words = set(re.findall(r'\w+', query.lower()))
        doc_words = set(re.findall(r'\w+', document.lower()))
        
        if not query_words or not doc_words:
            return 0.0
        
        intersection = len(query_words & doc_words)
        union = len(query_words | doc_words)
        
        return intersection / union if union > 0 else 0.0
    
    def rerank_results(
        self,
        query: str,
        results: Dict[str, List],
        reranker_fn: Optional[callable] = None
    ) -> Dict[str, List]:
        if reranker_fn is None or not results['documents']:
            return results
        
        scored_results = []
        for idx in range(len(results['documents'])):
            doc = results['documents'][idx]
            rerank_score = reranker_fn(query, doc)
            scored_results.append((idx, rerank_score))
        
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        reranked_results = {
            'ids': [],
            'documents': [],
            'metadatas': [],
            'distances': [],
            'rerank_scores': []
        }
        
        for idx, score in scored_results:
            reranked_results['ids'].append(results['ids'][idx])
            reranked_results['documents'].append(results['documents'][idx])
            reranked_results['metadatas'].append(results['metadatas'][idx])
            reranked_results['distances'].append(results['distances'][idx])
            reranked_results['rerank_scores'].append(score)
        
        return reranked_results
    
    def delete_documents(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)
    
    def update_document(
        self,
        doc_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        update_data = {'ids': [doc_id]}
        
        if text is not None:
            update_data['documents'] = [text]
        
        if metadata is not None:
            update_data['metadatas'] = [metadata]
        
        self.collection.update(**update_data)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        
        return {
            'collection_name': self.collection_name,
            'total_documents': count,
            'persist_directory': str(self.persist_directory)
        }
    
    def clear_collection(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def get_by_ids(self, ids: List[str]) -> Dict[str, List]:
        results = self.collection.get(ids=ids)
        
        return {
            'ids': results['ids'],
            'documents': results['documents'],
            'metadatas': results['metadatas']
        }


# Alias để tương thích với code cũ
VectorStore = ChromaVectorStore
