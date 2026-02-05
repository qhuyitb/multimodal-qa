"""
Benchmark retrieval systems
Compare naive vs smart chunking, BM25 vs semantic vs hybrid
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_from_disk
from src.services.hybrid_retrieval import HybridRetriever
from src.core.video_chunking import VideoChunker
from src.core.document_chunking import DocumentChunker
from src.utils.retrieval_metrics import evaluate_retrieval, compare_retrievers, print_comparison_table
from typing import List, Dict
import time


def create_naive_chunks(text: str, chunk_size: int = 512) -> List[str]:
    """Naive chunking: fixed size"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


def create_smart_chunks(text: str) -> List[str]:
    """Smart chunking: paragraph-based"""
    chunker = DocumentChunker(max_chunk_size=512)
    chunk_dicts = chunker.chunk_by_paragraph(text)
    return [c['text'] for c in chunk_dicts]


class NaiveRetriever:
    """Wrapper for naive chunking + retrieval"""
    def __init__(self):
        self.retriever = HybridRetriever(collection_name="naive_chunks")
    
    def index(self, documents: List[str]):
        chunks = []
        for doc in documents:
            chunks.extend(create_naive_chunks(doc))
        self.retriever.index_documents(chunks)
    
    def search(self, query: str, top_k: int = 10):
        return self.retriever.hybrid_search(query, top_k=top_k, alpha=0.7)


class SmartRetriever:
    """Wrapper for smart chunking + retrieval"""
    def __init__(self):
        self.retriever = HybridRetriever(collection_name="smart_chunks")
    
    def index(self, documents: List[str]):
        chunks = []
        for doc in documents:
            chunks.extend(create_smart_chunks(doc))
        self.retriever.index_documents(chunks)
    
    def search(self, query: str, top_k: int = 10):
        return self.retriever.hybrid_search(query, top_k=top_k, alpha=0.7)


class BM25OnlyRetriever:
    """BM25-only baseline"""
    def __init__(self):
        self.retriever = HybridRetriever(collection_name="bm25_only")
    
    def index(self, documents: List[str]):
        self.retriever.index_documents(documents)
    
    def search(self, query: str, top_k: int = 10):
        return self.retriever.hybrid_search(query, top_k=top_k, alpha=0.0)  # BM25 only


class SemanticOnlyRetriever:
    """Semantic-only baseline"""
    def __init__(self):
        self.retriever = HybridRetriever(collection_name="semantic_only")
    
    def index(self, documents: List[str]):
        self.retriever.index_documents(documents)
    
    def search(self, query: str, top_k: int = 10):
        return self.retriever.hybrid_search(query, top_k=top_k, alpha=1.0)  # Semantic only


def benchmark_on_viquad(num_samples: int = 100):
    """
    Benchmark on ViQuAD dataset
    """
    print("="*80)
    print("RETRIEVAL BENCHMARK ON VIQUAD")
    print("="*80)
    
    # Load dataset
    print("\nLoading ViQuAD dataset...")
    dataset = load_from_disk("datasets/processed/viquad_augmented")
    val_data = dataset['validation'].select(range(num_samples))
    
    # Prepare documents and test set
    documents = []
    test_set = []
    
    for i, example in enumerate(val_data):
        doc_id = f"doc_{i}"
        documents.append(example['context'])
        
        # Create test item
        test_set.append({
            "query": example['question'],
            "relevant_docs": {doc_id}  # The document containing the answer
        })
    
    print(f"Prepared {len(documents)} documents and {len(test_set)} queries")
    
    # Initialize retrievers
    print("\nInitializing retrievers...")
    
    retrievers = {
        "Naive Chunking": NaiveRetriever(),
        "Smart Chunking": SmartRetriever(),
        "BM25 Only": BM25OnlyRetriever(),
        "Semantic Only": SemanticOnlyRetriever(),
        "Hybrid (α=0.5)": HybridRetriever(collection_name="hybrid_balanced"),
        "Hybrid (α=0.7)": HybridRetriever(collection_name="hybrid_semantic"),
    }
    
    # Index documents
    print("\nIndexing documents...")
    for name, retriever in retrievers.items():
        print(f"  Indexing {name}...")
        start = time.time()
        if hasattr(retriever, 'index'):
            retriever.index(documents)
        else:
            retriever.index_documents(documents)
        print(f"    Done in {time.time() - start:.2f}s")
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    results = compare_retrievers(test_set, retrievers, k_values=[5, 10])
    
    print()
    print_comparison_table(results)
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Find best performer
    best_recall = max(results.items(), key=lambda x: x[1]['recall@10'])
    best_mrr = max(results.items(), key=lambda x: x[1]['mrr'])
    
    print(f"\nBest Recall@10: {best_recall[0]} ({best_recall[1]['recall@10']:.4f})")
    print(f"Best MRR: {best_mrr[0]} ({best_mrr[1]['mrr']:.4f})")
    
    # Improvement analysis
    naive_recall = results["Naive Chunking"]['recall@10']
    smart_recall = results["Smart Chunking"]['recall@10']
    improvement = (smart_recall - naive_recall) / naive_recall * 100
    
    print(f"\nSmart Chunking Improvement:")
    print(f"   Recall@10: {improvement:+.2f}%")
    
    bm25_mrr = results["BM25 Only"]['mrr']
    hybrid_mrr = results["Hybrid (α=0.7)"]['mrr']
    hybrid_improvement = (hybrid_mrr - bm25_mrr) / bm25_mrr * 100
    
    print(f"\nHybrid Search Improvement (vs BM25):")
    print(f"   MRR: {hybrid_improvement:+.2f}%")


def quick_benchmark():
    """Quick benchmark with sample data"""
    print("="*80)
    print("QUICK RETRIEVAL BENCHMARK")
    print("="*80)
    
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on training algorithms.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "Natural language processing enables computers to understand and generate human language.",
        "Computer vision allows machines to interpret and analyze visual information from images.",
        "Reinforcement learning trains agents by rewarding desired behaviors.",
    ]
    
    # Test queries
    test_set = [
        {
            "query": "How do neural networks work?",
            "relevant_docs": {"doc_1"}  # Deep learning document
        },
        {
            "query": "What is machine learning?",
            "relevant_docs": {"doc_0"}  # ML document
        },
    ]
    
    # Assign doc IDs
    for i in range(len(documents)):
        test_set[0]["relevant_docs"] = test_set[0]["relevant_docs"] | {f"doc_{i}"} if i == 1 else test_set[0]["relevant_docs"]
    
    # Initialize retrievers
    retrievers = {
        "BM25": BM25OnlyRetriever(),
        "Semantic": SemanticOnlyRetriever(),
        "Hybrid": HybridRetriever(collection_name="quick_test"),
    }
    
    # Index
    for name, retriever in retrievers.items():
        if hasattr(retriever, 'index'):
            retriever.index(documents)
        else:
            retriever.index_documents(documents)
    
    # Evaluate
    results = compare_retrievers(test_set, retrievers, k_values=[3, 5])
    
    print()
    print_comparison_table(results)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark retrieval systems")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                       help="Benchmark mode")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of samples for full benchmark")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        quick_benchmark()
    else:
        benchmark_on_viquad(args.samples)
