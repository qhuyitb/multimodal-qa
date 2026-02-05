"""
Updated Video Pipeline with Smart Chunking and Hybrid Retrieval
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.stt import create_stt_model
from models.qa_model import create_qa_model
from core.video_chunking import VideoChunker
from services.hybrid_retrieval import HybridRetriever
from services.reranker import CrossEncoderReranker
from typing import List, Dict, Optional
import json


class VideoPipelineV2:
    """
    Enhanced video pipeline with:
    - Smart chunking (timestamp-aware)
    - Hybrid retrieval (BM25 + Semantic)
    - Cross-encoder reranking (optional)
    - QA with source citation
    """
    
    def __init__(
        self,
        stt_model: str = "base",
        qa_checkpoint: str = "models/xlm_roberta_qa/stage2_best",
        use_reranking: bool = False
    ):
        # Models
        self.stt = create_stt_model(model_size=stt_model)
        self.qa_model = create_qa_model(checkpoint_path=qa_checkpoint, use_lora=False)
        
        # Chunking & Retrieval
        self.chunker = VideoChunker(
            max_chunk_size=512,
            overlap_ratio=0.1,
            similarity_threshold=0.7
        )
        self.retriever = HybridRetriever(collection_name="video_qa")
        
        # Optional reranking
        self.use_reranking = use_reranking
        if use_reranking:
            self.reranker = CrossEncoderReranker()
        
        self.video_index = {}  # {video_id: {chunks, metadata}}
    
    def process_video(
        self,
        video_path: str,
        video_id: str,
        chunking_strategy: str = "sentence"
    ) -> Dict:
        """
        Process video: STT → Chunking → Indexing
        
        Args:
            video_path: Path to video file
            video_id: Unique video identifier
            chunking_strategy: "sentence" or "topic"
        
        Returns:
            Processing results with chunks and metadata
        """
        print(f"Processing video: {video_id}")
        
        # 1. Speech-to-Text
        print("  [1/3] Transcribing audio...")
        transcript = self.stt.transcribe(video_path)
        
        # Convert to format for chunker
        transcript_segments = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            }
            for seg in transcript["segments"]
        ]
        
        # 2. Smart Chunking
        print(f"  [2/3] Chunking with strategy: {chunking_strategy}")
        if chunking_strategy == "sentence":
            chunks = self.chunker.chunk_by_sentence(transcript_segments)
        else:
            chunks = self.chunker.chunk_by_topic(transcript_segments)
        
        print(f"      Created {len(chunks)} chunks")
        
        # 3. Index to Vector Store
        print("  [3/3] Indexing to vector database...")
        
        # Prepare for indexing
        chunk_texts = [c["text"] for c in chunks]
        chunk_metadata = [
            {
                "video_id": video_id,
                "chunk_id": i,
                "start_time": c["start_time"],
                "end_time": c["end_time"],
                "duration": c["duration"]
            }
            for i, c in enumerate(chunks)
        ]
        
        self.retriever.index_documents(chunk_texts, chunk_metadata)
        
        # Store in memory
        self.video_index[video_id] = {
            "chunks": chunks,
            "metadata": {
                "num_chunks": len(chunks),
                "total_duration": transcript_segments[-1]["end"] if transcript_segments else 0,
                "language": transcript.get("language", "unknown")
            }
        }
        
        return {
            "video_id": video_id,
            "num_chunks": len(chunks),
            "metadata": self.video_index[video_id]["metadata"]
        }
    
    def answer_question(
        self,
        question: str,
        video_id: Optional[str] = None,
        top_k_retrieval: int = 5,
        top_k_rerank: int = 3,
        return_sources: bool = True
    ) -> Dict:
        """
        Answer question about video(s)
        
        Args:
            question: User question
            video_id: Optional specific video ID
            top_k_retrieval: Number of chunks to retrieve
            top_k_rerank: Number of chunks after reranking
            return_sources: Whether to return source citations
        
        Returns:
            Answer with confidence, sources, and timestamps
        """
        print(f"\nQuestion: {question}")
        
        # 1. Retrieve relevant chunks
        print(f"  [1/3] Retrieving top-{top_k_retrieval} relevant chunks...")
        
        # Use hybrid search
        retrieved = self.retriever.hybrid_search(
            question,
            top_k=top_k_retrieval,
            alpha=0.7  # Favor semantic search
        )
        
        # Filter by video_id if specified
        if video_id:
            retrieved = [
                r for r in retrieved
                if r['metadata'].get('video_id') == video_id
            ]
        
        if not retrieved:
            return {
                "answer": "No relevant information found.",
                "confidence": 0.0,
                "sources": []
            }
        
        # 2. Optional reranking
        if self.use_reranking and len(retrieved) > 1:
            print(f"  [2/3] Reranking to top-{top_k_rerank}...")
            retrieved = self.reranker.rerank(
                question,
                retrieved,
                top_k=top_k_rerank
            )
        else:
            retrieved = retrieved[:top_k_rerank]
        
        # 3. QA on each chunk and select best answer
        print(f"  [3/3] Generating answers from {len(retrieved)} chunks...")
        
        answers = []
        for chunk in retrieved:
            try:
                qa_result = self.qa_model.predict(
                    question,
                    chunk['document'],
                    top_k=1
                )
                
                if qa_result:
                    answers.append({
                        "text": qa_result[0]["text"],
                        "confidence": qa_result[0]["score"],
                        "chunk_score": chunk.get('rerank_score', chunk['score']),
                        "metadata": chunk['metadata']
                    })
            except Exception as e:
                print(f"      Error processing chunk: {e}")
                continue
        
        if not answers:
            return {
                "answer": "Could not generate answer.",
                "confidence": 0.0,
                "sources": []
            }
        
        # Select best answer (highest combined score)
        for ans in answers:
            ans['combined_score'] = ans['confidence'] * 0.7 + ans['chunk_score'] * 0.3
        
        best_answer = max(answers, key=lambda x: x['combined_score'])
        
        # Prepare sources
        sources = []
        if return_sources:
            for ans in answers[:3]:  # Top 3 sources
                sources.append({
                    "video_id": ans['metadata']['video_id'],
                    "timestamp": f"{ans['metadata']['start_time']:.1f}s - {ans['metadata']['end_time']:.1f}s",
                    "chunk_id": ans['metadata']['chunk_id'],
                    "relevance": ans['chunk_score']
                })
        
        return {
            "answer": best_answer["text"],
            "confidence": best_answer["confidence"],
            "combined_score": best_answer["combined_score"],
            "sources": sources
        }
    
    def get_video_info(self, video_id: str) -> Optional[Dict]:
        """Get indexed video information"""
        return self.video_index.get(video_id)
    
    def save_index(self, output_path: str):
        """Save video index to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.video_index, f, ensure_ascii=False, indent=2)
    
    def load_index(self, input_path: str):
        """Load video index from file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            self.video_index = json.load(f)


if __name__ == "__main__":
    # Example usage
    print("="*80)
    print("VIDEO PIPELINE V2 - WITH SMART CHUNKING & HYBRID RETRIEVAL")
    print("="*80)
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = VideoPipelineV2(
        stt_model="base",
        use_reranking=True
    )
    
    # Process video (example)
    video_path = "examples/videos/sample.mp4"
    
    if Path(video_path).exists():
        print(f"\nProcessing video: {video_path}")
        result = pipeline.process_video(
            video_path,
            video_id="sample_001",
            chunking_strategy="sentence"
        )
        
        print(f"\nProcessing complete:")
        print(f"   Chunks: {result['num_chunks']}")
        print(f"   Duration: {result['metadata']['total_duration']:.1f}s")
        
        # Ask questions
        questions = [
            "What is the main topic of this video?",
            "Can you summarize the key points?"
        ]
        
        for question in questions:
            answer = pipeline.answer_question(
                question,
                video_id="sample_001",
                return_sources=True
            )
            
            print(f"\nQ: {question}")
            print(f"A: {answer['answer']}")
            print(f"Confidence: {answer['confidence']:.2f}")
            if answer['sources']:
                print("Sources:")
                for src in answer['sources']:
                    print(f"  - {src['timestamp']} (relevance: {src['relevance']:.2f})")
    else:
        print(f"\nSample video not found: {video_path}")
        print("Place a video file at the path above to test the pipeline.")
