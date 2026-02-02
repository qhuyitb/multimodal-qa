"""
Smart Chunking for Video Transcripts
Maintains timestamp synchronization
"""

from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


class VideoChunker:
    """
    Smart chunking for video transcripts with timestamp preservation
    """
    
    def __init__(
        self,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        overlap_ratio: float = 0.1,
        similarity_threshold: float = 0.7
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_ratio = overlap_ratio
        self.similarity_threshold = similarity_threshold
        self.encoder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    def chunk_by_sentence(
        self,
        transcript: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Chunk transcript by sentence boundaries
        
        Args:
            transcript: List of {
                "start": float,
                "end": float,
                "text": str
            }
        
        Returns:
            List of chunks with metadata
        """
        chunks = []
        current_chunk = {
            "text": "",
            "start": None,
            "end": None,
            "sentences": []
        }
        
        for segment in transcript:
            # Split into sentences
            sentences = self._split_sentences(segment["text"])
            
            for sentence in sentences:
                sentence_info = {
                    "text": sentence,
                    "start": segment["start"],
                    "end": segment["end"]
                }
                
                # Check if adding this sentence exceeds max size
                if len(current_chunk["text"]) + len(sentence) > self.max_chunk_size:
                    # Save current chunk if it meets min size
                    if len(current_chunk["text"]) >= self.min_chunk_size:
                        chunks.append(self._finalize_chunk(current_chunk))
                        
                        # Start new chunk with overlap
                        overlap_sentences = self._get_overlap(current_chunk["sentences"])
                        current_chunk = self._create_new_chunk(overlap_sentences)
                
                # Add sentence to current chunk
                if current_chunk["start"] is None:
                    current_chunk["start"] = sentence_info["start"]
                
                current_chunk["text"] += sentence + " "
                current_chunk["end"] = sentence_info["end"]
                current_chunk["sentences"].append(sentence_info)
        
        # Add final chunk
        if current_chunk["text"]:
            chunks.append(self._finalize_chunk(current_chunk))
        
        return chunks
    
    def chunk_by_topic(
        self,
        transcript: List[Dict[str, any]],
        window_size: int = 5
    ) -> List[Dict[str, any]]:
        """
        Chunk by detecting topic shifts using semantic similarity
        
        Args:
            transcript: List of segments with text and timestamps
            window_size: Window size for topic detection
        
        Returns:
            Topic-coherent chunks
        """
        if not transcript:
            return []
        
        # Get embeddings for each segment
        texts = [seg["text"] for seg in transcript]
        embeddings = self.encoder.encode(texts)
        
        # Detect topic boundaries
        boundaries = [0]
        for i in range(window_size, len(embeddings) - window_size):
            # Compare before and after windows
            before_window = embeddings[i-window_size:i]
            after_window = embeddings[i:i+window_size]
            
            # Calculate similarity
            sim = self._window_similarity(before_window, after_window)
            
            # If similarity drops below threshold, mark as boundary
            if sim < self.similarity_threshold:
                boundaries.append(i)
        
        boundaries.append(len(transcript))
        
        # Create chunks based on boundaries
        chunks = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            chunk = self._create_chunk_from_segments(
                transcript[start_idx:end_idx]
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter (can be improved with spaCy)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap(self, sentences: List[Dict]) -> List[Dict]:
        """Get overlap sentences from previous chunk"""
        overlap_count = int(len(sentences) * self.overlap_ratio)
        return sentences[-overlap_count:] if overlap_count > 0 else []
    
    def _create_new_chunk(self, overlap_sentences: List[Dict]) -> Dict:
        """Create new chunk with overlap"""
        text = " ".join([s["text"] for s in overlap_sentences])
        start = overlap_sentences[0]["start"] if overlap_sentences else None
        end = overlap_sentences[-1]["end"] if overlap_sentences else None
        
        return {
            "text": text + " " if text else "",
            "start": start,
            "end": end,
            "sentences": overlap_sentences.copy()
        }
    
    def _finalize_chunk(self, chunk: Dict) -> Dict:
        """Finalize chunk with metadata"""
        return {
            "text": chunk["text"].strip(),
            "start_time": chunk["start"],
            "end_time": chunk["end"],
            "duration": chunk["end"] - chunk["start"] if chunk["start"] and chunk["end"] else 0,
            "num_sentences": len(chunk["sentences"]),
            "char_count": len(chunk["text"])
        }
    
    def _create_chunk_from_segments(
        self,
        segments: List[Dict]
    ) -> Dict:
        """Create chunk from list of segments"""
        text = " ".join([seg["text"] for seg in segments])
        return {
            "text": text,
            "start_time": segments[0]["start"],
            "end_time": segments[-1]["end"],
            "duration": segments[-1]["end"] - segments[0]["start"],
            "num_segments": len(segments),
            "char_count": len(text)
        }
    
    def _window_similarity(
        self,
        before: np.ndarray,
        after: np.ndarray
    ) -> float:
        """Calculate similarity between two windows"""
        before_mean = np.mean(before, axis=0).reshape(1, -1)
        after_mean = np.mean(after, axis=0).reshape(1, -1)
        return cosine_similarity(before_mean, after_mean)[0][0]


if __name__ == "__main__":
    # Test
    sample_transcript = [
        {"start": 0.0, "end": 3.5, "text": "Hello everyone. Welcome to this tutorial."},
        {"start": 3.5, "end": 7.2, "text": "Today we will learn about machine learning."},
        {"start": 7.2, "end": 11.0, "text": "Machine learning is a subset of AI."},
        {"start": 11.0, "end": 15.5, "text": "It allows computers to learn from data."},
    ]
    
    chunker = VideoChunker(max_chunk_size=100, overlap_ratio=0.1)
    
    print("=== Sentence-based Chunking ===")
    chunks = chunker.chunk_by_sentence(sample_transcript)
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Text: {chunk['text'][:80]}...")
        print(f"  Time: {chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s")
        print(f"  Sentences: {chunk['num_sentences']}")
    
    print("\n=== Topic-based Chunking ===")
    chunks = chunker.chunk_by_topic(sample_transcript, window_size=2)
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Text: {chunk['text'][:80]}...")
        print(f"  Time: {chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s")
