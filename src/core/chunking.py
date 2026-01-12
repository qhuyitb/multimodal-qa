import re
from nltk.tokenize import sent_tokenize
from typing import List, Tuple, Optional, Dict, Any
import nltk
from dataclasses import dataclass, field


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


@dataclass
class Chunk:
    text: str
    start_idx: int
    end_idx: int
    chunk_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self):
        return len(self.text)
    
    def __repr__(self):
        return f"Chunk(id={self.chunk_id}, len={len(self.text)}, start={self.start_idx}, end={self.end_idx})"


class SmartChunker:
    
    def __init__(
        self, 
        chunk_size: int = 512, 
        chunk_overlap: int = 100, 
        min_chunk_size: int = 100, 
        language: str = 'auto'
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.language = language
        
    def _detect_language(self, text: str) -> str:
        vietnamese_chars = re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text.lower())
        return 'vietnamese' if len(vietnamese_chars) > 10 else 'english'
    
    def _split_sentences(self, text: str) -> List[str]:
        lang = self.language if self.language != 'auto' else self._detect_language(text)
        
        if lang == 'vietnamese':
            sentences = sent_tokenize(text, language='english')
        else:
            sentences = sent_tokenize(text, language='english')
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _merge_sentences_into_chunks(self, sentences: List[str]) -> List[str]:
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                overlap_text = ' '.join(current_chunk)
                if len(overlap_text) > self.chunk_overlap:
                    overlap_sentences = []
                    overlap_length = 0
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_length += len(s)
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_length = overlap_length
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_document(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text or len(text) < self.min_chunk_size:
            return []
        
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = self._split_sentences(text)
        chunk_texts = self._merge_sentences_into_chunks(sentences)
        
        chunks = []
        current_pos = 0
        
        for idx, chunk_text in enumerate(chunk_texts):
            start_idx = text.find(chunk_text, current_pos)
            if start_idx == -1:
                start_idx = current_pos
            end_idx = start_idx + len(chunk_text)
            
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['type'] = 'document'
            chunk_metadata['sentence_count'] = len([s for s in sentences if s in chunk_text])
            
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=end_idx,
                chunk_id=idx,
                metadata=chunk_metadata
            ))
            
            current_pos = end_idx
        
        return chunks
    
    def chunk_video_transcript(
        self, 
        text: str, 
        timestamps: Optional[List[Tuple[float, float]]] = None,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        if not text or len(text) < self.min_chunk_size:
            return []
        
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = self._split_sentences(text)
        
        if timestamps and len(timestamps) == len(sentences):
            sentence_with_time = list(zip(sentences, timestamps))
        else:
            sentence_with_time = [(s, None) for s in sentences]
        
        chunks = []
        current_chunk_sentences = []
        current_chunk_timestamps = []
        current_length = 0
        chunk_id = 0
        
        for sentence, timestamp in sentence_with_time:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata['type'] = 'video_transcript'
                chunk_metadata['sentence_count'] = len(current_chunk_sentences)
                
                if current_chunk_timestamps and all(t is not None for t in current_chunk_timestamps):
                    chunk_metadata['start_time'] = current_chunk_timestamps[0][0]
                    chunk_metadata['end_time'] = current_chunk_timestamps[-1][1]
                
                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=0,
                    end_idx=len(chunk_text),
                    chunk_id=chunk_id,
                    metadata=chunk_metadata
                ))
                
                chunk_id += 1
                
                overlap_count = 0
                overlap_length = 0
                for i in range(len(current_chunk_sentences) - 1, -1, -1):
                    if overlap_length + len(current_chunk_sentences[i]) <= self.chunk_overlap:
                        overlap_count += 1
                        overlap_length += len(current_chunk_sentences[i])
                    else:
                        break
                
                if overlap_count > 0:
                    current_chunk_sentences = current_chunk_sentences[-overlap_count:]
                    current_chunk_timestamps = current_chunk_timestamps[-overlap_count:]
                    current_length = overlap_length
                else:
                    current_chunk_sentences = []
                    current_chunk_timestamps = []
                    current_length = 0
            
            current_chunk_sentences.append(sentence)
            current_chunk_timestamps.append(timestamp)
            current_length += sentence_length + 1
        
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['type'] = 'video_transcript'
            chunk_metadata['sentence_count'] = len(current_chunk_sentences)
            
            if current_chunk_timestamps and all(t is not None for t in current_chunk_timestamps):
                chunk_metadata['start_time'] = current_chunk_timestamps[0][0]
                chunk_metadata['end_time'] = current_chunk_timestamps[-1][1]
            
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=0,
                end_idx=len(chunk_text),
                chunk_id=chunk_id,
                metadata=chunk_metadata
            ))
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict[str, float]:
        if not chunks:
            return {}
        
        lengths = [len(chunk.text) for chunk in chunks]
        overlaps = []
        
        for i in range(len(chunks) - 1):
            current_text = chunks[i].text
            next_text = chunks[i + 1].text
            
            max_overlap = min(len(current_text), len(next_text))
            for overlap_len in range(max_overlap, 0, -1):
                if current_text[-overlap_len:] == next_text[:overlap_len]:
                    overlaps.append(overlap_len)
                    break
            else:
                overlaps.append(0)
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_length': sum(lengths) / len(lengths),
            'min_chunk_length': min(lengths),
            'max_chunk_length': max(lengths),
            'avg_overlap': sum(overlaps) / len(overlaps) if overlaps else 0,
            'overlap_ratio': (sum(overlaps) / sum(lengths)) if sum(lengths) > 0 else 0
        }

