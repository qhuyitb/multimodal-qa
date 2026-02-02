"""
Smart Chunking for Documents
Preserves structure (paragraphs, sections, code blocks)
"""

from typing import List, Dict, Optional
import re
from pathlib import Path


class DocumentChunker:
    """
    Smart chunking for documents with structure preservation
    """
    
    def __init__(
        self,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        overlap_tokens: int = 50
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_tokens = overlap_tokens
    
    def chunk_by_paragraph(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Chunk by paragraph boundaries
        
        Args:
            text: Document text
            metadata: Optional metadata (filename, page, etc.)
        
        Returns:
            List of chunks with preserved paragraphs
        """
        paragraphs = self._split_paragraphs(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            # If paragraph alone exceeds max size, split it
            if para_length > self.max_chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, metadata))
                    current_chunk = []
                    current_length = 0
                
                # Split large paragraph by sentences
                sub_chunks = self._split_large_paragraph(para)
                chunks.extend([self._create_chunk([c], metadata) for c in sub_chunks])
                continue
            
            # Check if adding paragraph exceeds max size
            if current_length + para_length > self.max_chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, metadata))
                
                # Start new chunk with overlap
                overlap = self._get_overlap_text(current_chunk)
                current_chunk = [overlap] if overlap else []
                current_length = len(overlap) if overlap else 0
            
            current_chunk.append(para)
            current_length += para_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, metadata))
        
        return chunks
    
    def chunk_by_section(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Chunk by detecting sections (headers)
        
        Supports:
        - Markdown headers (# ## ###)
        - HTML headers (<h1> <h2>)
        - Numbered sections (1. 1.1 etc)
        """
        sections = self._detect_sections(text)
        chunks = []
        
        for section in sections:
            # If section is too large, chunk by paragraphs
            if len(section["text"]) > self.max_chunk_size:
                sub_chunks = self.chunk_by_paragraph(
                    section["text"],
                    {**metadata, "section": section["header"]} if metadata else {"section": section["header"]}
                )
                chunks.extend(sub_chunks)
            else:
                chunk = {
                    "text": section["text"],
                    "header": section["header"],
                    "level": section["level"],
                    "char_count": len(section["text"])
                }
                if metadata:
                    chunk["metadata"] = metadata
                chunks.append(chunk)
        
        return chunks
    
    def recursive_chunk(
        self,
        text: str,
        max_size: int = None,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Recursive splitting strategy:
        1. Try split by section
        2. If too large, split by paragraph
        3. If still large, split by sentence
        
        This ensures chunks never exceed max_size
        """
        max_size = max_size or self.max_chunk_size
        
        if len(text) <= max_size:
            return [self._create_chunk([text], metadata)]
        
        # Try section split
        sections = self._detect_sections(text)
        if len(sections) > 1:
            chunks = []
            for section in sections:
                sub_chunks = self.recursive_chunk(section["text"], max_size, metadata)
                chunks.extend(sub_chunks)
            return chunks
        
        # Try paragraph split
        paragraphs = self._split_paragraphs(text)
        if len(paragraphs) > 1:
            return self.chunk_by_paragraph(text, metadata)
        
        # Fall back to sentence split
        sentences = self._split_sentences(text)
        return self._chunk_sentences(sentences, max_size, metadata)
    
    def chunk_markdown(
        self,
        markdown_text: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Special handling for Markdown documents
        - Preserve code blocks
        - Keep lists intact
        - Maintain header hierarchy
        """
        # Extract code blocks first
        code_blocks = []
        text_without_code = markdown_text
        
        code_pattern = r'```[\s\S]*?```'
        for match in re.finditer(code_pattern, markdown_text):
            code_blocks.append({
                "text": match.group(0),
                "start": match.start(),
                "end": match.end()
            })
        
        # Remove code blocks temporarily
        text_without_code = re.sub(code_pattern, '[CODE_BLOCK]', markdown_text)
        
        # Chunk the rest
        chunks = self.chunk_by_section(text_without_code, metadata)
        
        # Re-insert code blocks
        for chunk in chunks:
            if '[CODE_BLOCK]' in chunk["text"]:
                # Find and replace with actual code
                for cb in code_blocks:
                    chunk["text"] = chunk["text"].replace('[CODE_BLOCK]', cb["text"], 1)
        
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'([.!?]+\s+)', text)
        result = []
        for i in range(0, len(sentences) - 1, 2):
            result.append(sentences[i] + sentences[i + 1])
        if len(sentences) % 2 == 1:
            result.append(sentences[-1])
        return [s.strip() for s in result if s.strip()]
    
    def _detect_sections(self, text: str) -> List[Dict]:
        """Detect sections in document"""
        sections = []
        
        # Markdown headers pattern
        md_pattern = r'^(#{1,6})\s+(.+?)$'
        lines = text.split('\n')
        
        current_section = {"header": "", "level": 0, "text": "", "start_line": 0}
        
        for i, line in enumerate(lines):
            match = re.match(md_pattern, line)
            if match:
                # Save previous section
                if current_section["text"]:
                    sections.append(current_section.copy())
                
                # Start new section
                level = len(match.group(1))
                header = match.group(2)
                current_section = {
                    "header": header,
                    "level": level,
                    "text": "",
                    "start_line": i
                }
            else:
                current_section["text"] += line + "\n"
        
        # Add last section
        if current_section["text"]:
            sections.append(current_section)
        
        # If no sections found, treat entire text as one section
        if not sections:
            sections = [{"header": "", "level": 0, "text": text, "start_line": 0}]
        
        return sections
    
    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """Split large paragraph by sentences"""
        sentences = self._split_sentences(paragraph)
        chunks = []
        current = ""
        
        for sent in sentences:
            if len(current) + len(sent) > self.max_chunk_size:
                if current:
                    chunks.append(current.strip())
                current = sent
            else:
                current += " " + sent
        
        if current:
            chunks.append(current.strip())
        
        return chunks
    
    def _chunk_sentences(
        self,
        sentences: List[str],
        max_size: int,
        metadata: Optional[Dict]
    ) -> List[Dict]:
        """Chunk list of sentences"""
        chunks = []
        current = ""
        
        for sent in sentences:
            if len(current) + len(sent) > max_size:
                if current:
                    chunks.append(self._create_chunk([current], metadata))
                current = sent
            else:
                current += " " + sent
        
        if current:
            chunks.append(self._create_chunk([current], metadata))
        
        return chunks
    
    def _get_overlap_text(self, chunks: List[str]) -> str:
        """Get overlap text from previous chunks"""
        combined = " ".join(chunks)
        words = combined.split()
        overlap_words = words[-self.overlap_tokens:] if len(words) > self.overlap_tokens else words
        return " ".join(overlap_words)
    
    def _create_chunk(
        self,
        paragraphs: List[str],
        metadata: Optional[Dict]
    ) -> Dict:
        """Create chunk dictionary"""
        text = "\n\n".join(paragraphs)
        chunk = {
            "text": text,
            "char_count": len(text),
            "num_paragraphs": len(paragraphs)
        }
        if metadata:
            chunk["metadata"] = metadata
        return chunk


This is the second paragraph with more details.

## Section 1

This section has information about topic 1.

### Subsection 1.1

More detailed information here.

## Section 2

This section talks about topic 2. It has multiple paragraphs.

This is the second paragraph of section 2.
    """
    
    chunker = DocumentChunker(max_chunk_size=200)
    
    print("=== Paragraph-based Chunking ===")
    chunks = chunker.chunk_by_paragraph(sample_text)
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}: {chunk['char_count']} chars")
        print(f"Text: {chunk['text'][:80]}...")
    
    print("\n\n=== Section-based Chunking ===")
    chunks = chunker.chunk_by_section(sample_text)
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}: {chunk.get('header', 'No header')}")
        print(f"Level: {chunk.get('level', 0)}")
        print(f"Text: {chunk['text'][:80]}...")
