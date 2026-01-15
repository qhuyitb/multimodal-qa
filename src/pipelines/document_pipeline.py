"""
Document processing pipeline with translation support.
Handles extraction, language detection, optional translation, and indexing.
Supports generating translated or dual-language documents.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """
    End-to-end pipeline for document processing with multilingual support.
    """
    
    def __init__(
        self,
        extractor,
        language_detector=None,
        translation_service=None,
        vector_store=None,
        embedding_model=None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize document pipeline.
        
        Args:
            extractor: Document extractor
            language_detector: Language detection service
            translation_service: Translation service
            vector_store: Vector store for indexing
            embedding_model: Embedding model
            output_dir: Directory for output files
        """
        self.extractor = extractor
        self.language_detector = language_detector
        self.translation_service = translation_service
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.output_dir = output_dir or Path("data/output/documents")
        
        logger.info("Document pipeline initialized")
    
    def process(
        self,
        document_path: Path,
        target_language: Optional[str] = None,
        generate_translation: bool = False,
        generate_dual_language: bool = False,
        index_content: bool = True,
        index_translation: bool = False
    ) -> Dict[str, Any]:
        """
        Process a document through the full pipeline.
        
        Args:
            document_path: Path to input document
            target_language: Target language for translation
            generate_translation: Whether to generate translated document
            generate_dual_language: Whether to generate dual-language document
            index_content: Whether to index original content
            index_translation: Whether to index translated content
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing document: {document_path}")
        
        results = {
            "document_path": str(document_path),
            "success": False,
            "source_language": None,
            "target_language": target_language,
            "extracted_text": None,
            "translated_text": None,
            "output_files": [],
            "indexed": False
        }
        
        try:
            # Step 1: Extract text
            logger.info("Extracting text from document")
            extracted_text = self.extractor.extract(document_path)
            results["extracted_text"] = extracted_text
            
            if not extracted_text or len(extracted_text.strip()) < 10:
                logger.warning("Insufficient text extracted")
                return results
            
            # Step 2: Detect source language
            if self.language_detector:
                logger.info("Detecting document language")
                detection = self.language_detector.detect_from_text(extracted_text)
                source_language = detection.get("language", "unknown")
                confidence = detection.get("confidence", 0.0)
                results["source_language"] = source_language
                results["language_confidence"] = confidence
                logger.info(f"Detected language: {source_language} (confidence: {confidence:.2f})")
            else:
                source_language = "unknown"
            
            # Step 3: Save original text
            base_name = document_path.stem
            original_output = self.output_dir / document_path.suffix[1:] / f"{base_name}.{source_language}.txt"
            original_output.parent.mkdir(parents=True, exist_ok=True)
            
            with open(original_output, "w", encoding="utf-8") as f:
                f.write(extracted_text)
            results["output_files"].append(str(original_output))
            logger.info(f"Saved original text to {original_output}")
            
            # Step 4: Translation (if requested)
            translated_text = None
            if (generate_translation or generate_dual_language) and target_language:
                if self.translation_service and source_language != target_language:
                    logger.info(f"Translating document from {source_language} to {target_language}")
                    
                    # Split into chunks if too long
                    chunks = self._split_text_for_translation(extracted_text)
                    translated_chunks = self.translation_service.translate(
                        chunks,
                        source_lang=source_language,
                        target_lang=target_language,
                        batch_size=8
                    )
                    translated_text = "\n\n".join(translated_chunks)
                    results["translated_text"] = translated_text
                    
                    # Save translated document
                    if generate_translation:
                        trans_output = self.output_dir / document_path.suffix[1:] / f"{base_name}.{target_language}.txt"
                        with open(trans_output, "w", encoding="utf-8") as f:
                            f.write(translated_text)
                        results["output_files"].append(str(trans_output))
                        logger.info(f"Saved translated text to {trans_output}")
                    
                    # Save dual-language document
                    if generate_dual_language:
                        dual_output = self.output_dir / document_path.suffix[1:] / f"{base_name}.{source_language}_{target_language}.txt"
                        dual_text = self._create_dual_language_text(
                            extracted_text,
                            translated_text,
                            source_language,
                            target_language
                        )
                        with open(dual_output, "w", encoding="utf-8") as f:
                            f.write(dual_text)
                        results["output_files"].append(str(dual_output))
                        logger.info(f"Saved dual-language text to {dual_output}")
            
            # Step 5: Index content
            if index_content and self.vector_store:
                logger.info("Indexing original content")
                self._index_document(
                    extracted_text,
                    document_path,
                    source_language,
                    is_translation=False
                )
                results["indexed"] = True
            
            # Step 6: Index translation
            if index_translation and translated_text and self.vector_store:
                logger.info("Indexing translated content")
                self._index_document(
                    translated_text,
                    document_path,
                    target_language,
                    is_translation=True
                )
            
            results["success"] = True
            logger.info(f"Document processing completed successfully")
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def _split_text_for_translation(
        self,
        text: str,
        max_chars: int = 2000
    ) -> List[str]:
        """Split long text into chunks for translation."""
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            if current_length + para_length > max_chars and current_chunk:
                # Save current chunk and start new one
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # Add remaining paragraphs
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks
    
    def _create_dual_language_text(
        self,
        original: str,
        translated: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """Create dual-language document with original and translation."""
        header = f"=== Dual Language Document ===\n"
        header += f"Original Language: {source_lang}\n"
        header += f"Translation Language: {target_lang}\n"
        header += "=" * 50 + "\n\n"
        
        # Split into paragraphs for alignment
        orig_paras = original.split("\n\n")
        trans_paras = translated.split("\n\n")
        
        # Align paragraphs
        dual_text = header
        for i in range(max(len(orig_paras), len(trans_paras))):
            if i < len(orig_paras):
                dual_text += f"[{source_lang.upper()}]\n{orig_paras[i]}\n\n"
            if i < len(trans_paras):
                dual_text += f"[{target_lang.upper()}]\n{trans_paras[i]}\n\n"
            dual_text += "-" * 50 + "\n\n"
        
        return dual_text
    
    def _index_document(
        self,
        text: str,
        document_path: Path,
        language: str,
        is_translation: bool = False
    ):
        """Index document content in vector store."""
        if not self.vector_store or not self.embedding_model:
            logger.warning("Vector store or embedding model not available")
            return
        
        # Split into chunks
        from src.core.chunking import chunk_text
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks)
        
        # Prepare metadata
        metadata_list = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": str(document_path),
                "chunk_index": i,
                "language": language,
                "is_translation": is_translation,
                "document_type": document_path.suffix[1:]
            }
            metadata_list.append(metadata)
        
        # Add to vector store
        self.vector_store.add_documents(
            texts=chunks,
            embeddings=embeddings,
            metadata=metadata_list
        )
        
        logger.info(f"Indexed {len(chunks)} chunks from document")
    
    def batch_process(
        self,
        document_paths: List[Path],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            document_paths: List of document paths
            **kwargs: Arguments passed to process()
            
        Returns:
            List of processing results
        """
        results = []
        for doc_path in document_paths:
            result = self.process(doc_path, **kwargs)
            results.append(result)
        return results
    
    def process_with_custom_translation(
        self,
        document_path: Path,
        custom_translations: Dict[str, str],
        save_output: bool = True
    ) -> Dict[str, Any]:
        """
        Process document with user-provided custom translations.
        
        Args:
            document_path: Path to document
            custom_translations: Dictionary mapping target languages to translated texts
            save_output: Whether to save outputs
            
        Returns:
            Processing results
        """
        results = {
            "document_path": str(document_path),
            "custom_translations": {},
            "output_files": []
        }
        
        # Extract original text
        extracted_text = self.extractor.extract(document_path)
        
        # Detect source language
        if self.language_detector:
            detection = self.language_detector.detect_from_text(extracted_text)
            source_language = detection.get("language", "unknown")
        else:
            source_language = "unknown"
        
        # Save custom translations
        if save_output:
            base_name = document_path.stem
            
            for target_lang, translated_text in custom_translations.items():
                output_path = self.output_dir / document_path.suffix[1:] / f"{base_name}.{target_lang}.txt"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(translated_text)
                
                results["output_files"].append(str(output_path))
                results["custom_translations"][target_lang] = str(output_path)
                
                logger.info(f"Saved custom {target_lang} translation to {output_path}")
        
        return results


def create_document_pipeline(
    vector_store_path: Optional[Path] = None,
    enable_translation: bool = True,
    **kwargs
) -> DocumentPipeline:
    """
    Factory function to create document pipeline.
    
    Args:
        vector_store_path: Path to vector store
        enable_translation: Whether to enable translation
        **kwargs: Additional arguments
        
    Returns:
        Configured document pipeline
    """
    from src.extractors.document import DocumentExtractor
    from src.services.language_detector import LanguageDetector
    from src.models.translation import get_translation_service
    from src.services.vector_store import VectorStore
    from src.models.embedding import EmbeddingModel
    
    # Initialize components
    extractor = DocumentExtractor()
    language_detector = LanguageDetector()
    
    translation_service = None
    if enable_translation:
        translation_service = get_translation_service()
    
    vector_store = None
    embedding_model = None
    if vector_store_path:
        vector_store = VectorStore(vector_store_path)
        embedding_model = EmbeddingModel()
    
    return DocumentPipeline(
        extractor=extractor,
        language_detector=language_detector,
        translation_service=translation_service,
        vector_store=vector_store,
        embedding_model=embedding_model,
        **kwargs
    )
