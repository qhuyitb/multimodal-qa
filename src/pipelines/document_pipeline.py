from functools import lru_cache
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from utils.helpers import get_data_dir


class DocumentPipeline:
    """Pipeline xử lý tài liệu: trích xuất, dịch, và index"""
    
    def __init__(
        self,
        extractor,
        language_detector=None,
        translation_service=None,
        vector_store=None,
        embedding_model=None,
        output_dir: Optional[Path] = None
    ):
        self.extractor = extractor
        self.language_detector = language_detector
        self.translation_service = translation_service
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.output_dir = output_dir or get_data_dir("output/documents")
    
    def process(
        self,
        document_path: Path,
        target_language: Optional[str] = None,
        generate_translation: bool = False,
        generate_dual_language: bool = False,
        index_content: bool = True,
        index_translation: bool = False
    ) -> Dict[str, Any]:
        """Xử lý document: trích xuất, dịch, và index vào vector store"""
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
        
        extracted_text = self.extractor.extract(document_path)
        results["extracted_text"] = extracted_text
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            return results
        
        if self.language_detector:
            detection = self.language_detector.detect_from_text(extracted_text)
            source_language = detection.get("language", "unknown")
            confidence = detection.get("confidence", 0.0)
            results["source_language"] = source_language
            results["language_confidence"] = confidence
        else:
            source_language = "unknown"
        
        base_name = document_path.stem
        original_output = self._save_text_file(
            extracted_text,
            base_name,
            source_language,
            document_path.suffix[1:]
        )
        results["output_files"].append(str(original_output))
        
        translated_text = None
        if (generate_translation or generate_dual_language) and target_language and self.translation_service:
            if source_language != target_language:
                
                chunks = self._split_text_for_translation(extracted_text)
                translated_chunks = self.translation_service.translate(
                    chunks,
                    source_lang=source_language,
                    target_lang=target_language,
                    batch_size=8
                )
                translated_text = "\n\n".join(translated_chunks)
                results["translated_text"] = translated_text
                
                if generate_translation:
                    trans_output = self._save_text_file(
                        translated_text,
                        base_name,
                        target_language,
                        document_path.suffix[1:]
                    )
                    results["output_files"].append(str(trans_output))
                
                if generate_dual_language:
                    dual_text = self._create_dual_language_text(
                        extracted_text,
                        translated_text,
                        source_language,
                        target_language
                    )
                    dual_output = self._save_text_file(
                        dual_text,
                        base_name,
                        f"{source_language}_{target_language}",
                        document_path.suffix[1:]
                    )
                    results["output_files"].append(str(dual_output))
        
        if index_content and self.vector_store:
            self._index_document(
                extracted_text,
                document_path,
                source_language,
                is_translation=False
            )
            results["indexed"] = True
        
        if index_translation and translated_text and self.vector_store:
            self._index_document(
                translated_text,
                document_path,
                target_language,
                is_translation=True
            )
        
        results["success"] = True
        
        return results
    
    def _save_text_file(
        self,
        text: str,
        base_name: str,
        language: str,
        doc_type: str
    ) -> Path:
        output_path = self.output_dir / doc_type / f"{base_name}.{language}.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        return output_path
    
    def _split_text_for_translation(
        self,
        text: str,
        max_chars: int = 2000
    ) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        
        paragraphs = text.split("\n\n")
        chunks, current_chunk, current_length = [], [], 0
        
        for para in paragraphs:
            para_length = len(para)
            if para_length > max_chars:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk, current_length = [], 0
                chunks.append(para)
                continue
            
            if current_length + para_length + 2 > max_chars and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk, current_length = [para], para_length
            else:
                current_chunk.append(para)
                current_length += para_length + 2
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        return chunks or [text]
    
    def _create_dual_language_text(
        self,
        original: str,
        translated: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        orig_paras = original.split("\n\n")
        trans_paras = translated.split("\n\n")
        parts = ["=== Dual Language Document ===", f"Original Language: {source_lang}",
                 f"Translation Language: {target_lang}", "=" * 50, ""]
        separator = "-" * 50
        source_tag, target_tag = source_lang.upper(), target_lang.upper()
        
        for i in range(max(len(orig_paras), len(trans_paras))):
            if i < len(orig_paras):
                parts.extend([f"[{source_tag}]", orig_paras[i], ""])
            if i < len(trans_paras):
                parts.extend([f"[{target_tag}]", trans_paras[i], ""])
            parts.extend([separator, ""])
        return "\n".join(parts)
    
    def _index_document(
        self,
        text: str,
        document_path: Path,
        language: str,
        is_translation: bool = False
    ):
        if not self.vector_store or not self.embedding_model:
            return
        
        from core.chunking import chunk_text
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        
        if not chunks:
            return
        
        embeddings = self.embedding_model.encode(chunks)
        
        doc_type = document_path.suffix[1:]
        source = str(document_path)
        metadata_list = [
            {
                "source": source,
                "chunk_index": i,
                "language": language,
                "is_translation": is_translation,
                "document_type": doc_type
            }
            for i in range(len(chunks))
        ]
        
        self.vector_store.add_documents(
            texts=chunks,
            embeddings=embeddings,
            metadata=metadata_list
        )
    
    def batch_process(
        self,
        document_paths: List[Path],
        **kwargs
    ) -> List[Dict[str, Any]]:
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
        results = {"document_path": str(document_path), "custom_translations": {}, "output_files": []}
        extracted_text = self.extractor.extract(document_path)
        source_language = (self.language_detector.detect_from_text(extracted_text).get("language", "unknown")
                          if self.language_detector else "unknown")
        
        # Lưu bản dịch tùy chỉnh
        if save_output:
            base_name = document_path.stem
            doc_type = document_path.suffix[1:]
            
            for target_lang, translated_text in custom_translations.items():
                output_path = self._save_text_file(
                    translated_text,
                    base_name,
                    target_lang,
                    doc_type
                )
                
                results["output_files"].append(str(output_path))
                results["custom_translations"][target_lang] = str(output_path)
        
        return results


def create_document_pipeline(
    vector_store_path: Optional[Path] = None,
    enable_translation: bool = True,
    **kwargs
) -> DocumentPipeline:
    from extractors.document import DocumentExtractor
    from services.language_detector import LanguageDetector
    
    # Khởi tạo các thành phần cốt lõi
    extractor = DocumentExtractor()
    language_detector = LanguageDetector()
    
    translation_service = None
    if enable_translation:
        from models.translation import get_translation_service
        translation_service = get_translation_service()
    
    vector_store = None
    embedding_model = None
    if vector_store_path:
        from services.vector_store import VectorStore
        from models.embedding import EmbeddingModel
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
