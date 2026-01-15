from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class DocumentProcessRequest(BaseModel):
    
    document_path: str = Field(..., description="Path to document file")
    detect_language: bool = Field(True, description="Auto-detect document language")
    source_language: Optional[str] = Field(None, description="Override detected language")
    target_language: Optional[str] = Field(None, description="Target language for translation")
    generate_translation: bool = Field(False, description="Generate translated document")
    generate_dual_language: bool = Field(False, description="Generate dual-language document")
    index_content: bool = Field(True, description="Index original content")
    index_translation: bool = Field(False, description="Index translated content")


class DocumentLanguageInfo(BaseModel):
    
    language: str = Field(..., description="Detected/specified language")
    confidence: float = Field(..., description="Detection confidence")
    method: str = Field(..., description="Detection method")


class DocumentOutputFile(BaseModel):
    
    path: str = Field(..., description="File path")
    language: str = Field(..., description="Content language")
    file_type: str = Field(..., description="File type (original/translation/dual)")


class DocumentProcessResponse(BaseModel):
    
    document_path: str = Field(..., description="Input document path")
    success: bool = Field(..., description="Processing success status")
    language_info: Optional[DocumentLanguageInfo] = Field(None, description="Language detection result")
    output_files: List[DocumentOutputFile] = Field(default_factory=list, description="Generated output files")
    indexed: bool = Field(False, description="Whether content was indexed")
    extracted_text_length: Optional[int] = Field(None, description="Length of extracted text")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchDocumentProcessRequest(BaseModel):
    
    document_paths: List[str] = Field(..., description="List of document paths")
    detect_language: bool = Field(True, description="Auto-detect languages")
    target_language: Optional[str] = Field(None, description="Target language for all documents")
    generate_translation: bool = Field(False, description="Generate translations")
    index_content: bool = Field(True, description="Index content")


class BatchDocumentProcessResponse(BaseModel):
    
    results: List[DocumentProcessResponse] = Field(..., description="Processing results")
    total_processed: int = Field(..., description="Total documents processed")
    successful: int = Field(..., description="Number of successful processings")
    failed: int = Field(..., description="Number of failed processings")


class DocumentTranslateRequest(BaseModel):
    
    document_path: str = Field(..., description="Path to document file")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    output_format: str = Field("txt", description="Output format (txt/pdf/docx)")
    include_dual_language: bool = Field(False, description="Include both languages")


class DocumentTranslateResponse(BaseModel):
    
    success: bool = Field(..., description="Translation success status")
    source_language: str = Field(..., description="Source language")
    target_language: str = Field(..., description="Target language")
    output_path: str = Field(..., description="Path to translated document")
    translation_time: Optional[float] = Field(None, description="Translation time in seconds")
    character_count: Optional[int] = Field(None, description="Number of characters translated")
    error: Optional[str] = Field(None, description="Error message if failed")
