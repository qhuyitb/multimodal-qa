from fastapi import APIRouter, HTTPException, UploadFile, File
from pathlib import Path

from src.api.schemas.document import (
    DocumentProcessRequest,
    DocumentProcessResponse,
    BatchDocumentProcessRequest,
    BatchDocumentProcessResponse,
    DocumentTranslateRequest,
    DocumentTranslateResponse,
    DocumentLanguageInfo,
    DocumentOutputFile
)
from src.pipelines.document_pipeline import create_document_pipeline
from src.core.config import get_config

router = APIRouter(prefix="/document", tags=["Document Processing"])


@router.post("/process", response_model=DocumentProcessResponse)
async def process_document(request: DocumentProcessRequest):
    document_path = Path(request.document_path)
    if not document_path.exists():
        raise HTTPException(status_code=404, detail="Document file not found")
    
    config = get_config()
    vector_store_path = Path(config.get("paths", {}).get("vector_db", "data/vector_db"))
    enable_translation = config.get("translation", {}).get("enabled", True)
    
    pipeline = create_document_pipeline(
        vector_store_path=vector_store_path,
        enable_translation=enable_translation
    )
    
    result = pipeline.process(
        document_path=document_path,
        target_language=request.target_language,
        generate_translation=request.generate_translation,
        generate_dual_language=request.generate_dual_language,
        index_content=request.index_content,
        index_translation=request.index_translation
    )
    
    language_info = None
    if result.get("source_language"):
        language_info = DocumentLanguageInfo(
            language=result["source_language"],
            confidence=result.get("language_confidence", 0.0),
            method="auto-detection"
        )
    
    output_files = [
        DocumentOutputFile(
            path=path,
            language=result.get("source_language", "unknown"),
            file_type="original" if i == 0 else "translation"
        )
        for i, path in enumerate(result.get("output_files", []))
    ]
    
    return DocumentProcessResponse(
        document_path=str(document_path),
        success=result["success"],
        language_info=language_info,
        output_files=output_files,
        indexed=result.get("indexed", False),
        extracted_text_length=len(result.get("extracted_text", "")),
        error=result.get("error")
    )


@router.post("/batch-process", response_model=BatchDocumentProcessResponse)
async def batch_process_documents(request: BatchDocumentProcessRequest):
    config = get_config()
    vector_store_path = Path(config.get("paths", {}).get("vector_db", "data/vector_db"))
    enable_translation = config.get("translation", {}).get("enabled", True)
    
    pipeline = create_document_pipeline(
        vector_store_path=vector_store_path,
        enable_translation=enable_translation
    )
    
    document_paths = [Path(p) for p in request.document_paths]
    results = pipeline.batch_process(
        document_paths=document_paths,
        target_language=request.target_language,
        generate_translation=request.generate_translation,
        index_content=request.index_content
    )
    
    doc_responses = []
    successful = 0
    failed = 0
    
    for result in results:
        language_info = None
        if result.get("source_language"):
            language_info = DocumentLanguageInfo(
                language=result["source_language"],
                confidence=result.get("language_confidence", 0.0),
                method="auto-detection"
            )
        
        output_files = [
            DocumentOutputFile(
                path=path,
                language=result.get("source_language", "unknown"),
                file_type="processed"
            )
            for path in result.get("output_files", [])
        ]
        
        doc_response = DocumentProcessResponse(
            document_path=result["document_path"],
            success=result["success"],
            language_info=language_info,
            output_files=output_files,
            indexed=result.get("indexed", False),
            extracted_text_length=len(result.get("extracted_text", "")),
            error=result.get("error")
        )
        
        doc_responses.append(doc_response)
        
        if result["success"]:
            successful += 1
        else:
            failed += 1
    
    return BatchDocumentProcessResponse(
        results=doc_responses,
        total_processed=len(results),
        successful=successful,
        failed=failed
    )


@router.post("/translate", response_model=DocumentTranslateResponse)
async def translate_document(request: DocumentTranslateRequest):
    document_path = Path(request.document_path)
    if not document_path.exists():
        raise HTTPException(status_code=404, detail="Document file not found")
    
    pipeline = create_document_pipeline(enable_translation=True)
    
    result = pipeline.process(
        document_path=document_path,
        target_language=request.target_language,
        generate_translation=True,
        generate_dual_language=request.include_dual_language,
        index_content=False
    )
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Translation failed"))
    
    output_files = result.get("output_files", [])
    translated_file = None
    for f in output_files:
        if request.target_language in f:
            translated_file = f
            break
    
    if not translated_file:
        raise HTTPException(status_code=500, detail="Translated file not found")
    
    return DocumentTranslateResponse(
        success=True,
        source_language=request.source_language,
        target_language=request.target_language,
        output_path=translated_file,
        translation_time=None,
        character_count=len(result.get("translated_text", ""))
    )
