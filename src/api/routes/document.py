from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pathlib import Path
import aiofiles
import uuid
from datetime import datetime

from utils.helpers import get_data_dir
from api.schemas.document import (
    DocumentProcessRequest,
    DocumentProcessResponse,
    BatchDocumentProcessRequest,
    BatchDocumentProcessResponse,
    DocumentTranslateRequest,
    DocumentTranslateResponse,
    DocumentLanguageInfo,
    DocumentOutputFile
)
from pipelines.document_pipeline import create_document_pipeline
from core.config import get_config

router = APIRouter(prefix="/document", tags=["Document Processing"])
processing_tasks = {}


@router.post("/upload", response_model=dict)
async def upload_document(file: UploadFile = File(...)):
    """Upload document và trả về task_id"""
    allowed_extensions = {".pdf", ".docx", ".txt", ".doc"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    task_id = str(uuid.uuid4())
    ext_folder = "pdf" if file_ext == ".pdf" else "docx" if file_ext in [".docx", ".doc"] else "txt"
    upload_dir = get_data_dir("input/documents") / ext_folder
    upload_dir.mkdir(parents=True, exist_ok=True)
    doc_path = upload_dir / f"{task_id}{file_ext}"
    
    async with aiofiles.open(doc_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    
    processing_tasks[task_id] = {
        "status": "uploaded",
        "filename": file.filename,
        "path": str(doc_path),
        "uploaded_at": datetime.now().isoformat(),
        "progress": 0
    }
    
    return {
        "task_id": task_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "status": "uploaded"
    }


@router.post("/process/{task_id}", response_model=dict)
async def start_document_processing(
    task_id: str,
    background_tasks: BackgroundTasks,
    target_language: str = None,
    generate_translation: bool = False,
    generate_dual_language: bool = False,
    index_content: bool = True,
    index_translation: bool = False
):
    """Bắt đầu xử lý document ở background"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = processing_tasks[task_id]
    if task_info["status"] not in ["uploaded", "failed"]:
        return {"task_id": task_id, "status": task_info["status"]}
    
    processing_tasks[task_id]["status"] = "processing"
    processing_tasks[task_id]["started_at"] = datetime.now().isoformat()
    
    background_tasks.add_task(
        _process_document_background,
        task_id=task_id,
        document_path=Path(task_info["path"]),
        target_language=target_language,
        generate_translation=generate_translation,
        generate_dual_language=generate_dual_language,
        index_content=index_content,
        index_translation=index_translation
    )
    
    return {"task_id": task_id, "status": "processing"}


async def _process_document_background(
    task_id: str,
    document_path: Path,
    target_language: str,
    generate_translation: bool,
    generate_dual_language: bool,
    index_content: bool,
    index_translation: bool
):
    """Background task xử lý document"""
    config = get_config()
    vector_store_path = Path(config.get("paths", {}).get("vector_db")) if config.get("paths", {}).get("vector_db") else get_data_dir("vector_db")
    enable_translation = config.get("translation", {}).get("enabled", True)
    
    pipeline = create_document_pipeline(
        vector_store_path=vector_store_path,
        enable_translation=enable_translation
    )
    
    result = pipeline.process(
        document_path=document_path,
        target_language=target_language,
        generate_translation=generate_translation,
        generate_dual_language=generate_dual_language,
        index_content=index_content,
        index_translation=index_translation
    )
    
    processing_tasks[task_id]["status"] = "completed" if result["success"] else "failed"
    processing_tasks[task_id]["completed_at"] = datetime.now().isoformat()
    processing_tasks[task_id]["result"] = result
    processing_tasks[task_id]["progress"] = 100
    if not result["success"]:
        processing_tasks[task_id]["error"] = result.get("error", "Unknown error")


@router.get("/status/{task_id}", response_model=dict)
async def get_processing_status(task_id: str):
    """Lấy trạng thái xử lý document"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return processing_tasks[task_id]


@router.post("/process", response_model=DocumentProcessResponse)
async def process_document(request: DocumentProcessRequest):
    document_path = Path(request.document_path)
    if not document_path.exists():
        raise HTTPException(status_code=404, detail="Document file not found")
    
    config = get_config()
    vector_store_path = Path(config.get("paths", {}).get("vector_db")) if config.get("paths", {}).get("vector_db") else get_data_dir("vector_db")
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
    vector_store_path = Path(config.get("paths", {}).get("vector_db")) if config.get("paths", {}).get("vector_db") else get_data_dir("vector_db")
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
