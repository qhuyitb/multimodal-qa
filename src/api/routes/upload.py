"""API Upload và Index Tài liệu.

Upload files và tự động index vào ChromaDB cho QA.
Nhiệm vụ: đọc file -> chunk đơn giản -> lưu vào ChromaDB.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pathlib import Path
from datetime import datetime
from typing import List
import uuid
import re

from .chat import get_conversational_qa

router = APIRouter(prefix="/api/v1/upload", tags=["Upload & Index"])


def _simple_chunks(text: str, chunk_size: int = 500) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [normalized[i : i + chunk_size] for i in range(0, len(normalized), chunk_size)]


async def _process_file(file: UploadFile, conversational_qa):
    allowed_extensions = {".txt", ".pdf", ".docx"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File không được hỗ trợ. Chỉ chấp nhận: {', '.join(allowed_extensions)}"
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File rỗng")

    if file_ext == ".txt":
        content_text = content.decode('utf-8')
    elif file_ext == ".pdf":
        import pdfplumber, io
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            content_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif file_ext == ".docx":
        import docx, io
        doc = docx.Document(io.BytesIO(content))
        content_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    else:
        content_text = ""

    if not content_text.strip():
        raise HTTPException(status_code=400, detail="Không thể trích xuất text từ file")

    chunks = _simple_chunks(content_text, chunk_size=500)
    if not chunks:
        raise HTTPException(status_code=400, detail="Không thể chia nhỏ document")

    base_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    chunk_ids = [f"{base_id}_chunk_{i}" for i in range(len(chunks))]
    chunk_metadatas = [{
        "filename": file.filename,
        "type": file_ext[1:],
        "uploaded_at": datetime.now().isoformat(),
        "source": f"uploaded/{file.filename}",
        "chunk_index": i,
        "total_chunks": len(chunks)
    } for i in range(len(chunks))]

    conversational_qa.hybrid_retrieval.index_documents(
        documents=chunks,
        ids=chunk_ids,
        metadata=chunk_metadatas,
        append=True
    )

    return {
        "document_id": base_id,
        "filename": file.filename,
        "chunks": len(chunks),
        "size_bytes": len(content),
        "text_length": len(content_text),
    }


@router.post("/document")
async def upload_and_index_document(
    file: UploadFile = File(...),
    conversational_qa = Depends(get_conversational_qa)
):
    """Upload 1 file (TXT/PDF/DOCX) and index in-memory."""
    try:
        result = await _process_file(file, conversational_qa)
        return {
            "success": True,
            **result,
            "message": f"Đã upload và index tài liệu '{result['filename']}' ({result['chunks']} đoạn) thành công!"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý file: {str(e)}")


@router.post("/documents")
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    conversational_qa = Depends(get_conversational_qa)
):
    """Upload nhiều file trong một lần (TXT/PDF/DOCX)."""
    results = []
    for file in files:
        try:
            res = await _process_file(file, conversational_qa)
            res["success"] = True
            results.append(res)
        except HTTPException as he:
            results.append({"success": False, "filename": file.filename, "error": he.detail})
        except Exception as e:
            results.append({"success": False, "filename": file.filename, "error": str(e)})

    return {
        "uploaded": [r for r in results if r.get("success")],
        "failed": [r for r in results if not r.get("success")]
    }


@router.get("/stats")
async def get_upload_stats(conversational_qa = Depends(get_conversational_qa)):
    """Thống kê số lượng documents đã upload (in-memory)."""
    try:
        retriever = conversational_qa.hybrid_retrieval
        total_chunks = len(retriever.documents)
        metas = retriever.doc_metadata or []

        recent = []
        for i, meta in enumerate(metas):
            recent.append({
                "id": retriever.doc_ids[i] if i < len(retriever.doc_ids) else "",
                "filename": meta.get("filename", "Unknown"),
                "type": meta.get("type", "Unknown"),
                "uploaded_at": meta.get("uploaded_at", "Unknown")
            })
        recent = sorted(recent, key=lambda x: x.get("uploaded_at", ""), reverse=True)[:20]

        unique_files = {m.get("filename", "Unknown") for m in metas}

        return {
            "total_chunks": total_chunks,
            "total_files": len(unique_files),
            "recent_uploads": recent
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lấy thống kê: {str(e)}")


@router.post("/reload")
async def reload_documents(conversational_qa = Depends(get_conversational_qa)):
    """No-op reload for in-memory mode; keeps API compatible."""
    try:
        total_docs = len(conversational_qa.hybrid_retrieval.documents)
        return {
            "success": True,
            "total_documents": total_docs,
            "message": f"Đang dùng in-memory, không cần reload (hiện {total_docs} documents)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi reload documents: {str(e)}")


@router.post("/reset")
async def reset_documents(conversational_qa = Depends(get_conversational_qa)):
    """Xóa toàn bộ dữ liệu in-memory (DB về rỗng)."""
    try:
        conversational_qa.hybrid_retrieval.clear()
        return {"success": True, "message": "Đã xóa toàn bộ dữ liệu in-memory"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi reset: {str(e)}")
