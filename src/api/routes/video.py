from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pathlib import Path
import json
import aiofiles
import uuid
from datetime import datetime

from utils.helpers import get_data_dir
from api.schemas.video import (
    VideoProcessRequest,
    VideoProcessResponse,
    SubtitleGenerateRequest,
    SubtitleGenerateResponse,
    LanguageDetectionResult,
    SubtitleInfo,
    TranscriptSegment
)
from pipelines.video_pipeline import create_video_pipeline
from services.subtitle import SubtitleGenerator
from models.translation import get_translation_service
from core.config import get_config

router = APIRouter(prefix="/video", tags=["Video Processing"])
processing_tasks = {}


@router.post("/upload", response_model=dict)
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload video file và trả về task_id để tracking"""
    allowed_extensions = {".mp4", ".avi", ".mkv", ".mov", ".wmv"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    task_id = str(uuid.uuid4())
    upload_dir = get_data_dir("input/videos") / file_ext.lstrip(".")
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / f"{task_id}{file_ext}"
    
    async with aiofiles.open(video_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    
    processing_tasks[task_id] = {
        "status": "uploaded",
        "filename": file.filename,
        "path": str(video_path),
        "uploaded_at": datetime.now().isoformat(),
        "progress": 0
    }
    
    return {
        "task_id": task_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "status": "uploaded",
        "message": "Video uploaded successfully"
    }


@router.post("/process/{task_id}", response_model=dict)
async def start_video_processing(
    task_id: str,
    background_tasks: BackgroundTasks,
    detect_language: bool = True,
    source_language: str = None,
    target_languages: list[str] = None,
    generate_subtitles: bool = True,
    subtitle_formats: list[str] = ["srt", "vtt"],
    include_dual_language: bool = False,
    index_transcript: bool = True
):
    """Bắt đầu xử lý video ở background"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = processing_tasks[task_id]
    if task_info["status"] not in ["uploaded", "failed"]:
        return {"task_id": task_id, "status": task_info["status"]}
    
    processing_tasks[task_id]["status"] = "processing"
    processing_tasks[task_id]["started_at"] = datetime.now().isoformat()
    
    background_tasks.add_task(
        _process_video_background,
        task_id=task_id,
        video_path=Path(task_info["path"]),
        detect_language=detect_language,
        source_language=source_language,
        target_languages=target_languages or [],
        generate_subtitles=generate_subtitles,
        subtitle_formats=subtitle_formats,
        include_dual_language=include_dual_language,
        index_transcript=index_transcript
    )
    
    return {"task_id": task_id, "status": "processing"}


async def _process_video_background(
    task_id: str,
    video_path: Path,
    detect_language: bool,
    source_language: str,
    target_languages: list,
    generate_subtitles: bool,
    subtitle_formats: list,
    include_dual_language: bool,
    index_transcript: bool
):
    """Background task xử lý video"""
    config = get_config()
    pipeline = create_video_pipeline(
        vector_store_path=Path(config.get("paths", {}).get("vector_db")) if config.get("paths", {}).get("vector_db") else get_data_dir("vector_db"),
        enable_translation=config.get("translation", {}).get("enabled", True)
    )
    
    result = pipeline.process(
        video_path=video_path,
        detect_language=detect_language,
        source_language=source_language,
        target_languages=target_languages,
        generate_subtitles=generate_subtitles,
        subtitle_formats=subtitle_formats,
        include_dual_language=include_dual_language,
        index_transcript=index_transcript
    )
    
    processing_tasks[task_id]["status"] = "completed" if result["success"] else "failed"
    processing_tasks[task_id]["completed_at"] = datetime.now().isoformat()
    processing_tasks[task_id]["result"] = result
    processing_tasks[task_id]["progress"] = 100
    if not result["success"]:
        processing_tasks[task_id]["error"] = result.get("error", "Unknown error")


@router.get("/status/{task_id}", response_model=dict)
async def get_processing_status(task_id: str):
    """Lấy trạng thái xử lý video"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return processing_tasks[task_id]


@router.post("/process", response_model=VideoProcessResponse)
async def process_video(request: VideoProcessRequest):
    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    config = get_config()
    pipeline = create_video_pipeline(
        vector_store_path=Path(config.get("paths", {}).get("vector_db")) if config.get("paths", {}).get("vector_db") else get_data_dir("vector_db"),
        enable_translation=config.get("translation", {}).get("enabled", True)
    )
    
    result = pipeline.process(
        video_path=video_path,
        detect_language=request.detect_language,
        source_language=request.source_language,
        target_languages=request.target_languages or [],
        generate_subtitles=request.generate_subtitles,
        subtitle_formats=request.subtitle_formats,
        include_dual_language=request.include_dual_language,
        index_transcript=request.index_transcript
    )
    
    language_detection = None
    if result.get("language_detection"):
        lang_det = result["language_detection"]
        language_detection = LanguageDetectionResult(
            language=lang_det["language"],
            confidence=lang_det["confidence"],
            method=lang_det["method"],
            probabilities=lang_det.get("probabilities")
        )
    
    transcript = [
        TranscriptSegment(
            start=seg["start"],
            end=seg["end"],
            text=seg["text"]
        )
        for seg in result.get("transcript", [])
    ]
    
    subtitles = [
        SubtitleInfo(
            language=sub["language"],
            format=sub["format"],
            path=sub["path"],
            is_dual_language=sub.get("is_dual_language", False)
        )
        for sub in result.get("subtitles", [])
    ]
    
    return VideoProcessResponse(
        video_path=str(video_path),
        success=result["success"],
        language_detection=language_detection,
        transcript=transcript,
        subtitles=subtitles,
        indexed=result.get("indexed", False),
        transcript_path=result.get("transcript_path"),
        error=result.get("error")
    )


@router.post("/generate-subtitles", response_model=SubtitleGenerateResponse)
async def generate_subtitles(request: SubtitleGenerateRequest):
    transcript_path = Path(request.transcript_path)
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found")
    
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    
    transcript = transcript_data.get("segments", [])
    
    translation_service = get_translation_service()
    generator = SubtitleGenerator(translation_service=translation_service)
    
    output_dir = Path(request.output_dir) if request.output_dir else transcript_path.parent.parent / "subtitles"
    base_name = transcript_path.stem
    
    subtitle_files = generator.generate_multilingual_subtitles(
        transcript=transcript,
        source_lang=request.source_language,
        target_langs=request.target_languages,
        output_dir=output_dir,
        base_name=base_name,
        formats=request.formats,
        include_dual=request.include_dual_language
    )
    
    subtitles = []
    for lang, paths in subtitle_files.items():
        for path in paths:
            is_dual = "_" in lang
            fmt = path.suffix[1:]
            
            subtitles.append(SubtitleInfo(
                language=lang,
                format=fmt,
                path=str(path),
                is_dual_language=is_dual
            ))
    
    return SubtitleGenerateResponse(
        success=True,
        subtitles=subtitles
    )
