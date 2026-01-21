from fastapi import APIRouter, HTTPException, UploadFile, File
from pathlib import Path
import json

from src.utils.helpers import get_data_dir
from src.api.schemas.video import (
    VideoProcessRequest,
    VideoProcessResponse,
    SubtitleGenerateRequest,
    SubtitleGenerateResponse,
    LanguageDetectionResult,
    SubtitleInfo,
    TranscriptSegment
)
from src.pipelines.video_pipeline import create_video_pipeline
from src.services.subtitle import SubtitleGenerator
from src.models.translation import get_translation_service
from src.core.config import get_config

router = APIRouter(prefix="/video", tags=["Video Processing"])


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
