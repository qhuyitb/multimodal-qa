from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class VideoProcessRequest(BaseModel):
    
    video_path: str = Field(..., description="Path to video file")
    detect_language: bool = Field(True, description="Auto-detect video language")
    source_language: Optional[str] = Field(None, description="Override detected language")
    target_languages: Optional[List[str]] = Field(None, description="Languages for subtitle translation")
    generate_subtitles: bool = Field(True, description="Generate subtitle files")
    subtitle_formats: List[str] = Field(["srt", "vtt"], description="Subtitle formats to generate")
    include_dual_language: bool = Field(False, description="Generate dual-language subtitles")
    index_transcript: bool = Field(True, description="Index transcript for QA")


class TranscriptSegment(BaseModel):
    
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcript text")


class LanguageDetectionResult(BaseModel):
    
    language: str = Field(..., description="Detected language code")
    confidence: float = Field(..., description="Detection confidence")
    method: str = Field(..., description="Detection method used")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Language probabilities")


class SubtitleInfo(BaseModel):
    
    language: str = Field(..., description="Subtitle language")
    format: str = Field(..., description="Subtitle format (srt/vtt)")
    path: str = Field(..., description="Path to subtitle file")
    is_dual_language: bool = Field(False, description="Is dual-language subtitle")


class VideoProcessResponse(BaseModel):
    
    video_path: str = Field(..., description="Processed video path")
    success: bool = Field(..., description="Processing success status")
    language_detection: Optional[LanguageDetectionResult] = Field(None, description="Language detection result")
    transcript: List[TranscriptSegment] = Field(default_factory=list, description="Video transcript")
    subtitles: List[SubtitleInfo] = Field(default_factory=list, description="Generated subtitle files")
    indexed: bool = Field(False, description="Whether transcript was indexed")
    transcript_path: Optional[str] = Field(None, description="Path to transcript file")
    error: Optional[str] = Field(None, description="Error message if failed")


class SubtitleGenerateRequest(BaseModel):
    
    transcript_path: str = Field(..., description="Path to transcript JSON file")
    source_language: str = Field(..., description="Source language code")
    target_languages: List[str] = Field(..., description="Target languages for translation")
    formats: List[str] = Field(["srt", "vtt"], description="Output formats")
    include_dual_language: bool = Field(False, description="Generate dual-language subtitles")
    output_dir: Optional[str] = Field(None, description="Output directory")


class SubtitleGenerateResponse(BaseModel):
    
    success: bool = Field(..., description="Generation success status")
    subtitles: List[SubtitleInfo] = Field(default_factory=list, description="Generated subtitle files")
    error: Optional[str] = Field(None, description="Error message if failed")
