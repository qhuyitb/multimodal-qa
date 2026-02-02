from typing import List, Dict, Optional
from pathlib import Path
from datetime import timedelta


class Subtitle:
    
    def __init__(
        self,
        index: int,
        start_time: float,
        end_time: float,
        text: str
    ):
        self.index = index
        self.start_time = start_time
        self.end_time = end_time
        self.text = text
    
    def format_time_srt(self, seconds: float) -> str:
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millis = int((td.total_seconds() % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def format_time_vtt(self, seconds: float) -> str:
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millis = int((td.total_seconds() % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def to_srt(self) -> str:
        start = self.format_time_srt(self.start_time)
        end = self.format_time_srt(self.end_time)
        return f"{self.index}\n{start} --> {end}\n{self.text}\n"
    
    def to_vtt(self) -> str:
        start = self.format_time_vtt(self.start_time)
        end = self.format_time_vtt(self.end_time)
        return f"{start} --> {end}\n{self.text}\n"


class SubtitleGenerator:
    
    def __init__(self, translation_service=None):
        self.translation_service = translation_service
    
    def from_transcript(
        self,
        transcript: List[Dict],
        max_chars_per_line: int = 42,
        max_lines: int = 2
    ) -> List[Subtitle]:
        subtitles = []
        
        for i, segment in enumerate(transcript, 1):
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)
            text = segment.get("text", "").strip()
            
            lines = self._split_text(text, max_chars_per_line, max_lines)
            text = "\n".join(lines)
            
            subtitle = Subtitle(
                index=i,
                start_time=start,
                end_time=end,
                text=text
            )
            subtitles.append(subtitle)
        
        return subtitles
    
    def _split_text(self, text: str, max_chars: int, max_lines: int) -> List[str]:
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            
            # Check if adding word exceeds line limit
            if current_length + word_length + len(current_line) > max_chars:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_length = word_length
                else:
                    # Single word exceeds limit, force add
                    lines.append(word)
                    current_length = 0
            else:
                current_line.append(word)
                current_length += word_length
        
        # Add remaining words
        if current_line:
            lines.append(" ".join(current_line))
        
        # Limit to max lines
        return lines[:max_lines]
    
    def translate_subtitles(
        self,
        subtitles: List[Subtitle],
        source_lang: str,
        target_lang: str
    ) -> List[Subtitle]:
        if not self.translation_service:
            return subtitles
        
        texts = [sub.text for sub in subtitles]
        
        translated_texts = self.translation_service.translate(
            texts,
            source_lang=source_lang,
            target_lang=target_lang,
            batch_size=16
        )
        
        translated_subtitles = []
        for sub, trans_text in zip(subtitles, translated_texts):
            translated_sub = Subtitle(
                index=sub.index,
                start_time=sub.start_time,
                end_time=sub.end_time,
                text=trans_text
            )
            translated_subtitles.append(translated_sub)
        
        return translated_subtitles
    
    def create_dual_language_subtitles(
        self,
        subtitles: List[Subtitle],
        translated_subtitles: List[Subtitle]
    ) -> List[Subtitle]:
        dual_subtitles = []
        
        for orig, trans in zip(subtitles, translated_subtitles):
            dual_text = f"{orig.text}\n{trans.text}"
            dual_sub = Subtitle(
                index=orig.index,
                start_time=orig.start_time,
                end_time=orig.end_time,
                text=dual_text
            )
            dual_subtitles.append(dual_sub)
        
        return dual_subtitles
    
    def save_srt(self, subtitles: List[Subtitle], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for subtitle in subtitles:
                f.write(subtitle.to_srt())
                f.write("\n")
    
    def save_vtt(self, subtitles: List[Subtitle], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for subtitle in subtitles:
                f.write(subtitle.to_vtt())
                f.write("\n")
    
    def generate_multilingual_subtitles(
        self,
        transcript: List[Dict],
        source_lang: str,
        target_langs: List[str],
        output_dir: Path,
        base_name: str,
        formats: List[str] = ["srt", "vtt"],
        include_dual: bool = False
    ) -> Dict[str, List[Path]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        
        original_subtitles = self.from_transcript(transcript)
        results[source_lang] = []
        
        for fmt in formats:
            if fmt == "srt":
                output_path = output_dir / f"{base_name}.{source_lang}.srt"
                self.save_srt(original_subtitles, output_path)
                results[source_lang].append(output_path)
            elif fmt == "vtt":
                output_path = output_dir / f"{base_name}.{source_lang}.vtt"
                self.save_vtt(original_subtitles, output_path)
                results[source_lang].append(output_path)
        
        if self.translation_service:
            supported = self.translation_service.get_supported_languages()
            valid_targets = [lang for lang in target_langs if lang in supported and lang != source_lang]
            
            for target_lang in valid_targets:
                
                translated_subs = self.translate_subtitles(
                    original_subtitles,
                    source_lang,
                    target_lang
                )
                
                results[target_lang] = []
                
                # Save translated subtitles
                for fmt in formats:
                    if fmt == "srt":
                        output_path = output_dir / f"{base_name}.{target_lang}.srt"
                        self.save_srt(translated_subs, output_path)
                        results[target_lang].append(output_path)
                    elif fmt == "vtt":
                        output_path = output_dir / f"{base_name}.{target_lang}.vtt"
                        self.save_vtt(translated_subs, output_path)
                        results[target_lang].append(output_path)
                
                # Generate dual-language if requested
                if include_dual:
                    dual_subs = self.create_dual_language_subtitles(
                        original_subtitles,
                        translated_subs
                    )
                    dual_lang = f"{source_lang}_{target_lang}"
                    results[dual_lang] = []
                    
                    for fmt in formats:
                        if fmt == "srt":
                            output_path = output_dir / f"{base_name}.{dual_lang}.srt"
                            self.save_srt(dual_subs, output_path)
                            results[dual_lang].append(output_path)
                        elif fmt == "vtt":
                            output_path = output_dir / f"{base_name}.{dual_lang}.vtt"
                            self.save_vtt(dual_subs, output_path)
                            results[dual_lang].append(output_path)
        
        return results


def create_subtitles_from_transcript(
    transcript: List[Dict],
    output_path: Path,
    format: str = "srt",
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    translation_service=None
) -> Path:
    """
    Convenience function to create subtitles from transcript.
    
    Args:
        transcript: Transcript segments
        output_path: Output file path
        format: Subtitle format ("srt" or "vtt")
        source_lang: Source language (for translation)
        target_lang: Target language (for translation)
        translation_service: Translation service instance
        
    Returns:
        Path to generated subtitle file
    """
    generator = SubtitleGenerator(translation_service=translation_service)
    subtitles = generator.from_transcript(transcript)
    
    # Apply translation if requested
    if target_lang and source_lang and source_lang != target_lang:
        subtitles = generator.translate_subtitles(
            subtitles,
            source_lang,
            target_lang
        )
    
    # Save in requested format
    if format.lower() == "srt":
        generator.save_srt(subtitles, output_path)
    elif format.lower() == "vtt":
        generator.save_vtt(subtitles, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return output_path
