from typing import Dict, Optional, Any
from pathlib import Path


class LanguageDetector:
    """Service phát hiện ngôn ngữ từ text, audio, hoặc document"""
    
    def __init__(self):
        self._langdetect_available = False
        self._fasttext_model = None
        self._init_detectors()
    
    def _init_detectors(self):
        try:
            import langdetect
            self._langdetect_available = True
        except ImportError:
            pass
        
        try:
            import fasttext
            model_path = Path.home() / ".cache" / "fasttext" / "lid.176.bin"
            if model_path.exists():
                self._fasttext_model = fasttext.load_model(str(model_path))
        except ImportError:
            pass
    
    def detect_from_text(self, text: str, method: str = "auto") -> Dict[str, Any]:
        """Phát hiện ngôn ngữ từ text"""
        if not text or len(text.strip()) < 10:
            return {"language": "unknown", "confidence": 0.0}
        
        text = text.strip()
        
        if method == "auto":
            if self._fasttext_model:
                return self._detect_fasttext(text)
            elif self._langdetect_available:
                return self._detect_langdetect(text)
        elif method == "fasttext" and self._fasttext_model:
            return self._detect_fasttext(text)
        elif method == "langdetect" and self._langdetect_available:
            return self._detect_langdetect(text)
        
        return {"language": "unknown", "confidence": 0.0}
    
    def _detect_fasttext(self, text: str) -> Dict[str, Any]:
        text = text.replace("\n", " ").strip()
        predictions = self._fasttext_model.predict(text, k=1)
        lang = predictions[0][0].replace("__label__", "")
        conf = float(predictions[1][0])
        return {"language": lang, "confidence": conf}
    
    def _detect_langdetect(self, text: str) -> Dict[str, Any]:
        import langdetect
        from langdetect import DetectorFactory
        
        DetectorFactory.seed = 0
        lang = langdetect.detect(text)
        probs = langdetect.detect_langs(text)
        conf = max([p.prob for p in probs if p.lang == lang], default=0.0)
        return {"language": lang, "confidence": conf}
    
    def detect_from_audio(
        self,
        audio_path: Path,
        whisper_model: Optional[Any] = None,
        sample_duration: int = 30
    ) -> Dict[str, Any]:
        """Phát hiện ngôn ngữ từ file audio"""
        import whisper
        
        if whisper_model is None:
            whisper_model = whisper.load_model("base")
        
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio, length=sample_duration * 16000)
        mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
        
        _, probs = whisper_model.detect_language(mel)
        lang = max(probs, key=probs.get)
        conf = probs[lang]
        
        top5 = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return {
            "language": lang,
            "confidence": float(conf),
            "probabilities": {k: float(v) for k, v in top5.items()}
        }
    
    def detect_from_document(self, document_path: Path, sample_size: int = 5000) -> Dict[str, Any]:
        """Phát hiện ngôn ngữ từ document"""
        text = ""
        
        if document_path.suffix == ".txt":
            with open(document_path, "r", encoding="utf-8") as f:
                text = f.read(sample_size)
        
        elif document_path.suffix == ".pdf":
            import PyPDF2
            with open(document_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages[:5]:
                    text += page.extract_text()
                    if len(text) >= sample_size:
                        break
                text = text[:sample_size]
        
        elif document_path.suffix in [".docx", ".doc"]:
            from docx import Document
            doc = Document(document_path)
            text = " ".join([p.text for p in doc.paragraphs])[:sample_size]
        
        else:
            return {"language": "unknown", "confidence": 0.0}
        
        return self.detect_from_text(text)


def detect_language(content: Any, content_type: str = "text", **kwargs) -> Dict[str, Any]:
    detector = LanguageDetector()
    
    if content_type == "text":
        return detector.detect_from_text(content, **kwargs)
    elif content_type == "audio":
        return detector.detect_from_audio(content, **kwargs)
    elif content_type == "document":
        return detector.detect_from_document(content, **kwargs)
    
    return {"language": "unknown", "confidence": 0.0}