from typing import Dict, List, Optional, Union
from pathlib import Path
import time

# Chỉ hỗ trợ dịch giữa tiếng Anh và tiếng Việt
class TranslationService:
    
    MODEL_MAPPING = {
        ("en", "vi"): "Helsinki-NLP/opus-mt-en-vi",
        ("vi", "en"): "Helsinki-NLP/opus-mt-vi-en",
    }
    
    def __init__(self, device: str = "cpu", cache_dir: Optional[Path] = None, max_length: int = 512):
        self.device = device
        self.cache_dir = cache_dir
        self.max_length = max_length
        self._models = {}
        self._tokenizers = {}
    
    def _get_model_name(self, source_lang: str, target_lang: str) -> Optional[str]:
        source_lang = source_lang.lower()[:2]
        target_lang = target_lang.lower()[:2]
        
        if (source_lang, target_lang) in self.MODEL_MAPPING:
            return self.MODEL_MAPPING[(source_lang, target_lang)]
        
        return None
    
    def _load_model(self, source_lang: str, target_lang: str):
        model_name = self._get_model_name(source_lang, target_lang)
        
        if model_name in self._models:
            return self._models[model_name], self._tokenizers[model_name]
        
        from transformers import MarianMTModel, MarianTokenizer
        
        tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
        model = MarianMTModel.from_pretrained(model_name, cache_dir=self.cache_dir).to(self.device)
        
        self._models[model_name] = model
        self._tokenizers[model_name] = tokenizer
        
        return model, tokenizer
    
    def translate(self, text: Union[str, List[str]], source_lang: str, target_lang: str, batch_size: int = 8) -> Union[str, List[str]]:
        if not text:
            return text
        
        source_lang = source_lang.lower()[:2]
        target_lang = target_lang.lower()[:2]
        
        if source_lang == target_lang:
            return text
        
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        model_name = self._get_model_name(source_lang, target_lang)
        if not model_name:
            raise ValueError(f"Không hỗ trợ dịch từ {source_lang} sang {target_lang}. Chỉ hỗ trợ: en <-> vi")
        
        model, tokenizer = self._load_model(source_lang, target_lang)
        
        translations = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            outputs = model.generate(**inputs, max_length=self.max_length, num_beams=4, early_stopping=True)
            batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(batch_translations)
        
        return translations[0] if is_single else translations
    
    def translate_with_timing(self, text: Union[str, List[str]], source_lang: str, target_lang: str, **kwargs) -> Dict[str, any]:
        start_time = time.time()
        translation = self.translate(text, source_lang, target_lang, **kwargs)
        elapsed = time.time() - start_time
        
        char_count = len(text) if isinstance(text, str) else sum(len(t) for t in text)
        
        return {
            "translation": translation,
            "source_language": source_lang,
            "target_language": target_lang,
            "elapsed_time": elapsed,
            "characters": char_count,
            "chars_per_second": char_count / elapsed if elapsed > 0 else 0
        }
    
    def supports_language_pair(self, source_lang: str, target_lang: str) -> bool:
        source_lang = source_lang.lower()[:2]
        target_lang = target_lang.lower()[:2]
        if source_lang == target_lang:
            return True
        return self._get_model_name(source_lang, target_lang) is not None
    
    def get_supported_languages(self) -> List[str]:
        return ["en", "vi"]
    
    def clear_cache(self):
        self._models.clear()
        self._tokenizers.clear()


class TranslationCache:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = {}
        self._access_order = []
    
    def _make_key(self, text: str, source_lang: str, target_lang: str) -> str:
        return f"{source_lang}:{target_lang}:{hash(text)}"
    
    def get(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        key = self._make_key(text, source_lang, target_lang)
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, text: str, source_lang: str, target_lang: str, translation: str):
        key = self._make_key(text, source_lang, target_lang)
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        self._cache[key] = translation
        if key not in self._access_order:
            self._access_order.append(key)
    
    def clear(self):
        self._cache.clear()
        self._access_order.clear()


_service = None

def get_translation_service(**kwargs) -> TranslationService:
    global _service
    if _service is None:
        _service = TranslationService(**kwargs)
    return _service

def translate(text: Union[str, List[str]], source_lang: str, target_lang: str, **kwargs) -> Union[str, List[str]]:
    service = get_translation_service()
    return service.translate(text, source_lang, target_lang, **kwargs)
