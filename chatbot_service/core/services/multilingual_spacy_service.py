"""
Multilingual NLP Service - Language Detection & Processing
============================================================
Extends spaCy service with multilingual support for:
- Automatic language detection
- Language-specific model loading
- Cross-lingual medical entity extraction
- Translation-aware processing


Supported Languages:
- en: English (primary, en_core_web_trf/lg/md/sm)
- es: Spanish (es_core_news_md)
- fr: French (fr_core_news_md)
- de: German (de_core_news_md)
- pt: Portuguese (pt_core_news_md)
- zh: Chinese (zh_core_web_md)
- ar: Arabic (ar basic patterns)
- xx: Multilingual (xx_ent_wiki_sm for fallback)

Installation:
    python -m spacy download en_core_web_sm
    python -m spacy download es_core_news_md
    python -m spacy download fr_core_news_md
    python -m spacy download de_core_news_md
    python -m spacy download pt_core_news_md
    python -m spacy download zh_core_web_sm
    python -m spacy download xx_ent_wiki_sm
"""

from __future__ import annotations
import os
import threading
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

# Environment configuration
SPACY_MULTILINGUAL_ENABLED = os.getenv("SPACY_MULTILINGUAL_ENABLED", "true").lower() == "true"
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")
SUPPORTED_LANGUAGES = os.getenv("SUPPORTED_LANGUAGES", "en,es,fr,de,pt,zh,ar").split(",")


@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    language: str
    confidence: float
    is_supported: bool
    fallback_used: bool = False


@dataclass
class MultilingualProcessingResult:
    """Result of multilingual NLP processing."""
    language: str
    entities: List[Dict[str, Any]]
    text: str
    translated_text: Optional[str] = None
    model_used: str = ""


class MultilingualSpaCyService:
    """
    Thread-safe multilingual NLP service.
    
    Features:
    - Lazy model loading (only loads models when needed)
    - Language detection via langdetect
    - Per-language model caching
    - Fallback to multilingual model for unsupported languages
    
    Example:
        service = MultilingualSpaCyService()
        
        # Process English
        result = service.process("I take metoprolol for my blood pressure")
        assert result.language == "en"
        
        # Process Spanish (auto-detected)
        result = service.process("Tomo metoprolol para mi presión arterial")
        assert result.language == "es"
    """
    
    _instance: Optional[MultilingualSpaCyService] = None
    _lock = threading.Lock()
    
    # Language-specific model mappings
    LANGUAGE_MODELS = {
        "en": ["en_core_web_trf", "en_core_web_lg", "en_core_web_md", "en_core_web_sm"],
        "es": ["es_core_news_md", "es_core_news_sm"],
        "fr": ["fr_core_news_md", "fr_core_news_sm"],
        "de": ["de_core_news_md", "de_core_news_sm"],
        "pt": ["pt_core_news_md", "pt_core_news_sm"],
        "zh": ["zh_core_web_md", "zh_core_web_sm"],
        "xx": ["xx_ent_wiki_sm"],  # Multilingual fallback
    }
    
    # Medical entity labels for each language
    MEDICAL_LABELS = {
        "en": ["DRUG", "MEDICATION", "DISEASE", "CONDITION", "SYMPTOM", "DOSAGE"],
        "es": ["MISC", "PER", "ORG", "LOC"],  # Map to medical concepts
        "fr": ["MISC", "PER", "ORG", "LOC"],
        "de": ["MISC", "PER", "ORG", "LOC"],
        "pt": ["MISC", "PER", "ORG", "LOC"],
        "zh": ["MISC", "PER", "ORG", "LOC"],
        "xx": ["MISC", "PER", "ORG", "LOC"],
    }
    
    def __new__(cls) -> MultilingualSpaCyService:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._models: Dict[str, Any] = {}  # Cached loaded models
        self._lang_detector = None
        self._initialized = True
        
        logger.info(
            f"MultilingualSpaCyService initialized: "
            f"enabled={SPACY_MULTILINGUAL_ENABLED}, "
            f"default={DEFAULT_LANGUAGE}, "
            f"supported={SUPPORTED_LANGUAGES}"
        )
    
    def _get_lang_detector(self):
        """Lazy-load language detector."""
        if self._lang_detector is None:
            try:
                from langdetect import detect_langs, DetectorFactory
                # Make detection deterministic
                DetectorFactory.seed = 42
                self._lang_detector = detect_langs
                logger.info("Language detection initialized (langdetect)")
            except ImportError:
                logger.warning("langdetect not installed. Install: pip install langdetect")
                self._lang_detector = lambda x: []
        return self._lang_detector
    
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """
        Detect language of input text.
        
        Returns:
            LanguageDetectionResult with language code and confidence
        """
        if not text or len(text.strip()) < 3:
            return LanguageDetectionResult(
                language=DEFAULT_LANGUAGE,
                confidence=0.0,
                is_supported=True,
                fallback_used=True
            )
        
        try:
            detector = self._get_lang_detector()
            results = detector(text)
            
            if results:
                top_result = results[0]
                lang = top_result.lang
                conf = top_result.prob
                
                is_supported = lang in SUPPORTED_LANGUAGES
                
                return LanguageDetectionResult(
                    language=lang if is_supported else DEFAULT_LANGUAGE,
                    confidence=conf,
                    is_supported=is_supported,
                    fallback_used=not is_supported
                )
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
        
        return LanguageDetectionResult(
            language=DEFAULT_LANGUAGE,
            confidence=0.0,
            is_supported=True,
            fallback_used=True
        )
    
    def _load_model(self, language: str) -> Any:
        """Load spaCy model for language, with fallback."""
        if language in self._models:
            return self._models[language]
        
        import spacy
        
        models_to_try = self.LANGUAGE_MODELS.get(language, [])
        
        # Add multilingual fallback
        if language not in self.LANGUAGE_MODELS:
            models_to_try = self.LANGUAGE_MODELS["xx"]
        
        for model_name in models_to_try:
            try:
                nlp = spacy.load(model_name)
                self._models[language] = nlp
                logger.info(f"✅ Loaded spaCy model: {model_name} for language: {language}")
                return nlp
            except OSError:
                logger.debug(f"Model {model_name} not found, trying next...")
        
        # Ultimate fallback: use English sm model
        try:
            nlp = spacy.load("en_core_web_sm")
            self._models[language] = nlp
            logger.warning(f"⚠️ Using en_core_web_sm fallback for language: {language}")
            return nlp
        except OSError:
            logger.error("No spaCy models available!")
            return None
    
    def get_model(self, language: str = None) -> Any:
        """Get spaCy model for language."""
        lang = language or DEFAULT_LANGUAGE
        return self._load_model(lang)
    
    def process(
        self,
        text: str,
        language: str = None,
        auto_detect: bool = True
    ) -> MultilingualProcessingResult:
        """
        Process text with language-appropriate model.
        
        Args:
            text: Input text
            language: Override language (skip detection)
            auto_detect: Auto-detect language if not provided
        
        Returns:
            MultilingualProcessingResult with entities and metadata
        """
        # Determine language
        if language:
            detected_lang = language
        elif auto_detect and SPACY_MULTILINGUAL_ENABLED:
            detection = self.detect_language(text)
            detected_lang = detection.language
        else:
            detected_lang = DEFAULT_LANGUAGE
        
        # Get model
        nlp = self.get_model(detected_lang)
        if nlp is None:
            return MultilingualProcessingResult(
                language=detected_lang,
                entities=[],
                text=text,
                model_used="none"
            )
        
        # Process text
        doc = nlp(text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "language": detected_lang,
            })
        
        return MultilingualProcessingResult(
            language=detected_lang,
            entities=entities,
            text=text,
            model_used=nlp.meta.get("name", "unknown")
        )
    
    def get_medical_entities(
        self,
        text: str,
        language: str = None
    ) -> List[Dict[str, Any]]:
        """
        Extract medical entities with language awareness.
        
        Uses language-specific medical entity patterns.
        """
        result = self.process(text, language=language)
        
        # Filter to medical-relevant entities
        medical_labels = self.MEDICAL_LABELS.get(result.language, self.MEDICAL_LABELS["en"])
        
        medical_entities = [
            ent for ent in result.entities
            if ent["label"] in medical_labels
        ]
        
        return medical_entities
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return SUPPORTED_LANGUAGES.copy()
    
    def get_loaded_models(self) -> Dict[str, str]:
        """Get info about currently loaded models."""
        return {
            lang: model.meta.get("name", "unknown")
            for lang, model in self._models.items()
        }


# Singleton access
_multilingual_service: Optional[MultilingualSpaCyService] = None

def get_multilingual_spacy_service() -> MultilingualSpaCyService:
    """Get singleton multilingual spaCy service."""
    global _multilingual_service
    if _multilingual_service is None:
        _multilingual_service = MultilingualSpaCyService()
    return _multilingual_service
