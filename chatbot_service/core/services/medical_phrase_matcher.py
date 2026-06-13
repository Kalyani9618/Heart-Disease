"""
Medical Phrase Matcher
======================
Efficient matching for medical terminology using spaCy's PhraseMatcher.

Complexity: matching cost is linear in the number of document tokens and
proportional to the number of patterns. spaCy's PhraseMatcher is highly
optimized but is *not* constant-time (O(1)).
"""

from typing import List, Dict, Any, Union
import json
import os
import logging

try:
    from spacy.matcher import PhraseMatcher
    from spacy.language import Language
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except Exception:
    PhraseMatcher = Language = Doc = None  # type: ignore
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available in medical_phrase_matcher")

logger = logging.getLogger(__name__)

class MedicalPhraseMatcher:
    """Efficient matching for medical terminology (linear in doc tokens, proportional to patterns)."""
    
    def __init__(self, nlp: Language, data_dir: str = "data"):
        self.nlp = nlp
        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self.data_dir = data_dir
        self._load_terminology()
    
    def _load_terminology(self) -> None:
        """Load medical terms from JSON files."""
        # Load different term categories
        term_files = {
            "DRUG": "drugs.json",
            "SYMPTOM": "symptoms.json",
            "INTERACTION": "interactions.json",
        }
        
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(base_path, self.data_dir)
        
        for label, filename in term_files.items():
            filepath = os.path.join(data_path, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        terms = self._extract_terms(data)
                        # Create patterns efficiently using make_doc (faster, no pipeline overhead)
                        patterns = [self.nlp.make_doc(text) for text in terms]
                        self.matcher.add(label, patterns)
                        logger.info(f"Loaded {len(terms)} terms for {label}")
                except Exception as e:
                    logger.warning(f"Failed to load terms from {filename}: {e}")
            else:
                logger.warning(f"Terms file not found: {filepath}")
    
    def _extract_terms(self, data: Any) -> List[str]:
        """Extract term strings from various JSON structures."""
        terms = set()
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str) and len(key) > 2:
                    terms.add(key)
                if isinstance(value, (dict, list)):
                    terms.update(self._extract_terms(value))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str) and len(item) > 2:
                    terms.add(item)
                else:
                    terms.update(self._extract_terms(item))
        return list(terms)
    
    def find_matches(self, doc: Doc) -> List[Dict[str, Any]]:
        """Find all medical term matches in document."""
        matches = self.matcher(doc)
        results = []
        for match_id, start, end in matches:
            span = doc[start:end]
            results.append({
                "text": span.text,
                "label": self.nlp.vocab.strings[match_id],
                "start": span.start_char,
                "end": span.end_char,
            })
        return results
