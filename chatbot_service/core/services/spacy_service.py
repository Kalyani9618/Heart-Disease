"""
Centralized spaCy Service - Medical Safety Enhanced
====================================================
Thread-safe singleton with hybrid NLP pipeline:
- Transformer model for accuracy
- EntityRuler for deterministic drug detection
- Negation detection for medical safety
"""

from __future__ import annotations
import threading
from typing import Optional, List, Dict, Any
import logging

try:
    from spacy.tokens import Doc
except Exception:
    Doc = None  # type: ignore

logger = logging.getLogger(__name__)

class SpaCyService:
    """Thread-safe singleton for medical NLP pipeline."""
    
    _instance: Optional[SpaCyService] = None
    _lock = threading.Lock()
    
    # UPDATED: Model priority - prefer Transformer for healthcare
    MODEL_PRIORITY = [
        "en_core_web_trf",  # Transformer - highest accuracy (GPU recommended)
        "en_core_sci_md",   # Medical domain (if available)
        "en_core_web_lg",   # Large - good accuracy
        "en_core_web_md",   # Medium - balanced
        "en_core_web_sm",   # Small - fallback only
    ]
    
    def __new__(cls) -> SpaCyService:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._nlp = None
        self._model_name: str = ""
        self._medical_matcher = None
        self._initialized = True
    
    @property
    def nlp(self):
        """Lazy-load medical NLP pipeline on first access."""
        if self._nlp is None:
            self._load_pipeline()
        return self._nlp
    
    def _load_pipeline(self) -> None:
        """Load the medical NLP pipeline using factory."""
        # Import here to avoid circular imports
        from rag.nlp.factory import build_medical_nlp, get_medical_nlp
        
        self._nlp = get_medical_nlp()
        self._model_name = "custom_medical_pipeline"
        
        # Initialize PhraseMatcher
        try:
            from core.services.medical_phrase_matcher import MedicalPhraseMatcher
            self._medical_matcher = MedicalPhraseMatcher(self._nlp)
            logger.info("Initialized MedicalPhraseMatcher")
        except Exception as e:
            logger.warning(f"Failed to initialize MedicalPhraseMatcher: {e}")
            self._medical_matcher = None
            
        logger.info(f"Loaded medical NLP pipeline: {self._nlp.pipe_names}")

        # Initialize DrugInteractionDetector
        try:
            from core.services.interaction_detector import DrugInteractionDetector
            self._interaction_detector = DrugInteractionDetector()
            logger.info("Initialized DrugInteractionDetector")
        except Exception as e:
            logger.warning(f"Failed to initialize DrugInteractionDetector: {e}")
            self._interaction_detector = None
    
    def process(self, text: str):
        """Process single text through medical pipeline."""
        return self.nlp(text)
    
    def get_entities(self, text: str, include_negated: bool = True) -> List[Dict[str, Any]]:
        """
        Extract named entities with negation status.
        
        Args:
            text: Input text
            include_negated: Whether to include negated entities
        
        Returns:
            List of entity dictionaries with negation info
        """
        doc = self.process(text)
        entities = []
        
        for ent in doc.ents:
            if not include_negated and ent._.is_negated:
                continue
            
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "is_negated": ent._.is_negated,
                "negation_cue": ent._.negation_cue,
            })
        
        return entities
    
    def get_medications(self, text: str, exclude_negated: bool = True) -> List[Dict[str, Any]]:
        """
        Extract medication entities, optionally excluding negated ones.
        
        SAFETY: By default, excludes "I am NOT taking X" type statements.
        """
        doc = self.process(text)
        medications = []
        
        for ent in doc.ents:
            if ent.label_ not in ["DRUG", "MEDICATION"]:
                continue
            
            if exclude_negated and ent._.is_negated:
                logger.info(f"Excluded negated medication: {ent.text} (cue: {ent._.negation_cue})")
                continue
            
            medications.append({
                "text": ent.text,
                "is_negated": ent._.is_negated,
                "negation_cue": ent._.negation_cue,
            })
        
        return medications
    
    def get_medical_summary(self, text: str) -> Dict[str, Any]:
        """
        Get complete medical entity summary with negation awareness.
        """
        doc = self.process(text)
        
        summary = {
            "medications": {"active": [], "negated": []},
            "conditions": {"present": [], "denied": []},
            "symptoms": {"present": [], "denied": []},
            "dosages": [],
            "has_negations": doc._.has_negated_entities,
        }
        
        for ent in doc.ents:
            entry = {"text": ent.text, "label": ent.label_}
            
            if ent.label_ in ["DRUG", "MEDICATION"]:
                key = "negated" if ent._.is_negated else "active"
                summary["medications"][key].append(entry)
            
            elif ent.label_ in ["DISEASE", "CONDITION"]:
                key = "denied" if ent._.is_negated else "present"
                summary["conditions"][key].append(entry)
            
            elif ent.label_ in ["SYMPTOM", "PROBLEM"]:
                key = "denied" if ent._.is_negated else "present"
                summary["symptoms"][key].append(entry)
            
            elif ent.label_ == "DOSAGE":
                summary["dosages"].append(entry)
        
        # Add interactions check for active medications
        active_meds = [m["text"] for m in summary["medications"]["active"]]
        if self._interaction_detector and active_meds:
            interactions = self._interaction_detector.get_interaction_summary(active_meds)
            summary["interactions"] = interactions
        else:
            summary["interactions"] = {"found": False, "interactions": []}
        
        return summary

    def lemmatize(self, text: str) -> str:
        """Get lemmatized version of text."""
        doc = self.process(text)
        return " ".join(token.lemma_ for token in doc if not token.is_punct)
    
    def get_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        doc = self.process(text)
        return [sent.text.strip() for sent in doc.sents]
    
    def process_batch(self, texts: List[str], batch_size: int = 100) -> List[Any]:
        """Efficiently process multiple texts using nlp.pipe()."""
        return list(self.nlp.pipe(texts, batch_size=batch_size))

    def find_medical_terms(self, text: str) -> List[Dict[str, Any]]:
        """Find medical terminology in text using PhraseMatcher."""
        doc = self.process(text)
        if self._medical_matcher:
            return self._medical_matcher.find_matches(doc)
        return []

    def process_for_entities_only(self, text: str) -> object:
        """Fast processing when only entities needed."""
        # Disable parser and other unused components
        # Keep entity_ruler, ner, negation_detector, medical_annotator
        # MUST include sentencizer because negation_detector uses doc.sents
        pipes_to_enable = [
            "transformer", 
            "sentencizer",
            "medical_sentencizer",
            "entity_ruler", 
            "ner", 
            "negation_detector", 
            "medical_annotator"
        ]
        # Filter to only existing pipes
        existing_pipes = [p for p in pipes_to_enable if p in self.nlp.pipe_names]
        
        with self.nlp.select_pipes(enable=existing_pipes):
            return self.nlp(text)
    
    def process_for_sentences_only(self, text: str) -> object:
        """Fast processing when only sentence segmentation needed."""
        pipes_to_enable = ["transformer", "parser", "medical_sentencizer"]
        existing_pipes = [p for p in pipes_to_enable if p in self.nlp.pipe_names]
        
        with self.nlp.select_pipes(enable=existing_pipes):
            return self.nlp(text)

    def get_contextual_facts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract facts using dependency parsing (e.g. 'I have headache').
        More robust than regex for capturing subject-verb-object relationships.
        """
        doc = self.process(text)
        facts = []
        
        # Define patterns on the fly (or cache them in __init__)
        # Ideally move this to __init__ for performance
        from spacy.matcher import DependencyMatcher
        matcher = DependencyMatcher(self.nlp.vocab)
        
        patterns = {
            "SYMPTOM_PRESENT": [
                [
                    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": {"IN": ["have", "experience", "feel", "suffer"]}}},
                    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"LOWER": {"IN": ["i", "patient", "he", "she"]}}},
                    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "symptom", "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "attr", "pobj"]}}}
                ]
            ],
            "DRUG_INTAKE": [
                [
                    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": {"IN": ["take", "prescribe", "start", "use"]}}},
                    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "drug", "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "pobj"]}}}
                ]
            ],
            "DIAGNOSIS": [
                [
                    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": {"IN": ["diagnose"]}}},
                    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "condition", "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "oprd"]}}}
                ],
                [
                    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": {"IN": ["diagnose"]}}},
                    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {"DEP": "prep"}},
                    {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "condition", "RIGHT_ATTRS": {"DEP": "pobj"}}
                ]
            ]
        }
        
        for label, pattern_list in patterns.items():
            matcher.add(label, pattern_list)
            
        matches = matcher(doc)
        
        for match_id, token_ids in matches:
            label = self.nlp.vocab.strings[match_id]
            # Target index depends on pattern length
            target_idx = -1
            if label == "SYMPTOM_PRESENT":
                target_idx = 2
            elif label == "DRUG_INTAKE":
                target_idx = 1
            elif label == "DIAGNOSIS":
                # Check which pattern matched by length
                # Pattern 1 (direct): verb, condition -> len 2 -> index 1
                # Pattern 2 (prep): verb, prep, condition -> len 3 -> index 2
                target_idx = len(token_ids) - 1
                
            if target_idx >= 0 and target_idx < len(token_ids):
                token = doc[token_ids[target_idx]]
                # Get the full subtree for the target to capture "severe headache" instead of just "headache"
                span = doc[token.left_edge.i : token.right_edge.i + 1]
                
                facts.append({
                    "type": label,
                    "text": span.text,
                    "lemma": token.lemma_,
                    "sent": token.sent.text
                })
                
        return facts


def get_spacy_service() -> SpaCyService:
    """Get the singleton SpaCyService instance."""
    return SpaCyService()
