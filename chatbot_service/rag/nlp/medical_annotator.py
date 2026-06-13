"""
Medical Annotator Component
===========================
Enriches entities with medical categories and metadata.

Functionality:
- Categorizes drugs (e.g., antibiotic, beta-blocker)
- Maps symptoms to body systems
- Identifies severity indicators
"""


import logging
from typing import Dict, List, Optional

try:
    from spacy.tokens import Doc, Span, Token
    from spacy.language import Language
    SPACY_AVAILABLE = True
except Exception:
    Doc = Span = Token = Language = None  # type: ignore
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Simple medical categories mapping (could be loaded from external source)
DRUG_CATEGORIES = {
    "metoprolol": "beta-blocker",
    "lisinopril": "ace-inhibitor",
    "atorvastatin": "statin",
    "amoxicillin": "antibiotic",
    "ibuprofen": "nsaid",
    "acetaminophen": "analgesic",
    "metformin": "antidiabetic",
    "albuterol": "bronchodilator",
}

SYMPTOM_SYSTEMS = {
    "chest pain": "cardiovascular",
    "shortness of breath": "respiratory",
    "cough": "respiratory",
    "headache": "neurological",
    "nausea": "gastrointestinal",
    "rash": "dermatological",
    "fever": "systemic",
}

def medical_annotator(doc):
    """
    Annotate entities with medical metadata.
    
    Args:
        doc: spaCy Doc object
        
    Returns:
        Doc with enriched entities
    """
    for ent in doc.ents:
        # Drug categorization
        if ent.label_ in ["DRUG", "MEDICATION"]:
            category = _get_drug_category(ent.text)
            if category:
                ent._.set("medical_category", category)
                # Also set on tokens
                for token in ent:
                    token._.set("medical_category", category)
        
        # Symptom system mapping
        elif ent.label_ in ["SYMPTOM", "PROBLEM"]:
            system = _get_symptom_system(ent.text)
            if system:
                ent._.set("medical_category", system)
                for token in ent:
                    token._.set("medical_category", system)
                    
    return doc

def _get_drug_category(text: str) -> Optional[str]:
    """Get category for a drug name."""
    text_lower = text.lower()
    # Direct match
    if text_lower in DRUG_CATEGORIES:
        return DRUG_CATEGORIES[text_lower]
    
    # Partial match (e.g. "metoprolol tartrate")
    for drug, category in DRUG_CATEGORIES.items():
        if drug in text_lower:
            return category
            
    return None

def _get_symptom_system(text: str) -> Optional[str]:
    """Get body system for a symptom."""
    text_lower = text.lower()
    if text_lower in SYMPTOM_SYSTEMS:
        return SYMPTOM_SYSTEMS[text_lower]
        
    for symptom, system in SYMPTOM_SYSTEMS.items():
        if symptom in text_lower:
            return system
            
    return None


# Register spaCy component only when spaCy is available
if SPACY_AVAILABLE:
    try:
        Language.component("medical_annotator", func=medical_annotator)
    except Exception:
        pass  # Already registered or other issue
