"""
Medical NLP Pipeline Module
============================
Custom spaCy pipeline components for healthcare safety.

Components:
- factory.py: Build custom medical NLP pipeline
- negation_detector.py: Detect negated medical entities
- medical_annotator.py: Categorize and enrich medical entities

Usage:
    from rag.nlp.factory import build_medical_nlp, get_medical_nlp
    
    nlp = get_medical_nlp()
    doc = nlp("Patient is NOT taking aspirin.")
    
    for ent in doc.ents:
        print(f"{ent.text}: negated={ent._.is_negated}")
"""

from .factory import build_medical_nlp, get_medical_nlp, save_medical_model
from .negation_detector import negation_detector

__all__ = [
    "build_medical_nlp",
    "get_medical_nlp", 
    "save_medical_model",
    "negation_detector",
]
