"""
Symptom Checker
===============
Extracts and analyzes symptoms from text using spaCy.
"""


import logging
from typing import List, Dict, Any, Optional
from core.services.spacy_service import get_spacy_service

logger = logging.getLogger(__name__)

class SymptomChecker:
    """
    Analyzes text for symptoms and their attributes.
    """
    
    def __init__(self):
        self.spacy_service = get_spacy_service()
        
    def analyze_symptoms(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for symptoms.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing detected symptoms and their status (present/absent)
        """
        summary = self.spacy_service.get_medical_summary(text)
        
        return {
            "present_symptoms": summary["symptoms"]["present"],
            "denied_symptoms": summary["symptoms"]["denied"],
            "has_red_flags": self._check_red_flags(summary["symptoms"]["present"])
        }
        
    def _check_red_flags(self, symptoms: List[Dict[str, str]]) -> bool:
        """Check for red flag symptoms."""
        red_flags = {"chest pain", "difficulty breathing", "severe headache", "sudden weakness"}
        
        for symptom in symptoms:
            if symptom["text"].lower() in red_flags:
                return True
                
        return False

# Singleton
_checker = None

def get_symptom_checker() -> SymptomChecker:
    global _checker
    if _checker is None:
        _checker = SymptomChecker()
    return _checker
