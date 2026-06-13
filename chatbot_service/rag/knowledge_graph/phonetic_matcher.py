"""
Phonetic Matching for Medical Terms

Implements medical domain-aware similarity checking to prevent
lookalike drug/condition confusion.

Example:
    from rag.knowledge_graph.phonetic_matcher import PhoneticMatcher
    
    # Check if terms are opposite
    is_opposite = PhoneticMatcher.is_opposite_term("hypertension", "hypotension")
    # Returns: True
    
    # Calculate phonetic similarity
    sim = PhoneticMatcher.metaphone_similarity("Smith", "Smyth")
    # Returns: 1.0 (same phonetic encoding)
"""


import logging
from typing import Tuple, Set
from functools import lru_cache

logger = logging.getLogger(__name__)

try:
    from metaphone import doublemetaphone
    METAPHONE_AVAILABLE = True
except ImportError:
    METAPHONE_AVAILABLE = False
    logger.warning("metaphone library not available. Install with: pip install metaphone")


class PhoneticMatcher:
    """
    Phonetic matching and medical domain validation.
    
    Uses Metaphone algorithm combined with domain-specific rules
    to safely match medical terms despite typos or phonetic variations.
    """
    
    # Medical opposite pairs (critical safety check)
    MEDICAL_OPPOSITES: Set[Tuple[str, str]] = {
        ("hypertension", "hypotension"),
        ("tachycardia", "bradycardia"),
        ("hyperglycemia", "hypoglycemia"),
        ("hyperkalemia", "hypokalemia"),
        ("acidosis", "alkalosis"),
        ("systolic", "diastolic"),
        ("hyperthermia", "hypothermia"),
        ("hyponatremia", "hypernatremia"),
        ("tachypnea", "bradypnea"),
        ("polyuria", "oliguria"),
    }
    
    # Drug name lookalike pairs (observed confusions in literature)
    DRUG_LOOKALIKES: Set[Tuple[str, str]] = {
        ("lisinopril", "atenolol"),
        ("metoprolol", "metformin"),
        ("atorvastatin", "simvastatin"),
        ("amoxicillin", "ampicillin"),
        ("dopamine", "dobutamine"),
        ("epinephrine", "norepinephrine"),
        ("heparin", "warfarin"),
    }
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def metaphone_encode(term: str) -> Tuple[str, str]:
        """
        Encode term using Metaphone (phonetic algorithm).
        
        Returns:
            Tuple of (primary_encoding, secondary_encoding)
        
        Example:
            >>> PhoneticMatcher.metaphone_encode("Smith")
            ('SM0', 'XMT')
        """
        if not METAPHONE_AVAILABLE:
            return ("", "")
        
        try:
            return doublemetaphone(term.lower().strip())
        except Exception as e:
            logger.warning(f"Metaphone encoding failed for '{term}': {e}")
            return ("", "")
    
    @staticmethod
    def metaphone_similarity(s1: str, s2: str) -> float:
        """
        Calculate similarity based on Metaphone encoding.
        
        If both terms produce the same Metaphone encoding,
        they sound similar and get high score.
        
        Args:
            s1: First term
            s2: Second term
        
        Returns:
            Similarity score 0.0-1.0
        
        Example:
            >>> PhoneticMatcher.metaphone_similarity("Smith", "Smyth")
            1.0
            >>> PhoneticMatcher.metaphone_similarity("Smith", "Brown")
            0.0
        """
        if not METAPHONE_AVAILABLE:
            return 0.0
        
        enc1_primary, enc1_secondary = PhoneticMatcher.metaphone_encode(s1)
        enc2_primary, enc2_secondary = PhoneticMatcher.metaphone_encode(s2)
        
        # Check if primary encodings match (high confidence match)
        if enc1_primary and enc2_primary and enc1_primary == enc2_primary:
            return 1.0
        
        # Check if primary matches secondary (partial match)
        if (enc1_primary == enc2_secondary) or (enc1_secondary == enc2_primary):
            return 0.8
        
        # No match
        return 0.0
    
    @staticmethod
    def is_opposite_term(term1: str, term2: str) -> bool:
        """
        Check if two terms are medical opposites.
        
        This is CRITICAL for safety. "Hypertension" and "Hypotension"
        are one letter apart in Levenshtein distance, but opposite conditions.
        
        Args:
            term1: First term
            term2: Second term
        
        Returns:
            True if terms are medical opposites, False otherwise
        
        Example:
            >>> PhoneticMatcher.is_opposite_term("hypertension", "hypotension")
            True
        """
        t1_lower = term1.lower().strip()
        t2_lower = term2.lower().strip()
        
        for (opp1, opp2) in PhoneticMatcher.MEDICAL_OPPOSITES:
            if (t1_lower == opp1 and t2_lower == opp2) or \
               (t1_lower == opp2 and t2_lower == opp1):
                return True
        
        return False
    
    @staticmethod
    def is_drug_lookalike(drug1: str, drug2: str) -> bool:
        """
        Check if two drug names are documented lookalikes.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
        
        Returns:
            True if drugs are known lookalikes
        """
        d1_lower = drug1.lower().strip()
        d2_lower = drug2.lower().strip()
        
        for (look1, look2) in PhoneticMatcher.DRUG_LOOKALIKES:
            if (d1_lower == look1 and d2_lower == look2) or \
               (d1_lower == look2 and d2_lower == look1):
                return True
        
        return False
    
    @staticmethod
    def combined_medical_similarity(term1: str, term2: str) -> float:
        """
        Calculate combined similarity with medical domain knowledge.
        
        Weights:
        - Phonetic match: 40%
        - Levenshtein: 30%
        - Jaccard (tokens): 20%
        - Prefix match: 10%
        
        SAFETY: Returns 0.0 if terms are detected as opposites.
        
        Args:
            term1: First term
            term2: Second term
        
        Returns:
            Similarity score 0.0-1.0
        """
        # CRITICAL: Check for opposites first
        if PhoneticMatcher.is_opposite_term(term1, term2):
            logger.warning(f"Opposite terms detected: '{term1}' vs '{term2}' → score=0.0")
            return 0.0
        
        if PhoneticMatcher.is_drug_lookalike(term1, term2):
            logger.warning(f"Drug lookalike detected: '{term1}' vs '{term2}' → score=0.0")
            return 0.0
        
        # Import needed functions
        from rag.knowledge_graph.medical_ontology import FuzzyMatcher
        
        # Calculate component scores
        phon_score = PhoneticMatcher.metaphone_similarity(term1, term2)
        lev_score = FuzzyMatcher.levenshtein_similarity(term1, term2)
        jac_score = FuzzyMatcher.jaccard_similarity(term1, term2)
        
        # Prefix matching
        t1_norm = FuzzyMatcher.normalize(term1)
        t2_norm = FuzzyMatcher.normalize(term2)
        prefix_score = 0.0
        if len(t1_norm) >= 3 and len(t2_norm) >= 3:
            if t1_norm.startswith(t2_norm[:3]) or t2_norm.startswith(t1_norm[:3]):
                prefix_score = 0.5
            if t1_norm.startswith(t2_norm) or t2_norm.startswith(t1_norm):
                prefix_score = 1.0
        
        # Weighted combination
        combined = (
            (0.4 * phon_score) +
            (0.3 * lev_score) +
            (0.2 * jac_score) +
            (0.1 * prefix_score)
        )
        
        return combined
