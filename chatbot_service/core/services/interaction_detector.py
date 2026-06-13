"""
Drug Interaction Detector
=========================
Service for detecting potential adverse interactions between drugs.

Currently uses a local knowledge base of common interactions.
Designed to be extensible to external APIs (e.g., RxNorm, DrugBank).
"""


from typing import List, Dict, Any, Tuple, Set
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Interaction:
    severity: str  # "HIGH", "MODERATE", "LOW"
    description: str
    drugs: List[str]

class DrugInteractionDetector:
    """
    Detects interactions between a list of drugs.
    """
    
    # Mock Knowledge Base of Common Interactions
    # Format: frozenset({drug1, drug2}): (severity, description)
    # Keys must be lowercase
    INTERACTIONS_DB = {
        frozenset({"warfarin", "aspirin"}): ("HIGH", "Increased risk of bleeding."),
        frozenset({"warfarin", "ibuprofen"}): ("HIGH", "Increased risk of bleeding."),
        frozenset({"lisinopril", "potassium"}): ("MODERATE", "Risk of hyperkalemia."),
        frozenset({"sildenafil", "nitroglycerin"}): ("CRITICAL", "Risk of severe hypotension."),
        frozenset({"metformin", "contrast dye"}): ("MODERATE", "Risk of lactic acidosis."),
        frozenset({"simvastatin", "amiodarone"}): ("MODERATE", "Increased risk of myopathy."),
        frozenset({"ciprofloxacin", "theophylline"}): ("MODERATE", "Increased theophylline levels."),
        frozenset({"fluoxetine", "phenelzine"}): ("CRITICAL", "Risk of serotonin syndrome."),
        frozenset({"clopidogrel", "omeprazole"}): ("MODERATE", "Reduced efficacy of clopidogrel."),
        frozenset({"digoxin", "verapamil"}): ("MODERATE", "Increased digoxin levels."),
    }

    def __init__(self):
        self._cache = {}

    def check_interactions(self, drugs: List[str]) -> List[Interaction]:
        """
        Check for interactions between any pair of drugs in the list.
        
        Args:
            drugs: List of drug names (strings)
            
        Returns:
            List of Interaction objects
        """
        if len(drugs) < 2:
            return []
        
        interactions = []
        # Normalize drugs to lowercase and remove duplicates
        normalized_drugs = list(set(d.lower() for d in drugs))
        
        # Check every pair
        for i in range(len(normalized_drugs)):
            for j in range(i + 1, len(normalized_drugs)):
                drug1 = normalized_drugs[i]
                drug2 = normalized_drugs[j]
                
                # Check exact match in DB
                key = frozenset({drug1, drug2})
                if key in self.INTERACTIONS_DB:
                    severity, desc = self.INTERACTIONS_DB[key]
                    interactions.append(Interaction(
                        severity=severity,
                        description=desc,
                        drugs=[drug1, drug2]
                    ))
                    continue
                
                # Fuzzy/Partial match check (optional, simple version)
                # e.g. "aspirin 81mg" contains "aspirin"
                found_match = False
                for db_key, (severity, desc) in self.INTERACTIONS_DB.items():
                    db_drugs = list(db_key)
                    # Check if both db_drugs are present in the input pair (as substrings)
                    # This is O(N*M) where N is DB size, so keep DB small or optimize
                    if (self._is_match(db_drugs[0], drug1) and self._is_match(db_drugs[1], drug2)) or \
                       (self._is_match(db_drugs[0], drug2) and self._is_match(db_drugs[1], drug1)):
                        interactions.append(Interaction(
                            severity=severity,
                            description=desc,
                            drugs=[drug1, drug2]
                        ))
                        found_match = True
                        break
        
        return interactions

    def _is_match(self, db_drug: str, input_drug: str) -> bool:
        """Check if db_drug is in input_drug (e.g. 'aspirin' in 'aspirin 81mg')."""
        return db_drug in input_drug

    def get_interaction_summary(self, drugs: List[str]) -> Dict[str, Any]:
        """Get a summary of interactions for API response."""
        interactions = self.check_interactions(drugs)
        
        if not interactions:
            return {"found": False, "interactions": []}
        
        return {
            "found": True,
            "count": len(interactions),
            "interactions": [
                {
                    "severity": i.severity,
                    "description": i.description,
                    "drugs": i.drugs
                }
                for i in interactions
            ]
        }
