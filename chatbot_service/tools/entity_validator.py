"""
Entity Validator - Fuzzy Matcher Middleware for Tool Input Validation

This module prevents hallucinated entity names (drugs, foods, conditions) from
being passed to tools by validating them against known entities before execution.

Prevents scenarios like:
  LLM: "Check interaction between CardioFix and Aspirin"
  Query: get_drug_interactions(["CardioFix", "Aspirin"])
  
  Without validation:
    - Tool queries database for non-existent "CardioFix"
    - Returns empty result, wasting cycles
    
    
  With validation:
    - Fuzzy matcher finds no similar drug
    - Tool is not executed, or fuzzy-matched alternative is suggested
    - Response: "I couldn't find 'CardioFix' in the drug database. Did you mean 'Cardizem'?"

Usage:
    validator = EntityValidator.get_instance()
    
    # Validate drug names
    result = await validator.validate_drugs(["Aspirin", "CardioFix"])
    # Returns: {"Aspirin": {"valid": True, "id": "aspirin"}, 
    #           "CardioFix": {"valid": False, "similar": ["Cardizem"]}}
    
    # Use in tool middleware
    validated = await validator.validate_and_sanitize_drugs(user_input_drugs)
    if validated["all_valid"]:
        result = await get_drug_interactions(validated["normalized"])
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of entity validation."""
    entity: str
    entity_type: str  # 'drug', 'food', 'symptom', 'condition'
    valid: bool
    normalized_name: Optional[str] = None
    confidence: float = 0.0
    similar_entities: List[str] = field(default_factory=list)
    reason: Optional[str] = None


class EntityValidator:
    """Fuzzy matcher middleware for validating entity names against knowledge base."""
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __init__(self, data_dir: str = "data"):
        """Initialize validator with entity databases."""
        self.data_dir = Path(data_dir)
        self.drugs_db = {}
        self.foods_db = {}
        self.symptoms_db = {}
        self.conditions_db = {}
        self.loaded = False
        
    @classmethod
    def get_instance(cls, data_dir: str = "data"):
        """Get singleton instance with thread-safe lazy initialization."""
        if cls._instance is None:
            cls._instance = cls(data_dir)
        return cls._instance
    
    async def load_databases(self) -> bool:
        """Load entity databases from JSON files."""
        if self.loaded:
            return True
        
        try:
            async with self._lock:
                if self.loaded:  # Double-check pattern
                    return True
                
                # Load expanded_drugs.json
                drugs_path = self.data_dir / "expanded_drugs.json"
                if drugs_path.exists():
                    with open(drugs_path, 'r') as f:
                        drugs_list = json.load(f)
                        for drug in drugs_list:
                            # Store by generic name and all brand names
                            drug_id = drug.get("id", "").lower()
                            generic_name = drug.get("generic_name", "").lower()
                            
                            if drug_id:
                                self.drugs_db[drug_id] = drug
                                self.drugs_db[generic_name] = drug
                            
                            # Also index by brand names for fuzzy matching
                            for brand in drug.get("brand_names", []):
                                self.drugs_db[brand.lower()] = drug
                    
                    logger.info(f"✅ Loaded {len(set(d['id'] for d in drugs_list))} drugs")
                
                # Load symptoms.json
                symptoms_path = self.data_dir / "symptoms.json"
                if symptoms_path.exists():
                    with open(symptoms_path, 'r') as f:
                        symptoms_list = json.load(f)
                        for symptom in symptoms_list:
                            if isinstance(symptom, dict):
                                symptom_name = symptom.get("name", "").lower()
                                if symptom_name:
                                    self.symptoms_db[symptom_name] = symptom
                            elif isinstance(symptom, str):
                                self.symptoms_db[symptom.lower()] = {"name": symptom}
                    
                    logger.info(f"✅ Loaded {len(self.symptoms_db)} symptoms")
                
                # Load disease mapping cache
                disease_path = self.data_dir / "disease_mapping_cache.json"
                if disease_path.exists():
                    with open(disease_path, 'r') as f:
                        disease_data = json.load(f)
                        for condition_name, condition_info in disease_data.items():
                            self.conditions_db[condition_name.lower()] = condition_info
                    
                    logger.info(f"✅ Loaded {len(self.conditions_db)} conditions")
                
                self.loaded = True
                return True
        
        except Exception as e:
            logger.error(f"❌ Failed to load entity databases: {e}")
            return False
    
    def _fuzzy_match(
        self, 
        query: str, 
        candidates: Dict[str, Any],
        threshold: float = 0.6
    ) -> Tuple[Optional[str], float]:
        """
        Fuzzy match query against candidates using SequenceMatcher.
        
        Args:
            query: User input to match (e.g., "Aspirn" or "CardioFix")
            candidates: Dict of {name: entity} to match against
            threshold: Minimum similarity score (0-1) to consider a match
            
        Returns:
            Tuple of (best_match_key, similarity_score) or (None, 0.0)
        """
        query_lower = query.lower().strip()
        best_match = None
        best_score = 0.0
        
        for candidate_key in candidates.keys():
            ratio = SequenceMatcher(None, query_lower, candidate_key).ratio()
            if ratio > best_score:
                best_score = ratio
                best_match = candidate_key
        
        if best_score >= threshold:
            return best_match, best_score
        
        return None, 0.0
    
    def _get_similar_entities(
        self,
        query: str,
        candidates: Dict[str, Any],
        num_similar: int = 3,
        threshold: float = 0.5
    ) -> List[str]:
        """Get list of similar entity names for suggestion."""
        query_lower = query.lower().strip()
        scores = []
        
        for candidate_key in candidates.keys():
            ratio = SequenceMatcher(None, query_lower, candidate_key).ratio()
            if ratio >= threshold:
                scores.append((candidate_key, ratio))
        
        # Sort by score descending and return top N
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:num_similar]]
    
    async def validate_drugs(
        self, 
        drug_names: List[str],
        strict: bool = False
    ) -> Dict[str, ValidationResult]:
        """
        Validate a list of drug names against expanded_drugs.json.
        
        Args:
            drug_names: List of drug names to validate
            strict: If True, only exact matches are valid. If False, fuzzy match is used.
            
        Returns:
            Dict mapping drug names to ValidationResult objects
        """
        if not self.loaded:
            await self.load_databases()
        
        results = {}
        
        for drug_name in drug_names:
            drug_lower = drug_name.lower().strip()
            
            # Check for exact match first
            if drug_lower in self.drugs_db:
                drug_info = self.drugs_db[drug_lower]
                results[drug_name] = ValidationResult(
                    entity=drug_name,
                    entity_type="drug",
                    valid=True,
                    normalized_name=drug_info.get("generic_name", drug_name),
                    confidence=1.0
                )
            elif not strict:
                # Try fuzzy match
                match_key, score = self._fuzzy_match(drug_name, self.drugs_db, threshold=0.6)
                
                if match_key and score > 0.6:
                    drug_info = self.drugs_db[match_key]
                    similar = self._get_similar_entities(drug_name, self.drugs_db, num_similar=2)
                    
                    results[drug_name] = ValidationResult(
                        entity=drug_name,
                        entity_type="drug",
                        valid=False,
                        normalized_name=drug_info.get("generic_name", drug_name),
                        confidence=score,
                        similar_entities=similar,
                        reason=f"Unknown drug. Did you mean '{drug_info.get('generic_name', match_key)}'?"
                    )
                else:
                    # No match found
                    similar = self._get_similar_entities(drug_name, self.drugs_db, num_similar=3)
                    results[drug_name] = ValidationResult(
                        entity=drug_name,
                        entity_type="drug",
                        valid=False,
                        confidence=0.0,
                        similar_entities=similar,
                        reason=f"Drug '{drug_name}' not found in database."
                    )
            else:
                # Strict mode: no match
                results[drug_name] = ValidationResult(
                    entity=drug_name,
                    entity_type="drug",
                    valid=False,
                    confidence=0.0,
                    reason=f"Drug '{drug_name}' not found in database (strict mode)."
                )
        
        return results
    
    async def validate_symptoms(
        self,
        symptom_names: List[str],
        strict: bool = False
    ) -> Dict[str, ValidationResult]:
        """Validate symptom names against symptoms.json."""
        if not self.loaded:
            await self.load_databases()
        
        results = {}
        
        for symptom_name in symptom_names:
            symptom_lower = symptom_name.lower().strip()
            
            # Exact match
            if symptom_lower in self.symptoms_db:
                results[symptom_name] = ValidationResult(
                    entity=symptom_name,
                    entity_type="symptom",
                    valid=True,
                    normalized_name=symptom_lower,
                    confidence=1.0
                )
            elif not strict:
                match_key, score = self._fuzzy_match(symptom_name, self.symptoms_db, threshold=0.6)
                if match_key and score > 0.6:
                    similar = self._get_similar_entities(symptom_name, self.symptoms_db, num_similar=2)
                    results[symptom_name] = ValidationResult(
                        entity=symptom_name,
                        entity_type="symptom",
                        valid=False,
                        normalized_name=match_key,
                        confidence=score,
                        similar_entities=similar,
                        reason=f"Symptom may be '{match_key}'?"
                    )
                else:
                    similar = self._get_similar_entities(symptom_name, self.symptoms_db, num_similar=3)
                    results[symptom_name] = ValidationResult(
                        entity=symptom_name,
                        entity_type="symptom",
                        valid=False,
                        confidence=0.0,
                        similar_entities=similar
                    )
            else:
                results[symptom_name] = ValidationResult(
                    entity=symptom_name,
                    entity_type="symptom",
                    valid=False,
                    confidence=0.0
                )
        
        return results
    
    async def validate_and_sanitize_drugs(
        self,
        drug_names: List[str],
        auto_fix: bool = True
    ) -> Dict[str, Any]:
        """
        Validate drugs and return sanitized result for tool execution.
        
        Args:
            drug_names: List of drug names from user/LLM
            auto_fix: If True, use fuzzy-matched names. If False, only valid drugs.
            
        Returns:
            {
                "all_valid": bool,
                "normalized": List[str],  # Safe to pass to tool
                "warnings": List[str],
                "invalid": Dict[str, str]  # entity -> reason
            }
        """
        validation_results = await self.validate_drugs(drug_names, strict=False)
        
        normalized = []
        warnings = []
        invalid = {}
        
        for drug_name, validation in validation_results.items():
            if validation.valid:
                normalized.append(validation.normalized_name)
            elif auto_fix and validation.confidence > 0.6 and validation.normalized_name:
                # Use fuzzy-matched version
                normalized.append(validation.normalized_name)
                warnings.append(
                    f"'{drug_name}' interpreted as '{validation.normalized_name}' "
                    f"(confidence: {validation.confidence:.1%})"
                )
            else:
                invalid[drug_name] = validation.reason or "Unknown drug"
        
        return {
            "all_valid": len(invalid) == 0,
            "normalized": normalized,
            "warnings": warnings,
            "invalid": invalid
        }
    
    async def validate_conditions(
        self,
        condition_names: List[str],
        strict: bool = False
    ) -> Dict[str, ValidationResult]:
        """Validate medical conditions against disease_mapping_cache.json."""
        if not self.loaded:
            await self.load_databases()
        
        results = {}
        
        for condition_name in condition_names:
            condition_lower = condition_name.lower().strip()
            
            # Exact match
            if condition_lower in self.conditions_db:
                results[condition_name] = ValidationResult(
                    entity=condition_name,
                    entity_type="condition",
                    valid=True,
                    normalized_name=condition_lower,
                    confidence=1.0
                )
            elif not strict:
                match_key, score = self._fuzzy_match(condition_name, self.conditions_db, threshold=0.6)
                if match_key and score > 0.6:
                    similar = self._get_similar_entities(condition_name, self.conditions_db, num_similar=2)
                    results[condition_name] = ValidationResult(
                        entity=condition_name,
                        entity_type="condition",
                        valid=False,
                        normalized_name=match_key,
                        confidence=score,
                        similar_entities=similar
                    )
                else:
                    similar = self._get_similar_entities(condition_name, self.conditions_db, num_similar=3)
                    results[condition_name] = ValidationResult(
                        entity=condition_name,
                        entity_type="condition",
                        valid=False,
                        confidence=0.0,
                        similar_entities=similar
                    )
            else:
                results[condition_name] = ValidationResult(
                    entity=condition_name,
                    entity_type="condition",
                    valid=False,
                    confidence=0.0
                )
        
        return results


async def validate_tool_input(
    entity_type: str,  # 'drug', 'symptom', 'condition'
    entities: List[str],
    auto_fix: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to validate tool inputs before execution.
    
    Usage in tool middleware:
        drugs_input = ["Aspirin", "CardioFix", "Warfarin"]
        validation = await validate_tool_input("drug", drugs_input)
        
        if not validation["all_valid"]:
            logger.warning(f"Invalid drugs detected: {validation['invalid']}")
            # Handle invalid input
            return ToolResult(success=False, error=f"Invalid drugs: {validation['invalid']}")
        
        # Use sanitized input
        result = await get_drug_interactions(validation["normalized"])
    """
    validator = EntityValidator.get_instance()
    
    if entity_type == "drug":
        return await validator.validate_and_sanitize_drugs(entities, auto_fix=auto_fix)
    elif entity_type == "symptom":
        results = await validator.validate_symptoms(entities, strict=(not auto_fix))
        return {
            "all_valid": all(r.valid for r in results.values()),
            "normalized": [r.normalized_name for r in results.values() if r.normalized_name],
            "warnings": [r.reason for r in results.values() if not r.valid],
            "invalid": {k: v.reason for k, v in results.items() if not v.valid}
        }
    elif entity_type == "condition":
        results = await validator.validate_conditions(entities, strict=(not auto_fix))
        return {
            "all_valid": all(r.valid for r in results.values()),
            "normalized": [r.normalized_name for r in results.values() if r.normalized_name],
            "warnings": [r.reason for r in results.values() if not r.valid],
            "invalid": {k: v.reason for k, v in results.items() if not v.valid}
        }
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")
