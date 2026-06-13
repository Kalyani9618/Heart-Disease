"""
Medical Ontology Mapper for ICD-10 Standardization.

Provides mapping from free-text disease names to standardized ICD-10 codes
to ensure clean, deduplicated knowledge graph with proper medical ontology.

Features:
- Exact match against ICD-10 database and aliases
- Fuzzy matching (Levenshtein + Jaccard similarity)
- LLM fallback for unmatched diseases
- Aggressive caching to minimize API costs
- Analytics tracking for popular lookups and cost savings
- Cache warming for common diseases


Performance:
- Exact match: <1ms, $0.00
- Fuzzy match: 1-5ms, $0.00
- LLM fallback: ~200ms, ~$0.01
- Average cost: ~$5/month for typical usage (99% cache hits)

Example:
    mapper = get_medical_ontology_mapper()
    
    # Map disease name to ICD-10
    mapping = await mapper.map_disease("heart attack")
    # Returns: {
    #     "icd10_code": "I21.9",
    #     "standard_name": "Acute myocardial infarction, unspecified",
    #     "category": "Diseases of the circulatory system",
    #     "confidence": 1.0,
    #     "match_method": "exact"
    # }
    
    # Fuzzy match example
    mapping = await mapper.map_disease("hart atack")  # Typo
    # Still finds I21.9 via fuzzy matching!
    
    # Get analytics
    print(mapper.get_report())
"""

import json
import logging
import re
import time
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

from rag.knowledge_graph.phonetic_matcher import PhoneticMatcher


# ============================================================
# Fuzzy Matching Utilities
# ============================================================


class FuzzyMatcher:
    """
    Fuzzy string matching for disease names.
    
    Uses multiple strategies:
    1. Exact match (normalized)
    2. Levenshtein distance
    3. Token overlap (Jaccard similarity)
    4. Prefix matching
    """
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for matching."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance."""
        if len(s1) < len(s2):
            return FuzzyMatcher.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def levenshtein_similarity(s1: str, s2: str) -> float:
        """Calculate similarity based on Levenshtein distance."""
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        distance = FuzzyMatcher.levenshtein_distance(s1, s2)
        return 1 - (distance / max_len)
    
    @staticmethod
    def jaccard_similarity(s1: str, s2: str) -> float:
        """Calculate Jaccard similarity based on token overlap."""
        tokens1 = set(s1.split())
        tokens2 = set(s2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union)
    
    @staticmethod
    def combined_similarity(s1: str, s2: str) -> float:
        """
        Calculate combined similarity score with MEDICAL SAFETY CHECKS.
        
        Weights:
        - Levenshtein: 30%
        - Jaccard: 30%
        - Phonetic: 30%
        - Prefix match: 10%
        
        CRITICAL SAFETY:
        - Returns 0.0 if terms are medical opposites (hyper/hypo)
        - Returns 0.0 if terms are known drug lookalikes
        """
        s1_norm = FuzzyMatcher.normalize(s1)
        s2_norm = FuzzyMatcher.normalize(s2)
        
        if s1_norm == s2_norm:
            return 1.0
        
        # CRITICAL SAFETY CHECK: Reject opposite medical terms
        if PhoneticMatcher.is_opposite_term(s1, s2):
            logger.warning(
                f"⚠️ SAFETY: Rejecting match of '{s1}' to '{s2}' "
                f"(opposite medical terms - would be dangerous)"
            )
            return 0.0
        
        # SAFETY CHECK: Reject known drug lookalikes
        if PhoneticMatcher.is_drug_lookalike(s1, s2):
            logger.warning(
                f"⚠️ SAFETY: Rejecting match of '{s1}' to '{s2}' "
                f"(known drug lookalike pair)"
            )
            return 0.0
        
        # Calculate component scores
        lev_score = FuzzyMatcher.levenshtein_similarity(s1_norm, s2_norm)
        jac_score = FuzzyMatcher.jaccard_similarity(s1_norm, s2_norm)
        phon_score = PhoneticMatcher.metaphone_similarity(s1, s2)
        
        prefix_score = 0.0
        if len(s1_norm) >= 3 and len(s2_norm) >= 3:
            if s1_norm.startswith(s2_norm[:3]) or s2_norm.startswith(s1_norm[:3]):
                prefix_score = 0.5
        if s1_norm.startswith(s2_norm) or s2_norm.startswith(s1_norm):
            prefix_score = 1.0
        
        return (0.3 * lev_score) + (0.3 * jac_score) + (0.3 * phon_score) + (0.1 * prefix_score)


# ============================================================
# Analytics Tracking
# ============================================================


class OntologyAnalytics:
    """
    Analytics for disease mapping operations.
    
    Tracks:
    - Popular disease lookups
    - Cache hit/miss rates
    - LLM call costs
    - Processing times
    """
    
    # Estimated cost per LLM call for disease mapping
    COST_PER_LLM_CALL = 0.01
    
    def __init__(self):
        self.lookup_counts: Dict[str, int] = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        self.exact_matches = 0
        self.fuzzy_matches = 0
        self.llm_calls = 0
        self.total_llm_cost = 0.0
        self.processing_times: List[float] = []
        self.start_time = datetime.now()
    
    def record_lookup(
        self,
        disease_name: str,
        method: str,  # "cache", "exact", "fuzzy", "llm"
        llm_cost: float = 0.0
    ):
        """Record a disease lookup."""
        normalized = disease_name.lower().strip()
        self.lookup_counts[normalized] += 1
        
        if method == "cache":
            self.cache_hits += 1
        elif method == "exact":
            self.exact_matches += 1
            self.cache_misses += 1
        elif method == "fuzzy":
            self.fuzzy_matches += 1
            self.cache_misses += 1
        elif method == "llm":
            self.llm_calls += 1
            self.cache_misses += 1
            self.total_llm_cost += llm_cost or self.COST_PER_LLM_CALL
    
    def record_processing_time(self, time_ms: float):
        """Record processing time."""
        self.processing_times.append(time_ms)
    
    def get_popular_diseases(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently looked up diseases."""
        sorted_items = sorted(
            self.lookup_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_items[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analytics statistics."""
        total_lookups = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_lookups * 100) if total_lookups > 0 else 0
        
        avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            "total_lookups": total_lookups,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "exact_matches": self.exact_matches,
            "fuzzy_matches": self.fuzzy_matches,
            "llm_calls": self.llm_calls,
            "cache_hit_rate_percent": round(hit_rate, 1),
            "total_llm_cost": round(self.total_llm_cost, 4),
            "avg_processing_time_ms": round(avg_time, 2),
            "unique_diseases": len(self.lookup_counts),
            "uptime_hours": round((datetime.now() - self.start_time).total_seconds() / 3600, 2)
        }
    
    def get_report(self) -> str:
        """Get formatted analytics report."""
        stats = self.get_stats()
        popular = self.get_popular_diseases(5)
        
        report = f"""
📊 Medical Ontology Analytics Report
=====================================
Uptime: {stats['uptime_hours']} hours

Lookups:
  - Total: {stats['total_lookups']}
  - Unique diseases: {stats['unique_diseases']}
  - Cache hits: {stats['cache_hits']} ({stats['cache_hit_rate_percent']}%)

Match Methods:
  - Exact matches: {stats['exact_matches']}
  - Fuzzy matches: {stats['fuzzy_matches']}
  - LLM fallback: {stats['llm_calls']}

LLM Usage:
  - API calls: {stats['llm_calls']}
  - Total cost: ${stats['total_llm_cost']:.4f}

Performance:
  - Avg processing time: {stats['avg_processing_time_ms']:.2f}ms

Top 5 Most Looked Up Diseases:
"""
        for i, (disease, count) in enumerate(popular, 1):
            report += f"  {i}. {disease}: {count} lookups\n"
        
        return report.strip()


# ============================================================
# Main Mapper Class
# ============================================================


class MedicalOntologyMapper:
    """
    Comprehensive medical ontology mapper.
    
    Mapping order:
    1. Check cache (instant)
    2. Exact match against ICD-10 codes and aliases (fast)
    3. Fuzzy match against all aliases (fast)
    4. LLM fallback for unknown diseases (slow, costs money)
    
    Features:
    - 70+ ICD-10 cardiovascular codes
    - Fuzzy matching with 0.7 threshold
    - LLM fallback with aggressive caching
    - Analytics and cost tracking
    - Cache warming for common diseases
    """
    
    def __init__(
        self,
        icd10_file: str = None,
        cache_file: str = None,
        fuzzy_threshold: float = 0.7,
        enable_llm_fallback: bool = True
    ):
        """
        Initialize medical ontology mapper.
        
        Args:
            icd10_file: Path to ICD-10 codes JSON file
            cache_file: Path to mapping cache file
            fuzzy_threshold: Minimum similarity for fuzzy match (0.0-1.0)
            enable_llm_fallback: Whether to use LLM for unmatched diseases
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.enable_llm_fallback = enable_llm_fallback
        self.analytics = OntologyAnalytics()
        
        # Use centralized path configuration
        if icd10_file is None or cache_file is None:
            from core.config.rag_config import RAGConfig
            config = RAGConfig()
            data_dir = config.paths.data_dir
        else:
            data_dir = Path(__file__).parent.parent.parent / "data"
        
        self.icd10_file = Path(icd10_file) if icd10_file else data_dir / "icd10_codes.json"
        self.cache_file = Path(cache_file) if cache_file else data_dir / "disease_mapping_cache.json"
        
        # Load databases
        self.icd10_codes = self._load_icd10()
        self.cache = self._load_cache()
        
        # Build alias index for fast lookup
        self.alias_index = self._build_alias_index()
        
        logger.info(
            f"MedicalOntologyMapper initialized: "
            f"{len(self.icd10_codes)} ICD-10 codes, "
            f"{len(self.cache)} cached mappings, "
            f"{len(self.alias_index)} indexed aliases, "
            f"fuzzy_threshold={fuzzy_threshold}, "
            f"llm_fallback={'enabled' if enable_llm_fallback else 'disabled'}"
        )
    
    def _load_icd10(self) -> Dict[str, Dict]:
        """Load ICD-10 database."""
        if not self.icd10_file.exists():
            logger.warning(f"ICD-10 file not found: {self.icd10_file}")
            return {}
        
        try:
            with open(self.icd10_file, 'r', encoding='utf-8') as f:
                codes = json.load(f)
            logger.info(f"Loaded {len(codes)} ICD-10 codes from {self.icd10_file}")
            return codes
        except Exception as e:
            logger.error(f"Failed to load ICD-10 codes: {e}")
            return {}
    
    def _load_cache(self) -> Dict[str, Dict]:
        """Load mapping cache."""
        if not self.cache_file.exists():
            return {}
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            logger.info(f"Loaded {len(cache)} cached mappings")
            return cache
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return {}
    
    def _save_cache(self):
        """Save mapping cache to file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _build_alias_index(self) -> Dict[str, str]:
        """
        Build index from normalized aliases to ICD-10 codes.
        
        Returns:
            Dict mapping normalized alias → ICD-10 code
        """
        index = {}
        
        for code, data in self.icd10_codes.items():
            # Index by code
            index[code.lower()] = code
            
            # Index by standard name
            name = data.get("name", "").lower().strip()
            if name:
                index[name] = code
            
            # Index by aliases
            for alias in data.get("aliases", []):
                normalized = alias.lower().strip()
                if normalized:
                    index[normalized] = code
        
        return index
    
    def _check_cache(self, disease_name: str) -> Optional[Dict]:
        """Check if disease mapping is cached."""
        normalized = disease_name.lower().strip()
        return self.cache.get(normalized)
    
    def _exact_match(self, disease_name: str) -> Optional[str]:
        """Try exact match in alias index."""
        normalized = disease_name.lower().strip()
        return self.alias_index.get(normalized)
    
    def _fuzzy_match(self, disease_name: str) -> Optional[Tuple[str, float]]:
        """Try fuzzy match against all aliases."""
        normalized = disease_name.lower().strip()
        best_match = None
        best_score = 0.0
        
        for alias, code in self.alias_index.items():
            score = FuzzyMatcher.combined_similarity(normalized, alias)
            
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = code
        
        if best_match:
            return (best_match, best_score)
        return None
    
    async def _llm_map(self, disease_name: str) -> Optional[Dict]:
        """Use LLM to map disease to ICD-10."""
        if not self.enable_llm_fallback:
            return None
        
        try:
            from core.llm.llm_gateway import get_llm_gateway
            
            gateway = get_llm_gateway()
            
            prompt = f"""Map the following medical condition to its ICD-10 code.

Condition: "{disease_name}"

Provide your response as a JSON object with these fields:
{{
    "icd10_code": "string (e.g., I21.9)",
    "standard_name": "string (official ICD-10 name)",
    "category": "string (ICD-10 chapter/category)",
    "confidence": 0.0-1.0 (how confident are you in this mapping)
}}

Important:
- Use the most specific ICD-10 code available
- If the condition is ambiguous, use the most general relevant code
- If you cannot find a match, set icd10_code to null
- Be as accurate as possible with medical terminology

Return ONLY the JSON object, no additional text."""
            
            response = await gateway.generate(
                prompt=prompt,
                content_type="medical"
            )
            
            # Parse JSON response
            response_clean = response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            response_clean = response_clean.strip()
            
            mapping = json.loads(response_clean)
            
            if not isinstance(mapping, dict) or not mapping.get("icd10_code"):
                return None
            
            return mapping
        
        except Exception as e:
            logger.error(f"LLM mapping failed for '{disease_name}': {e}")
            return None
    
    def _create_mapping(
        self,
        code: str,
        original_name: str,
        method: str = "exact",
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """Create mapping result dictionary."""
        data = self.icd10_codes.get(code, {})
        
        return {
            "icd10_code": code,
            "standard_name": data.get("name", "Unknown"),
            "category": data.get("category", "Unknown"),
            "chapter": data.get("chapter", "Unknown"),
            "original_name": original_name,
            "confidence": confidence,
            "match_method": method,
            "cached_at": datetime.utcnow().isoformat()
        }
    
    def _cache_mapping(self, key: str, mapping: Dict):
        """Cache a mapping result."""
        self.cache[key] = mapping
        
        # Save cache periodically
        if len(self.cache) % 10 == 0:
            self._save_cache()
    
    async def map_disease(self, disease_name: str) -> Optional[Dict]:
        """
        Map disease name to ICD-10 code with medical safety checks.
        
        Order of attempts:
        1. Check cache (instant)
        2. Exact match in ICD-10 database (fast)
        3. Fuzzy match against aliases (fast)
        4. LLM fallback (slow, costs money)
        
        Args:
            disease_name: Disease name to map
            
        Returns:
            Mapping dictionary or None
        """
        start_time = time.time()
        normalized = disease_name.lower().strip()
        
        # 1. Check cache
        cached = self._check_cache(disease_name)
        if cached:
            self.analytics.record_lookup(disease_name, method="cache")
            self.analytics.record_processing_time((time.time() - start_time) * 1000)
            return cached
        
        # 2. Exact match
        code = self._exact_match(disease_name)
        if code:
            mapping = self._create_mapping(code, disease_name, method="exact", confidence=1.0)
            mapping["safety_score"] = 1.0  # Exact match is safe
            self._cache_mapping(normalized, mapping)
            self.analytics.record_lookup(disease_name, method="exact")
            self.analytics.record_processing_time((time.time() - start_time) * 1000)
            logger.debug(f"Exact match: '{disease_name}' → {code}")
            return mapping
        
        # 3. Fuzzy match with medical safety checks
        best_match = None
        best_score = 0.0
        best_safety_score = 0.0
        
        for alias, code in self.alias_index.items():
            # CRITICAL SAFETY CHECK: Reject opposite terms
            if PhoneticMatcher.is_opposite_term(disease_name, alias):
                logger.warning(
                    f"⚠️ SAFETY: Rejecting fuzzy match of '{disease_name}' to '{alias}' "
                    f"(opposite medical terms - would be dangerous)"
                )
                continue
            
            # Calculate combined similarity with domain knowledge
            combined = PhoneticMatcher.combined_medical_similarity(disease_name, alias)
            
            if combined > best_score:
                best_score = combined
                best_match = code
                best_safety_score = combined
        
        # Fuzzy threshold check (raised for safety)
        FUZZY_THRESHOLD = 0.75  # Higher than standard to reduce false positives
        
        if best_score >= FUZZY_THRESHOLD and best_match:
            mapping = self._create_mapping(
                best_match, disease_name,
                method="fuzzy",
                confidence=best_score
            )
            mapping["safety_score"] = best_safety_score
            
            # Log fuzzy matches for audit trail
            if best_score < 0.90:
                logger.info(
                    f"ℹ️ Fuzzy match: '{disease_name}' → '{best_match}' "
                    f"(conf={best_score:.2f}, safety={best_safety_score:.2f})"
                )
            
            self._cache_mapping(normalized, mapping)
            self.analytics.record_lookup(disease_name, method="fuzzy")
            self.analytics.record_processing_time((time.time() - start_time) * 1000)
            return mapping
        
        # 4. LLM fallback
        llm_result = await self._llm_map(disease_name)
        if llm_result:
            llm_result["original_name"] = disease_name
            llm_result["match_method"] = "llm"
            llm_result["cached_at"] = datetime.utcnow().isoformat()
            self._cache_mapping(normalized, llm_result)
            self.analytics.record_lookup(disease_name, method="llm")
            self.analytics.record_processing_time((time.time() - start_time) * 1000)
            logger.info(f"LLM match: '{disease_name}' → {llm_result.get('icd10_code')}")
            return llm_result
        
        # No match found
        self.analytics.record_processing_time((time.time() - start_time) * 1000)
        logger.warning(f"No ICD-10 mapping found for: {disease_name}")
        return None
    
    def warm_cache(self, common_diseases: List[str] = None):
        """
        Pre-populate cache with common diseases.
        
        Args:
            common_diseases: List of common disease names to cache
        """
        if common_diseases is None:
            common_diseases = [
                "heart attack", "MI", "STEMI", "NSTEMI",
                "heart failure", "CHF", "HFrEF", "HFpEF",
                "hypertension", "high blood pressure", "HTN",
                "atrial fibrillation", "AFib", "A-fib",
                "stroke", "CVA", "TIA",
                "diabetes", "T2DM", "diabetes mellitus",
                "coronary artery disease", "CAD",
                "angina", "chest pain",
                "arrhythmia", "palpitations",
                "DVT", "PE", "pulmonary embolism"
            ]
        
        logger.info(f"Warming cache with {len(common_diseases)} common diseases...")
        
        # Just do exact/fuzzy matches (no LLM calls during warmup)
        for disease in common_diseases:
            normalized = disease.lower().strip()
            
            # Skip if already cached
            if normalized in self.cache:
                continue
            
            # Try exact match
            code = self._exact_match(disease)
            if code:
                mapping = self._create_mapping(code, disease, method="exact")
                self._cache_mapping(normalized, mapping)
                continue
            
            # Try fuzzy match
            fuzzy_result = self._fuzzy_match(disease)
            if fuzzy_result:
                code, score = fuzzy_result
                mapping = self._create_mapping(code, disease, method="fuzzy", confidence=score)
                self._cache_mapping(normalized, mapping)
        
        self._save_cache()
        logger.info(f"Cache warming complete. {len(self.cache)} entries cached.")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mapping statistics."""
        stats = self.analytics.get_stats()
        stats["icd10_codes_loaded"] = len(self.icd10_codes)
        stats["cached_mappings"] = len(self.cache)
        stats["indexed_aliases"] = len(self.alias_index)
        return stats
    
    def get_report(self) -> str:
        """Get formatted analytics report."""
        return self.analytics.get_report()
    
    def get_popular_diseases(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently looked up diseases."""
        return self.analytics.get_popular_diseases(limit)
    
    def flush_cache(self):
        """Force save cache to disk."""
        self._save_cache()
        logger.info("Cache flushed to disk")


# ============================================================
# Singleton Instance
# ============================================================


_mapper_instance: Optional[MedicalOntologyMapper] = None


def get_medical_ontology_mapper() -> MedicalOntologyMapper:
    """
    Get singleton medical ontology mapper instance.
    
    Returns:
        MedicalOntologyMapper instance
    """
    global _mapper_instance
    
    if _mapper_instance is None:
        _mapper_instance = MedicalOntologyMapper()
        # Warm cache on first initialization
        _mapper_instance.warm_cache()
    
    return _mapper_instance


# Alias for backward compatibility
get_enhanced_ontology_mapper = get_medical_ontology_mapper
