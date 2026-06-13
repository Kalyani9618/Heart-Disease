"""
Source Validator for Trust Layer

Validates retrieved medical information against trusted sources and flags
potentially outdated or conflicting information.

Features:
- Source credibility assessment
- Medical guideline validation
- Conflict detection between sources
- Confidence scoring
- Recommendation generation
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class SourceCredibility(Enum):
    """Source credibility levels."""
    PEER_REVIEWED = 5          # Published in peer-reviewed journals
    MEDICAL_GUIDELINE = 4      # Official medical guidelines (AHA, AMA, etc)
    PROFESSIONAL_PUBLISHED = 3 # Published by medical professionals
    CLINICAL_PRACTICE = 2      # Based on clinical practice standards
    GENERAL_MEDICAL = 1        # General medical information
    UNVALIDATED = 0           # Not validated/unknown source


class ValidationLevel(Enum):
    """Information validation levels."""
    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONFLICTING = "conflicting"
    OUTDATED = "outdated"
    UNVALIDATED = "unvalidated"
    REQUIRES_DISCLAIMER = "requires_disclaimer"


@dataclass
class ValidationResult:
    """Result of source validation."""
    document_id: str
    validation_level: ValidationLevel
    credibility_score: float  # 0-1
    confidence: float  # 0-1
    issues: List[str]  # Any problems found
    recommendations: List[str]  # What to do about issues
    conflicting_sources: List[str] = field(default_factory=list)  # Other sources that conflict
    last_updated: Optional[datetime] = None
    needs_human_review: bool = False


@dataclass
class SourceMetadata:
    """Metadata about a source document."""
    document_id: str
    title: str
    publication_date: Optional[datetime] = None
    source_type: str = "unknown"  # journal, guideline, textbook, etc
    authors: List[str] = field(default_factory=list)
    credibility_level: SourceCredibility = SourceCredibility.UNVALIDATED
    medical_domains: List[str] = field(default_factory=list)  # cardiology, oncology, etc
    is_peer_reviewed: bool = False
    citation_count: Optional[int] = None
    last_updated: Optional[datetime] = None


class MedicalSourceValidator:
    """
    Validates medical sources and flags potentially problematic information.
    
    Checks:
    1. Source credibility (peer-reviewed, guidelines, etc)
    2. Information recency (outdated information)
    3. Conflicts with other sources
    4. Coverage of all relevant medical domains
    5. Clinical evidence strength
    """
    
    # Trusted medical organizations
    TRUSTED_ORGANIZATIONS = {
        "AHA": ("American Heart Association", 5),  # Highest credibility
        "ACC": ("American College of Cardiology", 5),
        "AMA": ("American Medical Association", 4),
        "NIH": ("National Institutes of Health", 5),
        "FDA": ("Food and Drug Administration", 5),
        "WHO": ("World Health Organization", 4),
        "ESC": ("European Society of Cardiology", 5),
    }
    
    # Maximum age for information (in days)
    MAX_AGES = {
        "guideline": 365 * 3,      # 3 years for guidelines
        "research": 365 * 5,       # 5 years for research
        "diagnostic": 365 * 2,     # 2 years for diagnostic info
        "treatment": 365 * 2,      # 2 years for treatment info
        "general": 365 * 10,       # 10 years for general info
    }
    
    def __init__(self):
        """Initialize the validator."""
        self.source_cache = {}  # Cache of validated sources
        self.conflict_graph = {}  # Track known conflicts
        logger.info("MedicalSourceValidator initialized")
    
    def validate_source(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict] = None,
        conflicting_sources: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Validate a medical source document.
        
        Args:
            document_id: Unique document identifier
            content: Document content to validate
            metadata: Optional metadata about the source
            conflicting_sources: Known conflicting sources
            
        Returns:
            ValidationResult with assessment and recommendations
        """
        issues = []
        recommendations = []
        conflicting = conflicting_sources or []
        
        # Convert SourceMetadata dataclass to dict if needed
        if metadata is not None and hasattr(metadata, '__dataclass_fields__'):
            from dataclasses import asdict
            metadata = asdict(metadata)
        
        # 1. Check source credibility
        credibility_score = self._assess_credibility(metadata or {}, content)
        
        # 2. Check for outdated information
        recency_issues, last_updated = self._check_recency(metadata or {}, content)
        if recency_issues:
            issues.extend(recency_issues)
            recommendations.append("Consider consulting more recent sources")
        
        # 3. Check for conflicts
        conflict_issues, conflicting_docs = self._check_conflicts(
            document_id, content, conflicting_sources or []
        )
        if conflict_issues:
            issues.extend(conflict_issues)
            recommendations.append("Review conflicting sources for latest evidence")
            conflicting = conflicting_docs
        
        # 4. Check medical domains covered
        domain_issues = self._check_domains(content)
        if domain_issues:
            issues.extend(domain_issues)
        
        # 5. Determine validation level
        validation_level = self._determine_validation_level(
            issues, credibility_score
        )
        
        # 6. Calculate confidence
        confidence = self._calculate_confidence(
            credibility_score, validation_level, bool(issues)
        )
        
        # Determine if human review needed
        needs_review = (
            validation_level in [
                ValidationLevel.CONFLICTING,
                ValidationLevel.OUTDATED,
                ValidationLevel.UNVALIDATED
            ]
            or credibility_score < 2.0
            or confidence < 0.6
        )
        
        return ValidationResult(
            document_id=document_id,
            validation_level=validation_level,
            credibility_score=credibility_score,
            confidence=confidence,
            issues=issues,
            recommendations=recommendations,
            conflicting_sources=conflicting,
            last_updated=last_updated,
            needs_human_review=needs_review,
        )
    
    def _assess_credibility(self, metadata: Dict, content: str) -> float:
        """
        Assess source credibility on scale 0-5.
        
        Checks:
        - Credibility level from metadata
        - Known trusted organizations
        - Peer-reviewed status
        - Citation history
        - Author credentials
        """
        score = 1.0  # Default baseline for unknown sources
        
        # Use credibility_level from metadata if available
        cred_level = metadata.get("credibility_level")
        if cred_level is not None:
            if isinstance(cred_level, SourceCredibility):
                score = float(cred_level.value)
            elif isinstance(cred_level, (int, float)):
                score = float(cred_level)
        
        # Check for trusted organizations
        for org_code, (org_name, org_score) in self.TRUSTED_ORGANIZATIONS.items():
            if org_code in content.upper() or org_name.upper() in content.upper():
                score = max(score, float(org_score))
        
        # Check peer-reviewed status
        if metadata.get("is_peer_reviewed"):
            score += 0.5
        
        # Check citation count
        citation_count = metadata.get("citation_count")
        if citation_count and citation_count > 100:
            score += 0.3
        elif citation_count and citation_count > 50:
            score += 0.1
        
        # Check publication date (newer = more credible, up to a point)
        pub_date = metadata.get("publication_date")
        if pub_date:
            age_days = (datetime.now() - pub_date).days
            if age_days < 365:  # Within 1 year
                score += 0.2
            elif age_days < 365 * 5:  # Within 5 years
                score += 0.1
        
        # Cap at 5.0
        return min(score, 5.0)
    
    def _check_recency(
        self, metadata: Dict, content: str
    ) -> Tuple[List[str], Optional[datetime]]:
        """Check if information is current or outdated."""
        issues = []
        last_updated = metadata.get("last_updated")
        
        if not last_updated and metadata.get("publication_date"):
            last_updated = metadata.get("publication_date")
        
        if last_updated:
            age_days = (datetime.now() - last_updated).days
            
            # Determine max age based on content type
            max_age = self.MAX_AGES.get("general", 365 * 10)
            
            if "guideline" in content.lower():
                max_age = self.MAX_AGES["guideline"]
            elif "treatment" in content.lower():
                max_age = self.MAX_AGES["treatment"]
            
            if age_days > max_age:
                issues.append(
                    f"Information is {age_days} days old (max recommended: {max_age})"
                )
        
        return issues, last_updated
    
    def _check_conflicts(
        self, doc_id: str, content: str, known_conflicts: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Check for conflicts with other sources."""
        issues = []
        conflicting = []
        
        # Check against known conflicts
        for conflict_doc in known_conflicts:
            issues.append(
                f"Conflicts with source: {conflict_doc}"
            )
            conflicting.append(conflict_doc)
        
        # Check for contradictory statements in content
        contradictory_phrases = [
            ("should not", "should"),
            ("contraindicated", "recommended"),
            ("avoid", "use"),
        ]
        
        for neg_phrase, pos_phrase in contradictory_phrases:
            has_neg = neg_phrase.lower() in content.lower()
            has_pos = pos_phrase.lower() in content.lower()
            
            if has_neg and has_pos:
                issues.append(
                    f"Contains both '{neg_phrase}' and '{pos_phrase}' statements"
                )
        
        return issues, conflicting
    
    def _check_domains(self, content: str) -> List[str]:
        """Check medical domains covered."""
        issues = []
        
        # Check if critical domains are missing
        critical_domains = ["diagnosis", "treatment", "monitoring"]
        
        for domain in critical_domains:
            if domain.lower() not in content.lower():
                issues.append(f"Missing coverage of {domain}")
        
        return issues
    
    def _determine_validation_level(
        self, issues: List[str], credibility: float
    ) -> ValidationLevel:
        """Determine overall validation level."""
        if not issues and credibility > 0.8:
            return ValidationLevel.FULLY_SUPPORTED
        elif not issues and credibility > 0.6:
            return ValidationLevel.PARTIALLY_SUPPORTED
        elif any("outdated" in issue.lower() for issue in issues):
            return ValidationLevel.OUTDATED
        elif any("conflict" in issue.lower() for issue in issues):
            return ValidationLevel.CONFLICTING
        elif credibility < 0.4:
            return ValidationLevel.UNVALIDATED
        else:
            return ValidationLevel.REQUIRES_DISCLAIMER
    
    def _calculate_confidence(
        self, credibility: float, validation: ValidationLevel, has_issues: bool
    ) -> float:
        """Calculate overall confidence in the source."""
        base = credibility
        
        # Adjust based on validation level
        if validation == ValidationLevel.FULLY_SUPPORTED:
            return min(base * 1.1, 1.0)
        elif validation == ValidationLevel.PARTIALLY_SUPPORTED:
            return base * 0.9
        elif validation == ValidationLevel.CONFLICTING:
            return base * 0.5
        elif validation == ValidationLevel.OUTDATED:
            return base * 0.6
        elif validation == ValidationLevel.UNVALIDATED:
            return base * 0.3
        else:  # REQUIRES_DISCLAIMER
            return base * 0.75
    
    def batch_validate(
        self, documents: List[Dict]
    ) -> List[ValidationResult]:
        """
        Validate multiple documents.
        
        Args:
            documents: List of dicts with 'id', 'content', 'metadata' keys
            
        Returns:
            List of ValidationResults
        """
        results = []
        
        for doc in documents:
            result = self.validate_source(
                document_id=doc.get("id", "unknown"),
                content=doc.get("content", ""),
                metadata=doc.get("metadata"),
            )
            results.append(result)
        
        # Check for cross-document conflicts
        for i, result1 in enumerate(results):
            for result2 in results[i+1:]:
                if self._has_conflict(result1, result2):
                    result1.conflicting_sources.append(result2.document_id)
                    result2.conflicting_sources.append(result1.document_id)
        
        return results
    
    def _has_conflict(
        self, result1: ValidationResult, result2: ValidationResult
    ) -> bool:
        """Check if two validation results conflict."""
        # Simple heuristic: different validation levels suggest conflict
        if result1.validation_level == ValidationLevel.CONFLICTING:
            return True
        if result2.validation_level == ValidationLevel.CONFLICTING:
            return True
        
        # Check for explicit conflict markers in issue lists
        conflict_keywords = ["contradict", "conflict", "disagree"]
        for keyword in conflict_keywords:
            for issue in result1.issues + result2.issues:
                if keyword in issue.lower():
                    return True
        
        return False


# Singleton instance
_validator_instance = None


def get_source_validator() -> MedicalSourceValidator:
    """Get singleton instance of source validator."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = MedicalSourceValidator()
    return _validator_instance
