"""
Conflict Detector for Trust Layer

Identifies and flags conflicting medical information from different sources.

Features:
- Contradiction detection
- Clinical guideline conflicts
- Evidence level conflicts
- Recommendation conflicts
- Severity assessment
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re

logger = logging.getLogger(__name__)


class ConflictSeverity(Enum):
    """Conflict severity levels."""
    CRITICAL = "critical"      # Could harm patient if conflicting info used
    HIGH = "high"              # Significant clinical difference
    MEDIUM = "medium"          # Notable difference, needs clarification
    LOW = "low"                # Minor difference, mostly cosmetic
    INFORMATIONAL = "info"     # Not really a conflict


@dataclass
class Conflict:
    """A detected conflict between sources."""
    document_id_1: str
    document_id_2: str
    severity: ConflictSeverity
    conflict_type: str  # contradiction, evidence_level, recommendation, etc
    statement_1: str  # From first document
    statement_2: str  # From second document
    explanation: str  # Why these conflict
    resolution: Optional[str]  # Suggested resolution


class MedicalConflictDetector:
    """
    Detects conflicts in medical information from multiple sources.
    
    Conflict types:
    1. Direct contradictions (do X vs don't do X)
    2. Evidence level conflicts (guideline vs. case report)
    3. Dosage/treatment conflicts
    4. Diagnostic criteria conflicts
    5. Risk assessment conflicts
    """
    
    # Contradiction pairs
    CONTRADICTION_PAIRS = [
        ("should", "should not"),
        ("recommended", "not recommended"),
        ("indicated", "contraindicated"),
        ("use", "avoid"),
        ("beneficial", "harmful"),
        ("safe", "dangerous"),
        ("effective", "ineffective"),
        ("increase", "decrease"),
        ("yes", "no"),
    ]
    
    # Clinical severity indicators
    CRITICAL_KEYWORDS = {
        "contraindicated": 5,
        "dangerous": 5,
        "fatal": 5,
        "lethal": 5,
        "emergency": 4,
        "life-threatening": 4,
        "critical": 4,
        "severe": 3,
        "serious": 3,
        "adverse": 3,
    }
    
    def __init__(self):
        """Initialize conflict detector."""
        logger.info("MedicalConflictDetector initialized")
    
    def detect_conflicts(
        self, documents: List[Dict]
    ) -> List[Conflict]:
        """
        Detect conflicts across multiple documents.
        
        Args:
            documents: List of dicts with 'id' and 'content' keys
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Compare all pairs
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                doc1 = documents[i]
                doc2 = documents[j]
                
                pairwise_conflicts = self._compare_documents(doc1, doc2)
                conflicts.extend(pairwise_conflicts)
        
        # Sort by severity
        conflicts.sort(
            key=lambda c: self._severity_score(c.severity),
            reverse=True
        )
        
        return conflicts
    
    def _compare_documents(self, doc1: Dict, doc2: Dict) -> List[Conflict]:
        """Compare two documents for conflicts."""
        conflicts = []
        
        doc_id_1 = doc1.get("id", "unknown")
        doc_id_2 = doc2.get("id", "unknown")
        content_1 = doc1.get("content", "").lower()
        content_2 = doc2.get("content", "").lower()
        
        # Check for direct contradictions
        contradiction_conflicts = self._find_contradictions(
            doc_id_1, content_1, doc_id_2, content_2
        )
        conflicts.extend(contradiction_conflicts)
        
        # Check for evidence level conflicts
        evidence_conflicts = self._find_evidence_conflicts(
            doc_id_1, content_1, doc_id_2, content_2
        )
        conflicts.extend(evidence_conflicts)
        
        # Check for dosage conflicts
        dosage_conflicts = self._find_dosage_conflicts(
            doc_id_1, content_1, doc_id_2, content_2
        )
        conflicts.extend(dosage_conflicts)
        
        # Check for recommendation conflicts
        recommendation_conflicts = self._find_recommendation_conflicts(
            doc_id_1, content_1, doc_id_2, content_2
        )
        conflicts.extend(recommendation_conflicts)
        
        return conflicts
    
    def _find_contradictions(
        self, id1: str, content1: str, id2: str, content2: str
    ) -> List[Conflict]:
        """Find direct contradictions."""
        conflicts = []
        
        for positive, negative in self.CONTRADICTION_PAIRS:
            # Check if one says positive and other says negative
            pos_in_1 = f" {positive} " in f" {content1} "
            pos_in_2 = f" {positive} " in f" {content2} "
            
            neg_in_1 = f" {negative} " in f" {content1} "
            neg_in_2 = f" {negative} " in f" {content2} "
            
            # If one has positive and other has negative, it's a conflict
            if (pos_in_1 and neg_in_2) or (neg_in_1 and pos_in_2):
                # Extract sentences with the keywords
                sent1 = self._extract_sentence(content1, positive or negative)
                sent2 = self._extract_sentence(content2, positive or negative)
                
                conflict = Conflict(
                    document_id_1=id1,
                    document_id_2=id2,
                    severity=self._assess_contradiction_severity(
                        content1, content2
                    ),
                    conflict_type="contradiction",
                    statement_1=sent1,
                    statement_2=sent2,
                    explanation=f"One document states '{positive}' while "
                                f"the other states '{negative}'",
                    resolution="Check primary sources and latest guidelines",
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _find_evidence_conflicts(
        self, id1: str, content1: str, id2: str, content2: str
    ) -> List[Conflict]:
        """Find conflicts in evidence levels."""
        conflicts = []
        
        # Check evidence level indicators
        evidence_levels = {
            "randomized controlled trial": 5,
            "meta-analysis": 5,
            "systematic review": 5,
            "clinical guideline": 4,
            "case series": 3,
            "case report": 2,
            "opinion": 1,
        }
        
        level1 = self._detect_evidence_level(content1, evidence_levels)
        level2 = self._detect_evidence_level(content2, evidence_levels)
        
        # If evidence levels differ significantly, it might be a conflict
        if abs(level1 - level2) >= 3:
            conflict = Conflict(
                document_id_1=id1,
                document_id_2=id2,
                severity=ConflictSeverity.MEDIUM,
                conflict_type="evidence_level",
                statement_1=f"Evidence level: {level1}/5",
                statement_2=f"Evidence level: {level2}/5",
                explanation="Sources have different evidence quality levels",
                resolution="Prioritize higher evidence level source",
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def _find_dosage_conflicts(
        self, id1: str, content1: str, id2: str, content2: str
    ) -> List[Conflict]:
        """Find conflicts in dosages or quantities."""
        conflicts = []
        
        # Extract dosages (simple pattern: "X mg/day", "X units/hour", etc)
        dosage_pattern = r'(\d+(?:\.\d+)?)\s*(mg|g|unit|ml|μg|ng|IU)(?:/day|/week|/hour)?'
        
        dosages1 = re.findall(dosage_pattern, content1, re.IGNORECASE)
        dosages2 = re.findall(dosage_pattern, content2, re.IGNORECASE)
        
        # Compare dosages
        if dosages1 and dosages2:
            dose_val_1 = float(dosages1[0][0])
            dose_val_2 = float(dosages2[0][0])
            
            # If dosages differ significantly (> 50%), flag as conflict
            if dose_val_1 > 0 and dose_val_2 > 0:
                ratio = max(dose_val_1, dose_val_2) / min(dose_val_1, dose_val_2)
                if ratio > 1.5:
                    conflict = Conflict(
                        document_id_1=id1,
                        document_id_2=id2,
                        severity=ConflictSeverity.CRITICAL,  # Dosage conflicts are serious
                        conflict_type="dosage",
                        statement_1=f"Dosage: {dosages1[0][0]} {dosages1[0][1]}",
                        statement_2=f"Dosage: {dosages2[0][0]} {dosages2[0][1]}",
                        explanation="Recommended dosages differ significantly",
                        resolution="Verify with pharmacist or official guidelines",
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _find_recommendation_conflicts(
        self, id1: str, content1: str, id2: str, content2: str
    ) -> List[Conflict]:
        """Find conflicts in recommendations."""
        conflicts = []
        
        # Common recommendation patterns
        patterns = [
            ("first-line", "second-line"),
            ("preferred", "alternative"),
            ("always", "never"),
            ("must", "must not"),
        ]
        
        for pattern1, pattern2 in patterns:
            if pattern1 in content1 and pattern2 in content2:
                conflict = Conflict(
                    document_id_1=id1,
                    document_id_2=id2,
                    severity=ConflictSeverity.HIGH,
                    conflict_type="recommendation",
                    statement_1=self._extract_sentence(content1, pattern1),
                    statement_2=self._extract_sentence(content2, pattern2),
                    explanation=f"Recommendations differ on precedence "
                                f"('{pattern1}' vs '{pattern2}')",
                    resolution="Check dates - newer guideline may supersede",
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _extract_sentence(self, text: str, keyword: str) -> str:
        """Extract sentence containing keyword."""
        sentences = text.split(".")
        for sentence in sentences:
            if keyword in sentence.lower():
                return sentence.strip()[:100]  # First 100 chars
        return f"(Contains '{keyword}')"
    
    def _assess_contradiction_severity(
        self, content1: str, content2: str
    ) -> ConflictSeverity:
        """Assess severity of a contradiction."""
        max_severity = ConflictSeverity.LOW
        
        # Check for critical keywords
        combined = content1 + " " + content2
        for keyword, score in self.CRITICAL_KEYWORDS.items():
            if keyword in combined:
                if score >= 5:
                    max_severity = ConflictSeverity.CRITICAL
                elif score >= 4:
                    max_severity = ConflictSeverity.HIGH
                elif score >= 3:
                    max_severity = ConflictSeverity.MEDIUM
        
        return max_severity
    
    def _detect_evidence_level(
        self, content: str, levels: Dict[str, int]
    ) -> int:
        """Detect evidence level in content."""
        max_level = 1  # Default: opinion
        
        for level_name, level_score in levels.items():
            if level_name in content.lower():
                max_level = max(max_level, level_score)
        
        return max_level
    
    def _severity_score(self, severity: ConflictSeverity) -> int:
        """Get numeric score for severity."""
        scores = {
            ConflictSeverity.CRITICAL: 5,
            ConflictSeverity.HIGH: 4,
            ConflictSeverity.MEDIUM: 3,
            ConflictSeverity.LOW: 2,
            ConflictSeverity.INFORMATIONAL: 1,
        }
        return scores.get(severity, 0)
    
    def generate_conflict_report(self, conflicts: List[Conflict]) -> str:
        """Generate human-readable conflict report."""
        if not conflicts:
            return "✅ No conflicts detected"
        
        report = f"⚠️ Detected {len(conflicts)} conflict(s):\n\n"
        
        for i, conflict in enumerate(conflicts, 1):
            report += f"{i}. [{conflict.severity.value.upper()}] {conflict.conflict_type}\n"
            report += f"   Source 1: {conflict.document_id_1}\n"
            report += f"   Source 2: {conflict.document_id_2}\n"
            report += f"   Issue: {conflict.explanation}\n"
            if conflict.resolution:
                report += f"   ✓ Resolution: {conflict.resolution}\n"
            report += "\n"
        
        return report


# Singleton instance
_detector_instance = None


def get_conflict_detector() -> MedicalConflictDetector:
    """Get singleton instance of conflict detector."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = MedicalConflictDetector()
    return _detector_instance
