"""
Safety Guardrails - enforces safety policies on all AI outputs.

All LLM responses MUST pass through this module before reaching users.
This ensures:
1. PII (Personally Identifiable Information) is redacted
2. Medical and nutrition disclaimers are appended
3. Content is logged for compliance auditing


Usage:
    from core.guardrails import SafetyGuardrail

    guardrail = SafetyGuardrail()
    safe_response = guardrail.process_output(raw_response, {"type": "medical"})
"""

import re
from typing import Dict, List, Tuple, Optional, Any
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SafetyGuardrail:
    """
    Enforces safety policies on AI outputs.

    Responsibilities:
    1. PII Redaction - SSN, phone, email, credit card
    2. Medical Disclaimers - Required for health-related content
    3. Nutrition Disclaimers - Required for diet/nutrition content
    4. Audit Logging - Log all outputs for compliance review
    """

    # PII detection patterns
    PII_PATTERNS = {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    }

    MEDICAL_DISCLAIMER = """

---
⚕️ **IMPORTANT HEALTH DISCLAIMER**

I am an AI assistant, not a licensed healthcare provider. The information provided is for educational purposes only and should not be considered medical advice. Always consult with a qualified healthcare professional before making any changes to your diet, exercise routine, or medications.

If you are experiencing a medical emergency, please call emergency services (911) immediately.
"""

    NUTRITION_DISCLAIMER = """

---
🥗 **NUTRITION NOTICE**

Nutritional information is estimated and may vary. This guidance is for informational purposes only. Consult a registered dietitian for personalized nutrition advice.
"""

    def __init__(self, strict_mode: bool = True):
        """
        Initialize SafetyGuardrail.

        Args:
            strict_mode: If True, log all outputs for compliance review
        """
        self.strict_mode = strict_mode
        self._compile_patterns()
        logger.info(f"SafetyGuardrail initialized (strict_mode={strict_mode})")

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for better performance."""
        self.compiled_patterns: Dict[str, re.Pattern] = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PII_PATTERNS.items()
        }

    def process_output(self, text: str, context: Dict) -> str:
        """
        Main entry point - process AI output through all safety checks.

        Args:
            text: Raw AI output
            context: Dict with 'type' (medical/nutrition/general)

        Returns:
            Sanitized output with appropriate disclaimers
        """
        # Step 1: Redact PII
        text = self.redact_pii(text)

        # Step 2: Add appropriate disclaimer
        content_type = context.get("type", "general")
        text = self.add_disclaimer(text, content_type)

        # Step 3: Safety classification (log for review in strict mode)
        if self.strict_mode:
            self.log_for_review(text, context)

        return text

    def redact_pii(self, text: str) -> str:
        """
        Redact personally identifiable information.

        Args:
            text: Input text to scan for PII

        Returns:
            Text with PII redacted as [REDACTED-TYPE]
        """
        redacted = text
        redactions_made: List[str] = []

        for pii_type, pattern in self.compiled_patterns.items():
            matches = pattern.findall(redacted)
            if matches:
                redactions_made.append(f"{pii_type}:{len(matches)}")
                redacted = pattern.sub(f"[REDACTED-{pii_type.upper()}]", redacted)

        if redactions_made:
            logger.warning(f"PII redacted: {', '.join(redactions_made)}")

        return redacted

    def add_disclaimer(self, text: str, content_type: str) -> str:
        """
        Add appropriate disclaimer based on content type.

        Args:
            text: Processed text
            content_type: One of 'medical', 'nutrition', 'general'

        Returns:
            Text with disclaimer appended
        """
        if content_type == "medical":
            return text + self.MEDICAL_DISCLAIMER
        elif content_type == "nutrition":
            return text + self.NUTRITION_DISCLAIMER
        return text

    def get_disclaimer(self, content_type: str = "medical") -> str:
        """
        Get disclaimer text for external use.

        Args:
            content_type: Type of disclaimer needed

        Returns:
            Disclaimer text
        """
        if content_type == "medical":
            return self.MEDICAL_DISCLAIMER.strip()
        elif content_type == "nutrition":
            return self.NUTRITION_DISCLAIMER.strip()
        return ""

    def log_for_review(self, text: str, context: Dict) -> None:
        """
        Log output for compliance review.

        This creates an audit trail for HIPAA compliance.
        """
        content_type = context.get("type", "unknown")
        user_id = context.get("user_id", "anonymous")

        logger.info(
            f"GUARDRAIL_AUDIT: type={content_type} "
            f"user={user_id} "
            f"len={len(text)} "
            f"has_disclaimer={content_type in ('medical', 'nutrition')}"
        )

    def check_safety(self, text: str) -> Dict[str, any]:
        """
        Check text for safety issues without modifying it.

        Useful for pre-flight validation before allowing content.

        Returns:
            Dict with 'safe' (bool) and 'issues' (list) keys
        """
        issues: List[str] = []

        # Check for PII
        for pii_type, pattern in self.compiled_patterns.items():
            if pattern.search(text):
                issues.append(f"Contains {pii_type}")

        return {"safe": len(issues) == 0, "issues": issues, "text_length": len(text)}


class SafetyGuardrails:
    """
    Safety checking with pre-retrieval filtering.
    
    Two-stage approach:
    1. Pre-Retrieval: Filter allergens before LLM sees context
    2. Post-Generation: Verify generated drugs against allergies
    """
    
    def __init__(self, vector_store, reranker, user_preferences_db):
        """
        Initialize safety guardrails.
        
        Args:
            vector_store: Vector storage service
            reranker: Document reranker
            user_preferences_db: User allergies storage
        """
        self.vector_store = vector_store
        self.reranker = reranker
        self.user_preferences_db = user_preferences_db
    
    async def retrieve_safe_context(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> Tuple[List[str], Dict]:
        """
        Retrieve context with ALLERGEN FILTERING.
        
        CRITICAL: Allergens removed BEFORE LLM sees them.
        
        Args:
            query: User question
            user_id: User ID for allergy lookup
            top_k: Number of results desired
            
        Returns:
            (safe_chunks, metadata)
        """
        # Step 1: Get user allergies
        allergies = await self.user_preferences_db.get_allergies(user_id)
        
        if not allergies:
            # No allergies - normal retrieval
            chunks = await self.vector_store.search(query, top_k=top_k)
            return [c['content'] for c in chunks], {"filtered": False}
        
        # Step 2: Standard vector search (get more to account for filtering)
        candidate_chunks = await self.vector_store.search(query, top_k=top_k * 3)
        
        # Step 3: FILTER OUT chunks containing allergens
        safe_chunks = []
        filtered_count = 0
        
        for chunk in candidate_chunks:
            is_safe = self._check_chunk_for_allergens(chunk['content'], allergies)
            
            if is_safe:
                safe_chunks.append(chunk)
                if len(safe_chunks) >= top_k:
                    break  # Got enough safe chunks
            else:
                filtered_count += 1
                logger.warning(
                    f"Filtered chunk containing allergen(s) for user {user_id}"
                )
        
        # Step 4: Rerank by relevance
        if safe_chunks:
            reranked = await self.reranker.rerank(safe_chunks, query)
            safe_content = [c['content'] for c in reranked[:top_k]]
        else:
            safe_content = []
        
        return safe_content, {
            "filtered": True,
            "filtered_count": filtered_count,
            "returned_chunks": len(safe_content),
            "allergens_tracked": len(allergies),
        }
    
    def _check_chunk_for_allergens(
        self,
        content: str,
        allergies: Dict[str, Dict],
    ) -> bool:
        """
        Check if chunk mentions any allergens.
        
        Returns: True if SAFE, False if allergen found
        
        Args:
            content: Document chunk content
            allergies: Dict of user allergies {allergen: details}
            
        Allergen Example:
            {
                "penicillin": {"severity": "severe", "type": "anaphylaxis"},
                "ibuprofen": {"severity": "moderate", "type": "GI upset"},
            }
        """
        content_lower = content.lower()
        
        for allergen in allergies.keys():
            # Word boundary regex (don't match "alli" in "allium")
            pattern = r'\\b' + re.escape(allergen.lower()) + r'\\b'
            
            if re.search(pattern, content_lower):
                logger.debug(
                    f"⚠️ Filtered chunk containing allergen: {allergen}"
                )
                return False  # UNSAFE
        
        return True  # SAFE
    
    async def process_response(
        self,
        response: str,
        user_id: str,
        context_used: List[str],
    ) -> Tuple[str, Dict]:
        """
        Post-Generation Safety Check.
        
        Double-checks generated response against allergies.
        
        Args:
            response: LLM-generated response
            user_id: User ID
            context_used: Context chunks used for generation
            
        Returns:
            (safe_response, safety_metadata)
        """
        allergies = await self.user_preferences_db.get_allergies(user_id)
        if not allergies:
            return response, {"safe": True}
            
        # Check generated response for allergens
        is_safe = self._check_chunk_for_allergens(response, allergies)
        
        if not is_safe:
            logger.critical(f"🚨 LLM generated response containing allergen for user {user_id}!")
            return (
                "I apologize, but I cannot provide that information as it may conflict with your known allergies. Please consult a healthcare professional.",
                {"safe": False, "blocked": True, "reason": "allergen_detected"}
            )
            
        return response, {"safe": True}


# ============================================================================
# ANSWER VALIDATORS (Merged from agents/validation.py)
# ============================================================================
# Provides structured validation for LLM responses before delivery.

@dataclass
class ValidationResult:
    """Result of a validation check."""
    validator_name: str
    passed: bool
    reason: str
    severity: str = "warning"  # "warning", "error", "info"
    suggested_fix: Optional[str] = None
    
    def __str__(self) -> str:
        status = "✅" if self.passed else "❌"
        return f"{status} {self.validator_name}: {self.reason}"


class AnswerValidator(ABC):
    """Abstract base class for answer validators."""
    
    name: str = "BaseValidator"
    severity: str = "warning"
    
    @abstractmethod
    async def validate(
        self,
        answer: str,
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate an answer."""
        pass


class MedicalSafetyValidator(AnswerValidator):
    """
    Validates medical responses for safety.
    
    Checks for potentially dangerous medical advice.
    """
    
    name = "MedicalSafetyValidator"
    severity = "error"
    
    # Patterns that indicate dangerous medical advice
    DANGEROUS_PATTERNS = [
        (r"stop\s+taking\s+(?:your\s+)?(?:prescribed\s+)?medication", 
         "Never advise stopping prescribed medication"),
        (r"increase\s+(?:the\s+)?dosage\s+(?:by\s+yourself|without)", 
         "Never advise changing dosage without doctor"),
        (r"ignore\s+(?:your\s+)?doctor'?s?\s+(?:advice|recommendation)", 
         "Never advise ignoring doctor's advice"),
        (r"you\s+(?:definitely|certainly)\s+have\s+\w+\s+(?:disease|cancer|syndrome)", 
         "AI should not diagnose conditions"),
        (r"this\s+(?:is|will)\s+(?:definitely|certainly)\s+(?:cure|heal)", 
         "AI should not guarantee cures"),
        (r"don'?t\s+(?:go\s+to|see|visit)\s+(?:a\s+)?doctor", 
         "Never advise against seeing a doctor"),
        (r"substitute\s+\w+\s+for\s+(?:your\s+)?(?:prescription|medication)", 
         "Never advise substituting prescriptions"),
    ]
    
    async def validate(
        self,
        answer: str,
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Check for unsafe medical advice."""
        
        answer_lower = answer.lower()
        
        for pattern, reason in self.DANGEROUS_PATTERNS:
            if re.search(pattern, answer_lower, re.IGNORECASE):
                return ValidationResult(
                    validator_name=self.name,
                    passed=False,
                    reason=f"Unsafe medical advice detected: {reason}",
                    severity="error",
                    suggested_fix="Rephrase to recommend professional consultation"
                )
        
        return ValidationResult(
            validator_name=self.name,
            passed=True,
            reason="Passed medical safety check",
            severity="info"
        )


class SourceCitationValidator(AnswerValidator):
    """Validates that medical claims have citations."""
    
    name = "SourceCitationValidator"
    severity = "warning"
    
    CLAIM_PATTERNS = [
        r"studies?\s+(?:show|indicate|suggest|found)",
        r"research\s+(?:shows|indicates|suggests|found)",
        r"according\s+to",
        r"has\s+been\s+(?:proven|shown|demonstrated)",
    ]
    
    CITATION_PATTERNS = [
        r"\[\d+\]",
        r"(?:pubmed|doi|pmid):\s*\S+",
        r"https?://\S+",
    ]
    
    async def validate(
        self,
        answer: str,
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Check for citations when claims are made."""
        
        has_claims = any(
            re.search(pattern, answer, re.IGNORECASE)
            for pattern in self.CLAIM_PATTERNS
        )
        
        if not has_claims:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                reason="No factual claims requiring citations",
                severity="info"
            )
        
        has_citations = any(
            re.search(pattern, answer, re.IGNORECASE)
            for pattern in self.CITATION_PATTERNS
        )
        
        context_citations = context.get("citations", [])
        
        if has_claims and not (has_citations or context_citations):
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                reason="Factual claims made without citations",
                severity="warning",
                suggested_fix="Add source citations for factual claims"
            )
        
        return ValidationResult(
            validator_name=self.name,
            passed=True,
            reason="Citations present for claims",
            severity="info"
        )


class FinalAnswerChecker:
    """
    Runs all validation checks on a final answer.
    
    Usage:
        checker = FinalAnswerChecker()
        passed, results = await checker.check(answer, context)
    """
    
    def __init__(
        self,
        validators: Optional[List[AnswerValidator]] = None,
        fail_on_warning: bool = False
    ):
        self.validators = validators or [
            MedicalSafetyValidator(),
            SourceCitationValidator(),
        ]
        self.fail_on_warning = fail_on_warning
    
    async def check(
        self,
        answer: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[ValidationResult]]:
        """Run all validation checks."""
        context = context or {}
        results = []
        all_passed = True
        
        for validator in self.validators:
            try:
                result = await validator.validate(answer, context)
                results.append(result)
                
                if not result.passed:
                    if result.severity == "error":
                        all_passed = False
                    elif result.severity == "warning" and self.fail_on_warning:
                        all_passed = False
                        
            except Exception as e:
                logger.error(f"Validator {validator.name} failed: {e}")
                all_passed = False
        
        return all_passed, results


def create_medical_answer_checker() -> FinalAnswerChecker:
    """Create a checker configured for medical responses."""
    return FinalAnswerChecker(
        validators=[
            MedicalSafetyValidator(),
            SourceCitationValidator(),
        ],
        fail_on_warning=False
    )
