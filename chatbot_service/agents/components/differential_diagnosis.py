"""
Differential Diagnosis Engine - Chain-of-Thought Medical Reasoning

Provides:
- Ranked differential diagnosis with reasoning
- Symptom-to-condition mapping
- Rule-out logic for unlikely causes
- Evidence-based likelihood scoring

Based on MedGemma clinical reasoning capabilities.
"""


import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class DiagnosisLikelihood(Enum):
    """Likelihood classification for differential diagnoses."""
    HIGHLY_LIKELY = "highly_likely"      # >70% probability
    LIKELY = "likely"                     # 40-70% probability
    POSSIBLE = "possible"                 # 15-40% probability
    UNLIKELY = "unlikely"                 # 5-15% probability
    RULED_OUT = "ruled_out"               # <5% probability


class SeverityLevel(Enum):
    """Clinical severity classification."""
    LIFE_THREATENING = "life_threatening"
    SEVERE = "severe"
    MODERATE = "moderate"
    MILD = "mild"
    BENIGN = "benign"


@dataclass
class ClinicalPresentation:
    """Structured clinical presentation for analysis."""
    chief_complaint: str
    symptoms: List[str]
    duration: Optional[str] = None
    onset: Optional[str] = None  # "sudden", "gradual"
    severity: Optional[str] = None  # "mild", "moderate", "severe"
    associated_symptoms: List[str] = field(default_factory=list)
    pertinent_negatives: List[str] = field(default_factory=list)
    vital_signs: Optional[Dict[str, Any]] = None
    medical_history: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    age: Optional[int] = None
    sex: Optional[str] = None


@dataclass
class DiagnosisCandidate:
    """A single diagnosis candidate with reasoning."""
    condition: str
    icd10_code: Optional[str]
    likelihood: DiagnosisLikelihood
    probability_estimate: float  # 0.0 to 1.0
    severity: SeverityLevel
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    key_discriminating_features: List[str]
    recommended_workup: List[str]
    reasoning: str


@dataclass
class DifferentialDiagnosisResult:
    """Complete differential diagnosis with reasoning trace."""
    presentation: ClinicalPresentation
    differentials: List[DiagnosisCandidate]
    most_likely: str
    cannot_miss: List[str]  # Life-threatening diagnoses to rule out
    reasoning_trace: List[str]
    recommended_immediate_actions: List[str]
    red_flags: List[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    confidence: float = 0.0

    def to_markdown(self) -> str:
        """Convert to formatted markdown for display."""
        lines = [
            "## Differential Diagnosis Analysis",
            "",
            f"**Chief Complaint**: {self.presentation.chief_complaint}",
            f"**Symptoms**: {', '.join(self.presentation.symptoms)}",
            "",
        ]
        
        if self.red_flags:
            lines.append("### 🚨 Red Flags")
            for flag in self.red_flags:
                lines.append(f"- {flag}")
            lines.append("")
        
        if self.cannot_miss:
            lines.append("### ⚠️ Cannot-Miss Diagnoses")
            for dx in self.cannot_miss:
                lines.append(f"- {dx}")
            lines.append("")
        
        lines.append("### Ranked Differential Diagnoses")
        lines.append("")
        
        for i, dx in enumerate(self.differentials[:5], 1):
            emoji = "🔴" if dx.likelihood == DiagnosisLikelihood.HIGHLY_LIKELY else \
                    "🟠" if dx.likelihood == DiagnosisLikelihood.LIKELY else \
                    "🟡" if dx.likelihood == DiagnosisLikelihood.POSSIBLE else "⚪"
            
            lines.append(f"#### {i}. {emoji} {dx.condition}")
            lines.append(f"- **Likelihood**: {dx.likelihood.value} ({dx.probability_estimate*100:.0f}%)")
            lines.append(f"- **Severity**: {dx.severity.value}")
            lines.append(f"- **Supporting Evidence**: {', '.join(dx.supporting_evidence[:3])}")
            if dx.contradicting_evidence:
                lines.append(f"- **Against**: {', '.join(dx.contradicting_evidence[:2])}")
            lines.append(f"- **Reasoning**: {dx.reasoning}")
            lines.append("")
        
        if self.recommended_immediate_actions:
            lines.append("### Recommended Immediate Actions")
            for action in self.recommended_immediate_actions:
                lines.append(f"1. {action}")
        
        lines.append("")
        lines.append("---")
        lines.append("*⚠️ This is AI-assisted clinical decision support. ")
        lines.append("Always apply clinical judgment and consult appropriate specialists.*")
        
        return "\n".join(lines)


class DifferentialDiagnosisEngine:
    """
    Chain-of-Thought Differential Diagnosis Generator.
    
    Uses structured clinical reasoning to:
    1. Parse symptoms and clinical context
    2. Generate initial differential list
    3. Apply Bayesian-style likelihood adjustment
    4. Identify cannot-miss diagnoses
    5. Recommend workup
    """
    
    # Common symptom-to-condition mappings for initialization
    SYMPTOM_CONDITION_MAP = {
        "chest_pain": [
            ("Acute Coronary Syndrome", SeverityLevel.LIFE_THREATENING),
            ("Pulmonary Embolism", SeverityLevel.LIFE_THREATENING),
            ("Aortic Dissection", SeverityLevel.LIFE_THREATENING),
            ("Pericarditis", SeverityLevel.MODERATE),
            ("Musculoskeletal Pain", SeverityLevel.MILD),
            ("GERD", SeverityLevel.MILD),
            ("Anxiety/Panic Attack", SeverityLevel.MILD),
        ],
        "shortness_of_breath": [
            ("Pulmonary Embolism", SeverityLevel.LIFE_THREATENING),
            ("Heart Failure", SeverityLevel.SEVERE),
            ("Pneumonia", SeverityLevel.MODERATE),
            ("Asthma/COPD Exacerbation", SeverityLevel.MODERATE),
            ("Anxiety", SeverityLevel.MILD),
        ],
        "headache": [
            ("Subarachnoid Hemorrhage", SeverityLevel.LIFE_THREATENING),
            ("Meningitis", SeverityLevel.LIFE_THREATENING),
            ("Stroke", SeverityLevel.LIFE_THREATENING),
            ("Migraine", SeverityLevel.MODERATE),
            ("Tension Headache", SeverityLevel.MILD),
        ],
        "abdominal_pain": [
            ("Appendicitis", SeverityLevel.SEVERE),
            ("Bowel Obstruction", SeverityLevel.SEVERE),
            ("Ruptured AAA", SeverityLevel.LIFE_THREATENING),
            ("Pancreatitis", SeverityLevel.MODERATE),
            ("Gastroenteritis", SeverityLevel.MILD),
        ],
    }
    
    # Red flag symptoms requiring urgent evaluation
    RED_FLAGS = {
        "chest_pain": ["radiating to arm/jaw", "associated with syncope", "tearing pain"],
        "headache": ["thunderclap onset", "worst headache of life", "neck stiffness", "altered mental status"],
        "abdominal_pain": ["rigid abdomen", "rebound tenderness", "hemodynamic instability"],
        "shortness_of_breath": ["cyanosis", "altered mental status", "unable to speak sentences"],
    }

    def __init__(self, llm_gateway=None, use_llm_reasoning: bool = True):
        """
        Initialize differential diagnosis engine.
        
        Args:
            llm_gateway: LLM for advanced reasoning
            use_llm_reasoning: Whether to use LLM for reasoning steps
        """
        self.llm_gateway = llm_gateway
        self.use_llm_reasoning = use_llm_reasoning and llm_gateway is not None

    async def generate_differential(
        self,
        presentation: ClinicalPresentation,
        max_diagnoses: int = 7,
    ) -> DifferentialDiagnosisResult:
        """
        Generate ranked differential diagnosis.
        
        Args:
            presentation: Clinical presentation data
            max_diagnoses: Maximum diagnoses to include
            
        Returns:
            DifferentialDiagnosisResult with ranked diagnoses
        """
        reasoning_trace = []
        
        # Step 1: Parse chief complaint
        reasoning_trace.append(f"[STEP 1] Analyzing chief complaint: '{presentation.chief_complaint}'")
        chief_category = self._categorize_chief_complaint(presentation.chief_complaint)
        reasoning_trace.append(f"  → Categorized as: {chief_category}")
        
        # Step 2: Identify red flags
        reasoning_trace.append("[STEP 2] Checking for red flag symptoms...")
        red_flags = self._identify_red_flags(presentation, chief_category)
        if red_flags:
            reasoning_trace.append(f"  ⚠️ RED FLAGS IDENTIFIED: {', '.join(red_flags)}")
        else:
            reasoning_trace.append("  → No immediate red flags detected")
        
        # Step 3: Generate initial differential
        reasoning_trace.append("[STEP 3] Generating initial differential list...")
        initial_differentials = self._get_initial_differentials(chief_category)
        reasoning_trace.append(f"  → {len(initial_differentials)} conditions under consideration")
        
        # Step 4: Apply clinical context to adjust probabilities
        reasoning_trace.append("[STEP 4] Applying clinical context adjustments...")
        adjusted_differentials = self._apply_clinical_context(
            initial_differentials, presentation
        )
        
        # Step 5: Use LLM for advanced reasoning if available
        if self.use_llm_reasoning:
            reasoning_trace.append("[STEP 5] Applying AI clinical reasoning...")
            adjusted_differentials = await self._llm_reasoning_adjustment(
                adjusted_differentials, presentation
            )
        
        # Step 6: Rank and filter
        reasoning_trace.append("[STEP 6] Ranking differentials by probability...")
        sorted_differentials = sorted(
            adjusted_differentials,
            key=lambda x: x.probability_estimate,
            reverse=True
        )[:max_diagnoses]
        
        # Step 7: Identify cannot-miss diagnoses
        cannot_miss = [
            dx.condition for dx in adjusted_differentials
            if dx.severity == SeverityLevel.LIFE_THREATENING
            and dx.probability_estimate > 0.05
        ]
        
        # Step 8: Generate recommendations
        immediate_actions = self._generate_immediate_actions(
            sorted_differentials, red_flags, presentation
        )
        
        result = DifferentialDiagnosisResult(
            presentation=presentation,
            differentials=sorted_differentials,
            most_likely=sorted_differentials[0].condition if sorted_differentials else "Unknown",
            cannot_miss=cannot_miss,
            reasoning_trace=reasoning_trace,
            recommended_immediate_actions=immediate_actions,
            red_flags=red_flags,
            confidence=sorted_differentials[0].probability_estimate if sorted_differentials else 0.0,
        )
        
        logger.info(f"Generated differential diagnosis: {result.most_likely}")
        return result

    def _categorize_chief_complaint(self, complaint: str) -> str:
        """Categorize chief complaint into symptom category."""
        complaint_lower = complaint.lower()
        
        if any(term in complaint_lower for term in ["chest", "heart", "cardiac"]):
            return "chest_pain"
        elif any(term in complaint_lower for term in ["breath", "dyspnea", "breathing"]):
            return "shortness_of_breath"
        elif any(term in complaint_lower for term in ["head", "migraine"]):
            return "headache"
        elif any(term in complaint_lower for term in ["abdom", "stomach", "belly"]):
            return "abdominal_pain"
        
        return "general"

    def _identify_red_flags(
        self,
        presentation: ClinicalPresentation,
        category: str
    ) -> List[str]:
        """Identify red flag symptoms requiring urgent evaluation."""
        red_flags = []
        all_symptoms = " ".join(presentation.symptoms + presentation.associated_symptoms).lower()
        
        category_flags = self.RED_FLAGS.get(category, [])
        for flag in category_flags:
            if flag.lower() in all_symptoms:
                red_flags.append(flag)
        
        # Universal red flags
        if presentation.vital_signs:
            if presentation.vital_signs.get("systolic_bp", 120) < 90:
                red_flags.append("Hypotension")
            if presentation.vital_signs.get("heart_rate", 80) > 120:
                red_flags.append("Tachycardia")
            if presentation.vital_signs.get("spo2", 98) < 92:
                red_flags.append("Hypoxia")
        
        return red_flags

    def _get_initial_differentials(self, category: str) -> List[DiagnosisCandidate]:
        """Get initial differential list based on symptom category."""
        conditions = self.SYMPTOM_CONDITION_MAP.get(category, [])
        
        differentials = []
        base_probability = 1.0 / len(conditions) if conditions else 0.1
        
        for condition, severity in conditions:
            differentials.append(DiagnosisCandidate(
                condition=condition,
                icd10_code=None,
                likelihood=DiagnosisLikelihood.POSSIBLE,
                probability_estimate=base_probability,
                severity=severity,
                supporting_evidence=[],
                contradicting_evidence=[],
                key_discriminating_features=[],
                recommended_workup=[],
                reasoning="Initial consideration based on chief complaint.",
            ))
        
        return differentials

    def _apply_clinical_context(
        self,
        differentials: List[DiagnosisCandidate],
        presentation: ClinicalPresentation
    ) -> List[DiagnosisCandidate]:
        """Apply clinical context to adjust probabilities."""
        for dx in differentials:
            # Age-based adjustments
            if presentation.age:
                if "Coronary" in dx.condition and presentation.age > 50:
                    dx.probability_estimate *= 1.5
                if "Coronary" in dx.condition and presentation.age < 30:
                    dx.probability_estimate *= 0.3
            
            # Sex-based adjustments
            if presentation.sex:
                if "Coronary" in dx.condition and presentation.sex.lower() == "male":
                    dx.probability_estimate *= 1.3
            
            # Symptom-specific adjustments
            symptoms_text = " ".join(presentation.symptoms).lower()
            
            if "crushing" in symptoms_text or "pressure" in symptoms_text:
                if "Coronary" in dx.condition or "Infarction" in dx.condition:
                    dx.probability_estimate *= 2.0
                    dx.supporting_evidence.append("Crushing/pressure quality typical of cardiac chest pain")
            
            if "tearing" in symptoms_text:
                if "Dissection" in dx.condition:
                    dx.probability_estimate *= 3.0
                    dx.supporting_evidence.append("Tearing pain highly suggestive of aortic dissection")
            
            if "pleuritic" in symptoms_text:
                if "Pericarditis" in dx.condition or "Embolism" in dx.condition:
                    dx.probability_estimate *= 1.5
                if "Coronary" in dx.condition:
                    dx.probability_estimate *= 0.5
                    dx.contradicting_evidence.append("Pleuritic pain less typical of ACS")
            
            # Update likelihood category
            dx.likelihood = self._probability_to_likelihood(dx.probability_estimate)
        
        return differentials

    async def _llm_reasoning_adjustment(
        self,
        differentials: List[DiagnosisCandidate],
        presentation: ClinicalPresentation
    ) -> List[DiagnosisCandidate]:
        """Use LLM for advanced clinical reasoning adjustments."""
        if not self.llm_gateway:
            return differentials
        
        prompt = f"""You are an expert diagnostician. Analyze this clinical presentation and adjust the differential diagnosis probabilities.

CLINICAL PRESENTATION:
- Chief Complaint: {presentation.chief_complaint}
- Symptoms: {', '.join(presentation.symptoms)}
- Duration: {presentation.duration or 'Not specified'}
- Onset: {presentation.onset or 'Not specified'}
- Age: {presentation.age or 'Not specified'}
- Sex: {presentation.sex or 'Not specified'}
- Pertinent Negatives: {', '.join(presentation.pertinent_negatives) or 'None reported'}

CURRENT DIFFERENTIAL (ranked by initial probability):
{chr(10).join(f"- {dx.condition}: {dx.probability_estimate*100:.0f}%" for dx in differentials[:5])}

For each diagnosis, provide:
1. Adjusted probability (0-100%)
2. Key supporting evidence from the presentation
3. Key contradicting evidence
4. One sentence reasoning

Format as JSON array.
"""
        
        try:
            response = await self.llm_gateway.generate(prompt)
            
            # Extract JSON from response (handle potential markdown code blocks)
            import json
            import re
            
            json_str = response
            if "```json" in response:
                match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
                if match:
                    json_str = match.group(1)
            elif "```" in response:
                match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
                if match:
                    json_str = match.group(1)
            
            try:
                adjustments = json.loads(json_str)
                
                # Update differentials based on LLM feedback
                for adj in adjustments:
                    # Find matching differential (fuzzy match)
                    target_dx = None
                    adj_condition = adj.get("condition", adj.get("diagnosis", "")).lower()
                    
                    for dx in differentials:
                        if dx.condition.lower() in adj_condition or adj_condition in dx.condition.lower():
                            target_dx = dx
                            break
                    
                    if target_dx:
                        # Update probability
                        if "adjusted_probability" in adj:
                            # Handle "80%" or 0.8 format
                            prob_val = adj["adjusted_probability"]
                            if isinstance(prob_val, str):
                                prob_val = float(prob_val.replace("%", "").strip()) / 100.0
                            target_dx.probability_estimate = float(prob_val)
                            target_dx.likelihood = self._probability_to_likelihood(target_dx.probability_estimate)
                        
                        # Add evidence
                        if "supporting_evidence" in adj:
                            ev = adj["supporting_evidence"]
                            if isinstance(ev, list):
                                target_dx.supporting_evidence.extend(ev)
                            elif isinstance(ev, str):
                                target_dx.supporting_evidence.append(ev)
                                
                        if "contradicting_evidence" in adj:
                            ev = adj["contradicting_evidence"]
                            if isinstance(ev, list):
                                target_dx.contradicting_evidence.extend(ev)
                            elif isinstance(ev, str):
                                target_dx.contradicting_evidence.append(ev)
                                
                        # Update reasoning
                        if "reasoning" in adj:
                            target_dx.reasoning = f"{target_dx.reasoning} [AI]: {adj['reasoning']}"
                            
                logger.info("LLM reasoning applied to differential diagnosis")
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM reasoning JSON: {json_str[:100]}...")
                
        except Exception as e:
            logger.error(f"LLM reasoning failed: {e}")
        
        return differentials

    def _probability_to_likelihood(self, prob: float) -> DiagnosisLikelihood:
        """Convert probability to likelihood category."""
        if prob > 0.7:
            return DiagnosisLikelihood.HIGHLY_LIKELY
        elif prob > 0.4:
            return DiagnosisLikelihood.LIKELY
        elif prob > 0.15:
            return DiagnosisLikelihood.POSSIBLE
        elif prob > 0.05:
            return DiagnosisLikelihood.UNLIKELY
        else:
            return DiagnosisLikelihood.RULED_OUT

    def _generate_immediate_actions(
        self,
        differentials: List[DiagnosisCandidate],
        red_flags: List[str],
        presentation: ClinicalPresentation
    ) -> List[str]:
        """Generate recommended immediate actions."""
        actions = []
        
        if red_flags:
            actions.append("STAT evaluation by physician")
        
        # Check for life-threatening conditions in top differential
        for dx in differentials[:3]:
            if dx.severity == SeverityLevel.LIFE_THREATENING:
                if "Coronary" in dx.condition or "Infarction" in dx.condition:
                    actions.append("12-lead ECG immediately")
                    actions.append("Troponin levels")
                    actions.append("Aspirin 325mg if no contraindication")
                elif "Embolism" in dx.condition:
                    actions.append("D-dimer or CT-PA")
                    actions.append("Evaluate Wells score")
                elif "Dissection" in dx.condition:
                    actions.append("CT angiography of chest")
                    actions.append("Blood pressure control")
                elif "Hemorrhage" in dx.condition or "Stroke" in dx.condition:
                    actions.append("Non-contrast CT head immediately")
                    actions.append("Neurology consult")
        
        if not actions:
            actions.append("Complete history and physical examination")
            actions.append("Basic labs as indicated")
        
        return list(dict.fromkeys(actions))  # Remove duplicates


# Convenience function for tool integration
async def generate_differential_diagnosis(
    symptoms: str,
    history: str = "",
    age: Optional[int] = None,
    sex: Optional[str] = None
) -> str:
    """
    Generate differential diagnosis from symptom description.
    
    Tool function for LangGraph integration.
    
    Args:
        symptoms: Description of symptoms
        history: Patient history
        age: Patient age
        sex: Patient sex
        
    Returns:
        Formatted markdown differential diagnosis
    """
    # Parse symptoms from natural language
    symptom_list = [s.strip() for s in symptoms.split(",")]
    chief_complaint = symptom_list[0] if symptom_list else symptoms
    
    presentation = ClinicalPresentation(
        chief_complaint=chief_complaint,
        symptoms=symptom_list,
        age=age,
        sex=sex,
        medical_history=history.split(",") if history else [],
    )
    
    from core.dependencies import DIContainer
    container = DIContainer.get_instance()
    engine = DifferentialDiagnosisEngine(llm_gateway=container.llm_gateway)
    result = await engine.generate_differential(presentation)
    
    return result.to_markdown()
