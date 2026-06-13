"""
Triage Support System - ESI Implementation
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)



class ESILevel(Enum):
    ESI_1 = 1  # Immediate: Life-saving intervention required
    ESI_2 = 2  # Emergent: High risk, confusion, severe pain
    ESI_3 = 3  # Urgent: 2+ resources needed
    ESI_4 = 4  # Less Urgent: 1 resource needed
    ESI_5 = 5  # Non-Urgent: No resources needed


class TriageCategory(Enum):
    IMMEDIATE = "immediate"
    EMERGENT = "emergent"
    URGENT = "urgent"
    LESS_URGENT = "less_urgent"
    NON_URGENT = "non_urgent"


@dataclass
class VitalSigns:
    heart_rate: Optional[int] = None
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None
    respiratory_rate: Optional[int] = None
    spo2: Optional[int] = None
    temperature: Optional[float] = None
    pain_score: Optional[int] = None  # 0-10
    gcs: Optional[int] = None  # Glasgow Coma Scale

    def has_critical_values(self) -> bool:
        """Check for critical vital signs (Adult)."""
        if self.heart_rate and (self.heart_rate > 120 or self.heart_rate < 50): return True
        if self.systolic_bp and (self.systolic_bp < 90 or self.systolic_bp >= 200): return True
        if self.spo2 and self.spo2 < 90: return True
        if self.respiratory_rate and (self.respiratory_rate > 30 or self.respiratory_rate < 10): return True
        if self.temperature and (self.temperature >= 40.0 or self.temperature <= 34.0): return True
        return False


@dataclass
class TriageAssessment:
    patient_id: str
    chief_complaint: str
    esi_level: ESILevel
    triage_category: TriageCategory
    priority_score: int  # 0-100 calculated score
    wait_time_guidance: str
    immediate_actions: List[str]
    resource_requirements: List[str]
    red_flags: List[str]
    reassessment_interval: str
    reasoning: List[str]

    def to_markdown(self) -> str:
        color_map = {
            ESILevel.ESI_1: "🔴",
            ESILevel.ESI_2: "🟠",
            ESILevel.ESI_3: "🟡",
            ESILevel.ESI_4: "🟢",
            ESILevel.ESI_5: "🔵",
        }
        icon = color_map.get(self.esi_level, "⚪")
        
        return f"""
# {icon} Triage Assessment: ESI Level {self.esi_level.value}

**Category**: {self.triage_category.value.upper()}
**Chief Complaint**: {self.chief_complaint}

## ⚡ Immediate Actions
{chr(10).join(f"- {action}" for action in self.immediate_actions)}

## 🚩 Red Flags
{chr(10).join(f"- {flag}" for flag in self.red_flags) if self.red_flags else "- None identified"}

## 🏥 Resource Requirements
{chr(10).join(f"- {res}" for res in self.resource_requirements) if self.resource_requirements else "- None predicted"}

---
*Guidance*: {self.wait_time_guidance}
*Reassess*: {self.reassessment_interval}
"""


class TriageSystem:
    """
    Emergency Severity Index (ESI) Implementation.
    """
    
    IMMEDIATE_KEYWORDS = ["arrest", "unresponsive", "apneic", "pulseless", "intubated", "overdose"]
    HIGH_RISK_KEYWORDS = ["chest pain", "stroke", "confusion", "lethargy", "disoriented", "severe pain", "suicidal", "worst headache"]

    async def assess(
        self,
        chief_complaint: str,
        patient_id: str = "anonymous",
        symptoms: List[str] = None,
        vitals: VitalSigns = None,
        age: Optional[int] = None,
        history: List[str] = None
    ) -> TriageAssessment:
        """
        Perform ESI Triage Assessment.
        """
        symptoms = symptoms or []
        history = history or []
        reasoning = []
        resources = []
        
        # Step 1: ESI-1 (Immediate Life Saving Intervention)
        if self._check_immediate_threat(chief_complaint, symptoms, vitals):
            reasoning.append("Patient requires immediate life-saving intervention (ESI-1)")
            return self._create_immediate_assessment(chief_complaint, vitals, patient_id, reasoning)

        # Step 2: ESI-2 (High Risk / Confusion / Severe Pain / Distress)
        high_risk_factors = self._check_high_risk(chief_complaint, symptoms, vitals, age)
        if high_risk_factors:
            reasoning.append(f"High risk factors identified: {', '.join(high_risk_factors)} (ESI-2)")
            esi_level = ESILevel.ESI_2
        else:
            # Step 3: Resource Prediction
            resources = self._estimate_resources(chief_complaint, symptoms, history)
            reasoning.append(f"Predicted resources: {len(resources)} ({', '.join(resources)})")
            
            esi_level = self._calculate_esi(high_risk_factors, resources, vitals)

        # Finalize Assessment
        triage_category = self._esi_to_category(esi_level)
        red_flags = self._identify_red_flags(chief_complaint, symptoms, vitals)
        immediate_actions = self._generate_actions(esi_level, chief_complaint, vitals)
        priority_score = self._calculate_priority_score(esi_level, red_flags, vitals)
        
        return TriageAssessment(
            patient_id=patient_id,
            chief_complaint=chief_complaint,
            esi_level=esi_level,
            triage_category=triage_category,
            priority_score=priority_score,
            wait_time_guidance=self._get_wait_guidance(esi_level),
            immediate_actions=immediate_actions,
            resource_requirements=resources if esi_level.value > 2 else ["Full workup"],
            red_flags=red_flags,
            reassessment_interval=self._get_reassess_interval(esi_level),
            reasoning=reasoning,
        )

    def _check_immediate_threat(
        self,
        complaint: str,
        symptoms: List[str],
        vitals: VitalSigns
    ) -> bool:
        """Check for ESI-1 (immediate) conditions."""
        all_text = f"{complaint} {' '.join(symptoms)}".lower()
        
        for keyword in self.IMMEDIATE_KEYWORDS:
            if keyword in all_text:
                return True
        
        if vitals:
            if vitals.gcs and vitals.gcs <= 8:
                return True
            if vitals.systolic_bp and vitals.systolic_bp < 70:
                return True
            if vitals.spo2 and vitals.spo2 < 85:
                return True
        
        return False

    def _check_high_risk(
        self,
        complaint: str,
        symptoms: List[str],
        vitals: VitalSigns,
        age: Optional[int]
    ) -> List[str]:
        """Identify high-risk factors for ESI-2."""
        risk_factors = []
        all_text = f"{complaint} {' '.join(symptoms)}".lower()
        
        for keyword in self.HIGH_RISK_KEYWORDS:
            if keyword in all_text:
                risk_factors.append(f"Complaint: {keyword}")
        
        if vitals and vitals.has_critical_values():
            risk_factors.append("Abnormal vital signs")
        
        if age and age > 65:
            # Higher suspicion for elderly with certain complaints
            if any(kw in all_text for kw in ["chest", "breath", "fall", "weak"]):
                risk_factors.append("Elderly with concerning symptoms")
        
        return risk_factors

    def _estimate_resources(
        self,
        complaint: str,
        symptoms: List[str],
        history: List[str]
    ) -> List[str]:
        """Estimate resources needed for workup."""
        resources = []
        all_text = f"{complaint} {' '.join(symptoms)}".lower()
        
        # Lab resources
        if any(kw in all_text for kw in ["chest", "heart", "breath", "pain"]):
            resources.append("Labs")
            resources.append("ECG")
        
        # Imaging
        if any(kw in all_text for kw in ["trauma", "fall", "accident", "fracture"]):
            resources.append("X-ray")
        
        if any(kw in all_text for kw in ["head", "stroke", "neuro"]):
            resources.append("CT scan")
        
        # IV access
        if any(kw in all_text for kw in ["dehydration", "vomiting", "bleeding", "sepsis"]):
            resources.append("IV fluids")
        
        return list(set(resources))

    def _calculate_esi(
        self,
        high_risk: List[str],
        resources: List[str],
        vitals: VitalSigns
    ) -> ESILevel:
        """Calculate ESI level based on factors."""
        if high_risk and (vitals and vitals.has_critical_values()):
            return ESILevel.ESI_2
        
        if high_risk:
            return ESILevel.ESI_2
        
        num_resources = len(resources)
        if num_resources >= 2:
            return ESILevel.ESI_3
        elif num_resources == 1:
            return ESILevel.ESI_4
        else:
            return ESILevel.ESI_5

    def _identify_red_flags(
        self,
        complaint: str,
        symptoms: List[str],
        vitals: VitalSigns
    ) -> List[str]:
        """Identify clinical red flags."""
        flags = []
        all_text = f"{complaint} {' '.join(symptoms)}".lower()
        
        red_flag_terms = {
            "crushing chest pain": "Possible ACS",
            "worst headache": "Possible SAH",
            "sudden weakness": "Possible stroke",
            "difficulty swallowing": "Airway concern",
            "blood in stool": "GI bleeding",
            "blood in vomit": "Upper GI bleeding",
        }
        
        for term, flag in red_flag_terms.items():
            if term in all_text:
                flags.append(flag)
        
        if vitals:
            if vitals.systolic_bp and vitals.systolic_bp < 90:
                flags.append("Hypotension")
            if vitals.spo2 and vitals.spo2 < 92:
                flags.append("Hypoxia")
            if vitals.heart_rate and vitals.heart_rate > 120:
                flags.append("Significant tachycardia")
        
        return flags

    def _create_immediate_assessment(
        self,
        complaint: str,
        vitals: VitalSigns,
        patient_id: str,
        reasoning: List[str]
    ) -> TriageAssessment:
        """Create ESI-1 immediate assessment."""
        return TriageAssessment(
            patient_id=patient_id,
            chief_complaint=complaint,
            esi_level=ESILevel.ESI_1,
            triage_category=TriageCategory.IMMEDIATE,
            priority_score=100,
            wait_time_guidance="IMMEDIATE - No wait",
            immediate_actions=[
                "Activate resuscitation team",
                "Bring to resuscitation bay",
                "Continuous monitoring",
                "Establish IV access",
                "Prepare for airway management",
            ],
            resource_requirements=["Resuscitation team", "Crash cart", "Continuous monitoring"],
            red_flags=["IMMEDIATE LIFE THREAT"],
            reassessment_interval="Continuous",
            reasoning=reasoning + ["Patient requires immediate life-saving intervention"],
        )

    def _generate_actions(
        self,
        esi: ESILevel,
        complaint: str,
        vitals: VitalSigns
    ) -> List[str]:
        """Generate immediate action list."""
        actions = []
        
        if esi.value <= 2:
            actions.append("Physician assessment immediately")
            actions.append("Place on cardiac monitor")
            actions.append("Establish IV access")
        elif esi.value == 3:
            actions.append("Physician assessment within 30 minutes")
            actions.append("Initiate workup per protocol")
        else:
            actions.append("Registration and vitals")
            actions.append("Assessment when available")
        
        # Complaint-specific actions
        if "chest" in complaint.lower():
            actions.insert(0, "12-lead ECG within 10 minutes")
        
        return actions

    def _esi_to_category(self, esi: ESILevel) -> TriageCategory:
        """Map ESI to triage category."""
        mapping = {
            ESILevel.ESI_1: TriageCategory.IMMEDIATE,
            ESILevel.ESI_2: TriageCategory.EMERGENT,
            ESILevel.ESI_3: TriageCategory.URGENT,
            ESILevel.ESI_4: TriageCategory.LESS_URGENT,
            ESILevel.ESI_5: TriageCategory.NON_URGENT,
        }
        return mapping.get(esi, TriageCategory.URGENT)

    def _calculate_priority_score(
        self,
        esi: ESILevel,
        red_flags: List[str],
        vitals: VitalSigns
    ) -> int:
        """Calculate 1-100 priority score."""
        base_scores = {
            ESILevel.ESI_1: 100,
            ESILevel.ESI_2: 80,
            ESILevel.ESI_3: 60,
            ESILevel.ESI_4: 40,
            ESILevel.ESI_5: 20,
        }
        
        score = base_scores.get(esi, 50)
        score += len(red_flags) * 5
        
        if vitals and vitals.has_critical_values():
            score += 10
        
        return min(score, 100)

    def _get_wait_guidance(self, esi: ESILevel) -> str:
        """Get wait time guidance by ESI level."""
        guidance = {
            ESILevel.ESI_1: "IMMEDIATE - No wait",
            ESILevel.ESI_2: "EMERGENT - Within 10 minutes",
            ESILevel.ESI_3: "URGENT - Within 30 minutes",
            ESILevel.ESI_4: "LESS URGENT - Within 1-2 hours",
            ESILevel.ESI_5: "NON-URGENT - Within 2-4 hours",
        }
        return guidance.get(esi, "Standard wait times apply")

    def _get_reassess_interval(self, esi: ESILevel) -> str:
        """Get reassessment interval by ESI level."""
        intervals = {
            ESILevel.ESI_1: "Continuous",
            ESILevel.ESI_2: "Every 15 minutes",
            ESILevel.ESI_3: "Every 30 minutes",
            ESILevel.ESI_4: "Every 60 minutes",
            ESILevel.ESI_5: "Every 2 hours or PRN",
        }
        return intervals.get(esi, "As needed")


# Convenience function for tool integration
async def triage_patient(
    symptoms: str,
    heart_rate: Optional[int] = None,
    blood_pressure: Optional[str] = None,
    spo2: Optional[int] = None,
    age: Optional[int] = None,
) -> str:
    """
    Perform triage assessment on a patient.
    
    Args:
        symptoms: Description of symptoms/chief complaint
        heart_rate: Heart rate in BPM
        blood_pressure: Blood pressure as "systolic/diastolic"
        spo2: Oxygen saturation percentage
        age: Patient age
        
    Returns:
        Formatted markdown triage assessment
    """
    # Parse blood pressure
    systolic, diastolic = None, None
    if blood_pressure and "/" in blood_pressure:
        parts = blood_pressure.split("/")
        systolic = int(parts[0])
        diastolic = int(parts[1])
    
    vitals = VitalSigns(
        heart_rate=heart_rate,
        systolic_bp=systolic,
        diastolic_bp=diastolic,
        spo2=spo2,
    )
    
    triage = TriageSystem()
    assessment = await triage.assess(
        chief_complaint=symptoms,
        vitals=vitals,
        age=age,
    )
    
    return assessment.to_markdown()
