"""
Medical Auto-Coding - SNOMED-CT and LOINC Mapping

Provides:
- Clinical text to SNOMED-CT code mapping
- Lab observation to LOINC code mapping
- ICD-10 suggestion
- Billing code automation
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from core.prompts.registry import get_prompt

logger = logging.getLogger(__name__)


class CodeSystem(Enum):
    """Medical coding systems."""
    SNOMED_CT = "http://snomed.info/sct"
    LOINC = "http://loinc.org"
    ICD10 = "http://hl7.org/fhir/sid/icd-10"
    ICD10_CM = "http://hl7.org/fhir/sid/icd-10-cm"
    CPT = "http://www.ama-assn.org/go/cpt"
    RxNorm = "http://www.nlm.nih.gov/research/umls/rxnorm"


@dataclass
class MedicalCode:
    """A medical code with metadata."""
    code: str
    system: CodeSystem
    display: str
    confidence: float
    is_primary: bool = False
    alternative_codes: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class CodingResult:
    """Complete coding result for clinical text."""
    original_text: str
    codes: List[MedicalCode]
    billable_codes: List[MedicalCode]
    warnings: List[str]
    requires_review: bool


class MedicalAutoCoder:
    """
    Auto-Coder for Medical Terminology.
    
    Maps clinical text to standardized medical codes:
    - SNOMED-CT for clinical findings
    - LOINC for lab observations
    - ICD-10 for diagnoses
    - CPT for procedures
    """

    # Common SNOMED-CT codes for quick lookup
    COMMON_SNOMED = {
        "hypertension": ("38341003", "Hypertensive disorder"),
        "diabetes type 2": ("44054006", "Type 2 diabetes mellitus"),
        "diabetes type 1": ("46635009", "Type 1 diabetes mellitus"),
        "chest pain": ("29857009", "Chest pain"),
        "shortness of breath": ("267036007", "Dyspnea"),
        "headache": ("25064002", "Headache"),
        "fever": ("386661006", "Fever"),
        "cough": ("49727002", "Cough"),
        "nausea": ("422587007", "Nausea"),
        "fatigue": ("84229001", "Fatigue"),
        "heart failure": ("84114007", "Heart failure"),
        "atrial fibrillation": ("49436004", "Atrial fibrillation"),
        "pneumonia": ("233604007", "Pneumonia"),
        "copd": ("13645005", "Chronic obstructive lung disease"),
        "asthma": ("195967001", "Asthma"),
    }

    # Common LOINC codes for labs
    COMMON_LOINC = {
        "hemoglobin": ("718-7", "Hemoglobin [Mass/volume] in Blood"),
        "hba1c": ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood"),
        "glucose": ("2345-7", "Glucose [Mass/volume] in Serum or Plasma"),
        "creatinine": ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma"),
        "potassium": ("2823-3", "Potassium [Moles/volume] in Serum or Plasma"),
        "sodium": ("2951-2", "Sodium [Moles/volume] in Serum or Plasma"),
        "troponin": ("10839-9", "Troponin I.cardiac [Mass/volume] in Serum or Plasma"),
        "bnp": ("30934-4", "Natriuretic peptide B [Mass/volume] in Serum or Plasma"),
        "wbc": ("6690-2", "Leukocytes [#/volume] in Blood"),
        "platelet": ("777-3", "Platelets [#/volume] in Blood"),
        "inr": ("6301-6", "INR in Platelet poor plasma"),
        "d-dimer": ("48066-5", "D-dimer [Mass/volume] in Platelet poor plasma"),
    }

    # ICD-10 mappings
    COMMON_ICD10 = {
        "hypertension": ("I10", "Essential (primary) hypertension"),
        "type 2 diabetes": ("E11", "Type 2 diabetes mellitus"),
        "type 1 diabetes": ("E10", "Type 1 diabetes mellitus"),
        "heart failure": ("I50", "Heart failure"),
        "atrial fibrillation": ("I48", "Atrial fibrillation and flutter"),
        "copd": ("J44", "Other chronic obstructive pulmonary disease"),
        "pneumonia": ("J18", "Pneumonia, unspecified organism"),
        "chest pain": ("R07", "Pain in throat and chest"),
        "acute mi": ("I21", "Acute myocardial infarction"),
    }

    def __init__(self, llm_gateway=None, use_llm_enhancement: bool = True):
        self.llm_gateway = llm_gateway
        self.use_llm_enhancement = use_llm_enhancement and llm_gateway is not None

    async def code_clinical_text(
        self,
        text: str,
        code_systems: List[CodeSystem] = None,
    ) -> CodingResult:
        """
        Map clinical text to medical codes.
        """
        if code_systems is None:
            code_systems = [CodeSystem.SNOMED_CT, CodeSystem.ICD10_CM]
        
        codes = []
        warnings = []
        text_lower = text.lower()
        
        # Rule-based matching first
        for term, (code, display) in self.COMMON_SNOMED.items():
            if term in text_lower and CodeSystem.SNOMED_CT in code_systems:
                codes.append(MedicalCode(
                    code=code,
                    system=CodeSystem.SNOMED_CT,
                    display=display,
                    confidence=0.9,
                    is_primary=len(codes) == 0,
                ))
        
        for term, (code, display) in self.COMMON_ICD10.items():
            if term in text_lower and CodeSystem.ICD10_CM in code_systems:
                codes.append(MedicalCode(
                    code=code,
                    system=CodeSystem.ICD10_CM,
                    display=display,
                    confidence=0.85,
                    is_primary=False,
                ))
        
        # LLM enhancement for complex cases
        if self.use_llm_enhancement and not codes:
            llm_codes = await self._llm_code_matching(text, code_systems)
            codes.extend(llm_codes)
            if llm_codes:
                warnings.append("Codes generated via AI - manual verification recommended")
        
        # Identify billable codes
        billable = [c for c in codes if c.system in [CodeSystem.ICD10_CM, CodeSystem.CPT]]
        
        return CodingResult(
            original_text=text,
            codes=codes,
            billable_codes=billable,
            warnings=warnings,
            requires_review=len(codes) == 0 or any(c.confidence < 0.8 for c in codes),
        )

    async def code_lab_observation(
        self,
        lab_name: str,
        value: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> List[MedicalCode]:
        """Map lab observation to LOINC code."""
        codes = []
        lab_lower = lab_name.lower()
        
        for term, (code, display) in self.COMMON_LOINC.items():
            if term in lab_lower:
                codes.append(MedicalCode(
                    code=code,
                    system=CodeSystem.LOINC,
                    display=display,
                    confidence=0.9,
                    is_primary=True,
                ))
                break
        
        return codes

    async def _llm_code_matching(
        self,
        text: str,
        systems: List[CodeSystem]
    ) -> List[MedicalCode]:
        """Use LLM to match text to medical codes."""
        if not self.llm_gateway:
            return []
        
        systems_str = ", ".join(s.name for s in systems)
        
        # Get base prompt from centralized registry
        base_prompt = get_prompt("tools", "medical_coding_specialist")
        
        prompt = f"""{base_prompt}

TEXT: {text}

CODING SYSTEMS TO USE: {systems_str}

For each distinct clinical finding in the text, provide:
1. Code system (SNOMED-CT, ICD-10, LOINC)
2. Code value
3. Display name
4. Confidence (0-100%)

Format as:
SYSTEM|CODE|DISPLAY|CONFIDENCE

Only provide codes you are confident about. If unsure, state "UNCERTAIN".
"""

        try:
            response = await self.llm_gateway.generate(prompt)
            return self._parse_llm_codes(response)
        except Exception as e:
            logger.error(f"LLM code matching failed: {e}")
            return []

    def _parse_llm_codes(self, response: str) -> List[MedicalCode]:
        """Parse LLM response into codes."""
        codes = []
        
        for line in response.strip().split("\n"):
            if "|" in line and "UNCERTAIN" not in line:
                parts = line.split("|")
                if len(parts) >= 4:
                    try:
                        system_str = parts[0].strip().upper()
                        system_map = {
                            "SNOMED-CT": CodeSystem.SNOMED_CT,
                            "SNOMED": CodeSystem.SNOMED_CT,
                            "ICD-10": CodeSystem.ICD10_CM,
                            "ICD10": CodeSystem.ICD10_CM,
                            "LOINC": CodeSystem.LOINC,
                        }
                        system = system_map.get(system_str, CodeSystem.SNOMED_CT)
                        
                        confidence = float(parts[3].replace("%", "").strip()) / 100.0
                        
                        codes.append(MedicalCode(
                            code=parts[1].strip(),
                            system=system,
                            display=parts[2].strip(),
                            confidence=confidence,
                            is_primary=len(codes) == 0,
                        ))
                    except (ValueError, IndexError):
                        continue
        
        return codes

    def to_fhir_codings(self, codes: List[MedicalCode]) -> List[Dict[str, Any]]:
        """Convert codes to FHIR Coding format."""
        return [
            {
                "system": code.system.value,
                "code": code.code,
                "display": code.display,
            }
            for code in codes
        ]


# Convenience function for tool integration
async def auto_code_clinical_note(
    clinical_text: str,
    include_billing: bool = True,
) -> str:
    """
    Auto-code a clinical note.
    
    Args:
        clinical_text: Clinical text to code
        include_billing: Include ICD-10 billing codes
        
    Returns:
        Formatted coding result
    """
    coder = MedicalAutoCoder()
    
    systems = [CodeSystem.SNOMED_CT]
    if include_billing:
        systems.append(CodeSystem.ICD10_CM)
    
    result = await coder.code_clinical_text(clinical_text, systems)
    
    lines = ["## Medical Coding Results", "", f"**Original Text**: {result.original_text}", ""]
    
    if result.codes:
        lines.append("### Assigned Codes")
        for code in result.codes:
            lines.append(f"- **{code.system.name}**: `{code.code}` - {code.display} ({code.confidence*100:.0f}%)")
    else:
        lines.append("*No codes matched. Manual coding required.*")
    
    if result.billable_codes:
        lines.append("\n### Billable Codes")
        for code in result.billable_codes:
            lines.append(f"- `{code.code}`: {code.display}")
    
    if result.warnings:
        lines.append("\n### ⚠️ Warnings")
        for warn in result.warnings:
            lines.append(f"- {warn}")
    
    if result.requires_review:
        lines.append("\n*⚠️ Manual review recommended before finalizing codes.*")
    
    return "\n".join(lines)
