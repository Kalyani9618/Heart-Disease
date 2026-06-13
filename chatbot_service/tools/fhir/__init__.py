"""
FHIR Integration Tools

Provides FHIR R4 client and LangGraph-compatible agent tool
for querying Electronic Health Record (EHR) systems.
"""

from .fhir_client import FHIRClient, FHIRConfig, PatientSummary, get_fhir_client
from .fhir_agent_tool import FHIRPatientQueryTool, get_fhir_tool

__all__ = [
    "FHIRClient",
    "FHIRConfig",
    "PatientSummary",
    "get_fhir_client",
    "FHIRPatientQueryTool",
    "get_fhir_tool",
]
