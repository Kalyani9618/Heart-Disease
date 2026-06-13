"""
FHIR Agent Tool - LangGraph Integration

Provides:
- Tool wrapper for FHIR Client
- Exposes patient data retrieval to the agent
"""


import logging
from typing import Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from tools.fhir.fhir_client import get_fhir_client

logger = logging.getLogger(__name__)

class FHIRPatientQueryInput(BaseModel):
    patient_id: str = Field(description="The ID of the patient to retrieve data for")

class FHIRPatientQueryTool(BaseTool):
    name: str = "fhir_patient_query"
    description: str = "Retrieves comprehensive patient data (conditions, medications, allergies, vitals) from the EHR system using FHIR."
    args_schema: Type[BaseModel] = FHIRPatientQueryInput

    def _run(self, patient_id: str) -> str:
        """Synchronous wrapper for async execution (not supported directly in all chains)."""
        # In a real async environment, we would use _arun. 
        # For now, we'll block or assume the agent uses ainvoke.
        raise NotImplementedError("Use _arun for async FHIR queries")

    async def _arun(self, patient_id: str) -> str:
        """Execute the tool asynchronously."""
        client = get_fhir_client()
        if not client:
            return "FHIR Client not configured. Cannot retrieve patient data."

        try:
            summary = await client.get_patient_summary(patient_id)
            if summary:
                return summary.to_context_string()
            else:
                return f"No data found for patient ID: {patient_id}"
        except Exception as e:
            logger.error(f"FHIR tool error: {e}")
            return f"Error retrieving patient data: {str(e)}"

def get_fhir_tool() -> BaseTool:
    """Get the FHIR tool instance."""
    return FHIRPatientQueryTool()
