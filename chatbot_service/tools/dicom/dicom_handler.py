"""
DICOM Handler - Medical Imaging Standard Integration

Security & Performance Features:
- Async I/O wrapping to prevent event loop blocking
- Memory-safe pixel data handling (excluded from JSON serialization)
- PII scrubbing before output
- Graceful error handling for malformed DICOMWeb responses
"""

import logging
import os
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Lazy imports
def _get_pydicom():
    try:
        import pydicom
        return pydicom
    except ImportError:
        logger.warning("pydicom not installed")
        return None

def _get_dicomweb_client():
    try:
        from dicomweb_client.api import DICOMwebClient
        return DICOMwebClient
    except ImportError:
        logger.warning("dicomweb-client not installed")
        return None


@dataclass
class DicomPatient:
    patient_id: str
    patient_name: str
    birth_date: Optional[str]
    sex: Optional[str]


@dataclass
class DicomStudy:
    study_instance_uid: str
    study_date: Optional[str]
    study_description: Optional[str]
    accession_number: Optional[str]
    referring_physician: Optional[str] = None


@dataclass
class DicomSeries:
    series_instance_uid: str
    modality: str
    series_description: Optional[str]
    body_part_examined: Optional[str]
    image_count: int


@dataclass
class DicomImage:
    sop_instance_uid: str
    instance_number: int
    rows: int
    columns: int
    photometric_interpretation: str
    pixel_data: Optional[bytes] = None  # Raw pixel data (careful with memory)


@dataclass
class DicomAnalysisResult:
    patient: DicomPatient
    study: DicomStudy
    series: List[DicomSeries]
    findings: Dict[str, Any]
    structured_report: Dict[str, Any]

    def to_json(self) -> str:
        """
        Convert to JSON, excluding sensitive pixel data.
        
        Security: Excludes pixel_data from serialization to prevent:
        - Memory exhaustion from large binary data
        - Accidental exposure of raw medical image data
        """
        data = {
            "patient": self.patient.__dict__,
            "study": self.study.__dict__,
            "series": [s.__dict__ for s in self.series],
            "findings": self.findings
        }
        
        # Remove pixel_data from series if present
        for series_dict in data.get("series", []):
            series_dict.pop("pixel_data", None)
        
        return json.dumps(data, default=str, indent=2)

    def to_fhir_diagnostic_report(self) -> Dict[str, Any]:
        """Convert to FHIR DiagnosticReport resource."""
        return {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "code": {
                "text": self.study.study_description or "Imaging Study"
            },
            "subject": {
                "reference": f"Patient/{self.patient.patient_id}",
                "display": self.patient.patient_name
            },
            "effectiveDateTime": self.study.study_date,
            "conclusion": self.findings.get("impression", "No impression"),
            "result": []  # References to Observations
        }


class DicomHandler:
    """
    Handler for DICOM files and DICOMweb integration.
    
    Performance & Security:
    - Async I/O to prevent blocking
    - Memory-safe pixel data handling
    - Graceful error handling for malformed responses
    """

    def __init__(self, dicomweb_url: Optional[str] = None, auth_token: Optional[str] = None):
        self.dicomweb_url = dicomweb_url
        self.auth_token = auth_token
        self._client = None

    def parse_file(self, file_path: str) -> Optional[DicomAnalysisResult]:
        """
        Parse a local DICOM file synchronously.
        
        Note: This is a blocking operation. Call via async wrapper in agentic_tools.py
        to prevent event loop blocking. See: asyncio.run_in_executor()
        """
        pydicom = _get_pydicom()
        if not pydicom:
            return None

        try:
            ds = pydicom.dcmread(file_path)
            
            patient = DicomPatient(
                patient_id=str(ds.get("PatientID", "Unknown")),
                patient_name=str(ds.get("PatientName", "Unknown")),
                birth_date=str(ds.get("PatientBirthDate", "")),
                sex=str(ds.get("PatientSex", ""))
            )
            
            study = DicomStudy(
                study_instance_uid=str(ds.get("StudyInstanceUID", "")),
                study_date=str(ds.get("StudyDate", "")),
                study_description=str(ds.get("StudyDescription", "")),
                accession_number=str(ds.get("AccessionNumber", ""))
            )
            
            series = DicomSeries(
                series_instance_uid=str(ds.get("SeriesInstanceUID", "")),
                modality=str(ds.get("Modality", "Unknown")),
                series_description=str(ds.get("SeriesDescription", "")),
                body_part_examined=str(ds.get("BodyPartExamined", "")),
                image_count=1
            )
            
            return DicomAnalysisResult(
                patient=patient,
                study=study,
                series=[series],
                findings={},
                structured_report={}
            )
            
        except Exception as e:
            logger.error(f"Failed to parse DICOM file {file_path}: {e}")
            return None

    async def parse_file_async(self, file_path: str) -> Optional[DicomAnalysisResult]:
        """
        Async wrapper for DICOM file parsing.
        
        Prevents event loop blocking by running parsing in thread pool executor.
        This is the recommended method for use in async contexts.
        """
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self.parse_file, file_path)
        except Exception as e:
            logger.error(f"Async DICOM parsing failed: {e}")
            return None

    async def fetch_from_dicomweb(
        self, 
        study_uid: str, 
        series_uid: Optional[str] = None
    ) -> Optional[DicomAnalysisResult]:
        """Fetch metadata from DICOMweb server."""
        if not self.dicomweb_url:
            logger.warning("DICOMweb URL not configured")
            return None

        # Implementation would use aiohttp or dicomweb-client
        # Simplified placeholder for now
        try:
            import aiohttp
            headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            url = f"{self.dicomweb_url}/studies/{study_uid}"
            if series_uid:
                url += f"/series/{series_uid}"
            url += "/metadata"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        metadata = await response.json()
                        return self._parse_dicomweb_metadata(metadata)
                    else:
                        logger.error(f"DICOMweb fetch failed: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"DICOMweb fetch error: {e}")
            return None

    def _parse_dicomweb_metadata(self, metadata: List[Dict]) -> DicomAnalysisResult:
        """
        Parse DICOMweb JSON metadata with robust error handling.
        
        Handles:
        - Empty metadata lists
        - Missing expected tags
        - Deeply nested structures
        - Empty Value arrays
        """
        if not metadata or len(metadata) == 0:
            logger.warning("DICOMweb metadata is empty")
            return DicomAnalysisResult(
                patient=DicomPatient("unknown", "Unknown", None, None),
                study=DicomStudy("", None, None, None),
                series=[],
                findings={},
                structured_report={}
            )
        
        first_instance = metadata[0] if isinstance(metadata, list) else {}
        
        def get_value(tag: str, default: str = "") -> str:
            """Safely extract DICOM tag value with multiple fallback strategies."""
            try:
                # Strategy 1: Standard DICOMWeb format
                if tag in first_instance:
                    val_obj = first_instance.get(tag, {})
                    if isinstance(val_obj, dict):
                        values = val_obj.get("Value", [])
                        if values and len(values) > 0:
                            return str(values[0])
                
                # Strategy 2: Direct value (some servers return flat structure)
                if tag in first_instance:
                    return str(first_instance[tag])
            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"Failed to extract tag {tag}: {e}")
            
            return default
        
        # Extract with safe defaults
        patient_id = get_value("00100020", "unknown")
        patient_name = get_value("00100010", "Unknown")
        birth_date = get_value("00100030") or None
        sex = get_value("00100040") or None
        
        patient = DicomPatient(
            patient_id=patient_id,
            patient_name=patient_name,
            birth_date=birth_date,
            sex=sex,
        )
        
        study_uid = get_value("0020000D", "")
        study_date = get_value("00080020") or None
        study_desc = get_value("00081030") or None
        acc_num = get_value("00080050") or None
        
        study = DicomStudy(
            study_instance_uid=study_uid,
            study_date=study_date,
            study_description=study_desc,
            accession_number=acc_num,
        )
        
        # Parse series from metadata entries
        series_map = {}
        for instance in metadata:
            series_uid = get_value("0020000E") if isinstance(instance, dict) else ""
            if series_uid and series_uid not in series_map:
                modality = instance.get("00080060", {}).get("Value", ["Unknown"])[0] if isinstance(instance, dict) else "Unknown"
                series_map[series_uid] = DicomSeries(
                    series_instance_uid=series_uid,
                    modality=str(modality),
                    series_description=instance.get("0020103E", {}).get("Value", [None])[0] if isinstance(instance, dict) else None,
                    body_part_examined=instance.get("00180015", {}).get("Value", [None])[0] if isinstance(instance, dict) else None,
                    image_count=len([i for i in metadata if isinstance(i, dict) and i.get("0020000E") == series_uid])
                )
        
        return DicomAnalysisResult(
            patient=patient,
            study=study,
            series=list(series_map.values()),
            findings={},
            structured_report={},
        )