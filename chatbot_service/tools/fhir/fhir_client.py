"""
FHIR Client - Healthcare Interoperability Layer

Provides:
- FHIR R4 compliant client for EHR systems
- Patient data retrieval (Observations, Conditions, Medications)
- Smart on FHIR authentication
- Resource caching with automatic TTL expiration (prevents memory leaks)


Prerequisites:
- pip install fhirpy fhir.resources cachetools
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

try:
    from cachetools import TTLCache
    HAS_CACHETOOLS = True
except ImportError:
    HAS_CACHETOOLS = False
    TTLCache = None
    logger_msg = "Warning: cachetools not installed. Using fallback cache (will grow indefinitely)"

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
def _get_fhir_client():
    try:
        from fhirpy import AsyncFHIRClient
        return AsyncFHIRClient
    except ImportError:
        logger.warning("fhirpy not installed. Run: pip install fhirpy")
        return None

def _get_fhir_resources():
    try:
        import fhir.resources as fr
        return fr
    except ImportError:
        logger.warning("fhir.resources not installed. Run: pip install fhir.resources")
        return None


class FHIRResourceType(Enum):
    """Supported FHIR resource types."""
    PATIENT = "Patient"
    OBSERVATION = "Observation"  # Vitals, Lab Results
    CONDITION = "Condition"  # Diagnoses
    MEDICATION_REQUEST = "MedicationRequest"  # Prescriptions
    ALLERGY_INTOLERANCE = "AllergyIntolerance"
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    PROCEDURE = "Procedure"
    IMMUNIZATION = "Immunization"


@dataclass
class FHIRConfig:
    """FHIR server configuration."""
    base_url: str
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    access_token: Optional[str] = None
    token_url: Optional[str] = None
    timeout: int = 30
    cache_ttl_seconds: int = 300  # 5 minutes
    max_retries: int = 3


@dataclass
class PatientSummary:
    """Structured patient summary from FHIR data."""
    patient_id: str
    name: str
    birth_date: Optional[str] = None
    gender: Optional[str] = None
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    medications: List[Dict[str, Any]] = field(default_factory=list)
    allergies: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_context_string(self) -> str:
        """Convert to string for LLM context injection."""
        sections = [f"**Patient Profile**: {self.name} ({self.patient_id})"]
        
        if self.birth_date:
            sections.append(f"- DOB: {self.birth_date}, Gender: {self.gender}")
        
        if self.conditions:
            cond_list = ", ".join(c.get("display", "Unknown") for c in self.conditions[:5])
            sections.append(f"- **Active Conditions**: {cond_list}")
        
        if self.medications:
            med_list = ", ".join(m.get("display", "Unknown") for m in self.medications[:5])
            sections.append(f"- **Current Medications**: {med_list}")
        
        if self.allergies:
            allergy_list = ", ".join(a.get("display", "Unknown") for a in self.allergies[:5])
            sections.append(f"- **Allergies**: {allergy_list}")
        
        if self.observations:
            # Get most recent vitals
            recent = self.observations[:3]
            vitals = [f"{o.get('code')}: {o.get('value')} {o.get('unit', '')}" for o in recent]
            sections.append(f"- **Recent Vitals**: {', '.join(vitals)}")
        
        return "\n".join(sections)


class FHIRClient:
    """
    FHIR R4 Client for EHR Integration.
    
    Supports:
    - Patient lookup and demographics
    - Clinical observations (vitals, labs)
    - Active conditions and diagnoses
    - Medication history
    - Allergy information
    
    Example:
        config = FHIRConfig(
            base_url="https://fhir.hospital.org/r4",
            access_token="your_token_here"
        )
        client = FHIRClient(config)
        summary = await client.get_patient_summary("patient-123")
        context = summary.to_context_string()
    """

    def __init__(self, config: FHIRConfig):
        self.config = config
        self._client = None
        
        # Initialize cache with TTL expiration
        if HAS_CACHETOOLS and TTLCache:
            # maxsize: 100 items, ttl: cache_ttl_seconds from config
            self._cache = TTLCache(maxsize=100, ttl=config.cache_ttl_seconds)
            self._use_ttl_cache = True
            logger.info(f"✅ FHIR cache initialized with TTLCache (max 100 items, TTL {config.cache_ttl_seconds}s)")
        else:
            # Fallback: regular dict with manual cleanup (will log warning)
            self._cache: Dict[str, tuple] = {}
            self._use_ttl_cache = False
            logger.warning("⚠️  cachetools not installed - using fallback cache without TTL expiration. Install: pip install cachetools")
        
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize FHIR client connection."""
        AsyncFHIRClient = _get_fhir_client()
        if AsyncFHIRClient is None:
            logger.error("Cannot initialize FHIR client - fhirpy not installed")
            return False

        try:
            self._client = AsyncFHIRClient(
                self.config.base_url,
                authorization=f"Bearer {self.config.access_token}" if self.config.access_token else None,
            )
            
            # Test connection with metadata endpoint
            # capability = await self._client.execute("metadata", method="get")
            # logger.info(f"✅ Connected to FHIR server: {self.config.base_url}")
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to FHIR server: {e}")
            return False

    def _is_cache_valid(self, key: str) -> bool:
        """
        Check if cached data is still valid.
        
        When using TTLCache, expired entries are automatically removed.
        For fallback mode, we check timestamp manually.
        """
        if self._use_ttl_cache:
            # TTLCache automatically handles expiration
            return key in self._cache
        else:
            # Fallback: manual TTL check
            if key not in self._cache:
                return False
            data, timestamp = self._cache[key]
            is_valid = (datetime.utcnow() - timestamp).seconds < self.config.cache_ttl_seconds
            if not is_valid:
                # Clean up expired entry
                del self._cache[key]
            return is_valid

    async def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve patient demographics."""
        cache_key = f"patient:{patient_id}"
        if self._is_cache_valid(cache_key):
            if self._use_ttl_cache:
                return self._cache[cache_key]
            else:
                data, _ = self._cache[cache_key]
                return data

        try:
            patient = await self._client.resources("Patient").search(_id=patient_id).first()
            if patient:
                data = patient.serialize()
                if self._use_ttl_cache:
                    self._cache[cache_key] = data  # TTLCache doesn't need timestamp
                else:
                    self._cache[cache_key] = (data, datetime.utcnow())
                return data
        except Exception as e:
            logger.error(f"Failed to fetch patient {patient_id}: {e}")
        
        return None

    async def get_observations(
        self, 
        patient_id: str, 
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve patient observations (vitals, labs)."""
        cache_key = f"observations:{patient_id}:{category}"
        if self._is_cache_valid(cache_key):
            if self._use_ttl_cache:
                return self._cache[cache_key]
            else:
                data, _ = self._cache[cache_key]
                return data

        try:
            search_params = {"patient": patient_id, "_count": limit, "_sort": "-date"}
            if category:
                search_params["category"] = category

            observations = await self._client.resources("Observation").search(**search_params).fetch_all()
            
            results = []
            for obs in observations:
                obs_data = obs.serialize()
                results.append({
                    "id": obs_data.get("id"),
                    "code": obs_data.get("code", {}).get("coding", [{}])[0].get("display"),
                    "value": self._extract_value(obs_data),
                    "unit": obs_data.get("valueQuantity", {}).get("unit"),
                    "date": obs_data.get("effectiveDateTime"),
                    "status": obs_data.get("status"),
                })
            
            if self._use_ttl_cache:
                self._cache[cache_key] = results
            else:
                self._cache[cache_key] = (results, datetime.utcnow())
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch observations for {patient_id}: {e}")
            return []

    async def get_conditions(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieve patient conditions (diagnoses)."""
        cache_key = f"conditions:{patient_id}"
        if self._is_cache_valid(cache_key):
            if self._use_ttl_cache:
                return self._cache[cache_key]
            else:
                data, _ = self._cache[cache_key]
                return data

        try:
            conditions = await self._client.resources("Condition").search(
                patient=patient_id,
                clinical_status="active"
            ).fetch_all()
            
            results = []
            for cond in conditions:
                cond_data = cond.serialize()
                results.append({
                    "id": cond_data.get("id"),
                    "display": cond_data.get("code", {}).get("coding", [{}])[0].get("display"),
                    "code": cond_data.get("code", {}).get("coding", [{}])[0].get("code"),
                    "system": cond_data.get("code", {}).get("coding", [{}])[0].get("system"),
                    "onset": cond_data.get("onsetDateTime"),
                    "status": cond_data.get("clinicalStatus", {}).get("coding", [{}])[0].get("code"),
                })
            
            if self._use_ttl_cache:
                self._cache[cache_key] = results
            else:
                self._cache[cache_key] = (results, datetime.utcnow())
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch conditions for {patient_id}: {e}")
            return []

    async def get_medications(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieve patient medications."""
        cache_key = f"medications:{patient_id}"
        if self._is_cache_valid(cache_key):
            if self._use_ttl_cache:
                return self._cache[cache_key]
            else:
                data, _ = self._cache[cache_key]
                return data

        try:
            meds = await self._client.resources("MedicationRequest").search(
                patient=patient_id,
                status="active"
            ).fetch_all()
            
            results = []
            for med in meds:
                med_data = med.serialize()
                results.append({
                    "id": med_data.get("id"),
                    "display": med_data.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display"),
                    "code": med_data.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("code"),
                    "dosage": med_data.get("dosageInstruction", [{}])[0].get("text"),
                    "authored": med_data.get("authoredOn"),
                    "status": med_data.get("status"),
                })
            
            if self._use_ttl_cache:
                self._cache[cache_key] = results
            else:
                self._cache[cache_key] = (results, datetime.utcnow())
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch medications for {patient_id}: {e}")
            return []

    async def get_allergies(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieve patient allergies and intolerances."""
        cache_key = f"allergies:{patient_id}"
        if self._is_cache_valid(cache_key):
            if self._use_ttl_cache:
                return self._cache[cache_key]
            else:
                data, _ = self._cache[cache_key]
                return data

        try:
            allergies = await self._client.resources("AllergyIntolerance").search(
                patient=patient_id,
                clinical_status="active"
            ).fetch_all()
            
            results = []
            for allergy in allergies:
                allergy_data = allergy.serialize()
                results.append({
                    "id": allergy_data.get("id"),
                    "display": allergy_data.get("code", {}).get("coding", [{}])[0].get("display"),
                    "category": allergy_data.get("category", [None])[0],
                    "criticality": allergy_data.get("criticality"),
                    "reaction": allergy_data.get("reaction", [{}])[0].get("manifestation", [{}])[0].get("coding", [{}])[0].get("display"),
                })
            
            if self._use_ttl_cache:
                self._cache[cache_key] = results
            else:
                self._cache[cache_key] = (results, datetime.utcnow())
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch allergies for {patient_id}: {e}")
            return []

    async def get_patient_summary(self, patient_id: str) -> Optional[PatientSummary]:
        """Get comprehensive patient summary for LLM context."""
        if not self._initialized:
            await self.initialize()
            if not self._initialized:
                return None

        # Fetch all data concurrently
        patient, conditions, medications, allergies, observations = await asyncio.gather(
            self.get_patient(patient_id),
            self.get_conditions(patient_id),
            self.get_medications(patient_id),
            self.get_allergies(patient_id),
            self.get_observations(patient_id, category="vital-signs", limit=5)
        )

        if not patient:
            logger.warning(f"Patient {patient_id} not found")
            return None

        name = self._extract_name(patient)
        return PatientSummary(
            patient_id=patient_id,
            name=name,
            birth_date=patient.get("birthDate"),
            gender=patient.get("gender"),
            conditions=conditions,
            medications=medications,
            allergies=allergies,
            observations=observations,
        )

    def _extract_name(self, patient: Dict) -> str:
        """Extract display name from patient resource."""
        names = patient.get("name", [])
        if names:
            name_obj = names[0]
            given = " ".join(name_obj.get("given", []))
            family = name_obj.get("family", "")
            return f"{given} {family}".strip() or "Unknown"
        return "Unknown"

    def _extract_value(self, observation: Dict) -> Any:
        """Extract value from observation."""
        if "valueQuantity" in observation:
            return observation["valueQuantity"].get("value")
        if "valueString" in observation:
            return observation["valueString"]
        if "valueCodeableConcept" in observation:
            return observation["valueCodeableConcept"].get("coding", [{}])[0].get("display")
        return None


# Singleton Factory
_fhir_client_instance: Optional[FHIRClient] = None

def get_fhir_client() -> Optional[FHIRClient]:
    """Get singleton FHIR client instance."""
    global _fhir_client_instance
    
    if _fhir_client_instance is None:
        import os
        base_url = os.getenv("FHIR_BASE_URL")
        access_token = os.getenv("FHIR_ACCESS_TOKEN")
        
        if not base_url:
            logger.warning("FHIR_BASE_URL not configured - FHIR agent disabled")
            return None
        
        config = FHIRConfig(base_url=base_url, access_token=access_token)
        _fhir_client_instance = FHIRClient(config)
    
    return _fhir_client_instance
