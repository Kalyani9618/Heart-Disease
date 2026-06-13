from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date

class RecallStatus:
    """Constants for recall status"""
    ONGOING = "Ongoing"
    COMPLETED = "Completed"
    TERMINATED = "Terminated"


class DrugRecall(BaseModel):
    """Model representing a drug recall event"""
    recall_number: str
    reason_for_recall: str
    status: str
    distribution_pattern: Optional[str] = None
    product_quantity: Optional[str] = None
    recall_initiation_date: Optional[str] = None
    state: Optional[str] = None
    product_type: Optional[str] = None
    event_id: Optional[str] = None
    product_description: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    recalling_firm: Optional[str] = None
    classification: Optional[str] = None
    code_info: Optional[str] = None
    initial_firm_notification: Optional[str] = None

class DrugLabel(BaseModel):
    """Model representing a drug label (package insert)"""
    id: Optional[str] = None
    set_id: Optional[str] = None
    version: Optional[str] = None
    effective_time: Optional[str] = None
    brand_name: Optional[List[str]] = None
    generic_name: Optional[List[str]] = None
    manufacturer_name: Optional[List[str]] = None
    product_type: Optional[List[str]] = None
    route: Optional[List[str]] = None
    substance_name: Optional[List[str]] = None
    
    # Clinical sections
    warnings: Optional[List[str]] = None
    adverse_reactions: Optional[List[str]] = None
    drug_interactions: Optional[List[str]] = None
    contraindications: Optional[List[str]] = None
    boxed_warning: Optional[List[str]] = None
    indications_and_usage: Optional[List[str]] = None
    dosage_and_administration: Optional[List[str]] = None
    pregnancy: Optional[List[str]] = None
    nursing_mothers: Optional[List[str]] = None
    pediatric_use: Optional[List[str]] = None
    geriatric_use: Optional[List[str]] = None

class SideEffect(BaseModel):
    """Model representing a reported side effect/adverse reaction"""
    term: str
    count: int
    meddra_version: Optional[str] = None
    
class SeverityStats(BaseModel):
    """Model for severity statistics (deaths, hospitalizations)"""
    total_reports: int
    death_reports: int
    hospitalization_reports: int
    life_threatening_reports: int
    disability_reports: int
    other_serious_reports: int

class FoodRecall(BaseModel):
    """Model representing a food recall event"""
    recall_number: str
    reason_for_recall: str
    status: str
    classification: Optional[str] = None
    distribution_pattern: Optional[str] = None
    product_description: Optional[str] = None
    product_quantity: Optional[str] = None
    recall_initiation_date: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    recalling_firm: Optional[str] = None
    code_info: Optional[str] = None

class FoodEvent(BaseModel):
    """Model representing a food/supplement adverse event (CAERS)"""
    report_number: str
    date_created: Optional[str] = None
    date_started: Optional[str] = None
    product_name: Optional[str] = None
    product_role: Optional[str] = None
    industry_name: Optional[str] = None
    consumer_age: Optional[str] = None
    consumer_gender: Optional[str] = None
    reactions: List[str] = Field(default_factory=list)
    outcomes: List[str] = Field(default_factory=list)

class FDAResult(BaseModel):
    """Standardized result wrapper for OpenFDA API responses"""
    meta: Dict[str, Any] = Field(default_factory=dict)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
