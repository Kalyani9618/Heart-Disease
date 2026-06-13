"""
OpenFDA Integration Module for HeartGuard Medical System
"""

from tools.openfda.models import (
    DrugLabel,
    DrugRecall,
    FoodRecall,
    FoodEvent,
    SideEffect,
    SeverityStats,
    RecallStatus,
    FDAResult
)

from tools.openfda.api_client import OpenFDAClient

from tools.openfda.drug_labels import DrugLabelQuerier

from tools.openfda.drug_enforcement import (
    DrugEnforcementQuerier,
)

from tools.openfda.drug_adverse_events import (
    DrugAdverseEventService,
    get_drug_side_effects,
    check_drug_reaction,
    check_drug_severity,
)

from tools.openfda.food_enforcement import (
    FoodEnforcementQuerier,
    check_food_recalls,
    check_allergen_recalls,
)

from tools.openfda.food_events import (
    FoodEventService,
    check_supplement_events,
    check_food_events,
)

from tools.openfda.openfda_safety_service import (
    OpenFDASafetyService,
    get_safety_service,
    reset_safety_service,
)

__all__ = [
    # Models
    "DrugLabel",
    "DrugRecall",
    "FoodRecall",
    "FoodEvent",
    "SideEffect",
    "SeverityStats",
    "RecallStatus",
    "FDAResult",
    # API Client
    "OpenFDAClient",
    # Drug Labels
    "DrugLabelQuerier",
    # Drug Enforcement
    "DrugEnforcementQuerier",
    # Drug Adverse Events
    "DrugAdverseEventService",
    "get_drug_side_effects",
    "check_drug_reaction",
    "check_drug_severity",
    # Food Enforcement
    "FoodEnforcementQuerier",
    "check_food_recalls",
    "check_allergen_recalls",
    # Food Events
    "FoodEventService",
    "check_supplement_events",
    "check_food_events",
    # Unified Safety Service
    "OpenFDASafetyService",
    "get_safety_service",
    "reset_safety_service",
]
