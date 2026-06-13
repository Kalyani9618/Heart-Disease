"""
Tools Module for HeartGuard Medical System

Provides tool integrations for medical queries and function calling:

- Tool Registry: LLM function calling system
- OpenFDA Integration: Real-time drug safety, recalls, enforcement actions
- Medical Knowledge Tools: Entity extraction, concept linking
- Analysis Tools: Medical document analysis, clinical relevance scoring

Features:
- Tool registration and discovery
- Schema generation for LLM function calling
- Parameter validation
- Execution with error handling
- Safe calculator (AST-based, no eval)
- Structured error handling with recovery suggestions

Example:
    >>> from tools.openfda import DrugLabelQuerier, DrugEnforcementQuerier
    >>> 
    >>> # Query drug information
    >>> labels = DrugLabelQuerier()
    >>> drug_info = labels.find_drug_by_name("Warfarin")
    >>> 
    >>> # Check for recalls and safety issues
    >>> enforcement = DrugEnforcementQuerier()
    >>> recalls = enforcement.check_drug_safety("Warfarin")
"""

from .tool_registry import (
    Tool,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    get_tool_registry,
    register_tool,
    execute_tool,
)

# P0 & P1 components
from .safe_calculator import SafeCalculator, safe_evaluate
from .tool_errors import ToolError, ToolErrorHandler, create_tool_error

# MedGemma Integration Imports
from .fhir.fhir_client import get_fhir_client
from .fhir.fhir_agent_tool import get_fhir_tool
from .dicom.dicom_handler import DicomHandler
from .medical_coding.auto_coder import auto_code_clinical_note

# Create global registry
tool_registry = ToolRegistry()

# Export
__all__ = [
    # Core registry
    "Tool",
    "ToolParameter",
    "ToolRegistry",
    "ToolResult",
    "get_tool_registry",
    "register_tool",
    "execute_tool",
    "tool_registry",
    
    # Safe calculator
    "SafeCalculator",
    "safe_evaluate",
    
    # Error handling
    "ToolError",
    "ToolErrorHandler",
    "create_tool_error",
    
    # MedGemma Integration
    "get_fhir_client",
    "get_fhir_tool",
    "DicomHandler",
    "auto_code_clinical_note",
]