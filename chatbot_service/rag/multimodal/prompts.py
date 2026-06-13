"""
Medical-Specific Prompt Templates for Multimodal Processing

CONSOLIDATED VERSION: All prompts are now centrally managed via PromptRegistry.
This module provides a convenient interface to access multimodal prompts.

Usage:
    from rag.multimodal.prompts import get_medical_prompt
    
    lab_prompt = get_medical_prompt("lab_results_table")
    vital_prompt = get_medical_prompt("vital_signs_table")
"""


from typing import Dict, Any
from core.prompts.registry import get_prompt

# Mapping of legacy prompt names to registry keys
PROMPT_MAPPING = {
    "TABLE_ANALYSIS_SYSTEM": ("multimodal", "table_analysis_system"),
    "LAB_RESULTS_TABLE": ("multimodal", "lab_results_table"),
    "VITAL_SIGNS_TABLE": ("multimodal", "vital_signs_table"),
    "MEDICATION_TABLE": ("multimodal", "medication_table"),
    "IMAGE_ANALYSIS_SYSTEM": ("multimodal", "image_analysis_system"),
    "ECG_ANALYSIS": ("multimodal", "ecg_analysis"),
    "MEDICAL_CHART_ANALYSIS": ("multimodal", "generic_table"),
    "PRESCRIPTION_IMAGE": ("multimodal", "generic_image"),
    "GENERIC_TABLE": ("multimodal", "generic_table"),
    "GENERIC_IMAGE": ("multimodal", "generic_image"),
    "EXTRACT_ENTITIES": ("multimodal", "extract_entities"),
    "MULTIMODAL_QUERY": ("multimodal", "query"),
}


def get_medical_prompt(prompt_name: str) -> str:
    """
    Get a medical prompt by name from the centralized PromptRegistry.
    
    Args:
        prompt_name: Name of the prompt (e.g., "LAB_RESULTS_TABLE")
        
    Returns:
        The prompt string from PromptRegistry
        
    Raises:
        KeyError: If prompt_name is not found in mapping
    """
    if prompt_name not in PROMPT_MAPPING:
        raise KeyError(
            f"Unknown prompt: {prompt_name}. "
            f"Available prompts: {', '.join(PROMPT_MAPPING.keys())}"
        )
    
    category, name = PROMPT_MAPPING[prompt_name]
    return get_prompt(category, name)


def get_all_medical_prompts() -> Dict[str, str]:
    """
    Get all medical prompts from the registry.
    
    Returns:
        Dictionary mapping prompt names to their contents
    """
    result = {}
    for prompt_name, (category, registry_name) in PROMPT_MAPPING.items():
        try:
            result[prompt_name] = get_prompt(category, registry_name)
        except KeyError as e:
            print(f"Warning: Could not load prompt {prompt_name}: {e}")
    return result


# Legacy MEDICAL_PROMPTS dictionary - now dynamically populated from registry
# This maintains backward compatibility with existing code
MEDICAL_PROMPTS: Dict[str, Any] = get_all_medical_prompts()

# =============================================================================
# OLD PROMPTS - NOW CENTRALIZED IN PROMPTREGISTRY
# =============================================================================
# The above prompts (LAB_RESULTS_TABLE, VITAL_SIGNS_TABLE, etc.) are now stored
# in core/prompts/system_prompts.py and accessed through PromptRegistry.
# The MEDICAL_PROMPTS dictionary above is dynamically populated from the registry.
