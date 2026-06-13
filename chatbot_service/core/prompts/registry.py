"""
PromptRegistry - Centralized Prompt Management System

Provides a unified interface for accessing, caching, and managing all system prompts
used throughout the Cardio AI application. This replaces hardcoded prompts scattered
across the codebase with a single source of truth.

Features:
- Lazy loading and caching
- Version tracking and history
- Environment-based overrides
- Audit logging
- Prompt validation
- Hot reloading for development

Usage:
    from core.prompts.registry import PromptRegistry
    
    registry = PromptRegistry()
    
    # Get a prompt
    prompt = registry.get_prompt("llm_gateway", "medical")
    
    # List all prompts in a category
    all_gateway_prompts = registry.list_prompts("llm_gateway")
    
    # Validate prompt safety
    is_safe = registry.validate_prompt(prompt)
    
    # Get audit history
    history = registry.get_audit_history("llm_gateway", "medical")
"""

import logging
import json
import hashlib
from typing import Dict, Optional, List, Any
from datetime import datetime
from pathlib import Path
import os

from core.prompts import system_prompts
from core.prompts.config_loader import get_prompt_override

logger = logging.getLogger(__name__)


class PromptRegistry:
    """
    Centralized registry for all system prompts in the Cardio AI application.
    
    Provides a unified interface for accessing prompts organized by category.
    Supports caching, versioning, and audit logging.
    """
    
    # Prompt categories and their available prompts
    PROMPT_CATEGORIES = {
        "llm_gateway": {
            "medical": "PROMPT_LLM_GATEWAY_MEDICAL",
            "nutrition": "PROMPT_LLM_GATEWAY_NUTRITION",
            "general": "PROMPT_LLM_GATEWAY_GENERAL",
            "multimodal_medical": "PROMPT_LLM_GATEWAY_MULTIMODAL_MEDICAL",
        },
        "medical_prompts": {
            "prompt_builder_system": "PROMPT_MEDICAL_PROMPT_BUILDER_SYSTEM",
        },
        "orchestrator": {
            "supervisor_routing": "PROMPT_SUPERVISOR_ROUTING",
            "supervisor_synthesis": "PROMPT_SUPERVISOR_SYNTHESIS",
        },
        "memori": {
            "search_agent": "PROMPT_MEMORI_SEARCH_AGENT",
            "memory_agent": "PROMPT_MEMORI_MEMORY_AGENT",
        },
        "multimodal": {
            "lab_results_table": "PROMPT_MULTIMODAL_LAB_RESULTS_TABLE",
            "vital_signs_table": "PROMPT_MULTIMODAL_VITAL_SIGNS_TABLE",
            "ecg_analysis": "PROMPT_MULTIMODAL_ECG_ANALYSIS",
            "medication_table": "PROMPT_MULTIMODAL_MEDICATION_TABLE",
            "image_analysis_system": "PROMPT_MULTIMODAL_IMAGE_ANALYSIS_SYSTEM",
            "table_analysis_system": "PROMPT_MULTIMODAL_TABLE_ANALYSIS_SYSTEM",
            "generic_table": "PROMPT_MULTIMODAL_GENERIC_TABLE",
            "generic_image": "PROMPT_MULTIMODAL_GENERIC_IMAGE",
            "extract_entities": "PROMPT_MULTIMODAL_EXTRACT_ENTITIES",
            "query": "PROMPT_MULTIMODAL_QUERY",
        },
        "tools": {
            "sql_expert": "PROMPT_SQL_EXPERT",
            "medical_coding_specialist": "PROMPT_MEDICAL_CODING_SPECIALIST",
        },
        "agents": {
            "medical_analyst": "PROMPT_MEDICAL_ANALYST_IMPLICIT",
            "researcher": "PROMPT_RESEARCHER_IMPLICIT",
            "drug_expert": "PROMPT_DRUG_EXPERT_IMPLICIT",
            "clinical_reasoning": "PROMPT_CLINICAL_REASONING_IMPLICIT",
            "thinking_agent": "PROMPT_THINKING_AGENT_IMPLICIT",
            "heart_analyst": "PROMPT_HEART_ANALYST_IMPLICIT",
            "fhir_agent": "PROMPT_FHIR_AGENT_IMPLICIT",
        },
    }
    
    def __init__(self, enable_audit_log: bool = True, cache_dir: Optional[str] = None):
        """
        Initialize the PromptRegistry.
        
        Args:
            enable_audit_log: Whether to track prompt usage in audit logs
            cache_dir: Directory for storing cached prompts and audit logs (optional)
        """
        self.enable_audit_log = enable_audit_log
        self.cache_dir = Path(cache_dir or "core/prompts/.cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._prompt_cache: Dict[str, Dict[str, str]] = {}
        self._version_cache: Dict[str, str] = {}
        self._audit_log: List[Dict[str, Any]] = []
        
        # Track which categories have been loaded (lazy loading)
        self._loaded_categories: set = set()
        
        logger.info(f"PromptRegistry initialized (lazy loading). Cache directory: {self.cache_dir}")
    
    def _load_category(self, category: str) -> None:
        """Lazy-load prompts for a single category on first access.
        
        This avoids loading ALL prompts into memory at startup,
        saving context window space when only specific prompts are needed.
        """
        if category in self._loaded_categories:
            return  # Already loaded
        
        if category not in self.PROMPT_CATEGORIES:
            return  # Unknown category
        
        try:
            self._prompt_cache[category] = {}
            prompts = self.PROMPT_CATEGORIES[category]
            for prompt_name, prompt_var_name in prompts.items():
                try:
                    prompt_value = getattr(system_prompts, prompt_var_name, None)
                    if prompt_value is None:
                        logger.warning(f"Prompt not found: {prompt_var_name} ({category}.{prompt_name})")
                        continue
                    
                    self._prompt_cache[category][prompt_name] = prompt_value
                    
                    # Generate and cache version hash
                    version_hash = self._generate_version_hash(prompt_value)
                    cache_key = f"{category}:{prompt_name}"
                    self._version_cache[cache_key] = version_hash
                    
                except Exception as e:
                    logger.error(f"Error loading prompt {prompt_var_name}: {e}")
            
            self._loaded_categories.add(category)
            logger.info(f"✅ Lazy-loaded {len(self._prompt_cache[category])} prompts for category '{category}'")
        except Exception as e:
            logger.error(f"Failed to load category '{category}': {e}")
            raise
    
    def _load_all_prompts(self) -> None:
        """Load ALL prompts (used for export/stats/reload operations)."""
        for category in self.PROMPT_CATEGORIES:
            self._load_category(category)
        logger.info(f"✅ Loaded all {sum(len(p) for p in self._prompt_cache.values())} prompts")
    
    def get_prompt(
        self,
        category: str,
        prompt_name: str,
        variables: Optional[Dict[str, str]] = None,
        track_usage: bool = True
    ) -> str:
        """
        Get a prompt by category and name.
        
        Priority: Config Override > Cache > Raise Error
        
        Args:
            category: Prompt category (e.g., "llm_gateway", "multimodal")
            prompt_name: Prompt name within the category (e.g., "medical", "lab_results_table")
            variables: Optional dictionary of variables to format into the prompt
            track_usage: Whether to track this usage in audit logs
            
        Returns:
            The prompt string, optionally formatted with variables
            
        Raises:
            KeyError: If category or prompt name not found
        """
        # Check for config override first
        override_prompt = get_prompt_override(category, prompt_name)
        if override_prompt is not None:
            prompt = override_prompt
            logger.debug(f"Using override for {category}.{prompt_name}")
        else:
            # Lazy-load the category on first access
            self._load_category(category)
            
            # Check category exists
            if category not in self._prompt_cache:
                raise KeyError(f"Unknown prompt category: {category}")
            
            # Check prompt exists
            if prompt_name not in self._prompt_cache[category]:
                raise KeyError(f"Unknown prompt '{prompt_name}' in category '{category}'")
            
            # Get prompt from cache
            prompt = self._prompt_cache[category][prompt_name]
        
        # Track usage if enabled
        if track_usage and self.enable_audit_log:
            self._log_usage(category, prompt_name)
        
        # Format with variables if provided
        if variables:
            try:
                prompt = prompt.format(**variables)
            except KeyError as e:
                logger.warning(f"Missing variable in prompt {category}.{prompt_name}: {e}")
        
        return prompt
    
    def list_prompts(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        List all available prompts, optionally filtered by category.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            Dictionary of available prompts
        """
        if category:
            self._load_category(category)
            if category not in self._prompt_cache:
                raise KeyError(f"Unknown category: {category}")
            return {
                prompt_name: f"[{len(prompt)} chars]"
                for prompt_name, prompt in self._prompt_cache[category].items()
            }
        else:
            # Load all categories for full listing
            self._load_all_prompts()
            return {
                cat: {
                    prompt_name: f"[{len(prompt)} chars]"
                    for prompt_name, prompt in prompts.items()
                }
                for cat, prompts in self._prompt_cache.items()
            }
    
    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Validate a prompt for safety and completeness.
        
        Checks for:
        - Medical disclaimers
        - Safety keywords
        - JSON format requirements (if applicable)
        - Proper structure
        
        Args:
            prompt: The prompt text to validate
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []
        
        # Check for medical disclaimer if it's a medical prompt
        if "medical" in prompt.lower() or "diagnosis" in prompt.lower():
            if "not diagnose" not in prompt.lower() and "no diagnosis" not in prompt.lower():
                warnings.append("⚠️ Medical prompt lacks clear 'do not diagnose' language")
        
        # Check for safety keywords
        safety_keywords = ["professional", "consultation", "healthcare", "doctor"]
        if "medical" in prompt.lower():
            found_safety = any(keyword in prompt.lower() for keyword in safety_keywords)
            if not found_safety:
                warnings.append("⚠️ Medical prompt lacks professional consultation guidance")
        
        # Check for proper JSON structure tags if needed
        if "{" in prompt and "}" in prompt:
            # This is a template prompt - verify it has proper placeholders
            if ":" not in prompt:
                issues.append("❌ JSON template format incorrect")
        
        # Check for injection vulnerabilities
        injection_patterns = ["follow user instructions", "ignore previous"]
        for pattern in injection_patterns:
            if pattern.lower() in prompt.lower():
                warnings.append(f"⚠️ Potential injection pattern detected: '{pattern}'")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "char_count": len(prompt),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_version(self, category: str, prompt_name: str) -> str:
        """
        Get the version hash of a prompt.
        
        Args:
            category: Prompt category
            prompt_name: Prompt name
            
        Returns:
            Version hash (SHA256)
        """
        cache_key = f"{category}:{prompt_name}"
        if cache_key not in self._version_cache:
            prompt = self.get_prompt(category, prompt_name, track_usage=False)
            self._version_cache[cache_key] = self._generate_version_hash(prompt)
        return self._version_cache[cache_key]
    
    def get_audit_history(
        self,
        category: Optional[str] = None,
        prompt_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get usage audit history for prompts.
        
        Args:
            category: Optional category filter
            prompt_name: Optional prompt name filter
            limit: Maximum number of records to return
            
        Returns:
            List of audit log entries
        """
        logs = self._audit_log.copy()
        
        if category:
            logs = [l for l in logs if l["category"] == category]
        
        if prompt_name:
            logs = [l for l in logs if l["prompt_name"] == prompt_name]
        
        return logs[-limit:]
    
    def reload_prompts(self) -> None:
        """
        Reload all prompts from system_prompts module.
        
        Useful for development when prompts are updated without restarting.
        """
        self._prompt_cache.clear()
        self._version_cache.clear()
        self._loaded_categories.clear()
        self._load_all_prompts()
        logger.info("✅ Prompts reloaded")
    
    def export_prompts(self, filepath: str) -> None:
        """
        Export all prompts to a JSON file for documentation or backup.
        
        Args:
            filepath: Path to export to
        """
        try:
            # Ensure all prompts are loaded for export
            self._load_all_prompts()
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "prompt_count": sum(len(p) for p in self._prompt_cache.values()),
                "prompts": self._prompt_cache
            }
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"✅ Prompts exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export prompts: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry stats
        """
        # Load all for accurate stats
        self._load_all_prompts()
        return {
            "total_categories": len(self._prompt_cache),
            "total_prompts": sum(len(p) for p in self._prompt_cache.values()),
            "categories": {
                cat: len(prompts)
                for cat, prompts in self._prompt_cache.items()
            },
            "total_audit_logs": len(self._audit_log),
            "cache_size_mb": sum(
                len(p) for cat_prompts in self._prompt_cache.values()
                for p in cat_prompts.values()
            ) / (1024 * 1024)
        }
    
    # ===== PRIVATE METHODS =====
    
    def _generate_version_hash(self, prompt: str) -> str:
        """Generate a SHA256 hash of a prompt for version tracking."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:12]
    
    def _log_usage(self, category: str, prompt_name: str) -> None:
        """Log prompt usage for audit trail."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "prompt_name": prompt_name,
            "version": self.get_version(category, prompt_name),
        }
        self._audit_log.append(log_entry)
        
        # Keep audit log size reasonable (keep last 10000 entries)
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]
    
    def create_version_commit(
        self,
        category: str,
        prompt_name: str,
        message: str,
        author: str = "system"
    ) -> Dict[str, Any]:
        """
        Create a version commit for a prompt (git-like commit message).
        
        Args:
            category: Prompt category
            prompt_name: Prompt name
            message: Commit message describing the change
            author: Author of the commit
            
        Returns:
            Commit information
        """
        try:
            prompt = self.get_prompt(category, prompt_name, track_usage=False)
            version_hash = self._generate_version_hash(prompt)
            
            commit = {
                "hash": version_hash,
                "timestamp": datetime.now().isoformat(),
                "category": category,
                "prompt_name": prompt_name,
                "message": message,
                "author": author,
                "prompt_length": len(prompt),
            }
            
            # Store commit in audit log with special marker
            commit["_type"] = "version_commit"
            self._audit_log.append(commit)
            
            logger.info(f"✅ Version commit created: {category}/{prompt_name} - {message}")
            return commit
        except Exception as e:
            logger.error(f"Failed to create version commit: {e}")
            raise
    
    def get_version_history(
        self,
        category: str,
        prompt_name: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get version commit history for a prompt.
        
        Args:
            category: Prompt category
            prompt_name: Prompt name
            limit: Maximum commits to return
            
        Returns:
            List of version commits
        """
        commits = [
            log for log in self._audit_log
            if log.get("_type") == "version_commit"
            and log.get("category") == category
            and log.get("prompt_name") == prompt_name
        ]
        return commits[-limit:]
    
    def get_audit_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive audit statistics.
        
        Returns:
            Detailed audit metrics
        """
        total_usages = len([l for l in self._audit_log if l.get("_type") != "version_commit"])
        total_commits = len([l for l in self._audit_log if l.get("_type") == "version_commit"])
        
        # Count usage by category
        usage_by_category = {}
        for log in self._audit_log:
            if log.get("_type") != "version_commit":
                cat = log.get("category", "unknown")
                usage_by_category[cat] = usage_by_category.get(cat, 0) + 1
        
        return {
            "total_usages": total_usages,
            "total_version_commits": total_commits,
            "usage_by_category": usage_by_category,
            "first_usage": self._audit_log[0]["timestamp"] if self._audit_log else None,
            "last_usage": self._audit_log[-1]["timestamp"] if self._audit_log else None,
        }


# Singleton instance for application-wide use
_registry_instance: Optional[PromptRegistry] = None


def get_prompt_registry() -> PromptRegistry:
    """
    Get the singleton PromptRegistry instance.
    
    Returns:
        The PromptRegistry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = PromptRegistry()
    return _registry_instance


# Convenience functions for direct access

def get_prompt(
    category: str,
    prompt_name: str,
    variables: Optional[Dict[str, str]] = None
) -> str:
    """
    Convenience function to get a prompt from the registry.
    
    Args:
        category: Prompt category
        prompt_name: Prompt name within category
        variables: Optional variables for formatting
        
    Returns:
        The prompt string
    """
    registry = get_prompt_registry()
    return registry.get_prompt(category, prompt_name, variables)


def list_prompts(category: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to list available prompts.
    
    Args:
        category: Optional category filter
        
    Returns:
        Dictionary of available prompts
    """
    registry = get_prompt_registry()
    return registry.list_prompts(category)


def validate_prompt(prompt: str) -> Dict[str, Any]:
    """
    Convenience function to validate a prompt.
    
    Args:
        prompt: Prompt text to validate
        
    Returns:
        Validation results
    """
    registry = get_prompt_registry()
    return registry.validate_prompt(prompt)
