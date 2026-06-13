"""
Prompt Configuration Override System

Allows environment-specific prompt overrides without code changes.

Usage:
    from core.prompts.config_loader import load_prompt_overrides
    
    overrides = load_prompt_overrides("production")
    override_prompt = overrides.get("llm_gateway", {}).get("medical")
    
    if override_prompt:
        # Use override instead of default
        prompt = override_prompt
"""

import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptConfigLoader:
    """Load and manage prompt configuration overrides."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the config loader.
        
        Args:
            config_dir: Directory containing prompt_overrides.yaml
        """
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "prompt_overrides.yaml"
        self.environment = os.getenv("PROMPT_OVERRIDES_ENV", "production")
        self.overrides: Dict[str, Dict[str, str]] = {}
        self.settings: Dict[str, Any] = {}
        self.rules: Dict[str, Any] = {}
    
    def load(self) -> bool:
        """
        Load configuration from yaml file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed. Prompt overrides will not be loaded.")
            logger.warning("Install with: pip install pyyaml")
            return False
        
        if not self.config_file.exists():
            logger.warning(f"Config file not found: {self.config_file}")
            return False
        
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if not config:
                logger.warning("Config file is empty")
                return False
            
            # Extract overrides for current environment
            environments = config.get("environments", {})
            env_config = environments.get(self.environment, {})
            self.overrides = env_config
            
            # Extract settings for current environment
            all_settings = config.get("settings", {})
            self.settings = all_settings.get(self.environment, {})
            
            # Extract rules for current environment
            special_rules = config.get("special_rules", {})
            self.rules = special_rules.get(self.environment, [])
            
            logger.info(
                f"✅ Loaded prompt overrides for environment: {self.environment} "
                f"({len(self.overrides)} categories)"
            )
            return True
        
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    def get_override(self, category: str, prompt_name: str) -> Optional[str]:
        """
        Get an override prompt if it exists.
        
        Args:
            category: Prompt category
            prompt_name: Prompt name
            
        Returns:
            Override prompt string, or None if no override
        """
        category_overrides = self.overrides.get(category, {})
        override = category_overrides.get(prompt_name)
        
        # Return None if override is explicitly set to null
        if override is None:
            return None
        
        # Return the override if it's a non-empty string
        if isinstance(override, str) and override.strip():
            logger.debug(f"Using override for {category}.{prompt_name}")
            return override
        
        return None
    
    def get_setting(self, setting_name: str, default: Any = None) -> Any:
        """
        Get a configuration setting.
        
        Args:
            setting_name: Setting name
            default: Default value if not found
            
        Returns:
            Setting value or default
        """
        return self.settings.get(setting_name, default)
    
    def has_rule(self, rule_name: str) -> bool:
        """
        Check if a special rule is enabled.
        
        Args:
            rule_name: Rule name to check
            
        Returns:
            True if rule is in the rules list
        """
        # Rules can be strings or dicts
        for rule in self.rules:
            if isinstance(rule, str):
                if rule == rule_name:
                    return True
            elif isinstance(rule, dict):
                if rule_name in rule and rule[rule_name]:
                    return True
        return False
    
    def apply_rule(self, rule_name: str, prompt: str) -> str:
        """
        Apply a special rule to a prompt.
        
        Args:
            rule_name: Rule to apply
            prompt: Prompt text
            
        Returns:
            Modified prompt
        """
        if rule_name == "add_dev_suffix":
            if self.has_rule("add_dev_suffix"):
                return f"[DEV MODE] {prompt}"
        
        return prompt
    
    def print_summary(self):
        """Print configuration summary."""
        print(f"""
╔════════════════════════════════════════════════════════╗
║   PROMPT CONFIGURATION OVERRIDE SUMMARY                ║
╚════════════════════════════════════════════════════════╝

Environment: {self.environment}
Config File: {self.config_file}
Override Categories: {len(self.overrides)}

Categories with Overrides:
""")
        for category, prompts in self.overrides.items():
            count = len([p for p in prompts.values() if p is not None])
            print(f"  - {category}: {count} overrides")
        
        print(f"\nSettings:")
        for setting, value in self.settings.items():
            print(f"  - {setting}: {value}")
        
        print(f"\nActive Rules:")
        if self.rules:
            for rule in self.rules:
                print(f"  - {rule}")
        else:
            print("  (none)")


# Global config loader instance
_config_loader: Optional[PromptConfigLoader] = None


def get_config_loader() -> PromptConfigLoader:
    """
    Get the global PromptConfigLoader instance.
    
    Returns:
        PromptConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = PromptConfigLoader()
        _config_loader.load()
    return _config_loader


def load_prompt_overrides(environment: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Load and return prompt overrides for an environment.
    
    Args:
        environment: Environment name (defaults to PROMPT_OVERRIDES_ENV env var)
        
    Returns:
        Dictionary of category -> prompt_name -> override_text
    """
    loader = get_config_loader()
    if environment:
        loader.environment = environment
        loader.load()
    return loader.overrides


def get_prompt_override(category: str, prompt_name: str) -> Optional[str]:
    """
    Convenience function to get a prompt override.
    
    Args:
        category: Prompt category
        prompt_name: Prompt name
        
    Returns:
        Override prompt or None
    """
    loader = get_config_loader()
    return loader.get_override(category, prompt_name)


if __name__ == "__main__":
    # Print configuration when run as main
    loader = get_config_loader()
    loader.print_summary()
