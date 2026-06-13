"""
Provider configuration for different LLM providers (OpenAI, Azure, custom)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger


class ProviderType(Enum):
    """Supported LLM provider types"""

    OPENAI = "openai"
    AZURE = "azure"
    CUSTOM = "custom"
    OPENAI_COMPATIBLE = "openai_compatible"  # For OpenAI-compatible APIs
    OPENROUTER = "openrouter"  # OpenRouter API (multiple models)
    LLAMA_LOCAL = "llama_local"  # Local Llama/llama.cpp server


@dataclass
class ProviderConfig:
    """
    Configuration for LLM providers with support for OpenAI, Azure, and custom endpoints.

    This class provides a unified interface for configuring different LLM providers
    while maintaining backward compatibility with existing OpenAI-only configuration.
    """

    # Common parameters
    api_key: str | None = None
    api_type: str | None = None  # "openai", "azure", or custom
    base_url: str | None = None  # Custom endpoint URL
    timeout: float | None = None
    max_retries: int | None = None

    # Azure-specific parameters
    azure_endpoint: str | None = None
    azure_deployment: str | None = None
    api_version: str | None = None
    azure_ad_token: str | None = None

    # OpenAI-specific parameters
    organization: str | None = None
    project: str | None = None

    # Model configuration
    model: str | None = None  # User can specify model, defaults to gpt-4o if not set

    # Additional headers for custom providers
    default_headers: dict[str, str] | None = None
    default_query: dict[str, Any] | None = None

    # HTTP client configuration
    http_client: Any | None = None

    @classmethod
    def from_openai(
        cls, api_key: str | None = None, model: str | None = None, **kwargs
    ):
        """Create configuration for standard OpenAI"""
        return cls(api_key=api_key, api_type="openai", model=model, **kwargs)

    @classmethod
    def from_azure(
        cls,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        model: str | None = None,
        **kwargs,
    ):
        """Create configuration for Azure OpenAI"""
        return cls(
            api_key=api_key,
            api_type="azure",
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            model=model,
            **kwargs,
        )

    @classmethod
    def from_custom(
        cls,
        base_url: str,
        api_key: str | None = None,
        model: str | None = None,
        **kwargs,
    ):
        """Create configuration for custom OpenAI-compatible endpoints"""
        return cls(
            api_key=api_key, api_type="custom", base_url=base_url, model=model, **kwargs
        )

    @classmethod
    def from_openrouter(
        cls,
        api_key: str | None = None,
        model: str | None = None,
        **kwargs,
    ):
        """
        Create configuration for OpenRouter API.
        
        OpenRouter provides access to multiple LLM providers (OpenAI, Anthropic, etc.)
        through a single API endpoint.
        
        Args:
            api_key: OpenRouter API key (starts with sk-or-)
            model: Model to use (e.g., 'openai/gpt-4o-mini', 'anthropic/claude-3-haiku')
        """
        import os
        
        # Get API key from env if not provided
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("MEMORI_LLM_API_KEY")
        
        # Default model for memory processing (cost-effective)
        if not model:
            model = os.getenv("MEMORI_LLM_MODEL", "openai/gpt-4o-mini")
        
        return cls(
            api_key=api_key,
            api_type="openrouter",
            base_url="https://openrouter.ai/api/v1",
            model=model,
            **kwargs,
        )

    @classmethod
    def from_llama_local(
        cls,
        base_url: str | None = None,
        model: str | None = None,
        **kwargs,
    ):
        """
        Create configuration for local Llama server (llama.cpp, Ollama, etc.)
        
        Args:
            base_url: Local server URL (default: http://127.0.0.1:8090/v1)
            model: Model name (default: medgemma-4b-it)
        """
        import os
        
        if not base_url:
            base_url = os.getenv("LLAMA_LOCAL_BASE_URL", "http://127.0.0.1:8090/v1")
        
        if not model:
            model = os.getenv("LLAMA_LOCAL_MODEL", "medgemma-4b-it")
        
        return cls(
            api_key=os.getenv("LLAMA_LOCAL_API_KEY", "sk-no-key-required"),
            api_type="llama_local",
            base_url=base_url,
            model=model,
            **kwargs,
        )

    def get_openai_client_kwargs(self) -> dict[str, Any]:
        """
        Get kwargs for OpenAI client initialization based on provider type.

        Returns:
            Dictionary of parameters to pass to OpenAI client constructor
        """
        kwargs = {}

        # Always include API key if provided
        if self.api_key:
            kwargs["api_key"] = self.api_key

        if self.api_type == "azure":
            # Azure OpenAI configuration
            if self.azure_endpoint:
                kwargs["azure_endpoint"] = self.azure_endpoint
            if self.azure_deployment:
                kwargs["azure_deployment"] = self.azure_deployment
            if self.api_version:
                kwargs["api_version"] = self.api_version
            if self.azure_ad_token:
                kwargs["azure_ad_token"] = self.azure_ad_token
            # For Azure, we need to use AzureOpenAI client
            kwargs["_use_azure_client"] = True

        elif self.api_type == "openrouter":
            # OpenRouter configuration (OpenAI-compatible)
            kwargs["base_url"] = self.base_url or "https://openrouter.ai/api/v1"
            # Add required headers for OpenRouter
            if not self.default_headers:
                kwargs["default_headers"] = {
                    "HTTP-Referer": "https://heartguard-ai.local",
                    "X-Title": "HeartGuard Memory System",
                }
            else:
                headers = dict(self.default_headers)
                headers.setdefault("HTTP-Referer", "https://heartguard-ai.local")
                headers.setdefault("X-Title", "HeartGuard Memory System")
                kwargs["default_headers"] = headers

        elif self.api_type == "llama_local":
            # Local Llama server configuration (llama.cpp, Ollama, etc.)
            kwargs["base_url"] = self.base_url or "http://127.0.0.1:8090/v1"
            # Local servers typically don't need auth, but we still pass the key
            if not self.api_key:
                kwargs["api_key"] = "sk-no-key-required"

        elif self.api_type == "custom" or self.api_type == "openai_compatible":
            # Custom endpoint configuration
            if self.base_url:
                kwargs["base_url"] = self.base_url

        elif self.api_type == "openai":
            # Standard OpenAI configuration
            if self.organization:
                kwargs["organization"] = self.organization
            if self.project:
                kwargs["project"] = self.project

        # Common parameters
        if self.timeout:
            kwargs["timeout"] = self.timeout
        if self.max_retries:
            kwargs["max_retries"] = self.max_retries
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        if self.default_query:
            kwargs["default_query"] = self.default_query
        if self.http_client:
            kwargs["http_client"] = self.http_client

        return kwargs

    def create_client(self):
        """
        Create the appropriate OpenAI client based on configuration.

        Returns:
            OpenAI or AzureOpenAI client instance
        """
        import openai

        kwargs = self.get_openai_client_kwargs()

        # Check if we should use Azure client
        if kwargs.pop("_use_azure_client", False):
            # Use Azure OpenAI client
            from openai import AzureOpenAI

            return AzureOpenAI(**kwargs)
        else:
            # Use standard OpenAI client (works for OpenAI and custom endpoints)
            return openai.OpenAI(**kwargs)

    def create_async_client(self):
        """
        Create the appropriate async OpenAI client based on configuration.

        Returns:
            AsyncOpenAI or AsyncAzureOpenAI client instance
        """
        import openai

        kwargs = self.get_openai_client_kwargs()

        # Check if we should use Azure client
        if kwargs.pop("_use_azure_client", False):
            # Use Azure OpenAI async client
            from openai import AsyncAzureOpenAI

            return AsyncAzureOpenAI(**kwargs)
        else:
            # Use standard async OpenAI client
            return openai.AsyncOpenAI(**kwargs)


def detect_provider_from_env() -> ProviderConfig:
    """
    Create provider configuration from environment variables.

    Priority order:
    1. MEMORI_LLM_API_TYPE explicitly set (openrouter, llama_local, openai, azure)
    2. OpenRouter if OPENROUTER_API_KEY or MEMORI_LLM_API_KEY with sk-or- prefix is set
    3. Llama Local if LLAMA_LOCAL_BASE_URL is set
    4. Standard OpenAI as fallback

    Returns:
        ProviderConfig instance configured from environment
    """
    import os

    # Check for explicit API type configuration
    api_type = os.getenv("MEMORI_LLM_API_TYPE", "").lower()
    
    # Priority 1: Explicit API type
    if api_type == "openrouter":
        logger.info("Provider configuration: Using OpenRouter (explicit)")
        return ProviderConfig.from_openrouter()
    elif api_type == "llama_local":
        logger.info("Provider configuration: Using Llama Local (explicit)")
        return ProviderConfig.from_llama_local()
    elif api_type == "azure":
        logger.info("Provider configuration: Using Azure OpenAI (explicit)")
        return ProviderConfig.from_azure()
    elif api_type == "openai":
        logger.info("Provider configuration: Using OpenAI (explicit)")
        return ProviderConfig.from_openai()
    
    # Priority 2: OpenRouter detection (check for OpenRouter API key)
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("MEMORI_LLM_API_KEY")
    if openrouter_key and openrouter_key.startswith("sk-or-"):
        logger.info("Provider configuration: Using OpenRouter (auto-detected from API key)")
        return ProviderConfig.from_openrouter(api_key=openrouter_key)
    
    # Priority 3: Llama Local detection
    llama_base_url = os.getenv("LLAMA_LOCAL_BASE_URL")
    if llama_base_url:
        logger.info("Provider configuration: Using Llama Local (auto-detected from base URL)")
        return ProviderConfig.from_llama_local(base_url=llama_base_url)
    
    # Priority 4: Standard OpenAI fallback
    model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o"
    logger.info("Provider configuration: Using standard OpenAI (default fallback)")
    return ProviderConfig.from_openai(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORGANIZATION"),
        project=os.getenv("OPENAI_PROJECT"),
        model=model,
    )
