"""
LLM Gateway - MedGemma-Only Implementation

This is the ONLY module that should directly call LLM providers.
All AI generation in the system MUST flow through this gateway.

Features:
- ✅ MedGemma (local medical LLM via OpenAI-compatible API)
- ✅ LangFuse (for debugging - optional)
- ✅ Guardrails (for safety)
- ✅ PII Detection (privacy protection)


Usage:
    from core.llm.llm_gateway import LLMGateway, get_llm_gateway

    gateway = get_llm_gateway()
    response = await gateway.generate(
        prompt="Explain heart health tips",
        content_type="medical"  # Adds medical disclaimer
    )

Configuration (via .env):
    MEDGEMMA_BASE_URL=http://127.0.0.1:8090/v1
    MEDGEMMA_MODEL=medgemma-4b-it
    MEDGEMMA_API_KEY=sk-no-key-required
    MEDGEMMA_TEMPERATURE=0.3
    MEDGEMMA_MAX_TOKENS=2048
"""

import os
import logging
import re
from typing import Optional, AsyncGenerator, Dict, Any

# Import PromptRegistry for centralized prompt management
from core.prompts.registry import get_prompt
from core.circuit_breaker import circuit_breaker

# LangChain components for MedGemma (OpenAI-compatible API)
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    ChatOpenAI = None
    LANGCHAIN_OPENAI_AVAILABLE = False
    logging.getLogger(__name__).error(
        "langchain-openai not installed! MedGemma integration requires this package. "
        "Install with: pip install langchain-openai"
    )

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangFuse imports (optional - for observability)
_langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"

if _langfuse_enabled:
    try:
        from langfuse import observe
        LANGFUSE_AVAILABLE = True
        logging.getLogger(__name__).info("Langfuse observability enabled")
    except ImportError:
        LANGFUSE_AVAILABLE = False
        def observe(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        logging.getLogger(__name__).warning("langfuse not installed. Observability features disabled.")
else:
    LANGFUSE_AVAILABLE = False
    # Create a no-op decorator when langfuse is disabled
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Import guardrails for safety processing
from .guardrails import SafetyGuardrail

logger = logging.getLogger(__name__)


class LLMGateway:
    """
    MedGemma-Only LLM Gateway - Single Source of Truth for LLM interactions.
    
    Connects to local MedGemma server via OpenAI-compatible API.
    All AI generation must flow through this gateway for safety and compliance.
    
    Benefits:
    - No cloud API dependencies (HIPAA-compliant local processing)
    - Simplified architecture (no fallback chains)
    - Medical-specialized model for healthcare accuracy
    - Full control over data and processing
    """

    def __init__(self):
        self.guardrails = SafetyGuardrail()
        
        # MedGemma Configuration (from environment)
        # Support both new MEDGEMMA_* and legacy LLAMA_LOCAL_* env vars
        self.medgemma_base_url = os.getenv(
            "MEDGEMMA_BASE_URL", 
            os.getenv("LLAMA_LOCAL_BASE_URL", "http://127.0.0.1:8090/v1")
        )
        self.medgemma_model = os.getenv(
            "MEDGEMMA_MODEL", 
            os.getenv("LLAMA_LOCAL_MODEL", "medgemma-4b-it")
        )
        if "medgemma" not in self.medgemma_model.lower():
            logger.warning(
                f"Invalid non-MedGemma model configured ('{self.medgemma_model}'). "
                "Forcing MEDGEMMA_MODEL=medgemma-4b-it."
            )
            self.medgemma_model = "medgemma-4b-it"
        # Backward-compatible alias used by some tracing hooks.
        self.model_name = self.medgemma_model
        self.medgemma_api_key = os.getenv(
            "MEDGEMMA_API_KEY", 
            os.getenv("LLAMA_LOCAL_API_KEY", "sk-no-key-required")
        )
        self.medgemma_temperature = float(os.getenv(
            "MEDGEMMA_TEMPERATURE", 
            os.getenv("LLAMA_LOCAL_TEMPERATURE", "0.3")
        ))
        self.medgemma_max_tokens = int(os.getenv(
            "MEDGEMMA_MAX_TOKENS", 
            os.getenv("LLAMA_LOCAL_MAX_TOKENS", "2048")
        ))
        
        # Initialize MedGemma via OpenAI-compatible API
        self.llm = None
        if LANGCHAIN_OPENAI_AVAILABLE and ChatOpenAI is not None:
            try:
                self.llm = ChatOpenAI(
                    model=self.medgemma_model,
                    api_key=self.medgemma_api_key,
                    base_url=self.medgemma_base_url,
                    temperature=self.medgemma_temperature,
                    max_tokens=self.medgemma_max_tokens,
                )
                logger.info(
                    f"✅ MedGemma initialized: model={self.medgemma_model}, "
                    f"endpoint={self.medgemma_base_url}"
                )
            except Exception as e:
                logger.error(f"❌ Failed to initialize MedGemma: {e}")
                logger.error(
                    f"   Ensure MedGemma server is running at {self.medgemma_base_url}\n"
                    f"   Start with: llama-server -m medgemma-4b.gguf --port 8090"
                )
        else:
            logger.error("❌ langchain-openai not installed - MedGemma unavailable!")

    def _get_model(self, provider: str = None):
        """
        Get the LLM model. Always returns MedGemma.
        Provider parameter kept for backward compatibility but is ignored.
        """
        if self.llm is None:
            raise RuntimeError(
                f"MedGemma not initialized. "
                f"Check server at {self.medgemma_base_url}"
            )
        return self.llm

    def _get_user_provider(self, user_id: Optional[str] = None) -> str:
        """
        Get the provider for a user. Always returns 'medgemma'.
        Kept for backward compatibility with existing code.
        """
        return "medgemma"
    
    def _contains_pii(self, text: str) -> bool:
        """
        Detect if text contains Personally Identifiable Information (PII).
        
        Patterns checked:
        - Social Security Number (XXX-XX-XXXX)
        - Email addresses
        - Phone numbers (XXX-XXX-XXXX, (XXX) XXX-XXXX)
        - Medical record numbers
        - Health insurance IDs
        
        Args:
            text: Text to analyze for PII
            
        Returns:
            True if PII detected, False otherwise
        """
        if not text:
            return False
        
        # Social Security Number pattern (XXX-XX-XXXX)
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        if re.search(ssn_pattern, text):
            logger.warning("PII detected: SSN pattern found")
            return True
        
        # Email address pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, text):
            logger.debug("PII detected: Email found")
            return True
        
        # US Phone number patterns
        phone_patterns = [
            r'\b\d{3}-\d{3}-\d{4}\b',  # XXX-XXX-XXXX
            r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # (XXX) XXX-XXXX
            r'\b\+1\s*\d{3}-\d{3}-\d{4}\b',  # +1 XXX-XXX-XXXX
            r'\b\d{3}\.\d{3}\.\d{4}\b',  # XXX.XXX.XXXX
        ]
        for pattern in phone_patterns:
            if re.search(pattern, text):
                logger.debug("PII detected: Phone number found")
                return True
        
        # Medical record number (MRN) pattern - typically 6-10 digits
        mrn_pattern = r'\bMRN\s*[:\s]+\d{6,10}\b'
        if re.search(mrn_pattern, text, re.IGNORECASE):
            logger.warning("PII detected: Medical Record Number found")
            return True
        
        # Health Insurance ID patterns
        insurance_patterns = [
            r'\bMember\s*ID\s*[:\s]+[A-Z0-9]{8,}\b',
            r'\bPolicy\s*#\s*[:\s]+\d{6,}\b',
            r'\bInsurance\s*ID\s*[:\s]+[A-Z0-9]{8,}\b',
        ]
        for pattern in insurance_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning("PII detected: Insurance ID found")
                return True
        
        return False
    
    def set_user_provider(self, user_id: str, provider: str) -> None:
        """
        Set the LLM provider preference for a user.
        
        Note: With MedGemma-only architecture, this is a no-op kept for backward compatibility.
        All requests use MedGemma regardless of provider setting.
        
        Args:
            user_id: User identifier
            provider: Provider to use (ignored - always uses MedGemma)
        """
        logger.debug(f"set_user_provider called for user {user_id} with provider {provider} - using MedGemma")

    @observe(name="medgemma-generation")  # ✅ LangFuse Observability
    async def generate(
        self, prompt: str, content_type: str = "general", user_id: Optional[str] = None
    ) -> str:
        """
        Generate text using MedGemma with safety checks.
        
        Note: With local-only processing, PII detection is logged for compliance
        but no provider switching occurs (all data stays on-premise).

        Args:
            prompt: The prompt to send to MedGemma
            content_type: "medical", "nutrition", or "general"
            user_id: Optional user ID for tracing

        Returns:
            Generated response with safety processing applied
        """
        # Log PII detection for compliance auditing (no routing needed with local LLM)
        if self._contains_pii(prompt):
            logger.info(
                f"PII detected in prompt (user: {user_id}) - processing locally via MedGemma (HIPAA-compliant)"
            )
        
        import time as _time
        _start = _time.perf_counter()
        
        try:
            raw_response = await self._execute_generation(prompt, content_type)
        except Exception as e:
            logger.error(f"MedGemma generation failed: {e}")
            raise
        
        _latency_ms = (_time.perf_counter() - _start) * 1000
        
        # Record in AgentTracer for observability
        try:
            from app_lifespan import get_agent_tracer
            tracer = get_agent_tracer()
            if tracer:
                tracer.record_llm_call(
                    model=self.model_name,
                    prompt=prompt[:200],
                    response=raw_response[:200],
                    tokens_used=len(raw_response.split()),
                    latency_ms=_latency_ms,
                )
        except Exception:
            pass  # Tracing must never break generation

        # Apply Guardrails ✅
        return self.guardrails.process_output(
            raw_response, {"type": content_type, "user_id": user_id}
        )

    @circuit_breaker(service_name="llm", fallback_result="I'm sorry, the AI service is currently unavailable. Please try again later.")
    async def _execute_generation(self, prompt: str, content_type: str) -> str:
        """Execute generation with MedGemma."""
        model = self._get_model()

        # Apply appropriate system prompt based on content type using PromptRegistry
        # Only load the ONE prompt needed for this content_type (saves context window)
        prompt_key = content_type if content_type in ("medical", "nutrition", "general") else "general"
        system_prompt = get_prompt("llm_gateway", prompt_key)

        # Create Chain with LangChain
        chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        system_prompt,
                    ),
                    ("human", "{input}"),
                ]
            )
            | model
            | StrOutputParser()
        )

        # Execute
        raw_response = await chain.ainvoke({"input": prompt})

        return raw_response

    async def generate_stream(
        self, prompt: str, content_type: str = "general", user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Streaming generation for chat interfaces using MedGemma.
        
        Note: PII detection logged for compliance but all processing is local.
        """
        # Log PII detection for compliance
        if self._contains_pii(prompt):
            logger.info(
                f"PII detected in streaming prompt (user: {user_id}) - processing locally via MedGemma"
            )
        
        model = self._get_model()

        # Only load the ONE prompt needed for this content_type (saves context window)
        prompt_key = content_type if content_type in ("medical", "nutrition", "general") else "general"
        system_prompt = get_prompt("llm_gateway", prompt_key)

        # Create Chain with LangChain
        chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        system_prompt,
                    ),
                    ("human", "{input}"),
                ]
            )
            | model
            | StrOutputParser()
        )

        async for chunk in chain.astream({"input": prompt}):
            yield chunk

    @observe(name="medgemma-multimodal")
    async def generate_multimodal(
        self, 
        prompt: str, 
        image_data: str, 
        content_type: str = "medical", 
        user_id: Optional[str] = None
    ) -> str:
        """
        Generate text from multimodal input (text + image) using MedGemma.
        
        Note: MedGemma-4B may have limited multimodal capabilities.
        Consider using MedGemma-27B for better vision support.
        
        Args:
            prompt: Text prompt
            image_data: Base64 encoded image string or URL
            content_type: Content type for system prompt selection
            user_id: User ID for tracing
            
        Returns:
            Generated response
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        
        model = self._get_model()
        
        # System prompt from PromptRegistry
        system_msg = SystemMessage(content=get_prompt("llm_gateway", "multimodal_medical"))
        
        # Prepare content
        # Check if image_data is a URL or base64
        if image_data.startswith("http"):
            image_url = image_data
        else:
            # Assume base64, ensure prefix
            if not image_data.startswith("data:image"):
                image_url = f"data:image/jpeg;base64,{image_data}"
            else:
                image_url = image_data
                
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
        
        human_msg = HumanMessage(content=content)
        
        try:
            # Direct invocation of the model with messages
            response = await model.ainvoke([system_msg, human_msg])
            
            # Handle response types (some return string, some AIMessage)
            if hasattr(response, "content"):
                return response.content
            return str(response)
            
        except Exception as e:
            logger.error(f"Multimodal generation failed with MedGemma: {e}")
            raise

    def supports_multimodal(self) -> bool:
        """
        Check if current MedGemma model supports vision/multimodal.
        MedGemma-27B supports multimodal, MedGemma-4B has limited support.
        """
        return "27b" in self.medgemma_model.lower()

    def get_status(self) -> Dict[str, Any]:
        """Return the health status of the LLM Gateway."""
        return {
            "status": "online" if self.llm is not None else "offline",
            "provider": "medgemma",
            "model": self.medgemma_model,
            "base_url": self.medgemma_base_url,
            "medgemma_available": self.llm is not None,
            "multimodal_supported": self.supports_multimodal(),
        }


# Singleton Accessor
_gateway_instance = None


def get_llm_gateway() -> LLMGateway:
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = LLMGateway()
    return _gateway_instance