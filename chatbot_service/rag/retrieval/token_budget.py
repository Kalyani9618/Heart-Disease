"""
Token Budget Manager for Multi-Model Context Window Management.

Provides intelligent token counting and budget allocation across different
language models (GPT-4, Gemma, LLaMA, etc.) with proper tokenizer selection.

Features:
- Model-specific tokenizer selection (tiktoken for OpenAI, HuggingFace for others)
- Budget allocation for multi-component prompts (query, context, history)
- Token limit enforcement
- Fallback to character approximation for unknown models

Performance:
- tiktoken: <1ms per tokenization
- HuggingFace: 1-5ms per tokenization
- Character approximation: <0.1ms


Example:
    manager = TokenBudgetManager(model_name="gpt-4", max_tokens=4096)
    
    allocation = manager.allocate(
        query="What is the diagnosis?",
        medical_context="Patient symptoms include...",
        memories="Previous visit showed...",
        history="Last conversation..."
    )
"""

import logging
from typing import Dict, Optional, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


class CharacterApproximationTokenizer:
    """Fallback tokenizer using character count approximation."""
    
    def __init__(self, chars_per_token: int = 4):
        self.chars_per_token = chars_per_token
    
    def encode(self, text: str) -> list:
        """Approximate token count from character count."""
        if not text:
            return []
        # Return list of "fake" tokens (just indices)
        token_count = max(1, len(text) // self.chars_per_token)
        return list(range(token_count))


@lru_cache(maxsize=10)
def _get_model_tokenizer(model_name: Optional[str]) -> Any:
    """
    Get appropriate tokenizer for the given model.
    
    Uses:
    - tiktoken for OpenAI models (gpt-4, gpt-3.5-turbo)
    - HuggingFace for Gemma, LLaMA, etc.
    - Character approximation as fallback
    
    Args:
        model_name: Name of the model
        
    Returns:
        Tokenizer instance with encode() method
    """
    if model_name is None:
        logger.debug("No model specified, using character approximation")
        return CharacterApproximationTokenizer()
    
    model_lower = model_name.lower()
    
    # OpenAI models: use tiktoken
    if any(name in model_lower for name in ["gpt-4", "gpt-3", "gpt-3.5", "openai"]):
        try:
            import tiktoken
            if "gpt-4" in model_lower:
                return tiktoken.encoding_for_model("gpt-4")
            else:
                return tiktoken.encoding_for_model("gpt-3.5-turbo")
        except ImportError:
            logger.warning("tiktoken not available, using character approximation for OpenAI models")
            return CharacterApproximationTokenizer()
        except Exception as e:
            logger.warning(f"Failed to load tiktoken: {e}")
            return CharacterApproximationTokenizer()
    
    # Gemma, LLaMA, Mistral: try HuggingFace transformers
    if any(name in model_lower for name in ["gemma", "llama", "mistral", "codellama"]):
        try:
            from transformers import AutoTokenizer
            
            # Map to actual HuggingFace model names
            hf_model_map = {
                "gemma": "google/gemma-2-2b-it",
                "llama": "meta-llama/Llama-2-7b-hf",
                "mistral": "mistralai/Mistral-7B-v0.1",
            }
            
            hf_model_name = None
            for key, value in hf_model_map.items():
                if key in model_lower:
                    hf_model_name = value
                    break
            
            if hf_model_name:
                tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
                return tokenizer
        except ImportError:
            logger.debug("transformers not available for HuggingFace tokenizers")
        except Exception as e:
            logger.debug(f"Failed to load HuggingFace tokenizer: {e}")
    
    # Fallback to character approximation
    logger.debug(f"Using character approximation for model: {model_name}")
    return CharacterApproximationTokenizer()


class TokenBudgetManager:
    """
    Manages token budget allocation across prompt components.
    
    Intelligently allocates token budget for:
    - Query (user input)
    - Medical context (retrieved documents)
    - Memories (long-term user facts)
    - History (conversation history)
    
    Priority order: Query > Medical Context > Memories > History
    """
    
    # Default allocation percentages
    DEFAULT_ALLOCATIONS = {
        "query": 0.10,          # 10% for query
        "medical_context": 0.50, # 50% for medical documents
        "memories": 0.20,        # 20% for memories
        "history": 0.15,         # 15% for history
        "reserved": 0.05,        # 5% reserved for system overhead
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: int = 2048,
        allocations: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize TokenBudgetManager.
        
        Args:
            model_name: Model name for tokenizer selection
            max_tokens: Maximum token budget
            allocations: Custom allocation percentages
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.allocations = allocations or self.DEFAULT_ALLOCATIONS
        
        # Get appropriate tokenizer
        self._tokenizer = _get_model_tokenizer(model_name)
        
        logger.debug(f"TokenBudgetManager initialized: model={model_name}, max_tokens={max_tokens}")
    
    def count_tokens(self, text: Optional[str]) -> int:
        """
        Count tokens in text using model-specific tokenizer.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Token count
        """
        if not text:
            return 0
        
        try:
            tokens = self._tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}, using character approximation")
            return len(text) // 4
    
    def allocate(
        self,
        query: str,
        medical_context: str,
        memories: str = "",
        history: str = "",
    ) -> Dict[str, str]:
        """
        Allocate token budget across prompt components.
        
        Returns truncated versions of each component if they exceed
        their allocated budget.
        
        Args:
            query: User query
            medical_context: Retrieved medical documents
            memories: Long-term memories
            history: Conversation history
            
        Returns:
            Dict with allocated (possibly truncated) components
        """
        # Calculate available tokens for each component
        reserved_tokens = int(self.max_tokens * self.allocations.get("reserved", 0.05))
        available_tokens = self.max_tokens - reserved_tokens
        
        allocations = {
            "query": int(available_tokens * self.allocations.get("query", 0.10)),
            "medical_context": int(available_tokens * self.allocations.get("medical_context", 0.50)),
            "memories": int(available_tokens * self.allocations.get("memories", 0.20)),
            "history": int(available_tokens * self.allocations.get("history", 0.15)),
        }
        
        result = {}
        
        # Allocate each component
        result["query"] = self._truncate_to_budget(query, allocations["query"])
        result["medical_context"] = self._truncate_to_budget(medical_context, allocations["medical_context"])
        result["memories"] = self._truncate_to_budget(memories, allocations["memories"])
        result["history"] = self._truncate_to_budget(history, allocations["history"])
        
        # Redistribute unused tokens from shorter components
        unused_tokens = 0
        for key, max_tokens in allocations.items():
            actual_tokens = self.count_tokens(result[key])
            if actual_tokens < max_tokens:
                unused_tokens += max_tokens - actual_tokens
        
        # Give unused tokens to medical_context if it was truncated
        if unused_tokens > 0 and self.count_tokens(medical_context) > allocations["medical_context"]:
            new_budget = allocations["medical_context"] + unused_tokens
            result["medical_context"] = self._truncate_to_budget(medical_context, new_budget)
        
        return result
    
    def _truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token budget.
        
        Uses binary search for efficient truncation.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum allowed tokens
            
        Returns:
            Truncated text
        """
        if not text:
            return ""
        
        current_tokens = self.count_tokens(text)
        if current_tokens <= max_tokens:
            return text
        
        # Binary search for optimal truncation point
        low, high = 0, len(text)
        
        while low < high:
            mid = (low + high + 1) // 2
            truncated = text[:mid]
            if self.count_tokens(truncated) <= max_tokens:
                low = mid
            else:
                high = mid - 1
        
        truncated_text = text[:low]
        
        # Add truncation indicator if we actually truncated
        if len(truncated_text) < len(text):
            # Make room for ellipsis
            while self.count_tokens(truncated_text + "...") > max_tokens and truncated_text:
                truncated_text = truncated_text[:-10]
            truncated_text += "..."
        
        return truncated_text
    
    def get_remaining_budget(self, used_tokens: int) -> int:
        """
        Get remaining token budget after some usage.
        
        Args:
            used_tokens: Tokens already used
            
        Returns:
            Remaining tokens
        """
        return max(0, self.max_tokens - used_tokens)
    
    def is_within_budget(self, text: str) -> bool:
        """
        Check if text fits within total budget.
        
        Args:
            text: Text to check
            
        Returns:
            True if within budget
        """
        return self.count_tokens(text) <= self.max_tokens
