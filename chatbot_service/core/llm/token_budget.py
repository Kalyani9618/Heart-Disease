"""
P2.3 Token Budget Utilities - Pre-allocation helpers for LLM context management.

Provides fast token estimation and budget calculation to avoid context
window overflow and optimize LLM generation speed.
"""

import os
from typing import Dict



class TokenBudgetCalculator:
    """P2.3: Token budget pre-allocation for optimal LLM usage.
    
    Calculates how much context can be safely passed to LLM based on
    prompt size, document count, and model limits.
    
    Usage:
        calc = TokenBudgetCalculator()
        budget = calc.calculate("What is atrial fibrillation?", num_documents=5)
        if budget["safe"]:
            # Safe to proceed with all documents
            pass
        else:
            # Reduce document count or truncate
            pass
    """
    
    # Model context window limits
    TOKEN_BUDGETS = {
        "medgemma-4b-it": {"context": 8192, "output": 2048},
        "gemma3:1b": {"context": 8192, "output": 2048},
        "gemini-1.5-flash": {"context": 1000000, "output": 8192},
        "gemini-2.0-flash": {"context": 1000000, "output": 8192},
        "gpt-4o": {"context": 128000, "output": 4096},
        "gpt-4o-mini": {"context": 128000, "output": 4096},
        "claude-3-sonnet": {"context": 200000, "output": 4096},
        "default": {"context": 4096, "output": 1024},
    }
    
    def __init__(self, model_name: str = None):
        """Initialize with optional model name override."""
        self.model_name = model_name or os.getenv("LLAMA_LOCAL_MODEL", "medgemma-4b-it")
        self.limits = self.TOKEN_BUDGETS.get(self.model_name, self.TOKEN_BUDGETS["default"])
    
    def estimate_tokens(self, text: str) -> int:
        """Fast token estimation using 4 chars = 1 token heuristic.
        
        Accuracy: ~80% for English text
        For precise counts, use actual tokenizer.
        """
        if not text:
            return 0
        return len(text) // 4 + 1
    
    def calculate(
        self,
        prompt: str,
        num_documents: int = 5,
        avg_doc_tokens: int = 200,
        system_prompt_tokens: int = 200,
    ) -> Dict:
        """Calculate optimal token allocation for a request.
        
        Args:
            prompt: User prompt text
            num_documents: Expected number of retrieved documents
            avg_doc_tokens: Average tokens per document
            system_prompt_tokens: Tokens reserved for system prompt
            
        Returns:
            Dict with:
            - prompt_tokens: Estimated prompt tokens
            - context_tokens: Estimated document tokens
            - output_budget: Available tokens for generation
            - total_used: Total tokens used
            - total_limit: Model context limit
            - safe: True if within 90% of limit
        """
        prompt_tokens = self.estimate_tokens(prompt)
        context_tokens = num_documents * avg_doc_tokens
        
        total_used = prompt_tokens + context_tokens + system_prompt_tokens
        available = self.limits["context"] - total_used
        output_budget = min(available, self.limits["output"])
        
        return {
            "prompt_tokens": prompt_tokens,
            "context_tokens": context_tokens,
            "system_prompt_tokens": system_prompt_tokens,
            "output_budget": max(100, output_budget),  # Minimum 100 tokens
            "total_used": total_used,
            "total_limit": self.limits["context"],
            "model": self.model_name,
            "safe": total_used < self.limits["context"] * 0.9,
        }
    
    def get_max_documents(
        self,
        prompt: str,
        avg_doc_tokens: int = 200,
        target_output_tokens: int = 500,
    ) -> int:
        """Calculate maximum number of documents that fit in context.
        
        Useful for dynamically adjusting top_k based on prompt size.
        """
        prompt_tokens = self.estimate_tokens(prompt)
        system_tokens = 200  # Reserve for system prompt
        
        available = self.limits["context"] - prompt_tokens - system_tokens - target_output_tokens
        max_docs = available // avg_doc_tokens
        
        return max(1, min(max_docs, 20))  # Between 1 and 20 docs


# Singleton instance for convenience
_calculator_instance = None


def get_token_calculator(model_name: str = None) -> TokenBudgetCalculator:
    """Get singleton token calculator instance."""
    global _calculator_instance
    if _calculator_instance is None:
        _calculator_instance = TokenBudgetCalculator(model_name)
    return _calculator_instance
