"""
Hallucination Grader.
Determines if an answer is grounded in the retrieved medical context.

P3.2 Optimization: Result caching to avoid repeated grading of similar answer/context pairs.
"""
import hashlib
from typing import Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from core.llm.llm_gateway import get_llm_gateway

class GradeResult(BaseModel):
    is_grounded: bool = Field(description="True if answer is supported by context.")
    explanation: str = Field(description="Why it is or isn't grounded.")

GRADER_PROMPT = """
You are a strict medical fact-checker.
Context: {context}
Answer: {answer}

Determine if the Answer is fully supported by the Context.
- If the answer contains medical claims NOT in the context, it is a Hallucination (False).
- If the answer is safe and supported, it is Grounded (True).

Output JSON.
"""

class HallucinationGrader:
    # P3.2: Grade result cache
    _cache: Dict[str, bool] = {}
    _cache_max_size: int = 500
    
    def __init__(self):
        self.llm = get_llm_gateway()
        self.parser = JsonOutputParser(pydantic_object=GradeResult)

    def _cache_key(self, answer: str, context: str) -> str:
        """P3.2: Create cache key from answer/context hash."""
        # Use first 500 chars of each to create deterministic key
        combined = f"{answer[:500]}||{context[:500]}"
        return hashlib.md5(combined.encode()).hexdigest()

    async def grade(self, answer: str, context: str) -> bool:
        """Returns True if grounded, False if hallucination.
        
        P3.2: Results are cached to avoid repeated LLM calls for
        similar answer/context pairs.
        """
        # P3.2: Check cache first
        cache_key = self._cache_key(answer, context)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        prompt = PromptTemplate(
            template=GRADER_PROMPT,
            input_variables=["context", "answer"]
        ).format(context=context, answer=answer)

        try:
            response = await self.llm.generate(prompt)
            result = self.parser.parse(response)
            is_grounded = result['is_grounded']
            
            # P3.2: Cache result
            if len(self._cache) >= self._cache_max_size:
                # Simple eviction: clear half the cache
                keys = list(self._cache.keys())[:self._cache_max_size // 2]
                for k in keys:
                    del self._cache[k]
            self._cache[cache_key] = is_grounded
            
            return is_grounded
        except Exception:
            # Fail safe: Assume ungrounded if check fails
            return False
    
    def clear_cache(self):
        """Clear the grading cache."""
        self._cache.clear()
