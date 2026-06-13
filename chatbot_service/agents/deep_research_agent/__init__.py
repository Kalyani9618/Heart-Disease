"""
Deep Research Agent

Provides three research modes:
1. Linear (original deep_research.py) - Simple pipeline
2. CoT (Chain-of-Thought) - ReasoningResearcher with adaptive search
3. AoT (Atom-of-Thought) - AtomicResearcher with plan decomposition
"""

from .models import ResearchInsight
from .deep_research import ResearchMode, run_linear_research, run_cot_research, run_aot_research
from .reasoning_researcher import ReasoningResearcher
from .atomic_researcher import AtomicResearcher

__all__ = [
    "ResearchInsight",
    "ResearchMode",
    "run_linear_research",
    "run_cot_research",
    "run_aot_research",
    "ReasoningResearcher",
    "AtomicResearcher",
]
