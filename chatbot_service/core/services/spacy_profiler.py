"""
SpaCy Profiler
==============
Profile spaCy pipeline performance to identify bottlenecks.
"""


import time
from typing import Dict, Any, List
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class SpaCyProfiler:
    """Profile spaCy pipeline performance."""
    
    def __init__(self, nlp):
        self.nlp = nlp
        self.stats = {}
    
    @contextmanager
    def profile_component(self, name: str):
        """Context manager to profile a component."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        if name not in self.stats:
            self.stats[name] = []
        self.stats[name].append(elapsed)
    
    def profile_pipeline(self, texts: List[str]) -> Dict[str, Any]:
        """Profile full pipeline on sample texts."""
        results = {}
        
        for name in self.nlp.pipe_names:
            times = []
            for text in texts:
                # Time each component
                try:
                    with self.nlp.select_pipes(enable=[name]):
                        start = time.perf_counter()
                        self.nlp(text)
                        elapsed = time.perf_counter() - start
                        times.append(elapsed)
                except Exception as e:
                    logger.warning(f"Profiling failed for {name}: {e}")
            
            if times:
                results[name] = {
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "total": sum(times)
                }
        
        return results
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest pipeline optimizations based on profiling."""
        suggestions = []
        
        if "parser" in self.stats and not self._needs_parser():
            suggestions.append("Consider disabling 'parser' if not using dependency trees")
        
        if "tagger" in self.stats and not self._needs_tagger():
            suggestions.append("Consider disabling 'tagger' if not using POS tags")
        
        return suggestions
    
    def _needs_parser(self) -> bool:
        # Heuristic check
        return True
    
    def _needs_tagger(self) -> bool:
        # Heuristic check
        return True
