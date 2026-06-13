"""
Analyzer Protocol and Registry for Pluggable NLP Analyzers

Enables:
- Adding new analyzers without modifying core code
- Swapping implementations for testing
- Health checks across all analyzers
- Unified interface for all analyzer types
- Model versioning per analyzer

Pattern: Strategy Pattern + Registry Pattern
Best for: Extensibility, testing, A/B testing
"""


import logging
import time
import asyncio
from typing import Protocol, Any, Dict, List, Optional, runtime_checkable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AnalyzerType(Enum):
    """Types of NLP analyzers"""

    INTENT = "intent"
    SENTIMENT = "sentiment"
    ENTITY = "entity"
    RISK = "risk"
    CUSTOM = "custom"


@dataclass
class AnalysisResult:
    """
    Standardized result from any analyzer.

    Attributes:
        analyzer_name: Name of the analyzer that produced this result
        analyzer_type: Type of analysis (intent, sentiment, etc)
        confidence: Confidence score (0-1)
        details: Type-specific analysis details
        processing_time_ms: Time taken for analysis
        model_version: Version of model used
        error: Optional error if analysis failed
    """

    analyzer_name: str
    analyzer_type: AnalyzerType
    confidence: float
    details: Dict[str, Any]
    processing_time_ms: float
    model_version: str = "1.0"
    error: Optional[str] = None

    @property
    def is_error(self) -> bool:
        """Check if analysis failed"""
        return self.error is not None

    @property
    def is_confident(self, threshold: float = 0.6) -> bool:
        """Check if confidence exceeds threshold"""
        return self.confidence >= threshold


@runtime_checkable
class Analyzer(Protocol):
    """
    Protocol interface for all NLP analyzers.

    Any class implementing these methods can be used as an analyzer:
    - name: str property
    - version: str property
    - analyze: async method
    - health_check: async method

    Example:
        class CustomAnalyzer:
            @property
            def name(self) -> str:
                return "my_analyzer"

            @property
            def version(self) -> str:
                return "1.0"

            async def analyze(self, text: str) -> AnalysisResult:
                ...

            async def health_check(self) -> bool:
                ...
    """

    @property
    def name(self) -> str:
        """Unique analyzer name"""
        ...

    @property
    def version(self) -> str:
        """Model version"""
        ...

    async def analyze(self, text: str) -> AnalysisResult:
        """
        Analyze text and return standardized result.

        Args:
            text: Input text to analyze

        Returns:
            AnalysisResult with details
        """
        ...

    async def health_check(self) -> bool:
        """
        Check if analyzer is healthy and ready to serve.

        Returns:
            True if healthy, False if degraded
        """
        ...


class BaseAnalyzer(ABC):
    """
    Abstract base class for implementing analyzers.

    Provides common functionality:
    - Error handling
    - Timing measurement
    - Health tracking
    """

    def __init__(self, name: str, analyzer_type: AnalyzerType):
        self._name = name
        self._analyzer_type = analyzer_type
        self._version = "1.0"
        self._last_error: Optional[str] = None
        self._error_count = 0
        self._success_count = 0
        self._is_healthy = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def type(self) -> AnalyzerType:
        return self._analyzer_type

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        total = self._success_count + self._error_count
        if total == 0:
            return 0.0
        return self._error_count / total

    @abstractmethod
    async def _do_analyze(self, text: str) -> AnalysisResult:
        """Subclass implementation of analysis"""
        ...

    async def analyze(self, text: str) -> AnalysisResult:
        """
        Analyze with error handling and tracking.

        Args:
            text: Input text

        Returns:
            AnalysisResult (may have error set)
        """
        start_time = time.time()

        try:
            result = await self._do_analyze(text)
            self._success_count += 1
            self._is_healthy = True
            return result

        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)

            # Mark unhealthy if error rate too high
            if self.error_rate > 0.1:  # 10% error rate
                self._is_healthy = False

            elapsed_ms = (time.time() - start_time) * 1000

            logger.error(
                f"Analyzer '{self.name}' error: {e}",
                extra={
                    "analyzer": self.name,
                    "error_type": type(e).__name__,
                    "processing_time_ms": elapsed_ms,
                },
            )

            # Return error result
            return AnalysisResult(
                analyzer_name=self.name,
                analyzer_type=self.type,
                confidence=0.0,
                details={},
                processing_time_ms=elapsed_ms,
                model_version=self.version,
                error=str(e),
            )

    async def health_check(self) -> bool:
        """
        Check analyzer health.

        Returns:
            True if healthy, False if degraded
        """
        return self._is_healthy and self.error_rate < 0.1

    def reset_metrics(self) -> None:
        """Reset error tracking metrics"""
        self._success_count = 0
        self._error_count = 0
        self._last_error = None
        self._is_healthy = True


class AnalyzerRegistry:
    """
    Registry for all analyzers in the system.

    Features:
    - Register analyzers dynamically
    - Run all analyzers in parallel
    - Health checks for all
    - Analyzer lookup by name or type

    Example:
        registry = AnalyzerRegistry()
        registry.register(intent_analyzer)
        registry.register(sentiment_analyzer)

        results = await registry.run_all(text)
        health = await registry.health_check()
    """

    def __init__(self):
        self._analyzers: Dict[str, Analyzer] = {}
        self._by_type: Dict[AnalyzerType, List[str]] = {}

    def register(self, analyzer: Analyzer) -> None:
        """
        Register an analyzer.

        Args:
            analyzer: Analyzer instance (must implement Analyzer protocol)

        Raises:
            ValueError: If analyzer doesn't implement protocol
            ValueError: If analyzer name already registered
        """
        # Verify analyzer implements protocol
        if not isinstance(analyzer, Analyzer):
            raise ValueError(
                f"Analyzer must implement Analyzer protocol. " f"Got {type(analyzer)}"
            )

        name = analyzer.name

        if name in self._analyzers:
            raise ValueError(
                f"Analyzer '{name}' already registered. "
                "Use unregister() first to replace."
            )

        self._analyzers[name] = analyzer

        # Track by type if BaseAnalyzer
        if isinstance(analyzer, BaseAnalyzer):
            analyzer_type = analyzer.type
            if analyzer_type not in self._by_type:
                self._by_type[analyzer_type] = []
            self._by_type[analyzer_type].append(name)

        logger.info(f"Registered analyzer: {name}")

    def unregister(self, analyzer_name: str) -> None:
        """Unregister an analyzer"""
        if analyzer_name not in self._analyzers:
            raise ValueError(f"Analyzer '{analyzer_name}' not registered")

        del self._analyzers[analyzer_name]

        # Remove from type index
        for analyzer_list in self._by_type.values():
            if analyzer_name in analyzer_list:
                analyzer_list.remove(analyzer_name)

        logger.info(f"Unregistered analyzer: {analyzer_name}")

    def get(self, analyzer_name: str) -> Analyzer:
        """
        Get analyzer by name.

        Args:
            analyzer_name: Name of analyzer

        Returns:
            Analyzer instance

        Raises:
            ValueError: If analyzer not found
        """
        if analyzer_name not in self._analyzers:
            raise ValueError(
                f"Analyzer '{analyzer_name}' not found. "
                f"Available: {list(self._analyzers.keys())}"
            )
        return self._analyzers[analyzer_name]

    def get_by_type(self, analyzer_type: AnalyzerType) -> List[Analyzer]:
        """Get all analyzers of a specific type"""
        names = self._by_type.get(analyzer_type, [])
        return [self._analyzers[name] for name in names]

    def list_analyzers(self) -> Dict[str, Dict[str, str]]:
        """
        List all registered analyzers with metadata.

        Returns:
            Dict mapping analyzer name to metadata
        """
        return {
            name: {
                "version": analyzer.version,
                "type": (
                    analyzer.type.value
                    if isinstance(analyzer, BaseAnalyzer)
                    else "unknown"
                ),
            }
            for name, analyzer in self._analyzers.items()
        }

    async def run_all(self, text: str) -> Dict[str, AnalysisResult]:
        """
        Run all registered analyzers on text in parallel.

        Args:
            text: Input text to analyze

        Returns:
            Dict mapping analyzer name to AnalysisResult
        """
        if not self._analyzers:
            logger.warning("No analyzers registered")
            return {}

        # Create tasks for all analyzers
        tasks = {
            name: analyzer.analyze(text) for name, analyzer in self._analyzers.items()
        }

        # Run in parallel with error handling
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Map results back to analyzer names
        output = {}
        for (name, _), result in zip(tasks.items(), results):
            if isinstance(result, Exception):
                output[name] = AnalysisResult(
                    analyzer_name=name,
                    analyzer_type=AnalyzerType.CUSTOM,
                    confidence=0.0,
                    details={},
                    processing_time_ms=0.0,
                    error=str(result),
                )
            else:
                output[name] = result

        return output

    async def health_check(self) -> Dict[str, Dict[str, Any]]:
        """
        Check health of all analyzers.

        Returns:
            Dict with health status per analyzer
        """
        health_results = {}

        for name, analyzer in self._analyzers.items():
            try:
                is_healthy = await analyzer.health_check()
                health_results[name] = {
                    "healthy": is_healthy,
                    "version": analyzer.version,
                }

                if isinstance(analyzer, BaseAnalyzer):
                    health_results[name].update(
                        {
                            "error_rate": analyzer.error_rate,
                            "last_error": analyzer.last_error,
                        }
                    )

            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                health_results[name] = {
                    "healthy": False,
                    "version": analyzer.version,
                    "error": str(e),
                }

        return health_results

    async def warm_up(self) -> None:
        """
        Warm up all analyzers with dummy analysis.

        Useful for:
        - Loading models into memory
        - Pre-compiling code
        - Validating setup
        """
        logger.info("Warming up analyzers...")

        dummy_text = "This is a test message for warming up."

        for name in self._analyzers:
            try:
                await self._analyzers[name].analyze(dummy_text)
                logger.info(f"Warmed up: {name}")
            except Exception as e:
                logger.warning(f"Warm-up failed for {name}: {e}")

    def reset_metrics(self) -> None:
        """Reset metrics for all analyzers"""
        for analyzer in self._analyzers.values():
            if isinstance(analyzer, BaseAnalyzer):
                analyzer.reset_metrics()

    @property
    def analyzer_count(self) -> int:
        """Get total number of registered analyzers"""
        return len(self._analyzers)


# Global analyzer registry
_global_registry: Optional[AnalyzerRegistry] = None


def initialize_global_registry() -> AnalyzerRegistry:
    """Create and return global registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = AnalyzerRegistry()
    return _global_registry


def get_analyzer_registry() -> AnalyzerRegistry:
    """Get global analyzer registry"""
    global _global_registry
    if _global_registry is None:
        raise RuntimeError(
            "Registry not initialized. Call initialize_global_registry()"
        )
    return _global_registry


__all__ = [
    "Analyzer",
    "BaseAnalyzer",
    "AnalysisResult",
    "AnalyzerType",
    "AnalyzerRegistry",
    "initialize_global_registry",
    "get_analyzer_registry",
]
