"""
Agents module for Cardio AI Chatbot Service.

Structure:
- langgraph_orchestrator: Main LangGraph-based orchestrator
- components/: Reusable agent building blocks
  - thinking: Chain-of-thought reasoning
  - vision: Image analysis
  - planning: Multi-step planning
  - managed: Sub-agent coordination
- utils/: Helper utilities
  - visualization: Graph diagrams
"""

# Core orchestrator
from .langgraph_orchestrator import LangGraphOrchestrator

# Components
from .components import (
    # Thinking
    ThinkingAgent,
    ThinkingBlock,
    ThinkingResult,
    ReasoningType,
    
    # Vision
    VisionCapableMixin,
    MedicalImageAnalyzer,
    ImageInput,
    
    # Planning
    PlanningMixin,
    PlanStep,
    PlanStepStatus,
    ExecutionPlan,
    
    # Managed
    ManagedAgent,
    AgentManager,
    DelegationResult,
)

# Utils
from .utils import (
    GraphVisualizer,
    GraphNode,
    GraphEdge,
    get_orchestrator_graph,
)

# Evaluation (kept in agents/ root)
from .evaluation import (
    ResponseEvaluator,
    EvaluationResult,
    EvaluationSummary,
    EvaluationMetric,
    BatchEvaluator,
    create_response_evaluator,
)

__all__ = [
    # Orchestrator
    "LangGraphOrchestrator",
    
    # Components
    "ThinkingAgent",
    "ThinkingBlock",
    "ThinkingResult",
    "ReasoningType",
    "VisionCapableMixin",
    "MedicalImageAnalyzer",
    "ImageInput",
    "PlanningMixin",
    "PlanStep",
    "PlanStepStatus",
    "ExecutionPlan",
    "ManagedAgent",
    "AgentManager",
    "DelegationResult",
    
    # Utils
    "GraphVisualizer",
    "GraphNode",
    "GraphEdge",
    "get_orchestrator_graph",
    
    # Evaluation
    "ResponseEvaluator",
    "EvaluationResult",
    "EvaluationSummary",
    "EvaluationMetric",
    "BatchEvaluator",
    "create_response_evaluator",
]
