"""
Graph Visualization Route - Expose agent workflow graph as Mermaid diagram.

Endpoints:
- GET /graph/mermaid - Returns Mermaid diagram text
- GET /graph/json - Returns graph structure as JSON
- GET /graph/metrics - Returns tracing/metrics summary
"""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, JSONResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/mermaid", response_class=PlainTextResponse, summary="Get Mermaid diagram of orchestrator graph")
async def get_graph_mermaid():
    """Returns the orchestrator agent graph as a Mermaid diagram string."""
    try:
        from agents.utils.visualization import get_orchestrator_graph
        mermaid = get_orchestrator_graph()
        return PlainTextResponse(content=mermaid, media_type="text/plain")
    except Exception as e:
        logger.error(f"Failed to generate Mermaid graph: {e}")
        return PlainTextResponse(content=f"Error: {e}", status_code=500)


@router.get("/json", summary="Get orchestrator graph as JSON structure")
async def get_graph_json():
    """Returns the orchestrator graph nodes/edges as JSON for programmatic use."""
    try:
        from agents.utils.visualization import GraphVisualizer
        viz = GraphVisualizer()

        graph_def = {
            "nodes": [
                {"name": "START", "type": "start", "description": "Entry point"},
                {"name": "router", "type": "conditional", "description": "SemanticRouterV2 intent classification"},
                {"name": "supervisor", "type": "supervisor", "description": "LLM-based task delegation"},
                {"name": "medical_analyst", "type": "worker", "description": "RAG for medical queries"},
                {"name": "researcher", "type": "worker", "description": "Deep research & web search"},
                {"name": "data_analyst", "type": "worker", "description": "SQL queries for vitals"},
                {"name": "drug_expert", "type": "worker", "description": "Drug interactions via GraphRAG"},
                {"name": "heart_analyst", "type": "worker", "description": "Heart disease risk assessment"},
                {"name": "thinking_agent", "type": "worker", "description": "Deep reasoning with tool use"},
                {"name": "fhir_agent", "type": "worker", "description": "EHR/FHIR data queries"},
                {"name": "clinical_reasoning", "type": "worker", "description": "Differential diagnosis & triage"},
                {"name": "profile_manager", "type": "worker", "description": "User profile management"},
                {"name": "END", "type": "end", "description": "Exit point"},
            ],
            "edges": [
                {"from": "START", "to": "router"},
                {"from": "router", "to": "supervisor", "label": "complex", "conditional": True},
                {"from": "router", "to": "medical_analyst", "label": "medical", "conditional": True},
                {"from": "router", "to": "data_analyst", "label": "vitals", "conditional": True},
                {"from": "router", "to": "drug_expert", "label": "drug", "conditional": True},
                {"from": "router", "to": "thinking_agent", "label": "reasoning", "conditional": True},
                {"from": "router", "to": "researcher", "label": "research", "conditional": True},
                {"from": "router", "to": "heart_analyst", "label": "heart", "conditional": True},
                {"from": "router", "to": "clinical_reasoning", "label": "clinical", "conditional": True},
                {"from": "supervisor", "to": "medical_analyst"},
                {"from": "supervisor", "to": "researcher"},
                {"from": "supervisor", "to": "data_analyst"},
                {"from": "supervisor", "to": "drug_expert"},
                {"from": "supervisor", "to": "heart_analyst"},
                {"from": "supervisor", "to": "thinking_agent"},
                {"from": "supervisor", "to": "fhir_agent"},
                {"from": "supervisor", "to": "clinical_reasoning"},
                {"from": "supervisor", "to": "profile_manager"},
                {"from": "medical_analyst", "to": "supervisor"},
                {"from": "researcher", "to": "supervisor"},
                {"from": "data_analyst", "to": "supervisor"},
                {"from": "drug_expert", "to": "supervisor"},
                {"from": "heart_analyst", "to": "supervisor"},
                {"from": "thinking_agent", "to": "supervisor"},
                {"from": "fhir_agent", "to": "supervisor"},
                {"from": "clinical_reasoning", "to": "supervisor"},
                {"from": "profile_manager", "to": "supervisor"},
                {"from": "supervisor", "to": "END", "label": "FINISH", "conditional": True},
            ]
        }

        return JSONResponse(content=graph_def)
    except Exception as e:
        logger.error(f"Failed to generate graph JSON: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/metrics", summary="Get tracing and performance metrics summary")
async def get_graph_metrics():
    """Returns aggregated tracing metrics and Prometheus metrics summary."""
    result = {}

    # Tracing metrics
    try:
        from core.observability.tracing import get_tracer
        tracer = get_tracer()
        result["tracing"] = tracer.get_metrics()
        result["recent_traces"] = tracer.get_recent_traces(limit=5)
    except Exception as e:
        result["tracing"] = {"error": str(e)}

    # Prometheus metrics
    try:
        from core.monitoring.prometheus_metrics import get_metrics
        metrics = get_metrics()
        result["prometheus"] = metrics.get_metrics_dict()
    except Exception as e:
        result["prometheus"] = {"error": str(e)}

    return JSONResponse(content=result)
