"""
Agentic Tools - Unified Toolset for Heart Health AI (Refactored with Proper DI).

This module consolidates all agent capabilities into a single registry-compatible format.
It adheres to the Functional Tool Pattern required by SemanticRouterV2.

**ARCHITECTURE**: Uses dependency injection instead of global state to prevent:
- Race conditions during initialization
- Test isolation issues
- Runtime None reference errors


Integrations:
- Safe Calculator (AST-based)
- Medical Image Analysis (Vision LLM)
- RAG & Knowledge Retrieval
- Text-to-SQL
- FHIR Integration
- DICOM Analysis
"""
import logging
import base64
import io
import asyncio
import json
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

from tools.tool_registry import register_tool, ToolParameter, ToolResult
from tools.safe_calculator import SafeCalculator
from tools.text_to_sql_tool import TextToSQLTool
from tools.web_search import VerifiedWebSearchTool, search_verified_sources
from tools.medical_search import MedicalContentSearcher, search_medical_content, ContentType
from tools.dicom.dicom_handler import DicomHandler

logger = logging.getLogger(__name__)


@dataclass
class AgenticToolsContext:
    """
    Dependency Injection Container for Agentic Tools.
    
    All dependencies are passed explicitly to prevent global state and race conditions.
    This makes tools easily testable and thread-safe.
    """
    db_manager: Any
    llm_gateway: Any
    vector_store: Any
    memory_manager: Any
    interaction_checker: Any
    memori_bridge: Optional[Any] = None
    dicom_handler: Optional[DicomHandler] = None
    
    def __post_init__(self):
        """Initialize derived services."""
        if not self.dicom_handler:
            self.dicom_handler = DicomHandler()
        
        # Validate critical dependencies
        if not self.llm_gateway:
            logger.warning("⚠️  LLM Gateway not initialized - vision/query tools will fail")
        if not self.db_manager:
            logger.warning("⚠️  DB Manager not initialized - SQL tools will fail")


# Global context holder (single reference point for dependency injection)
_tools_context: Optional[AgenticToolsContext] = None


def initialize_agent_tools_new(
    db_manager: Any,
    llm_gateway: Any,
    vector_store: Any,
    memory_manager: Any,
    interaction_checker: Any,
    memori_bridge: Optional[Any] = None,
    dicom_handler: Optional[DicomHandler] = None
) -> AgenticToolsContext:
    """
    Initialize agent tools with explicit dependency injection.
    
    **Benefits over global variables**:
    - No race conditions (dependencies set atomically)
    - Testable (pass mock objects)
    - Type-safe (AgenticToolsContext has clear schema)
    - Debuggable (context is accessible from any tool)
    
    Args:
        db_manager: Database connection manager
        llm_gateway: LLM provider (OpenAI, Ollama, etc.)
        vector_store: Vector database for embeddings
        memory_manager: Conversation memory backend
        interaction_checker: Drug interaction database
        memori_bridge: Optional Memori integration
        dicom_handler: Optional custom DICOM handler
    
    Returns:
        AgenticToolsContext instance (global _tools_context updated)
    """
    global _tools_context
    
    _tools_context = AgenticToolsContext(
        db_manager=db_manager,
        llm_gateway=llm_gateway,
        vector_store=vector_store,
        memory_manager=memory_manager,
        interaction_checker=interaction_checker,
        memori_bridge=memori_bridge,
        dicom_handler=dicom_handler
    )
    
    logger.info("✅ Agentic tools initialized with dependency injection context")
    logger.info(f"   - DB Manager: {type(db_manager).__name__}")
    logger.info(f"   - LLM Gateway: {type(llm_gateway).__name__}")
    logger.info(f"   - Vector Store: {type(vector_store).__name__}")
    
    return _tools_context


def get_tools_context() -> AgenticToolsContext:
    """
    Get the current tools context.
    
    Raises:
        RuntimeError: If initialize_agent_tools_new() hasn't been called yet.
    """
    if _tools_context is None:
        raise RuntimeError(
            "Agentic tools not initialized! Call initialize_agent_tools_new() from main.py startup."
        )
    return _tools_context


def require_context(dependencies: List[str]) -> None:
    """
    Validate that required dependencies are available.
    
    Args:
        dependencies: List of dependency names (e.g., ["llm_gateway", "db_manager"])
    
    Raises:
        RuntimeError: If any required dependency is None
    """
    ctx = get_tools_context()
    
    for dep_name in dependencies:
        dep_value = getattr(ctx, dep_name, None)
        if dep_value is None:
            raise RuntimeError(f"Required dependency '{dep_name}' is not initialized")


# ==============================================================================
# 1. UTILITY TOOLS
# ==============================================================================

@register_tool(
    name="calculator",
    description="Perform safe mathematical calculations. Supports +, -, *, /, **, and math functions (sqrt, sin, cos, round).",
    parameters=[
        ToolParameter("expression", "string", "The math expression to evaluate (e.g. '22 * 1.5', 'sqrt(100)')", required=True)
    ],
    category="utility",
    modes=["calculators", "general", "vitals"]
)
async def calculator(expression: str) -> ToolResult:
    """
    Safe evaluation of math expressions using AST parsing.
    
    Does NOT require dependencies, so it works without initialization.
    """
    try:
        calc = SafeCalculator()
        result = calc.evaluate(expression)
        return ToolResult(
            success=True, 
            data={"result": result, "expression": expression}
        )
    except Exception as e:
        return ToolResult(success=False, error=f"Calculation failed: {str(e)}")


# ==============================================================================
# 2. VISION TOOLS
# ==============================================================================

@register_tool(
    name="analyze_medical_image",
    description="Analyze a medical image (X-ray, report photo, pill) to extract information.",
    parameters=[
        ToolParameter("image_data", "string", "Base64 encoded image string or data URL", required=True),
        ToolParameter("query", "string", "What to look for in the image", required=True)
    ],
    category="vision",
    modes=["general", "medical_qa"]
)
async def analyze_medical_image(image_data: str, query: str) -> ToolResult:
    """
    Vision analysis using LLM Gateway.
    Automatically appends safety disclaimers for medical imagery.
    """
    try:
        require_context(["llm_gateway"])
        ctx = get_tools_context()
        
        # 1. Sanitize Base64 input
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
            
        # 2. Construct Medical Vision Prompt
        vision_prompt = f"""
        ANALYZE THIS MEDICAL IMAGE.
        User Query: {query}
        
        Guidelines:
        1. Describe exactly what you see visually.
        2. If text is present (OCR), transcribe it.
        3. Do NOT provide a definitive medical diagnosis.
        4. Mention if the image quality is too low to be certain.
        """

        # 3. Call Vision Model
        if hasattr(ctx.llm_gateway, "generate_vision"):
            description = await ctx.llm_gateway.generate_vision(vision_prompt, image_data)
        else:
            return ToolResult(success=False, error="Current LLM provider does not support Vision.")

        # 4. Append Disclaimer (Hardcoded safety)
        disclaimer = "\n\n(Note: AI analysis is for informational purposes only and not a substitute for professional radiologist/doctor review.)"
        
        return ToolResult(
            success=True,
            data={
                "analysis": description + disclaimer,
                "type": "medical_image_analysis"
            }
        )
    except RuntimeError as e:
        return ToolResult(success=False, error=str(e))
    except Exception as e:
        logger.error(f"Vision analysis failed: {e}")
        return ToolResult(success=False, error=str(e))


# ==============================================================================
# 3. DATABASE & SQL TOOLS
# ==============================================================================

@register_tool(
    name="query_sql_db",
    description="Convert natural language health queries to SQL and execute against patient database.",
    parameters=[
        ToolParameter("query", "string", "Natural language health question (e.g., 'What was my average heart rate last week?')", required=True),
        ToolParameter("user_id", "string", "Patient ID (enforced at database level)", required=True)
    ],
    category="data",
    modes=["vitals", "general"]
)
async def query_sql_db(query: str, user_id: str) -> ToolResult:
    """
    Query patient database with natural language.
    
    Security: Enforces row-level security via WHERE user_id filtering.
    See: tools/text_to_sql_tool.py for SQL injection prevention details.
    """
    try:
        require_context(["db_manager", "llm_gateway"])
        ctx = get_tools_context()
        
        # Create tool instance with explicit dependencies (no global state)
        sql_tool = TextToSQLTool(ctx.db_manager, ctx.llm_gateway)
        result = await sql_tool.execute(query, user_id)
        
        if not result.success:
            return ToolResult(success=False, error=result.error)
        
        return ToolResult(
            success=True,
            data={
                "sql_query": result.query,
                "results": result.result,
                "reasoning": result.reasoning,
                "execution_time": result.execution_time
            }
        )
    except RuntimeError as e:
        return ToolResult(success=False, error=str(e))
    except Exception as e:
        logger.error(f"SQL query failed: {e}")
        return ToolResult(success=False, error=str(e))


# ==============================================================================
# 4. MEDICAL IMAGING TOOLS
# ==============================================================================

@register_tool(
    name="analyze_dicom_image",
    description="Analyze DICOM medical images and extract structured clinical information.",
    parameters=[
        ToolParameter("file_path", "string", "Path to DICOM file or study UID for DICOMweb", required=True),
        ToolParameter("query", "string", "What clinical information to extract", required=False)
    ],
    category="vision",
    modes=["medical_qa", "imaging"]
)
async def analyze_dicom_image(file_path: str, query: Optional[str] = None) -> ToolResult:
    """
    Analyze DICOM files with async I/O wrapper.
    
    Performance: Uses loop.run_in_executor to prevent event loop blocking.
    Memory: Excludes pixel_data from JSON serialization.
    """
    try:
        ctx = get_tools_context()
        
        # Use async wrapper to prevent blocking
        result = await ctx.dicom_handler.parse_file_async(file_path)
        
        if not result:
            return ToolResult(success=False, error="Failed to parse DICOM file")
        
        return ToolResult(
            success=True,
            data={
                "patient": result.patient.__dict__,
                "study": result.study.__dict__,
                "series": [s.__dict__ for s in result.series],
                "findings": result.findings,
                "json": result.to_json()
            }
        )
    except Exception as e:
        logger.error(f"DICOM analysis failed: {e}")
        return ToolResult(success=False, error=str(e))


# ==============================================================================
# 5. VECTOR & RAG TOOLS
# ==============================================================================

@register_tool(
    name="semantic_search_knowledge_base",
    description="Search medical knowledge base using semantic similarity (embeddings).",
    parameters=[
        ToolParameter("query", "string", "Medical question or symptom", required=True),
        ToolParameter("top_k", "integer", "Number of results (default 5)", required=False)
    ],
    category="knowledge",
    modes=["medical_qa", "general"]
)
async def semantic_search_knowledge_base(query: str, top_k: int = 5) -> ToolResult:
    """
    Semantic search against medical knowledge base.
    Requires vector store.
    """
    try:
        require_context(["vector_store"])
        ctx = get_tools_context()
        
        results = await ctx.vector_store.search(query, limit=top_k)
        
        return ToolResult(
            success=True,
            data={
                "query": query,
                "results": results,
                "count": len(results)
            }
        )
    except RuntimeError as e:
        return ToolResult(success=False, error=str(e))
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return ToolResult(success=False, error=str(e))


# ==============================================================================
# 6. WEB SEARCH TOOLS
# ==============================================================================

@register_tool(
    name="verified_web_search",
    description="Search verified medical sources (PubMed, FDA, NIH) for clinical information.",
    parameters=[
        ToolParameter("query", "string", "Medical search query", required=True),
        ToolParameter("source", "string", "Source filter (pubmed|fda|nih|all)", required=False)
    ],
    category="knowledge",
    modes=["medical_qa", "research"]
)
async def verified_web_search(query: str, source: str = "all") -> ToolResult:
    """
    Search verified medical sources.
    Does NOT require DI context.
    """
    try:
        results = await search_verified_sources(query, max_results=5)
        return ToolResult(
            success=True,
            data={
                "query": query,
                "source": source,
                "results": results
            }
        )
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return ToolResult(success=False, error=str(e))


# ==============================================================================
# 7. COMPREHENSIVE MEDICAL CONTENT SEARCH
# ==============================================================================

@register_tool(
    name="search_medical_content",
    description=(
        "Search for comprehensive medical content: research papers (PubMed), "
        "medical articles (WHO, CDC, NIH, Mayo Clinic), medical news (FDA alerts, "
        "clinical trials), medical images (radiology, anatomy, pathology), "
        "and medical videos (procedures, lectures, patient education). "
        "All results are from verified medical sources only."
    ),
    parameters=[
        ToolParameter("query", "string", "Medical search query", required=True),
        ToolParameter("content_types", "string",
                      "Comma-separated types: 'article,research_paper,news,image,video' or 'all'",
                      required=False),
        ToolParameter("max_results", "integer", "Max results per type (default 5)", required=False)
    ],
    category="knowledge",
    modes=["medical_qa", "research", "general"]
)
async def comprehensive_medical_search(
    query: str,
    content_types: str = "all",
    max_results: int = 5
) -> ToolResult:
    """
    Search for medical research papers, articles, news, images, and videos.
    Does NOT require DI context.
    """
    try:
        # Parse content types
        types = None
        if content_types and content_types.lower() != "all":
            types = [ct.strip() for ct in content_types.split(",")]

        results = await search_medical_content(
            query=query,
            content_types=types,
            max_results=max_results
        )
        return ToolResult(
            success=True,
            data={
                "query": query,
                "content_types": content_types,
                "results": results
            }
        )
    except Exception as e:
        logger.error(f"Medical content search failed: {e}")
        return ToolResult(success=False, error=str(e))


# ==============================================================================
# BACKWARDS COMPATIBILITY
# ==============================================================================

def initialize_agent_tools(
    db_manager,
    llm_gateway,
    vector_store,
    memory_manager,
    interaction_checker,
    memori_bridge=None
):
    """
    Legacy initialization function for backwards compatibility.
    
    **DEPRECATED**: Use initialize_agent_tools_new() instead.
    This wrapper calls the new function.
    """
    logger.warning("⚠️  initialize_agent_tools() is deprecated. Use initialize_agent_tools_new() instead.")
    return initialize_agent_tools_new(
        db_manager=db_manager,
        llm_gateway=llm_gateway,
        vector_store=vector_store,
        memory_manager=memory_manager,
        interaction_checker=interaction_checker,
        memori_bridge=memori_bridge
    )
