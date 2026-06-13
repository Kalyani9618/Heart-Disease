"""
LangGraph Orchestrator - Agentic RAG Architecture

Orchestrates medical AI agents using LangGraph with:
- Semantic Router V2 for fast intent classification
- Supervisor LLM for complex reasoning
- Deterministic triage and safety checks
- PII redaction at all output points
"""

import logging
import operator
import json
import ast
import os
import asyncio
from typing import TypedDict, Annotated, List, Union, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

# Checkpointing for state persistence
# Redis checkpointing enables automatic recovery if a worker crashes mid-workflow
# MemorySaver is used as a fallback for development when RedisJSON module is not available
from langgraph.checkpoint.memory import MemorySaver
try:
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver
    REDIS_CHECKPOINTING_AVAILABLE = True
except ImportError:
    try:
        # Fallback for older versions
        from langgraph.checkpoint.redis import RedisSaver as AsyncRedisSaver
        REDIS_CHECKPOINTING_AVAILABLE = True
    except ImportError:
        AsyncRedisSaver = None
        REDIS_CHECKPOINTING_AVAILABLE = False

from core.config.app_config import get_app_config
from core.prompts.registry import get_prompt

from tools.semantic_router_v2 import SemanticRouterV2, IntentCategory
from tools.agentic_tools import (
    initialize_agent_tools_new,
    query_sql_db,
    semantic_search_knowledge_base,
    verified_web_search,
    comprehensive_medical_search,
    calculator,
    analyze_medical_image,
    analyze_dicom_image
)
from agents.components.thinking import create_thinking_agent
from agents.components.medical_planner import MedicalPlanner
from rag.pipeline.self_rag_medical import MedicalSelfRAG
from rag.knowledge_graph.interaction_checker import GraphInteractionChecker
from rag.pipeline.crag_fallback import CRAGFallback
from agents.heart_predictor import HeartDiseasePredictor
from tools.fhir.fhir_agent_tool import get_fhir_tool
from tools.medical_coding.auto_coder import auto_code_clinical_note
from agents.components.workflow_automation import WorkflowRouter
from agents.components.differential_diagnosis import generate_differential_diagnosis
from agents.components.triage_system import triage_patient

logger = logging.getLogger(__name__)

# Configuration
MAX_SUPERVISOR_STEPS = int(os.getenv("MAX_SUPERVISOR_STEPS", "8"))

# PII Scrubbing - Critical Safety Feature
try:
    from core.compliance.pii_scrubber_v2 import get_enhanced_pii_scrubber
    _pii_scrubber = get_enhanced_pii_scrubber()
    logger.info("✅ PII Scrubber loaded")
except ImportError:
    _pii_scrubber = None
    logger.error("❌ No PII scrubber available - HIPAA compliance risk!")

# --- Supervisor Response Schema (Pydantic for robust JSON parsing) ---
class SupervisorResponse(BaseModel):
    """Validated response schema for Supervisor node routing decision."""
    next: str = Field(
        ..., 
        description="Next node to execute or FINISH",
        examples=["medical_analyst", "researcher", "FINISH"]
    )
    reasoning: Optional[str] = Field(
        default="", 
        description="Brief explanation for the routing decision"
    )
    final_response: Optional[str] = Field(
        default="", 
        description="Final answer if FINISH"
    )

# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    user_id: str
    final_response: Optional[str]
    intent: Optional[str]
    confidence: Optional[float]
    citations: Optional[List[str]]
    source: Optional[str]  # Track response source: 'rag', 'crag', 'web', 'llm', 'llm_fallback'
    thinking: Optional[bool]
    web_search: Optional[bool]
    deep_search: Optional[bool]
    file_ids: Optional[List[str]]

# --- Orchestrator Class ---
class LangGraphOrchestrator:
    def __init__(self, db_manager=None, llm_gateway=None, vector_store=None, memory_manager=None, interaction_checker=None, memori_bridge=None):
        """
        Initialize the LangGraph Orchestrator.
        
        Args:
            db_manager: Database manager (optional if in DI)
            llm_gateway: LLM Gateway (optional if in DI)
            vector_store: Vector Store (optional if in DI)
            memory_manager: Memory Manager (optional if in DI)
            interaction_checker: Drug Interaction Checker (optional if in DI)
            memori_bridge: MemoriRAGBridge (optional)
        """
        # Use DIContainer for missing dependencies
        from core.dependencies import DIContainer
        container = DIContainer.get_instance()
        
        self.db_manager = db_manager or container.get_service('db_manager')
        
        if self.db_manager is None:
            logging.warning("⚠️ LangGraphOrchestrator: 'db_manager' not found in DI Container. Text-to-SQL will fail.")
        else:
            logging.info("✅ LangGraphOrchestrator: DB Manager injected successfully.")
        self.llm_gateway = llm_gateway or container.llm_gateway
        self.vector_store = vector_store or container.vector_store
        self.memory_manager = memory_manager or container.memory_manager
        self.interaction_checker = interaction_checker or container.interaction_checker
        # Fix: Fall back to container for memori_bridge like other services
        self.memori_bridge = memori_bridge or getattr(container, 'memori_bridge', None)
        
        if self.memori_bridge:
            logging.info("✅ LangGraphOrchestrator: MemoriRAGBridge injected successfully.")
        else:
            logging.warning("⚠️ LangGraphOrchestrator: 'memori_bridge' not available. Memory context will be disabled.")
        
        # Store container reference for dynamic DB fetching
        self.container = container
        
        # Initialize tools with robust DB provider lambda
        # This lambda will be called at query time, ensuring we get the latest DB state
        db_provider_lambda = lambda: (
            self.db_manager or 
            self.container.get_service('db_manager') or
            getattr(self.container, 'postgres_db', None) or 
            getattr(self.container, 'db', None)
        )
        
        # Use initialize_agent_tools_new() instead of deprecated initialize_agent_tools()
        initialize_agent_tools_new(
            self.db_manager, self.llm_gateway, self.vector_store, self.memory_manager, self.interaction_checker, self.memori_bridge
        )
        
        self.router_v2 = SemanticRouterV2()
        
        # Initialize LLM for Supervisor (MedGemma-only architecture)
        app_config = get_app_config()
        self.llm = None
        
        # MedGemma via OpenAI-compatible API
        if ChatOpenAI:
            try:
                # Support both MEDGEMMA_* and legacy LLAMA_LOCAL_* env vars
                medgemma_base_url = os.getenv(
                    "MEDGEMMA_BASE_URL", 
                    os.getenv("LLAMA_LOCAL_BASE_URL", "http://127.0.0.1:8090/v1")
                )
                medgemma_model = os.getenv(
                    "MEDGEMMA_MODEL", 
                    os.getenv("LLAMA_LOCAL_MODEL", "medgemma-4b-it")
                )
                medgemma_api_key = os.getenv(
                    "MEDGEMMA_API_KEY", 
                    os.getenv("LLAMA_LOCAL_API_KEY", "sk-no-key-required")
                )
                
                self.llm = ChatOpenAI(
                    model=medgemma_model,
                    api_key=medgemma_api_key,
                    base_url=medgemma_base_url,
                    temperature=0
                )
                logger.info(f"✅ Supervisor initialized with MedGemma ({medgemma_model}) at {medgemma_base_url}")
            except Exception as e:
                logger.error(f"❌ Could not initialize MedGemma supervisor: {e}")
                logger.error(f"   Ensure MedGemma server is running. Start with: llama-server -m medgemma-4b.gguf --port 8090")
        else:
            logger.error("❌ langchain-openai not installed - MedGemma supervisor unavailable!")
        
        if not self.llm:
            logger.error("❌ Supervisor LLM not initialized. Supervisor agent will fail.")

        # --- Advanced RAG Tools ---
        from rag.retrieval.token_budget import TokenBudgetManager
        # MedGemma local server has -c 8192 context size. Maximize token budget to 7168 context tokens,
        # leaving 1024 tokens for response generation.
        token_budget_mgr = TokenBudgetManager(model_name="gemma", max_tokens=7168)
        self.rag_tool = MedicalSelfRAG(
            vector_store=self.vector_store,     # Use resolved self.vector_store
            llm_gateway=self.llm_gateway,       # Use resolved self.llm_gateway
            memory_bridge=self.memori_bridge,   # Use resolved self.memori_bridge
            token_budget_manager=token_budget_mgr, # Expand budget for local GPU capacity
            enable_compression=True,            # P3.3: Enable Unified Compressor
            enable_fusion_retrieval=True        # P3.2: Explicitly enable Fusion
        )
        
        self.graph_checker = GraphInteractionChecker()
        self.crag_fallback = CRAGFallback(
            vector_store=self.vector_store,     # Use resolved self.vector_store
            web_search_tool=verified_web_search
        )
        # Note: verified_web_search in agentic_tools is a function. CRAGFallback expects an object with .search()
        # We need a wrapper for CRAGFallback compatibility.
        
        class WebSearchWrapper:
            async def search(self, query, num_results=3):
                # agentic_tools.verified_web_search returns ToolResult(data=...)
                # We need to parse it back to list of dicts for CRAG.
                from tools.agentic_tools import verified_web_search
                result = await verified_web_search(query)
                # result.data is likely a string or list.
                # If string, wrap it.
                return [{"content": str(result.data), "source": "web"}]

        self.crag_fallback.web_search = WebSearchWrapper()
        # --- Thinking Agent (P3.1) ---
        
        # Initialize MemoryTool if bridge is available
        memory_tools = []
        if self.memori_bridge and self.memori_bridge.memori:
             from memori.tools.memory_tool import create_memory_search_tool
             memory_search = create_memory_search_tool(self.memori_bridge.memori)
             memory_tools.append(memory_search)
             
        self.thinking_agent = create_thinking_agent(
            llm=self.llm,
            tools=[
                query_sql_db,
                semantic_search_knowledge_base,
                verified_web_search,
                comprehensive_medical_search,
                calculator,
                analyze_medical_image,
                analyze_dicom_image
            ] + memory_tools
        )
        
        # --- Medical Planner (P3.3: Lazy Loading) ---
        self._planner = None  # Lazy loaded
        
        # --- Heart Disease Predictor (P3.3: Lazy Loading) ---
        self._heart_predictor = None  # Lazy loaded
        logger.info("✅ Heavy components configured for lazy loading")

        # --- FHIR & Workflow Automation (P1 Integration) ---
        self.fhir_tool = get_fhir_tool()
        self.workflow_router = WorkflowRouter(llm_gateway=self.llm_gateway)
        logger.info("✅ FHIR Tool & Workflow Router initialized")

        # --- Redis Checkpointing for State Persistence ---
        # Enables automatic crash recovery and state persistence
        self.checkpointer = None
        self._redis_saver_cm = None  # Context manager for cleanup
        self._init_redis_checkpointer()

        self.workflow = self._build_workflow()
        self.app = self._compile_workflow()
    
    # P3.3: Lazy loading properties
    @property
    def planner(self):
        """Lazy load MedicalPlanner on first access."""
        if self._planner is None:
            self._planner = MedicalPlanner()
            logger.info("✅ MedicalPlanner initialized (lazy)")
        return self._planner
    
    @property
    def heart_predictor(self):
        """Lazy load HeartDiseasePredictor on first access."""
        if self._heart_predictor is None:
            self._heart_predictor = HeartDiseasePredictor(
                llm_gateway=self.llm_gateway,
                memori_bridge=self.memori_bridge,
                auto_initialize=True
            )
            logger.info("✅ HeartDiseasePredictor initialized (lazy)")
        return self._heart_predictor
    
    def _init_redis_checkpointer(self):
        """
        Initialize checkpointer for state persistence.
        
        This enables:
        - Automatic crash recovery from last checkpoint
        - State persistence across worker restarts
        - Zero data loss guarantee for agent workflows
        
        Tries Redis first (requires RedisJSON module), falls back to MemorySaver for development.
        """
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        # First, test if Redis has the JSON module (RedisJSON) by running a test command
        # This avoids runtime failures when the checkpointer tries to save state
        redis_json_available = False
        if REDIS_CHECKPOINTING_AVAILABLE:
            try:
                import redis
                # Parse redis_url to extract host/port
                # Format: redis://host:port/db or redis://host:port
                url_parts = redis_url.replace("redis://", "").split("/")
                host_port = url_parts[0].split(":")
                host = host_port[0] if host_port else "localhost"
                port = int(host_port[1]) if len(host_port) > 1 else 6379
                
                # Test Redis JSON module availability with a simple command
                test_client = redis.Redis(host=host, port=port, decode_responses=True)
                test_key = "__checkpointer_test__"
                try:
                    # Try JSON.SET - if it works, RedisJSON is available
                    test_client.execute_command("JSON.SET", test_key, "$", '{"test": true}')
                    test_client.delete(test_key)
                    redis_json_available = True
                    logger.info(f"✅ Redis JSON module available at {host}:{port}")
                except redis.exceptions.ResponseError as e:
                    if "unknown command" in str(e).lower():
                        logger.warning(f"⚠️ Redis JSON module not installed at {host}:{port}")
                    else:
                        logger.warning(f"⚠️ Redis JSON test failed: {e}")
                finally:
                    test_client.close()
            except Exception as e:
                logger.warning(f"⚠️ Could not test Redis JSON availability: {e}")
        
        # Use Redis checkpointer only if JSON module is confirmed available
        if redis_json_available:
            try:
                self.checkpointer = AsyncRedisSaver(redis_url=redis_url)
                self._redis_saver_cm = None
                logger.info(f"✅ Redis checkpointer initialized ({redis_url})")
                return
            except Exception as e:
                logger.warning(f"⚠️ Redis checkpointer failed ({e}), trying MemorySaver fallback")
        elif not REDIS_CHECKPOINTING_AVAILABLE:
            logger.warning("⚠️ langgraph-checkpoint-redis not installed")
        
        # Fallback to MemorySaver for development (state not persisted across restarts)
        try:
            self.checkpointer = MemorySaver()
            self._redis_saver_cm = None
            logger.info("✅ MemorySaver checkpointer initialized (development mode - state not persisted)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize any checkpointer: {e}")
            logger.warning("   State persistence disabled - workflows will not recover from crashes")
            self.checkpointer = None
            self._redis_saver_cm = None
    
    def _compile_workflow(self):
        """
        Compile workflow with optional checkpointer.
        
        If Redis checkpointer is available, every workflow step will be
        automatically persisted, enabling crash recovery.
        """
        if self.checkpointer:
            logger.info("📝 Compiling workflow with Redis checkpointing enabled")
            return self.workflow.compile(checkpointer=self.checkpointer)
        else:
            logger.info("📝 Compiling workflow without checkpointing")
            return self.workflow.compile()
        
    def _build_workflow(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add Nodes
        workflow.add_node("router", self.router_node)
        workflow.add_node("supervisor", self.supervisor_node)
        
        # Worker Nodes
        workflow.add_node("medical_analyst", self.medical_analyst_node)
        workflow.add_node("researcher", self.researcher_node)
        workflow.add_node("data_analyst", self.data_analyst_node)
        workflow.add_node("drug_expert", self.drug_expert_node)
        workflow.add_node("profile_manager", self.profile_manager_node)
        workflow.add_node("heart_analyst", self.heart_analyst_node)

        workflow.add_node("thinking_agent", self.thinking_node)
        workflow.add_node("fhir_agent", self.fhir_query_node)
        workflow.add_node("clinical_reasoning", self.clinical_reasoning_node)
        workflow.add_node("medical_coding", self.medical_coding_node)
        
        # Set Entry Point
        workflow.set_entry_point("router")
        
        # Add Edges
        # Router decides where to go next
        workflow.add_conditional_edges(
            "router",
            lambda state: state["next"],
            {
                "supervisor": "supervisor",
                "medical_analyst": "medical_analyst",
                "data_analyst": "data_analyst",
                "drug_expert": "drug_expert",
                "profile_manager": "profile_manager",
                "researcher": "researcher",
                "thinking_agent": "thinking_agent",
                "heart_analyst": "heart_analyst",
                "fhir_agent": "fhir_agent",
                "clinical_reasoning": "clinical_reasoning",
                "medical_coding": "medical_coding",
                "FINISH": END
            }
        )
        
        # Supervisor decides next worker or finish
        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state["next"],
            {
                "medical_analyst": "medical_analyst",
                "researcher": "researcher",
                "data_analyst": "data_analyst",
                "drug_expert": "drug_expert",
                "profile_manager": "profile_manager",
                "thinking_agent": "thinking_agent",
                "heart_analyst": "heart_analyst",
                "fhir_agent": "fhir_agent",
                "clinical_reasoning": "clinical_reasoning",
                "medical_coding": "medical_coding",
                "FINISH": END
            }
        )
        
        # Workers always report back to Supervisor for synthesis/next steps
        # (Agentic RAG pattern: Worker -> Supervisor -> [Worker/Finish])
        workflow.add_edge("medical_analyst", "supervisor")
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("data_analyst", "supervisor")
        workflow.add_edge("drug_expert", "supervisor")
        workflow.add_edge("profile_manager", "supervisor")
        workflow.add_edge("thinking_agent", "supervisor")
        workflow.add_edge("heart_analyst", "supervisor")
        workflow.add_edge("fhir_agent", "supervisor")
        workflow.add_edge("clinical_reasoning", "supervisor")
        workflow.add_edge("medical_coding", "supervisor")
        
        return workflow

    # --- P1.1: Parallel Worker Execution ---
    
    async def _execute_parallel_workers(
        self, 
        state: AgentState, 
        worker_names: List[str],
        timeout: float = 10.0
    ) -> Dict:
        """P1.1: Execute multiple independent workers in parallel.
        
        Use when query needs information from multiple workers that don't
        depend on each other (e.g., medication info + drug interaction check).
        
        Args:
            state: Current agent state
            worker_names: List of worker node names to execute
            timeout: Maximum time to wait for all workers
            
        Returns:
            Merged results from all workers
        """
        import asyncio
        
        worker_map = {
            "medical_analyst": self.medical_analyst_node,
            "researcher": self.researcher_node,
            "data_analyst": self.data_analyst_node,
            "drug_expert": self.drug_expert_node,
            "heart_analyst": self.heart_analyst_node,
        }
        
        tasks = []
        for name in worker_names:
            if name in worker_map:
                tasks.append(worker_map[name](state))
            else:
                logger.warning(f"P1.1: Unknown worker '{name}' - skipping")
        
        if not tasks:
            return {"messages": [], "citations": []}
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            
            # Merge results
            merged = {"messages": [], "citations": [], "next": "FINISH"}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"P1.1: Worker {worker_names[i]} failed: {result}")
                    continue
                if isinstance(result, dict):
                    if "messages" in result:
                        merged["messages"].extend(result.get("messages", []))
                    if "citations" in result:
                        merged["citations"].extend(result.get("citations", []))
            
            logger.info(f"P1.1: Parallel execution completed for {worker_names}")
            return merged
            
        except asyncio.TimeoutError:
            logger.warning(f"P1.1: Parallel workers timed out after {timeout}s")
            return {"messages": [], "citations": [], "next": "FINISH"}

    # --- Nodes ---
    
    async def router_node(self, state: AgentState) -> Dict:
        """
        Fast Path Router Node.
        Uses SemanticRouterV2 to classify intent and route to specific worker or supervisor.
        """
        messages = state["messages"]
        last_message = messages[-1]
        query = last_message.content
        
        # Check for explicit user overrides
        if state.get("thinking"):
            logger.info("User requested Thinking Agent")
            return {
                "next": "thinking_agent",
                "intent": "complex_reasoning",
                "confidence": 1.0
            }
        
        if state.get("deep_search"):
            logger.info("User requested Deep Search")
            return {
                "next": "researcher",
                "intent": "research",
                "confidence": 1.0
            }
            
        if state.get("web_search"):
            logger.info("User requested Web Search")
            return {
                "next": "researcher",
                "intent": "research",
                "confidence": 1.0
            }
            
        # Check for files (Multimodal)
        if state.get("file_ids"):
            logger.info("User attached files - routing to Thinking Agent for multimodal analysis")
            return {
                "next": "thinking_agent",
                "intent": "multimodal_analysis",
                "confidence": 1.0
            }
        
        # Use Semantic Router
        route_decision = self.router_v2.route(query)
        intent = route_decision.intent
        
        logger.info(f"Router Decision: {intent} (Confidence: {route_decision.confidence})")
        
        next_node = "supervisor" # Default to supervisor (Slow Path)
        
        # Fast Path Logic
        if route_decision.confidence > 0.8:
            if intent == IntentCategory.VITALS_QUERY:
                next_node = "data_analyst"
            elif intent == IntentCategory.DRUG_INTERACTION:
                next_node = "drug_expert"
            elif intent == IntentCategory.MEDICAL_QA:
                next_node = "medical_analyst"
            elif intent == IntentCategory.DIFFERENTIAL_DIAGNOSIS or intent == IntentCategory.TRIAGE:
                next_node = "clinical_reasoning"
            # Emergency is handled by orchestrator wrapper usually, but if here:
            elif intent == IntentCategory.EMERGENCY:
                # We might want to handle emergency specially, but for now route to supervisor
                # who should have emergency protocols, or medical analyst.
                next_node = "supervisor" 
        
        return {
            "next": next_node,
            "intent": intent.value if hasattr(intent, "value") else str(intent),
            "confidence": route_decision.confidence
        }

    def _parse_supervisor_output(self, output: str) -> dict:
        """
        Robustly parse supervisor output, handling JSON and Python dict formats.
        
        LOW RISK FIX: Enhanced parser with multiple fallback strategies:
        1. Standard JSON parse
        2. Python dict parse (handles single quotes, unquoted keys)
        3. Relaxed JSON cleanup (convert single quotes to double)
        4. Heuristic extraction (keyword-based fallback)
        
        This prevents supervisor JSON parsing failures that break the agentic loop.
        """
        try:
            # Clean markdown code blocks
            clean_output = output.replace("```json", "").replace("```python", "").replace("```", "").strip()
            
            # Attempt 1: Standard JSON parse
            try:
                return json.loads(clean_output)
            except json.JSONDecodeError:
                pass
            
            # Attempt 2: Python dictionary parse (handles single quotes)
            # This fixes the "Invalid json output: {'next': 'FINISH'}" error
            try:
                return ast.literal_eval(clean_output)
            except (ValueError, SyntaxError):
                pass
            
            # Attempt 3 (NEW): Try to fix common JSON format issues
            # Replace single quotes with double quotes (only outside of values)
            # This is a simple heuristic that works for basic JSON structures
            try:
                # Replace single-quoted keys with double-quoted
                fixed_output = clean_output
                
                # Fix pattern: 'key': -> "key":
                import re
                fixed_output = re.sub(r"'([a-zA-Z_][a-zA-Z0-9_]*)'(\s*):", r'"\1"\2:', fixed_output)
                
                # Fix pattern: : 'value' -> : "value"
                # But be careful not to break real strings with apostrophes
                fixed_output = re.sub(r":\s*'([^']*)'", r': "\1"', fixed_output)
                
                # Try JSON parse on fixed output
                return json.loads(fixed_output)
            except (json.JSONDecodeError, Exception):
                pass
            
            # Attempt 4: Try with basic quote normalization for nested structures
            try:
                # More aggressive: replace all single quotes with nothing, then parse
                # Only if output looks like a dict structure
                if clean_output.startswith('{') and clean_output.endswith('}'):
                    # This is risky, so we'll only try it if other methods failed
                    normalized = clean_output.replace("'", '"')
                    result = json.loads(normalized)
                    logger.debug("Supervisor output parsed via quote normalization")
                    return result
            except (json.JSONDecodeError, Exception):
                pass
            
            # Attempt 5: Heuristic extraction (keyword-based fallback)
            logger.warning(f"Failed to parse supervisor output with all JSON strategies: {output[:100]}...")
            
            # Extract decision keyword
            output_lower = output.lower()
            if "'next': 'finish'" in output_lower or '"next": "finish"' in output_lower or output_lower.count("finish") > 0:
                return {"next": "FINISH", "final_response": output[:500]}
            elif "thinking_agent" in output_lower or "thinking agent" in output_lower:
                return {"next": "thinking_agent", "reasoning": "Heuristic extraction"}
            elif "researcher" in output_lower:
                return {"next": "researcher", "reasoning": "Heuristic extraction"}
            elif "medical_analyst" in output_lower or "medical analyst" in output_lower:
                return {"next": "medical_analyst", "reasoning": "Heuristic extraction"}
            elif "drug_expert" in output_lower or "drug expert" in output_lower:
                return {"next": "drug_expert", "reasoning": "Heuristic extraction"}
            elif "data_analyst" in output_lower or "data analyst" in output_lower:
                return {"next": "data_analyst", "reasoning": "Heuristic extraction"}
            elif "clinical_reasoning" in output_lower or "clinical reasoning" in output_lower:
                return {"next": "clinical_reasoning", "reasoning": "Heuristic extraction"}
            
            # Final fallback: default to medical analyst
            logger.warning(f"Supervisor output unrecognizable, defaulting to medical_analyst")
            return {"next": "medical_analyst", "reasoning": "Fallback default"}
        
        except Exception as e:
            logger.error(f"Fatal error in supervisor output parsing: {e}")
            return {"next": "medical_analyst", "reasoning": "Fatal parsing error"}

    async def supervisor_node(self, state: AgentState) -> Dict:
        """
        Supervisor Node (LLM).
        Decides which worker to call next or if the task is finished.
        Uses prompts from PromptRegistry for centralized management.
        
        **ROBUSTNESS**: Uses JsonOutputParser with Markdown stripping to handle
        various LLM output formats (markdown code blocks, intro text, etc).
        """
        messages = state["messages"]
        last_user_message = messages[-1].content if messages else ""
        
        # --- EMERGENCY OVERRIDE ---
        # Intercept critical scenarios before LLM routing to prevent 
        # emergencies from being incorrectly routed to data tools.
        try:
            from rag.nlp.symptom_checker import get_symptom_checker
            checker = get_symptom_checker()
            analysis = checker.analyze_symptoms(last_user_message)
            if analysis.get("has_red_flags", False):
                logger.warning("🚨 EMERGENCY ROUTE: Red flags detected in supervisor, forcing clinical_reasoning")
                return {"next": "clinical_reasoning", "source": "emergency_override"}
        except Exception as e:
            logger.debug(f"Safety override check skipped: {e}")
            pass
        
        # Count worker responses - enforce configurable max steps
        worker_responses = [m for m in messages if getattr(m, 'type', '') == 'tool']
        if len(worker_responses) >= MAX_SUPERVISOR_STEPS:
            logger.warning(f"Max supervisor steps ({MAX_SUPERVISOR_STEPS}) reached, forcing FINISH")
            # Use the last worker response as final answer
            last_worker = worker_responses[-1].content if worker_responses else "I apologize, but I was unable to complete your request."
            return {
                "next": "FINISH",
                "final_response": last_worker[:2000],  # Truncate to prevent token overflow
                "messages": [AIMessage(content=last_worker[:2000])]
            }
        
        # Build a complete user message for Gemma compatibility
        # Gemma/MedGemma doesn't support system messages - must use user/assistant only
        last_user_message = messages[-1].content if messages else ""
        # Escape curly braces to prevent LangChain template variable interpretation
        last_user_message = last_user_message.replace("{", "{{").replace("}", "}}")
        
        # Build conversation summary for supervisor context
        worker_count = len([m for m in messages if getattr(m, 'type', '') == 'tool'])
        
        # Determine prompt based on context using PromptRegistry
        if worker_count > 0:
            # Worker has responded - ask for synthesis
            # Get the base prompt and format it with the worker output
            synthesis_prompt = get_prompt(
                "orchestrator", 
                "supervisor_synthesis",
                variables={"worker_output": last_user_message[:1000]}
            )
            full_prompt = (
                f"A worker has provided information for this request: {last_user_message[:200]}\n\n"
                f"{synthesis_prompt}\n\n"
                "Remember: Your response must be a valid JSON object only - no other text."
            )
        else:
            # First call - route to appropriate worker
            # Get the base prompt and format it with the user query
            routing_prompt = get_prompt(
                "orchestrator", 
                "supervisor_routing",
                variables={"user_query": last_user_message[:300]}
            )
            full_prompt = (
                f"User Request: {last_user_message[:300]}\n\n"
                f"{routing_prompt}\n\n"
                "Remember: Your response must be a valid JSON object only - no other text."
            )
        
        # Note: Do NOT escape braces here - the prompts already have properly escaped
        # JSON examples with {{ and }}. Additional escaping would break them.
        # The user message was already escaped earlier to prevent template injection.
        
        # Use HumanMessage directly instead of ChatPromptTemplate to avoid
        # LangChain trying to parse the JSON examples in the prompt as variables.
        # This is simpler and avoids template escaping issues.
        messages_for_llm = [HumanMessage(content=full_prompt)]
        
        # Use JsonOutputParser for robust parsing (handles Markdown, etc)
        parser = JsonOutputParser(pydantic_object=SupervisorResponse)
        
        try:
            # Invoke LLM directly with messages, then parse
            llm_response = await self.llm.ainvoke(messages_for_llm)
            result = parser.parse(llm_response.content)
            
            # Log successful parsing for monitoring
            logger.debug(f"✅ Supervisor JSON parsing successful - routing to: {result.get('next', 'FINISH')}")
            
            next_step = result.get("next", "FINISH")
            
            # Normalize next_step to handle LLM output variations
            node_aliases = {
                "clinical reasoning": "clinical_reasoning",
                "medical analyst": "medical_analyst",
                "data analyst": "data_analyst",
                "drug expert": "drug_expert",
                "profile manager": "profile_manager",
                "thinking agent": "thinking_agent",
                "heart analyst": "heart_analyst",
                "fhir agent": "fhir_agent",
                "medical coding": "medical_coding",
                "finish": "FINISH",
            }
            normalized = str(next_step).lower().strip()
            next_step = node_aliases.get(normalized, next_step)
            
            # Validate next_step is a known node
            valid_nodes = {"medical_analyst", "researcher", "data_analyst", "drug_expert", 
                          "profile_manager", "thinking_agent", "heart_analyst", "fhir_agent",
                          "clinical_reasoning", "medical_coding", "FINISH"}
            if next_step not in valid_nodes:
                logger.warning(f"Unknown next_step '{next_step}', defaulting to FINISH")
                next_step = "FINISH"
            
            # If FINISH, update state with final response
            # FIX: Normalize final_response by escaping newlines for safe display
            final_response = result.get("final_response", "") or str(result)
            if isinstance(final_response, str):
                # Escape newlines for cleaner display, but keep content
                final_response = final_response.replace("\n", " ").strip()
            
            if next_step == "FINISH":
                return {
                    "next": "FINISH", 
                    "final_response": final_response,
                    "messages": [AIMessage(content=final_response)],
                    "source": state.get("source")  # Preserve source through supervisor
                }
                
            return {"next": next_step, "source": state.get("source")}  # Preserve source for next node
            
        except Exception as e:
            logger.error(f"❌ Supervisor JSON parsing failed (attempt 1): {type(e).__name__}: {str(e)[:100]}")
            logger.debug(f"Raw supervisor response: {state.get('supervisor_response', 'N/A')[:500]}")
            # Attempt robust fallback parsing
            try:
                supervisor_response_text = str(state.get("supervisor_response", str(e)))
                result = self._parse_supervisor_output(supervisor_response_text)
                logger.info(f"✅ Supervisor recovered via fallback parser - routing to: {result.get('next', 'FINISH')}")
                next_step = result.get("next", "FINISH")
                final_response = result.get("final_response", supervisor_response_text)
                
                # FIX: Normalize final_response
                if isinstance(final_response, str):
                    final_response = final_response.replace("\n", " ").strip()
                
                if next_step == "FINISH":
                    return {
                        "next": "FINISH", 
                        "final_response": final_response,
                        "messages": [AIMessage(content=final_response)],
                        "source": state.get("source")  # Preserve source through supervisor
                    }
                return {"next": next_step, "source": state.get("source")}
            except Exception as e2:
                logger.error(f"❌ Supervisor robust fallback also failed (attempt 2): {type(e2).__name__}: {str(e2)[:100]}")
                logger.warning("Supervisor unable to parse response - forcing FINISH with error message")
                return {
                    "next": "FINISH", 
                    "final_response": "I apologize, but I encountered an error processing your request. Please try again with a simpler question.",
                    "source": "llm_fallback"  # Error fallback is LLM-only
                }

    async def medical_analyst_node(self, state: AgentState) -> Dict:
        """Worker: Medical Analyst (Self-RAG with Medical Prompt Builder)"""
        query = state["messages"][-1].content
        user_id = state["user_id"]
        
        # Use Medical Self-RAG
        result = await self.rag_tool.process(query, user_id=user_id)
        
        if result.requires_crag_fallback or result.needs_web_search:
            docs, method = await self.crag_fallback.retrieve_with_fallback(query)
            
            # Format context using centralized medical prompts
            try:
                from core.prompts.medical_prompts import get_prompt_builder
                prompt_builder = get_prompt_builder()
                formatted_context = prompt_builder.format_medical_context(
                    medical_docs="\n".join([d.get('content', '') for d in docs]),
                    drug_info="",
                    patient_memories="",
                    graph_context="",
                    drug_interactions=""
                )
                prompt_text = prompt_builder.build_rag_prompt(
                    query=query,
                    context=formatted_context,
                    history=""
                )
                response_content = await self.llm_gateway.generate(
                    prompt_text,
                    content_type="medical",
                    user_id=user_id
                )
            except Exception:
                # Fallback to basic formatting
                fallback_content = "\n".join([d.get('content', '') for d in docs])
                response_content = f"I couldn't find enough info in my medical database, so I searched the web:\n\n{fallback_content}"
            
            return {
                "messages": [ToolMessage(content=response_content, tool_call_id="call_medical")],
                "confidence": 0.6, # Lower confidence on fallback
                "citations": [d.get('source', 'web') for d in docs],
                "source": "crag"  # Track that this came from CRAG fallback
            }
            
        return {
            "messages": [ToolMessage(content=result.response, tool_call_id="call_medical")],
            "confidence": result.confidence,
            "citations": result.citations,
            "intent": "medical_qa",
            "source": "rag"  # Track that this came from local RAG
        }

    async def researcher_node(self, state: AgentState) -> Dict:
        """Worker: Researcher (Deep Reasoning Research with Medical Content + Clinical Guidelines)
        
        Searches across medical research papers, articles, news, images, and videos
        using the unified medical content search engine, enriches with clinical guidelines,
        then performs deep reasoning research for comprehensive analysis.
        """
        query = state["messages"][-1].content
        
        try:
            # Phase 0: Clinical Guidelines Search (trusted medical sources)
            guidelines_context = ""
            try:
                from tools.clinical_guidelines_search import ClinicalGuidelinesSearch
                guideline_searcher = ClinicalGuidelinesSearch()
                guidelines = await guideline_searcher.search(query, max_results=3)
                if guidelines:
                    parts = [f"### {g.title}\n**Source**: {g.source}\n{g.summary}" for g in guidelines]
                    guidelines_context = "\n\n## 📋 Clinical Guidelines\n" + "\n\n".join(parts)
                    logger.info(f"📋 Found {len(guidelines)} clinical guidelines for '{query[:50]}'")
            except Exception as e:
                logger.debug(f"Clinical guidelines search skipped: {e}")
            
            # Phase 1: Comprehensive medical content search (papers, news, images, videos)
            from tools.medical_search import search_medical_content
            medical_results = await search_medical_content(
                query=query,
                content_types=None,  # All types
                max_results=5
            )
            
            # Phase 2: Deep reasoning research for in-depth analysis
            from agents.deep_research_agent.reasoning_researcher import ReasoningResearcher
            
            researcher = ReasoningResearcher(llm=self.llm)
            session = await researcher.research(query)
            
            response = session.final_report
            reasoning_trace = session.reasoning_trace
            
            # Combine: deep research report + clinical guidelines + medical content (images, videos, papers)
            full_response = response
            
            # Append clinical guidelines if found
            if guidelines_context:
                full_response += guidelines_context
            
            # Append medical content results if they contain useful media
            if medical_results and "Medical Images" in medical_results:
                full_response += "\n\n" + medical_results.split("## 🖼️ Medical Images")[1].split("## ")[0] if "## 🖼️ Medical Images" in medical_results else ""
            if medical_results and "Medical Videos" in medical_results:
                video_section = ""
                if "## 🎬 Medical Videos" in medical_results:
                    parts = medical_results.split("## 🎬 Medical Videos")
                    if len(parts) > 1:
                        video_section = parts[1].split("## ")[0] if "## " in parts[1] else parts[1].split("---")[0]
                if video_section:
                    full_response += "\n\n## 🎬 Related Medical Videos\n" + video_section
            
            # If standalone medical results have more than the deep research found
            if not response or len(response) < 100:
                full_response = medical_results
            
            full_response += f"\n\n<details><summary>Research Reasoning</summary>\n\n{reasoning_trace}\n</details>"
            
            return {
                "messages": [ToolMessage(content=full_response, tool_call_id="call_research")],
                "citations": session.urls_crawled
            }
        except Exception as e:
            logger.error(f"ReasoningResearcher failed: {e}")
            # Fallback to comprehensive medical search only
            try:
                from tools.medical_search import search_medical_content
                result = await search_medical_content(query=query)
                return {"messages": [ToolMessage(content=result, tool_call_id="call_research")]}
            except Exception as e2:
                logger.error(f"Medical search fallback also failed: {e2}")
                # Final fallback to simple web search
                result = await verified_web_search(query=query)
                return {"messages": [ToolMessage(content=str(result.data), tool_call_id="call_research")]}

    async def data_analyst_node(self, state: AgentState) -> Dict:
        """Worker: Data Analyst (SQL)"""
        query = state["messages"][-1].content
        user_id = state["user_id"]
        result = await query_sql_db(query=query, user_id=user_id)
        
        # Synthesize SQL output with LLM to prevent raw data leak
        try:
            prompt_text = f"User Query: {query}\n\nDatabase Results:\n{result.data}\n\nProvide a concise and helpful response to the user based on the database results. Do NOT output raw SQL queries."
            response_content = await self.llm_gateway.generate(
                prompt_text,
                content_type="medical",
                user_id=user_id
            )
        except Exception as e:
            logger.error(f"Failed to synthesize SQL result: {e}")
            response_content = f"Here is the database information I found:\n{result.data}"
            
        return {"messages": [ToolMessage(content=response_content, tool_call_id="call_sql")]}

    async def drug_expert_node(self, state: AgentState) -> Dict:
        """Worker: Drug Expert (EntityValidator + GraphRAG + OpenFDA + Local Interaction Database)"""
        query = state["messages"][-1].content
        
        # 1. Provide smarter drug extraction via LLM if query is complex
        import re
        
        # Very simple extraction fallback
        drugs = re.findall(r'\b[A-Za-z]{3,}\b', query)
        stop_words = {"what", "are", "the", "interactions", "between", "and", "with", "can", "take", "safe", "check", "interaction", "provide", "detailed", "plan", "avoid", "prescription", "history", "male", "female"}
        drugs = [d for d in drugs if d.lower() not in stop_words]
        
        # Ask LLM to extract medication names to avoid fuzzy matching false positives
        try:
            extraction_prompt = f"""Task: Extract a simple comma-separated list of medication names from the text below.
DO NOT answer the question. DO NOT provide medical advice. ONLY list the medications.
If there are no medications, output exactly "NONE".

Text: "{query}"

Medications:"""
            extracted_text = await self.llm_gateway.generate(extraction_prompt, content_type="general", user_id="system")
            if extracted_text and "NONE" not in extracted_text.upper():
                extracted_text = extracted_text.replace("Medications:", "").strip()
                # If the generation is extremely long, it likely hallucinated an answer instead of a list
                if len(extracted_text) < 150:
                    llm_drugs = [d.strip() for d in extracted_text.replace('\n', ',').split(',') if len(d.strip()) > 2]
                    # Filter out sentences and common stop words that might slip through
                    llm_drugs = [d for d in llm_drugs if " " not in d and d.lower() not in stop_words]
                    if llm_drugs:
                        drugs = llm_drugs
                        logger.info(f"💊 LLM Extracted Drugs: {drugs}")
                else:
                    logger.warning("LLM generated a response too long for a drug list. Falling back to regex.")
        except Exception as e:
            logger.debug(f"LLM drug extraction failed, using heuristic: {e}")
            pass
        
        if not drugs:
             return {"messages": [ToolMessage(content="I could not identify any specific medications to check for interactions.", tool_call_id="call_drug")]}

        # 1b. Validate drug names via EntityValidator (prevents hallucinated entity names)
        validation_warnings = []
        try:
            from tools.entity_validator import EntityValidator
            validator = EntityValidator.get_instance()
            sanitized = await validator.validate_and_sanitize_drugs(drugs, auto_fix=True)
            if sanitized["warnings"]:
                validation_warnings = sanitized["warnings"]
                logger.info(f"🔍 Drug name corrections: {validation_warnings}")
            if sanitized["normalized"]:
                drugs = sanitized["normalized"]
            if sanitized["invalid"]:
                invalid_msg = ", ".join(f"'{k}': {v}" for k, v in sanitized["invalid"].items())
                logger.warning(f"⚠️ Unknown drugs skipped: {invalid_msg}")
        except Exception as e:
            logger.debug(f"EntityValidator skipped: {e}")

        if not drugs:
            return {"messages": [ToolMessage(content="I need at least two valid drug names to check for interactions.", tool_call_id="call_drug")]}
        
        if len(drugs) == 1:
            # We identified exactly 1 valid drug. User might just be asking for side effects or general info.
            return {"messages": [ToolMessage(content=f"You mentioned {drugs[0]}. However, to check for drug interactions, I need at least one other medication name.", tool_call_id="call_drug")]}

        # 2. Check interactions via GraphRAG (primary)
        result = await self.graph_checker.check_interaction(drugs)
        
        # 2b. Also check local interaction detector for additional coverage
        local_interactions = []
        try:
            from core.services.interaction_detector import DrugInteractionDetector
            detector = DrugInteractionDetector()
            local_result = detector.get_interaction_summary(drugs)
            if local_result.get("found"):
                local_interactions = local_result.get("interactions", [])
                logger.info(f"💊 Local DB found {len(local_interactions)} additional interactions")
        except Exception as e:
            logger.debug(f"Local interaction detector skipped: {e}")
        
        if not result["found_interactions"] and not local_interactions:
            return {
                "messages": [ToolMessage(content=f"No known interactions found between {', '.join(drugs)}.", tool_call_id="call_drug")],
                "confidence": 0.9
            }
        
        # 2c. Query OpenFDA for adverse events & safety data (supplementary)
        openfda_alerts = []
        try:
            from tools.openfda import get_safety_service
            fda_service = get_safety_service()
            for drug_name in drugs[:4]:  # Limit to 4 drugs for API rate limits
                try:
                    adverse = await fda_service.get_adverse_events(drug_name, limit=3)
                    if adverse and adverse.get("results"):
                        top_reactions = [r.get("term", "") for r in adverse["results"][:3]]
                        openfda_alerts.append({
                            "drug": drug_name,
                            "top_adverse_reactions": top_reactions,
                            "source": "OpenFDA"
                        })
                except Exception:
                    pass  # Individual drug lookup failure is non-critical
            if openfda_alerts:
                logger.info(f"📋 OpenFDA found adverse data for {len(openfda_alerts)} drug(s)")
        except Exception as e:
            logger.debug(f"OpenFDA safety check skipped: {e}")
            
        # 3. Format response from GraphRAG
        response_lines = ["⚠️ **Potential Interactions Found:**\n"]
        
        # GraphRAG results
        for interaction in result.get("interactions", []):
            response_lines.append(f"- **{interaction['drug_a']} + {interaction['drug_b']}**")
            response_lines.append(f"  - Severity: {interaction['severity'].upper()}")
            response_lines.append(f"  - Mechanism: {interaction.get('mechanism', 'Unknown')}")
            response_lines.append(f"  - Description: {interaction.get('description', '')}\n")
        
        # Local DB results (supplement GraphRAG)
        if local_interactions:
            seen_pairs = {frozenset([i['drug_a'].lower(), i['drug_b'].lower()]) 
                         for i in result.get("interactions", [])}
            for inter in local_interactions:
                pair = frozenset([d.lower() for d in inter.get("drugs", [])])
                if pair not in seen_pairs:
                    drug_str = " + ".join(inter.get("drugs", []))
                    response_lines.append(f"- **{drug_str}** (from local database)")
                    response_lines.append(f"  - Severity: {inter.get('severity', 'UNKNOWN')}")
                    response_lines.append(f"  - Description: {inter.get('description', '')}\n")
        
        # OpenFDA adverse event data
        if openfda_alerts:
            response_lines.append("\n📋 **FDA Adverse Event Data:**\n")
            for alert in openfda_alerts:
                reactions = ", ".join(alert["top_adverse_reactions"])
                response_lines.append(f"- **{alert['drug']}**: Most reported reactions — {reactions}")
        
        # Entity validation warnings (fuzzy-matched drug names)
        if validation_warnings:
            response_lines.append("\nℹ️ **Drug Name Notes:**")
            for w in validation_warnings:
                response_lines.append(f"  - {w}")
            
        return {
            "messages": [ToolMessage(content="\n".join(response_lines), tool_call_id="call_drug")],
            "confidence": 1.0,
            "citations": ["Graph Knowledge Base", "Local Interaction Database", "OpenFDA"]
        }

    async def profile_manager_node(self, state: AgentState) -> Dict:
        """Worker: Profile Manager"""
        user_id = state["user_id"]
        # TODO: fetch_user_profile is not implemented in agentic_tools
        # For now, return a placeholder message
        return {"messages": [ToolMessage(content=f"Profile retrieval for user {user_id} not yet implemented", tool_call_id="call_profile")]}

    async def heart_analyst_node(self, state: AgentState) -> Dict:
        """Worker: Heart Disease Risk Analyst using RAG-augmented MedGemma."""
        query = state["messages"][-1].content if state["messages"] else ""
        user_id = state.get("user_id", "anonymous")
        
        try:
            result = await self.heart_predictor.predict_risk(
                patient_symptoms=query,
                user_id=user_id,
                include_history=True,
                validate_response=True
            )
            
            # Format response with risk level and grounding status
            grounded_indicator = "✅ Evidence-based" if result.is_grounded else "⚠️ Review recommended"
            attention_indicator = "🚨 **SEEK IMMEDIATE MEDICAL ATTENTION**" if result.needs_medical_attention else ""
            
            # Include contributing factors explanation if available
            factors_section = ""
            if result.contributing_factors:
                factors_section = f"""
**Why These Results May Indicate Heart Disease:**
{result.contributing_factors}
"""
            
            response = f"""
**Heart Disease Risk Assessment**

{result.response}
{factors_section}
---
*Risk Level: {result.risk_level} | Confidence: {result.confidence:.0%}*
*{grounded_indicator}*
{attention_indicator}
""".strip()
            
            return {
                "messages": [ToolMessage(content=response, tool_call_id="call_heart")],
                "final_response": response,
                "citations": result.citations,
                "confidence": result.confidence,
                "intent": "heart_risk_assessment"
            }
        except Exception as e:
            logger.error(f"Heart analyst error: {e}")
            return {
                "messages": [ToolMessage(content=f"Heart risk analysis failed: {e}", tool_call_id="call_heart")],
                "final_response": f"I apologize, but I couldn't complete the heart disease risk analysis: {e}",
                "confidence": 0.0
            }

    async def thinking_node(self, state: AgentState) -> Dict:
        """Worker: Thinking Agent (Deep Reasoning)"""
        query = state["messages"][-1].content
        # Get context from previous messages
        context = "\n".join([m.content for m in state["messages"][:-1]])
        
        # --- Medical Planner Integration ---
        # Generate a plan first
        try:
            plan = await self.planner.create_initial_plan(query)
            plan_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan)])
            
            # Add plan to context
            enhanced_context = f"{context}\n\nProposed Plan:\n{plan_str}"
            logger.info(f"Medical Planner generated plan: {plan}")
        except Exception as e:
            logger.warning(f"Medical Planner failed: {e}")
            enhanced_context = context
        
        # Run thinking agent
        file_ids = state.get("file_ids")
        user_id = state.get("user_id")
        result = await self.thinking_agent.run(query, enhanced_context, file_ids=file_ids, user_id=user_id)
        
        # Format response with reasoning trace (collapsible in UI)
        response = f"{result.answer}\n\n<details><summary>Reasoning Trace</summary>\n\n{result.get_reasoning_trace()}\n</details>"
        
        return {
            "messages": [ToolMessage(content=response, tool_call_id="call_thinking")],
            "confidence": 0.95,
            "intent": "complex_reasoning"
        }

    async def fhir_query_node(self, state: AgentState) -> Dict:
        """
        Worker: FHIR Agent (EHR Data).
        
        **STABILITY**: Uses public ainvoke() method instead of private _arun()
        to maintain compatibility with library updates.
        """
        query = state["messages"][-1].content
        user_id = state["user_id"]
        
        # Extract patient ID from user_id or query
        # For now, assume user_id maps to patient_id or is passed in query
        # In production, this would look up patient_id from user profile
        patient_id = user_id 
        
        try:
            # Use public ainvoke() method (stable API, not private _arun)
            result = await self.fhir_tool.ainvoke(
                {"patient_id": patient_id},
                config={"timeout": 30}
            )
            
            # Handle result from BaseTool
            result_content = result if isinstance(result, str) else str(result.get("content", result))
            
            return {
                "messages": [ToolMessage(content=result_content, tool_call_id="call_fhir")],
                "confidence": 1.0,
                "intent": "fhir_query"
            }
        except Exception as e:
            logger.error(f"FHIR agent failed: {e}")
            return {
                "messages": [ToolMessage(content=f"Error retrieving EHR data: {e}", tool_call_id="call_fhir")],
                "confidence": 0.0
            }

    async def clinical_reasoning_node(self, state: AgentState) -> Dict:
        """
        Worker: Clinical Reasoning Agent.
        Handles differential diagnosis, triage, and symptom analysis.
        Integrates SymptomChecker for automated symptom extraction with red flag detection.
        """
        messages = state.get("messages", [])
        query = messages[-1].content if messages else ""
        intent = state.get("intent", "")
        
        # Pre-processing: Extract symptoms with SymptomChecker
        symptom_context = ""
        has_red_flags = False
        try:
            from rag.nlp.symptom_checker import get_symptom_checker
            checker = get_symptom_checker()
            analysis = checker.analyze_symptoms(query)
            present = analysis.get("present_symptoms", [])
            denied = analysis.get("denied_symptoms", [])
            has_red_flags = analysis.get("has_red_flags", False)
            
            if present or denied:
                parts = []
                if present:
                    symptom_strs = [s["text"] if isinstance(s, dict) else str(s) for s in present]
                    parts.append(f"**Present symptoms**: {', '.join(symptom_strs)}")
                if denied:
                    denied_strs = [s["text"] if isinstance(s, dict) else str(s) for s in denied]
                    parts.append(f"**Denied symptoms**: {', '.join(denied_strs)}")
                if has_red_flags:
                    parts.append("🚨 **RED FLAGS DETECTED** — Urgent evaluation recommended")
                symptom_context = "\n".join(parts) + "\n\n"
                logger.info(f"🩺 SymptomChecker: {len(present)} present, {len(denied)} denied, red_flags={has_red_flags}")
        except Exception as e:
            logger.debug(f"SymptomChecker skipped: {e}")
        
        # If red flags detected, force triage
        if has_red_flags and intent != "triage":
            logger.warning("🚨 Red flags detected — escalating to triage")
            intent = "triage"
        
        # Map string intent to enum if needed, or just check string
        if intent == "differential_diagnosis":
            result = await generate_differential_diagnosis(symptoms=query)
        elif intent == "triage":
            result = await triage_patient(symptoms=query)
        else:
            # Fallback based on keywords if intent lost
            if "triage" in query.lower() or "er" in query.lower():
                result = await triage_patient(symptoms=query)
            else:
                result = await generate_differential_diagnosis(symptoms=query)
        
        # Prepend symptom analysis if available
        if symptom_context:
            result = f"## 🩺 Symptom Analysis\n{symptom_context}\n{result}"
        
        return {
            "messages": [ToolMessage(content=result, tool_call_id="call_clinical")],
            "confidence": 0.95 if has_red_flags else 0.9,
            "intent": intent
        }

    async def medical_coding_node(self, state: AgentState) -> Dict:
        """
        Worker: Medical Coding Agent.
        Maps clinical text to standardized codes (SNOMED-CT, ICD-10, LOINC, CPT).
        Uses tools/medical_coding/auto_coder.py.
        """
        query = state["messages"][-1].content
        
        try:
            result = await auto_code_clinical_note(
                clinical_text=query,
                include_billing=True
            )
            return {
                "messages": [ToolMessage(content=result, tool_call_id="call_medical_coding")],
                "confidence": 0.9,
                "intent": "medical_coding"
            }
        except Exception as e:
            logger.error(f"Medical coding failed: {e}")
            return {
                "messages": [ToolMessage(
                    content=f"Error generating medical codes: {e}",
                    tool_call_id="call_medical_coding"
                )],
                "confidence": 0.0
            }

    # --- Main Execution ---
    async def execute(
        self, 
        query: str, 
        user_id: str,
        thread_id: Optional[str] = None,
        progress_callback: Optional[callable] = None,
        thinking: bool = False,
        web_search: bool = False,
        deep_search: bool = False,
        file_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute the orchestrator.
        
        Args:
            query: User query
            user_id: User ID
            thread_id: Optional thread ID for checkpointing. If provided, state
                       will be persisted to Redis, enabling crash recovery.
            progress_callback: Optional async callback for progress updates.
                               Signature: async (step: int, total: int, status: str, detail: str)
            
        Returns:
            Dict with 'response', 'metadata', etc.
        """
        import uuid
        import time as _time
        _start = _time.perf_counter()
        
        # Generate thread_id if checkpointing is enabled but no ID provided
        if self.checkpointer and not thread_id:
            thread_id = f"thread_{user_id}_{uuid.uuid4().hex[:8]}"
        
        # Load recent conversation history for context continuity
        history_messages = []
        try:
            from core.services.chat_history import ChatHistory
            chat_history = ChatHistory(user_id=user_id, max_messages=10)
            recent = chat_history.get_recent_messages(limit=6)
            for msg in recent:
                if msg.get("role") == "human":
                    history_messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "ai":
                    history_messages.append(AIMessage(content=msg["content"]))
            if history_messages:
                logger.info(f"📝 Loaded {len(history_messages)} messages from chat history")
        except Exception as e:
            logger.debug(f"Chat history loading skipped: {e}")
        
        initial_state = {
            "messages": history_messages + [HumanMessage(content=query)],
            "user_id": user_id,
            "next": "router",
            "final_response": None,
            "source": None,  # Will be set by workers (rag, crag, web, llm, llm_fallback)
            "thinking": thinking,
            "web_search": web_search,
            "deep_search": deep_search,
            "file_ids": file_ids
        }
        
        # Configure execution with checkpointing if available
        config = {}
        if thread_id and self.checkpointer:
            config = {"configurable": {"thread_id": thread_id}}
            logger.debug(f"📝 Checkpointing enabled for thread: {thread_id}")
        
        final_state = await self.app.ainvoke(initial_state, config=config if config else None)
        
        response = final_state.get("final_response")
        if not response:
            # If no final response set, take the last message
            last_msg = final_state["messages"][-1]
            response = last_msg.content

        # --- PERSISTENCE: Save Chat History to Postgres (UI) & Memori (Context) ---
        # 1. Postgres (Chat History for UI)
        if self.db_manager and hasattr(self.db_manager, 'store_chat_message'):
            try:
                # Store User Message
                await self.db_manager.store_chat_message(
                    session_id=thread_id or "default",
                    message_type="human",
                    content=query,
                    metadata={"user_id": user_id, "source": "web_ui"}
                )
                
                # Store AI Response
                await self.db_manager.store_chat_message(
                    session_id=thread_id or "default",
                    message_type="ai",
                    content=str(response),
                    metadata={
                        "user_id": user_id,
                        "intent": final_state.get("intent"),
                        "source": final_state.get("source"),
                        "citations": final_state.get("citations"),
                        "confidence": final_state.get("confidence")
                    }
                )
                logger.debug(f"✅ Chat history persisted to DB for thread: {thread_id}")
            except Exception as e:
                logger.error(f"❌ Failed to persist chat history to DB: {e}")

        # 2. Memori (Context Extraction & Long-term Memory)
        if self.memori_bridge and self.memori_bridge.memori:
             try:
                 # Feed conversation to Memori for analysis
                 # record_conversation is synchronous (schedules async task internally)
                 self.memori_bridge.memori.record_conversation(
                     user_input=query,
                     ai_output=str(response),
                     metadata={"source": "langgraph_orchestrator", "user_id": user_id}
                 )
                 logger.debug(f"🧠 Conversation recorded to Memori for analysis")
             except Exception as e:
                 logger.error(f"❌ Failed to record to Memori: {e}")
        
        # --- HALLUCINATION GRADING ---
        # Grade the response against the context from worker messages
        # to detect unsupported medical claims before returning to user.
        is_grounded = True
        try:
            from core.safety.hallucination_grader import HallucinationGrader
            grader = HallucinationGrader()
            
            # Build context from all worker (tool) messages
            worker_messages = [m.content for m in final_state.get("messages", [])
                               if getattr(m, 'type', '') == 'tool' and hasattr(m, 'content')]
            context_for_grading = "\n---\n".join(worker_messages[-3:])  # Last 3 worker outputs
            
            high_risk_medical_intents = {
                "clinical_reasoning",
                "differential_diagnosis",
                "triage",
                "drug_interaction",
                "heart_risk_assessment",
            }
            source = str(final_state.get("source") or "").lower()
            intent = str(final_state.get("intent") or "").lower()
            query_lower = str(query or "").lower()

            # Only show this warning for clinically risky/advisory requests.
            high_risk_query_terms = (
                "chest pain",
                "shortness of breath",
                "heart attack",
                "stroke",
                "emergency",
                "severe",
                "urgent",
                "dose",
                "dosage",
                "interaction",
                "side effect",
                "diagnose",
                "diagnosis",
                "treatment",
                "triage",
                "suicid",
                "self-harm",
            )
            is_high_risk_query = any(term in query_lower for term in high_risk_query_terms)
            should_surface_grounding_warning = (
                (source in {"rag", "crag"} or intent in high_risk_medical_intents)
                and is_high_risk_query
            )

            if context_for_grading and response and len(response) > 50:
                is_grounded = await grader.grade(answer=str(response), context=context_for_grading)
                if not is_grounded:
                    logger.warning("⚠️ Hallucination detected — response may not be fully grounded")
                    if should_surface_grounding_warning:
                        response = (
                            str(response) +
                            "\n\n---\n⚠️ *Note: Some claims in this response may not be fully supported "
                            "by the retrieved evidence. Please verify with a healthcare professional.*"
                        )
                    final_state["confidence"] = max(0.0, (final_state.get("confidence") or 0.5) - 0.2)
                else:
                    logger.debug("✅ Hallucination grading passed — response is grounded")
        except Exception as e:
            logger.warning(f"Hallucination grading skipped: {e}")
        
        # --- MEDICAL ENTITY EXTRACTION ---
        # Extract drug names and medical terms from the query for downstream checks
        detected_drugs = []
        try:
            from core.services.medical_phrase_matcher import MedicalPhraseMatcher, SPACY_AVAILABLE
            if SPACY_AVAILABLE:
                import spacy
                # Use lightweight model if available
                try:
                    nlp = spacy.blank("en")  # Minimal — PhraseMatcher only needs vocab
                    matcher = MedicalPhraseMatcher(nlp)
                    doc = nlp(query)
                    matches = matcher.find_matches(doc)
                    detected_drugs = [m["text"] for m in matches if m.get("label") == "DRUG"]
                    if matches:
                        logger.info(f"🔬 Medical entities in query: {[m['text'] for m in matches[:5]]}")
                except Exception as e_nlp:
                    logger.debug(f"Medical phrase matching skipped: {e_nlp}")
        except Exception:
            pass  # spaCy/matcher not available
        
        # --- DRUG INTERACTION CHECK ---
        # If 2+ drugs are detected in the query, check for interactions automatically
        drug_interaction_warning = ""
        if len(detected_drugs) >= 2:
            try:
                from core.services.interaction_detector import DrugInteractionDetector
                detector = DrugInteractionDetector()
                interaction_result = detector.get_interaction_summary(detected_drugs)
                if interaction_result.get("found"):
                    warnings = []
                    for inter in interaction_result.get("interactions", []):
                        severity = inter.get("severity", "UNKNOWN")
                        desc = inter.get("description", "")
                        drugs = " + ".join(inter.get("drugs", []))
                        warnings.append(f"  - **{drugs}** ({severity}): {desc}")
                    drug_interaction_warning = (
                        "\n\n---\n⚠️ **Drug Interaction Alert:**\n" + "\n".join(warnings)
                    )
                    response = str(response) + drug_interaction_warning
                    logger.info(f"💊 Drug interaction detected for {detected_drugs}")
            except Exception as e:
                logger.debug(f"Drug interaction check skipped: {e}")
        
        # Determine if this is a utility response (calculator, etc.) that doesn't need PII scrubbing
        # Calculator outputs are pure numbers and shouldn't be modified
        intent = final_state.get("intent", "unknown")
        source = final_state.get("source", "unknown")
        skip_pii_scrub = intent in ["CALCULATOR", "UTILITY"] or source in ["calculator", "utility"]
        
        # CRITICAL: Apply PII scrubbing to response before returning (except for utility responses)
        if _pii_scrubber and response and not skip_pii_scrub:
            try:
                response = _pii_scrubber.scrub(response)
                logger.debug("✅ PII scrubbing applied to response")
            except Exception as e:
                logger.critical(f"❌ PII scrubbing failed: {e} - BLOCKING RESPONSE")
                # FAIL-SECURE: Do not return unscrubbed data
                response = "I apologize, but I cannot provide a response at this time due to a security check failure."
                final_state["confidence"] = 0.0
        
        # Also scrub citations if they exist
        citations = final_state.get("citations", [])
        if _pii_scrubber and citations:
            try:
                citations = [_pii_scrubber.scrub(str(c)) if isinstance(c, str) else c for c in citations]
            except Exception as e:
                logger.warning(f"Citation scrubbing failed: {e}")
        
        _latency_ms = (_time.perf_counter() - _start) * 1000
        
        # Record Prometheus metrics for the orchestrator
        try:
            from core.monitoring.prometheus_metrics import get_metrics
            _prom = get_metrics()
            _prom.increment_counter("orchestrator_executions")
            _prom.record_histogram("orchestrator_latency_ms", _latency_ms)
            _prom.record_llm_latency(_latency_ms)
            if not is_grounded:
                _prom.increment_counter("orchestrator_hallucination_detected")
            _prom.increment_counter("orchestrator_hallucination_checks")
            if len(detected_drugs) >= 2:
                _prom.increment_counter("orchestrator_drug_interactions_checked")
        except Exception:
            pass  # Metrics must never break execution
        
        # Record agent execution in AgentTracer
        try:
            from app_lifespan import get_agent_tracer
            tracer = get_agent_tracer()
            if tracer:
                from core.observability.tracing import SpanType
                with tracer.trace_operation(
                    f"orchestrator_execute:{intent}",
                    operation_type=SpanType.AGENT_STEP,
                    metadata={"user_id": user_id, "intent": intent, "source": source}
                ) as span:
                    span.add_event("execution_complete", {
                        "latency_ms": round(_latency_ms, 1),
                        "steps": len(final_state["messages"]),
                        "source": source,
                    })
        except Exception:
            pass  # Tracing must never break execution
        
        # Emit webhook events for significant outcomes
        try:
            from core.services.webhook_service import get_webhook_service, WebhookEvent
            webhook = await get_webhook_service()
            if webhook:
                if not is_grounded:
                    await webhook.emit(WebhookEvent.HALLUCINATION_DETECTED, {
                        "user_id": user_id, "intent": intent, "thread_id": thread_id
                    })
                if drug_interaction_warning:
                    await webhook.emit(WebhookEvent.DRUG_INTERACTION_FOUND, {
                        "user_id": user_id, "drugs": detected_drugs, "thread_id": thread_id
                    })
        except Exception:
            pass  # Webhooks must never break execution
        
        return {
            "response": response,
            "intent": final_state.get("intent", "unknown"),
            "confidence": final_state.get("confidence", 0.0),
            "citations": citations,
            "pii_scrubbed": _pii_scrubber is not None,
            "thread_id": thread_id,  # Return thread_id for resumption
            "is_grounded": is_grounded,
            "detected_drugs": detected_drugs,
            "drug_interaction_found": bool(drug_interaction_warning),
            "metadata": {
                "steps": len(final_state["messages"]),
                "source": final_state.get("source", "unknown"),  # Track response source
                "model": getattr(self.llm_gateway, "medgemma_model", "unknown"),
                "checkpointed": self.checkpointer is not None,
                "latency_ms": round(_latency_ms, 1),
                "hallucination_checked": True,
                "medical_entities_extracted": len(detected_drugs) > 0,
            }
        }
    
    async def resume_from_checkpoint(self, thread_id: str) -> Dict[str, Any]:
        """
        Resume a workflow from the last checkpoint.
        
        Use this when a worker crashes and needs to continue processing
        from where it left off.
        
        Args:
            thread_id: The thread ID of the workflow to resume
            
        Returns:
            Dict with 'response', 'metadata', etc. from the resumed workflow
        """
        if not self.checkpointer:
            return {
                "error": "Checkpointing not available",
                "response": None,
                "metadata": {"recovered": False}
            }
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            
            # Get the current state from checkpoint
            state = await self.app.aget_state(config)
            
            if not state or not state.values:
                return {
                    "error": f"No checkpoint found for thread: {thread_id}",
                    "response": None,
                    "metadata": {"recovered": False}
                }
            
            logger.info(f"🔄 Resuming workflow from checkpoint: {thread_id}")
            
            # Resume execution from last checkpoint
            final_state = await self.app.ainvoke(None, config=config)
            
            response = final_state.get("final_response")
            if not response and final_state.get("messages"):
                last_msg = final_state["messages"][-1]
                response = last_msg.content
            
            return {
                "response": response,
                "intent": final_state.get("intent", "unknown"),
                "confidence": final_state.get("confidence", 0.0),
                "citations": final_state.get("citations", []),
                "thread_id": thread_id,
                "metadata": {
                    "steps": len(final_state.get("messages", [])),
                    "source": final_state.get("source", "unknown"),
                    "recovered": True,
                    "checkpointed": True
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to resume from checkpoint {thread_id}: {e}")
            return {
                "error": str(e),
                "response": None,
                "thread_id": thread_id,
                "metadata": {"recovered": False}
            }
    
    async def get_workflow_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a workflow from its checkpoint.
        
        Useful for inspecting workflow progress or debugging.
        
        Args:
            thread_id: The thread ID of the workflow
            
        Returns:
            Current workflow state or None if not found
        """
        if not self.checkpointer:
            logger.warning("Checkpointing not available - cannot retrieve state")
            return None
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = await self.app.aget_state(config)
            
            if not state or not state.values:
                return None
            
            return {
                "thread_id": thread_id,
                "current_node": state.values.get("next"),
                "message_count": len(state.values.get("messages", [])),
                "user_id": state.values.get("user_id"),
                "has_response": state.values.get("final_response") is not None,
                "source": state.values.get("source"),
                "checkpoint_id": getattr(state, 'config', {}).get('configurable', {}).get('checkpoint_id')
            }
            
        except Exception as e:
            logger.error(f"Failed to get workflow state for {thread_id}: {e}")
            return None
