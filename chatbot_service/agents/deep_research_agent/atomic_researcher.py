"""
Atomic Researcher - Atom-of-Thought Research Agent

Decomposes complex research into atomic tasks and executes with reasoning.
Combines PlanningMixin (AoT) with ThinkingAgent (CoT) for maximum capability.
"""


import asyncio
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Import existing components
try:
    from agents.components.thinking import ThinkingAgent, ThinkingResult
    from agents.components.planning import (
        PlanningMixin, ExecutionPlan, PlanStep, PlanStepStatus
    )
    from tools.web_search import VerifiedWebSearchTool
    from .search_tool import get_search_results
    from .crawler_tool import deep_crawl_research
    from .reporter import synthesize_report
    from .models import ResearchInsight
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from agents.components.thinking import ThinkingAgent, ThinkingResult
    from agents.components.planning import (
        PlanningMixin, ExecutionPlan, PlanStep, PlanStepStatus
    )
    from tools.web_search import VerifiedWebSearchTool
    from agents.deep_research_agent.search_tool import get_search_results
    from agents.deep_research_agent.crawler_tool import deep_crawl_research
    from agents.deep_research_agent.reporter import synthesize_report
    from agents.deep_research_agent.models import ResearchInsight

logger = logging.getLogger(__name__)


@dataclass
class AtomResult:
    """Result from processing a single atom."""
    atom_description: str
    success: bool
    output: str
    thinking_trace: str
    duration_ms: float


@dataclass
class AtomicResearchSession:
    """Complete atomic research session."""
    query: str
    plan: Optional[ExecutionPlan] = None
    atom_results: List[AtomResult] = field(default_factory=list)
    final_synthesis: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def get_completed_atoms(self) -> List[AtomResult]:
        return [a for a in self.atom_results if a.success]
    
    def to_markdown(self) -> str:
        lines = [
            f"# Atomic Research: {self.query}",
            f"\n*Started: {self.started_at.strftime('%Y-%m-%d %H:%M')}*\n",
        ]
        
        if self.plan:
            lines.append("## Execution Plan")
            lines.append(self.plan.to_markdown())
            lines.append("")
        
        lines.append("## Atom Execution Results\n")
        for i, atom in enumerate(self.atom_results, 1):
            status = "✅" if atom.success else "❌"
            lines.append(f"### Atom {i}: {status} {atom.atom_description}")
            lines.append(f"*Duration: {atom.duration_ms:.0f}ms*\n")
            lines.append(f"**Output:**\n{atom.output[:500]}...\n")
            lines.append(f"**Thinking:**\n```\n{atom.thinking_trace[:500]}...\n```\n")
        
        if self.final_synthesis:
            lines.append("## Final Synthesis\n")
            lines.append(self.final_synthesis)
        
        return "\n".join(lines)


class AtomicResearcher(PlanningMixin):
    """
    Atom-of-Thought Research Agent.
    
    Combines hierarchical planning with chain-of-thought execution:
    1. Decomposes research query into atomic steps (plan)
    2. Executes each step with a ThinkingAgent
    3. Can dynamically replan based on results
    4. Synthesizes findings into final report
    
    Example:
        researcher = AtomicResearcher(llm)
        session = await researcher.execute_atomic_research(
            "Compare GPT-4 and Claude 3 for medical applications"
        )
        
        print(session.plan.to_markdown())  # See the decomposition
        print(session.to_markdown())        # Full results
    """
    
    # Specialized planning prompt for research decomposition
    RESEARCH_PLANNING_PROMPT = """
You are a research strategist. Decompose this research query into 4-8 atomic steps.

Research Query: {query}

Each atom should be:
- Specific and actionable
- Independent or minimally dependent on others
- Achievable with web search and content extraction

Example atoms:
- "Search for recent academic papers on [topic]"
- "Find industry reports from [source type]"
- "Identify key researchers/companies in [field]"
- "Extract technical specifications from [sources]"
- "Compare findings across [categories]"

Respond in JSON format:
{{
    "steps": [
        {{
            "description": "Specific atomic task",
            "assigned_agent": "researcher",
            "tools_needed": ["search", "crawl"],
            "dependencies": []
        }}
    ]
}}
"""
    
    def __init__(
        self,
        llm,
        planning_interval: int = 5,
        max_thinking_rounds: int = 5,
        verbose: bool = True,
    ):
        """
        Initialize atomic researcher.
        
        Args:
            llm: Language model for planning and thinking
            planning_interval: Steps before considering replan
            max_thinking_rounds: Max rounds per atom
            verbose: Log execution details
        """
        # Initialize PlanningMixin
        PlanningMixin.__init__(
            self,
            planning_interval=planning_interval,
            max_plan_steps=10,
            llm=llm
        )
        
        self.llm = llm
        self.max_thinking_rounds = max_thinking_rounds
        self.verbose = verbose
        
        # Search tool
        self.search_tool = VerifiedWebSearchTool(use_cache=True)
        
        # Current session
        self.session: Optional[AtomicResearchSession] = None
    
    async def execute_atomic_research(self, query: str) -> AtomicResearchSession:
        """
        Execute research by decomposing into atoms.
        
        Args:
            query: Complex research query
            
        Returns:
            AtomicResearchSession with all results
        """
        logger.info(f"⚛️ Starting atomic research: {query}")
        
        self.session = AtomicResearchSession(query=query)
        
        # Phase 1: Generate Plan (Atoms)
        logger.info("📋 Phase 1: Generating atomic plan...")
        plan = await self._generate_research_plan(query)
        self.session.plan = plan
        
        logger.info(f"Generated {len(plan.steps)} atoms")
        if self.verbose:
            print(plan.to_markdown())
        
        # Phase 2: Execute Each Atom
        logger.info("⚡ Phase 2: Executing atoms...")
        accumulated_context = []
        
        while not plan.is_complete():
            step = plan.get_next_step()
            if not step:
                logger.warning("No executable step found (dependencies not met)")
                break
            
            step.start()
            logger.info(f"⚛️ Processing: {step.description}")
            
            # Execute with ThinkingAgent
            atom_result = await self._execute_atom(
                step.description,
                context=accumulated_context
            )
            
            self.session.atom_results.append(atom_result)
            
            if atom_result.success:
                step.complete(result=atom_result.output)
                accumulated_context.append({
                    "atom": step.description,
                    "result": atom_result.output[:500]
                })
            else:
                step.fail(reason=atom_result.output)
            
            # Check if dynamic replan needed
            replan = await self.maybe_replan({
                "original_query": query,
                "accumulated_context": accumulated_context
            })
            if replan:
                logger.info(f"📋 Replanned: {len(replan.steps)} new atoms")
                plan = replan
                self.session.plan = plan
        
        # Phase 3: Synthesize
        logger.info("📝 Phase 3: Synthesizing findings...")
        self.session.final_synthesis = await self._synthesize_atoms(
            query, 
            self.session.get_completed_atoms()
        )
        
        self.session.completed_at = datetime.utcnow()
        logger.info(f"✅ Atomic research complete: {len(self.session.atom_results)} atoms processed")
        
        return self.session
    
    async def _generate_research_plan(self, query: str) -> ExecutionPlan:
        """Generate atomic research plan."""
        # Use custom prompt for research
        original_prompt = self.PLANNING_PROMPT
        self.PLANNING_PROMPT = self.RESEARCH_PLANNING_PROMPT
        
        try:
            plan = await self.generate_plan(
                query=query,
                available_agents=["researcher", "analyst"]
            )
            return plan
        finally:
            self.PLANNING_PROMPT = original_prompt
    
    async def _execute_atom(
        self,
        atom_description: str,
        context: List[Dict]
    ) -> AtomResult:
        """
        Execute a single atom using ThinkingAgent.
        
        Args:
            atom_description: What to do
            context: Results from previous atoms
            
        Returns:
            AtomResult with output and reasoning
        """
        start_time = datetime.utcnow()
        
        # Create thinking agent with research tools
        tools = self._create_atom_tools()
        
        agent = ThinkingAgent(
            llm=self.llm,
            tools=tools,
            max_thinking_rounds=self.max_thinking_rounds,
            verbose=self.verbose
        )
        
        # Context summary
        context_str = "Previous findings:\n"
        for ctx in context[-3:]:  # Last 3 atoms for context
            context_str += f"- {ctx['atom']}: {ctx['result'][:200]}...\n"
        
        prompt = f"""Execute this specific research task:

TASK: {atom_description}

{context_str if context else "This is the first task."}

Use the available tools to complete this task. Think carefully and explain your approach."""

        try:
            result = await agent.run(prompt)
            
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AtomResult(
                atom_description=atom_description,
                success=True,
                output=result.answer,
                thinking_trace=result.get_reasoning_trace(),
                duration_ms=duration
            )
            
        except Exception as e:
            logger.error(f"Atom execution failed: {e}")
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AtomResult(
                atom_description=atom_description,
                success=False,
                output=f"Error: {e}",
                thinking_trace="",
                duration_ms=duration
            )
    
    def _create_atom_tools(self) -> List:
        """Create tools for atom execution."""
        
        async def search_web(query: str) -> str:
            """Search the web for information."""
            try:
                urls = await get_search_results(query, max_results=3, search_pdfs=False)
                if urls:
                    return f"Found {len(urls)} results:\n" + "\n".join(f"- {u}" for u in urls)
                return "No results found."
            except Exception as e:
                return f"Search failed: {e}"
        
        async def search_papers(query: str) -> str:
            """Search for academic papers (PDFs)."""
            try:
                urls = await get_search_results(query, max_results=3, search_pdfs=True)
                if urls:
                    return f"Found {len(urls)} papers:\n" + "\n".join(f"- {u}" for u in urls)
                return "No papers found."
            except Exception as e:
                return f"Search failed: {e}"
        
        async def extract_content(url: str, question: str) -> str:
            """Extract specific information from a URL."""
            try:
                insights = await deep_crawl_research(
                    seed_urls=[url],
                    extraction_prompt=question,
                    max_depth=1
                )
                if insights:
                    return insights[0].summary[:1000]
                return "Could not extract content."
            except Exception as e:
                return f"Extraction failed: {e}"
        
        # Set metadata
        search_web.__name__ = "search_web"
        search_web.name = "search_web"
        search_web.description = "Search the web for general information"
        
        search_papers.__name__ = "search_papers"
        search_papers.name = "search_papers"
        search_papers.description = "Search for academic papers and PDFs"
        
        extract_content.__name__ = "extract_content"
        extract_content.name = "extract_content"
        extract_content.description = "Extract specific information from a URL"
        
        return [search_web, search_papers, extract_content]
    
    async def _synthesize_atoms(
        self,
        query: str,
        completed_atoms: List[AtomResult]
    ) -> str:
        """Synthesize findings from all atoms."""
        if not completed_atoms:
            return "No atoms were successfully completed."
        
        synthesis_prompt = f"""Synthesize these research findings into a cohesive, evidence-based report.

Original Query: {query}

Findings from {len(completed_atoms)} research atoms:

"""
        for i, atom in enumerate(completed_atoms, 1):
            synthesis_prompt += f"""
### Atom {i}: {atom.atom_description}
{atom.output[:800]}
"""
        
        synthesis_prompt += """

Create a comprehensive report that:
1. **Key Findings** - Summarize the most important discoveries, ranked by evidence strength
2. **Cross-Validation** - Identify findings confirmed by multiple atoms/sources
3. **Contradictions** - Note any disagreements between sources with context
4. **Evidence Gaps** - What questions remain unanswered?
5. **Clinical Implications** - Actionable takeaways (if medical)
6. **Limitations** - What are the limitations of this research?
7. **Source Quality** - Rate the overall quality of evidence gathered

Write the report in well-structured markdown format.
For medical topics, always include safety disclaimers.
Cite specific sources where possible."""

        try:
            response = await self.llm.ainvoke(synthesis_prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"Synthesis error: {e}"
