"""
Planning Mixin - Adds planning capabilities to agents.

Provides:
- PlanStep: Single step in a plan
- PlanningMixin: Mixin class for planning support
- Auto-replanning at configurable intervals

Based on smolagents planning_interval pattern.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class PlanStepStatus(Enum):
    """Status of a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """
    A single step in an execution plan.
    
    Attributes:
        description: What this step does
        status: Current status
        assigned_agent: Which agent should execute
        tools_needed: Tools that might be used
        dependencies: Step indices this depends on
        result: Result after execution
    """
    description: str
    status: PlanStepStatus = PlanStepStatus.PENDING
    assigned_agent: Optional[str] = None
    tools_needed: List[str] = field(default_factory=list)
    dependencies: List[int] = field(default_factory=list)
    result: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "status": self.status.value,
            "assigned_agent": self.assigned_agent,
            "tools_needed": self.tools_needed,
            "dependencies": self.dependencies,
            "result": self.result
        }
    
    def start(self):
        """Mark step as started."""
        self.status = PlanStepStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()
    
    def complete(self, result: Optional[str] = None):
        """Mark step as completed."""
        self.status = PlanStepStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.utcnow()
    
    def fail(self, reason: Optional[str] = None):
        """Mark step as failed."""
        self.status = PlanStepStatus.FAILED
        self.result = reason
        self.completed_at = datetime.utcnow()


@dataclass
class ExecutionPlan:
    """
    A complete execution plan.
    
    Attributes:
        goal: Original goal/query
        steps: List of plan steps
        created_at: When plan was created
        version: Plan version (increments on replan)
    """
    goal: str
    steps: List[PlanStep]
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    
    def get_next_step(self) -> Optional[PlanStep]:
        """Get next pending step that has no unmet dependencies."""
        for i, step in enumerate(self.steps):
            if step.status == PlanStepStatus.PENDING:
                # Check dependencies
                deps_met = all(
                    self.steps[dep].status == PlanStepStatus.COMPLETED
                    for dep in step.dependencies
                    if dep < len(self.steps)
                )
                if deps_met:
                    return step
        return None
    
    def get_progress(self) -> Dict[str, int]:
        """Get plan progress statistics."""
        status_counts = {status.value: 0 for status in PlanStepStatus}
        for step in self.steps:
            status_counts[step.status.value] += 1
        return status_counts
    
    def is_complete(self) -> bool:
        """Check if all steps are done."""
        return all(
            step.status in [PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED]
            for step in self.steps
        )
    
    def to_markdown(self) -> str:
        """Format plan as markdown."""
        lines = [f"## Plan: {self.goal}", f"Version: {self.version}\n"]
        
        status_emoji = {
            PlanStepStatus.PENDING: "⏳",
            PlanStepStatus.IN_PROGRESS: "🔄",
            PlanStepStatus.COMPLETED: "✅",
            PlanStepStatus.FAILED: "❌",
            PlanStepStatus.SKIPPED: "⏭️"
        }
        
        for i, step in enumerate(self.steps, 1):
            emoji = status_emoji.get(step.status, "•")
            agent = f" [{step.assigned_agent}]" if step.assigned_agent else ""
            lines.append(f"{i}. {emoji} {step.description}{agent}")
        
        return "\n".join(lines)


class PlanningMixin:
    """
    Mixin that adds planning capabilities to agents.
    
    Features:
    - Automatic planning at start
    - Re-planning at configurable intervals
    - Plan progress tracking
    
    Usage:
        class MyAgent(PlanningMixin, BaseAgent):
            def __init__(self):
                PlanningMixin.__init__(self, planning_interval=5)
                ...
                
            async def run(self, query):
                # Generate initial plan
                plan = await self.generate_plan(query)
                
                while not plan.is_complete():
                    # Check if replan needed
                    should_replan = await self.maybe_replan(state)
                    if should_replan:
                        plan = should_replan
                    
                    # Execute next step
                    step = plan.get_next_step()
                    ...
    """
    
    # Prompt for plan generation
    PLANNING_PROMPT = """
Based on the user's query, create a step-by-step plan to answer it.

User Query: {query}

Current Context:
{context}

Available Agents:
{agents}

Create a plan with 3-7 specific, actionable steps.
For each step, specify:
- What to do
- Which agent should do it (if applicable)
- What tools might be needed

Respond in JSON format:
{{
    "steps": [
        {{
            "description": "Description of step",
            "assigned_agent": "agent_name or null",
            "tools_needed": ["tool1", "tool2"]
        }}
    ]
}}
"""
    
    def __init__(
        self,
        planning_interval: int = 5,
        max_plan_steps: int = 10,
        llm = None
    ):
        """
        Initialize planning capability.
        
        Args:
            planning_interval: Re-plan every N steps
            max_plan_steps: Maximum steps in a plan
            llm: LLM for plan generation
        """
        self.planning_interval = planning_interval
        self.max_plan_steps = max_plan_steps
        self._planning_llm = llm
        self.current_plan: Optional[ExecutionPlan] = None
        self.steps_since_plan = 0
    
    async def generate_plan(
        self,
        query: str,
        context: Optional[str] = None,
        available_agents: Optional[List[str]] = None
    ) -> ExecutionPlan:
        """
        Generate an execution plan for a query.
        
        Args:
            query: User query
            context: Current context
            available_agents: List of available agents
            
        Returns:
            ExecutionPlan with steps
        """
        if not self._planning_llm:
            # Fallback: simple single-step plan
            return ExecutionPlan(
                goal=query,
                steps=[PlanStep(description=f"Answer: {query}")]
            )
        
        # Format prompt
        prompt = self.PLANNING_PROMPT.format(
            query=query,
            context=context or "No context available",
            agents=", ".join(available_agents or ["general_agent"])
        )
        
        try:
            response = await self._planning_llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            plan_data = self._parse_plan_response(response_text)
            
            steps = []
            for step_data in plan_data.get("steps", [])[:self.max_plan_steps]:
                steps.append(PlanStep(
                    description=step_data.get("description", ""),
                    assigned_agent=step_data.get("assigned_agent"),
                    tools_needed=step_data.get("tools_needed", [])
                ))
            
            self.current_plan = ExecutionPlan(goal=query, steps=steps)
            self.steps_since_plan = 0
            
            logger.info(f"Generated plan with {len(steps)} steps")
            return self.current_plan
            
        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            # Fallback plan
            return ExecutionPlan(
                goal=query,
                steps=[PlanStep(description=f"Directly answer: {query}")]
            )
    
    async def maybe_replan(
        self,
        state: Dict[str, Any]
    ) -> Optional[ExecutionPlan]:
        """
        Check if replanning is needed and generate new plan if so.
        
        Args:
            state: Current agent state
            
        Returns:
            New plan if replanning occurred, None otherwise
        """
        self.steps_since_plan += 1
        
        # Check if interval reached
        if self.steps_since_plan >= self.planning_interval:
            logger.info(f"Replanning after {self.steps_since_plan} steps")
            
            # Get current progress
            completed = [
                s for s in self.current_plan.steps
                if s.status == PlanStepStatus.COMPLETED
            ] if self.current_plan else []
            
            # Generate new plan with context
            context = f"Completed steps: {[s.description for s in completed]}"
            new_plan = await self.generate_plan(
                query=state.get("original_query", ""),
                context=context
            )
            
            new_plan.version = (self.current_plan.version + 1) if self.current_plan else 1
            self.current_plan = new_plan
            
            return new_plan
        
        return None
    
    def _parse_plan_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        # Try to find JSON in response
        try:
            # Look for JSON block
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            elif "{" in response:
                # Find JSON object
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
            else:
                json_str = response
            
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            logger.warning(f"Could not parse plan JSON: {e}")
            return {"steps": []}
    
    def get_current_plan(self) -> Optional[ExecutionPlan]:
        """Get the current execution plan."""
        return self.current_plan
    
    def update_step_status(
        self,
        step_index: int,
        status: PlanStepStatus,
        result: Optional[str] = None
    ):
        """Update the status of a plan step."""
        if self.current_plan and step_index < len(self.current_plan.steps):
            step = self.current_plan.steps[step_index]
            step.status = status
            if result:
                step.result = result
            if status == PlanStepStatus.IN_PROGRESS:
                step.started_at = datetime.utcnow()
            elif status in [PlanStepStatus.COMPLETED, PlanStepStatus.FAILED]:
                step.completed_at = datetime.utcnow()
