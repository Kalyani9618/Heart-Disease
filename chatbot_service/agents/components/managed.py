"""
Managed Agents - Sub-agent coordination pattern.

Provides:
- ManagedAgent: Wrapper for sub-agents
- AgentManager: Coordinates multiple agents
- Task delegation and result aggregation

Based on smolagents managed_agents pattern.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class ManagedAgent:
    """
    A managed sub-agent that can be delegated tasks.
    
    Attributes:
        name: Unique agent identifier
        description: What this agent does (for LLM to choose)
        agent: The actual agent instance
        capabilities: List of capability tags
        model_id: LLM model used (if any)
    """
    name: str
    description: str
    agent: Any  # The actual agent instance
    capabilities: List[str] = field(default_factory=list)
    model_id: Optional[str] = None
    
    # Execution stats
    total_calls: int = 0
    successful_calls: int = 0
    total_latency_ms: float = 0.0
    
    def get_avg_latency(self) -> float:
        """Get average latency in ms."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls
    
    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "model_id": self.model_id,
            "stats": {
                "total_calls": self.total_calls,
                "successful_calls": self.successful_calls,
                "avg_latency_ms": self.get_avg_latency(),
                "success_rate": self.get_success_rate()
            }
        }


@dataclass 
class DelegationResult:
    """Result of delegating a task to an agent."""
    agent_name: str
    task: str
    success: bool
    result: Any
    latency_ms: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "task": self.task,
            "success": self.success,
            "result": self.result if self.success else None,
            "error": self.error,
            "latency_ms": self.latency_ms
        }


class AgentManager:
    """
    Manages sub-agents for complex task delegation.
    
    Features:
    - Agent registry with descriptions
    - Task delegation with routing
    - Parallel execution support
    - Result aggregation
    
    Usage:
        # Create managed agents
        web_agent = ManagedAgent(
            name="web_agent",
            description="Searches the web for information",
            agent=WebSearchAgent(),
            capabilities=["web_search", "url_fetch"]
        )
        
        # Create manager
        manager = AgentManager([web_agent, data_agent])
        
        # Delegate tasks
        result = await manager.delegate("web_agent", "Search for aspirin side effects")
        
        # Or let LLM choose agent
        result = await manager.auto_delegate(
            "What are the side effects of aspirin?",
            llm=my_llm
        )
    """
    
    # Prompt for auto-delegation
    DELEGATION_PROMPT = """
Choose the best agent to handle this task.

Task: {task}

Available Agents:
{agent_descriptions}

Respond with ONLY the agent name, nothing else.
"""
    
    def __init__(
        self,
        managed_agents: Optional[List[ManagedAgent]] = None,
        default_timeout: float = 30.0
    ):
        """
        Initialize the agent manager.
        
        Args:
            managed_agents: List of managed agents
            default_timeout: Default task timeout in seconds
        """
        self.agents: Dict[str, ManagedAgent] = {}
        self.default_timeout = default_timeout
        
        if managed_agents:
            for agent in managed_agents:
                self.register_agent(agent)
    
    def register_agent(self, agent: ManagedAgent):
        """
        Register a managed agent.
        
        Args:
            agent: ManagedAgent to register
        """
        self.agents[agent.name] = agent
        logger.info(f"Registered managed agent: {agent.name}")
    
    def unregister_agent(self, name: str):
        """Remove an agent from management."""
        if name in self.agents:
            del self.agents[name]
            logger.info(f"Unregistered agent: {name}")
    
    def get_agent(self, name: str) -> Optional[ManagedAgent]:
        """Get a managed agent by name."""
        return self.agents.get(name)
    
    def get_agent_descriptions(self) -> str:
        """
        Get formatted descriptions of all agents.
        
        For use in LLM prompts to help choose agents.
        """
        descriptions = []
        for agent in self.agents.values():
            caps = ", ".join(agent.capabilities) if agent.capabilities else "general"
            descriptions.append(
                f"- {agent.name}: {agent.description} (capabilities: {caps})"
            )
        return "\n".join(descriptions)
    
    def get_agents_by_capability(self, capability: str) -> List[ManagedAgent]:
        """Get all agents with a specific capability."""
        return [
            agent for agent in self.agents.values()
            if capability in agent.capabilities
        ]
    
    async def delegate(
        self,
        agent_name: str,
        task: str,
        timeout: Optional[float] = None,
        **kwargs
    ) -> DelegationResult:
        """
        Delegate a task to a specific agent.
        
        Args:
            agent_name: Name of the agent to delegate to
            task: Task description
            timeout: Task timeout (uses default if None)
            **kwargs: Additional arguments for the agent
            
        Returns:
            DelegationResult with success/failure and result
        """
        if agent_name not in self.agents:
            return DelegationResult(
                agent_name=agent_name,
                task=task,
                success=False,
                result=None,
                latency_ms=0,
                error=f"Unknown agent: {agent_name}"
            )
        
        managed = self.agents[agent_name]
        timeout = timeout or self.default_timeout
        start_time = datetime.utcnow()
        
        try:
            # Execute with timeout
            agent = managed.agent
            
            # Try different execution patterns
            if hasattr(agent, 'arun'):
                result = await asyncio.wait_for(
                    agent.arun(task, **kwargs),
                    timeout=timeout
                )
            elif hasattr(agent, 'run'):
                # Sync agent - run in executor
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: agent.run(task, **kwargs)),
                    timeout=timeout
                )
            elif hasattr(agent, 'execute'):
                result = await asyncio.wait_for(
                    agent.execute(task, **kwargs),
                    timeout=timeout
                )
            elif callable(agent):
                if asyncio.iscoroutinefunction(agent):
                    coro = agent(task, **kwargs)
                else:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    coro = loop.run_in_executor(None, lambda: agent(task, **kwargs))
                
                result = await asyncio.wait_for(
                    coro,
                    timeout=timeout
                )
            else:
                raise ValueError(f"Agent {agent_name} has no run/execute method")
            
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update stats
            managed.total_calls += 1
            managed.successful_calls += 1
            managed.total_latency_ms += latency
            
            return DelegationResult(
                agent_name=agent_name,
                task=task,
                success=True,
                result=result,
                latency_ms=latency
            )
            
        except asyncio.TimeoutError:
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            managed.total_calls += 1
            managed.total_latency_ms += latency
            
            return DelegationResult(
                agent_name=agent_name,
                task=task,
                success=False,
                result=None,
                latency_ms=latency,
                error=f"Task timed out after {timeout}s"
            )
            
        except Exception as e:
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            managed.total_calls += 1
            managed.total_latency_ms += latency
            
            logger.error(f"Agent {agent_name} failed: {e}")
            
            return DelegationResult(
                agent_name=agent_name,
                task=task,
                success=False,
                result=None,
                latency_ms=latency,
                error=str(e)
            )
    
    async def delegate_parallel(
        self,
        tasks: List[Dict[str, str]],
        timeout: Optional[float] = None
    ) -> List[DelegationResult]:
        """
        Delegate multiple tasks in parallel.
        
        Args:
            tasks: List of {"agent": name, "task": description}
            timeout: Optional timeout per task
            
        Returns:
            List of DelegationResult
        """
        async def run_task(task_spec: Dict[str, str]) -> DelegationResult:
            return await self.delegate(
                task_spec["agent"],
                task_spec["task"],
                timeout=timeout
            )
        
        results = await asyncio.gather(
            *[run_task(t) for t in tasks],
            return_exceptions=True
        )
        
        # Convert exceptions to DelegationResult
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed.append(DelegationResult(
                    agent_name=tasks[i].get("agent", "unknown"),
                    task=tasks[i].get("task", ""),
                    success=False,
                    result=None,
                    latency_ms=0,
                    error=str(result)
                ))
            else:
                processed.append(result)
        
        return processed
    
    async def auto_delegate(
        self,
        task: str,
        llm,
        timeout: Optional[float] = None
    ) -> DelegationResult:
        """
        Automatically choose best agent for a task using LLM.
        
        Args:
            task: Task description
            llm: LLM to use for agent selection
            timeout: Task timeout
            
        Returns:
            DelegationResult from chosen agent
        """
        if not self.agents:
            return DelegationResult(
                agent_name="none",
                task=task,
                success=False,
                result=None,
                latency_ms=0,
                error="No agents registered"
            )
        
        # Use LLM to choose agent
        prompt = self.DELEGATION_PROMPT.format(
            task=task,
            agent_descriptions=self.get_agent_descriptions()
        )
        
        try:
            response = await llm.ainvoke(prompt)
            chosen_agent = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            # Find matching agent (fuzzy match)
            matched = None
            for name in self.agents:
                if name.lower() in chosen_agent.lower() or chosen_agent.lower() in name.lower():
                    matched = name
                    break
            
            if not matched:
                # Default to first agent
                matched = list(self.agents.keys())[0]
                logger.warning(f"Could not match agent '{chosen_agent}', using {matched}")
            
            return await self.delegate(matched, task, timeout)
            
        except Exception as e:
            logger.error(f"Auto-delegation failed: {e}")
            # Fallback to first agent
            first_agent = list(self.agents.keys())[0]
            return await self.delegate(first_agent, task, timeout)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all managed agents."""
        return {
            "agents": [agent.to_dict() for agent in self.agents.values()],
            "total_agents": len(self.agents)
        }
