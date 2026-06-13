"""
Thinking Agent - Agent that thinks before acting.

Provides:
- ThinkingAgent: Uses <think> blocks before tool calls
- Structured reasoning traces
- Better decision making through explicit reasoning

Based on Test-Time Compute paper and DeepSeek-R1 patterns.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from enum import Enum
import re
import json
import logging

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Get current UTC time (replacement for deprecated datetime.utcnow())."""
    return datetime.now(timezone.utc)


class ReasoningType(Enum):
    """Types of reasoning in thinking process."""
    ANALYSIS = "analysis"        # Understanding the problem
    PLANNING = "planning"        # Deciding what to do
    EVALUATION = "evaluation"    # Assessing options
    REFLECTION = "reflection"    # Reviewing decisions
    CONCLUSION = "conclusion"    # Final decision


@dataclass
class ThinkingBlock:
    """
    A block of reasoning/thinking.
    
    Attributes:
        content: The thinking content
        reasoning_type: Type of reasoning
        key_insights: Important conclusions from this block
        timestamp: When this thinking occurred
    """
    content: str
    reasoning_type: ReasoningType = ReasoningType.ANALYSIS
    key_insights: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=_utc_now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "reasoning_type": self.reasoning_type.value,
            "insights": self.key_insights,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        return f"<think>\n{self.content}\n</think>"


@dataclass
class ThinkingResult:
    """
    Complete result from thinking agent.
    
    Attributes:
        answer: Final answer
        thinking_blocks: All reasoning blocks
        tool_calls: Tools that were called
        total_thinking_time_ms: Time spent thinking
    """
    answer: str
    thinking_blocks: List[ThinkingBlock]
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    total_thinking_time_ms: float = 0.0
    
    def get_reasoning_trace(self) -> str:
        """Get formatted reasoning trace."""
        lines = ["## Reasoning Trace\n"]
        for i, block in enumerate(self.thinking_blocks, 1):
            lines.append(f"### Step {i} ({block.reasoning_type.value})")
            lines.append(f"```\n{block.content}\n```")
            if block.key_insights:
                lines.append("**Key Insights:**")
                for insight in block.key_insights:
                    lines.append(f"- {insight}")
            lines.append("")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "thinking": [b.to_dict() for b in self.thinking_blocks],
            "tool_calls": self.tool_calls,
            "thinking_time_ms": self.total_thinking_time_ms
        }


class ThinkingAgent:
    """
    Agent that thinks before acting.
    
    Uses explicit <think>...</think> blocks to:
    1. Analyze the problem
    2. Plan the approach
    3. Evaluate options
    4. Reflect on decisions
    
    This leads to more reliable and interpretable responses.
    
    Usage:
        agent = ThinkingAgent(llm=my_llm, tools=[tool1, tool2])
        result = await agent.run("What drug interactions should I watch for?")
        
        print(result.answer)
        print(result.get_reasoning_trace())
    """
    
    THINKING_PROMPT = """You are a highly skilled medical reasoning assistant that follows a structured thinking process before acting.

IMPORTANT: Before taking any action, you MUST think through the problem systematically.
Wrap your reasoning in <think>...</think> tags.

Your thinking MUST follow this structured framework:

## STEP 1: UNDERSTAND (What is being asked?)
- Restate the question in your own words
- Identify key medical terms, conditions, or concepts
- Note any ambiguity or missing information

## STEP 2: GATHER (What do I know vs. what do I need?)
- List relevant medical knowledge you already have
- Identify specific gaps that require tool use
- Prioritize: safety-critical information first

## STEP 3: REASON (Apply clinical/medical logic)
- Consider differential diagnoses or multiple explanations
- Evaluate evidence quality and source reliability  
- Apply Bayesian reasoning: prior probability + new evidence
- Flag RED FLAGS: any potentially dangerous conditions or interactions

## STEP 4: VERIFY (Self-check your reasoning)
- Does my conclusion align with the evidence?
- Have I considered alternative explanations?
- Am I making unsupported assumptions?
- Could this advice cause harm if wrong?
- Rate your confidence: [certain/confident/likely/unsure/uncertain]

## STEP 5: ACT (Decide next action)
- Call a tool if more information is needed
- Provide answer if sufficiently confident
- Always include safety disclaimers for medical advice

Available tools:
{tool_descriptions}

After thinking, either:
- Call a tool: <tool_call>tool_name({{"param": "value"}})</tool_call>
- Provide final answer: <answer>Your complete response</answer>

User query: {query}

Previous context:
{context}

Begin with your structured thinking:"""

    def __init__(
        self,
        llm,
        tools: Optional[List[Any]] = None,
        max_thinking_rounds: int = 3,  # P1.2: Reduced from 5 to 3
        verbose: bool = True,
        early_exit_confidence: float = 0.85,  # P1.2: Exit early if confident
        max_context_chars: int = 12000,  # Context window management
        enable_self_verification: bool = True,  # Chain-of-Verification
    ):
        """
        Initialize the thinking agent.
        
        Args:
            llm: Language model for reasoning
            tools: List of tools available
            max_thinking_rounds: Max rounds of think-act (default: 3)
            verbose: Log thinking process
            early_exit_confidence: Exit early if confidence > this (default: 0.85)
            max_context_chars: Maximum context size before summarization
            enable_self_verification: Enable chain-of-verification step
        """
        self.llm = llm
        self.tools = {}
        for t in (tools or []):
            name = getattr(t, 'name', None) or getattr(t, '__name__', str(t))
            self.tools[name] = t
            
        self.max_thinking_rounds = max_thinking_rounds
        self.verbose = verbose
        self.early_exit_confidence = early_exit_confidence
        self.max_context_chars = max_context_chars
        self.enable_self_verification = enable_self_verification
        
        self.user_id = None  # To be set per run
        self.thinking_history: List[ThinkingBlock] = []
        self.tool_call_history: List[Dict[str, Any]] = []
        
        # P2.3: Token budget management
        self._token_budget = None
        try:
            from core.llm.token_budget import get_token_calculator
            self._token_budget = get_token_calculator()
            logger.info(f"✅ TokenBudget loaded (model={self._token_budget.model_name}, context={self._token_budget.limits['context']})")
        except Exception as e:
            logger.debug(f"TokenBudget not available: {e}")
    
    async def run(self, query: str, context: str = "", file_ids: Optional[List[str]] = None, user_id: Optional[str] = None) -> ThinkingResult:
        """
        Run the thinking agent.
        
        Args:
            query: User query
            context: Previous context
            file_ids: Optional list of file IDs to process
            user_id: Optional user ID for context and tool calling
            
        Returns:
            ThinkingResult with answer and reasoning
        """
        start_time = datetime.utcnow()
        self.user_id = user_id
        self.thinking_history = []
        self.tool_call_history = []
        
        current_context = context
        
        # Add file context if files are present
        if file_ids:
            file_context = f"\n\n[ATTACHED FILES]: The user has attached {len(file_ids)} file(s) with IDs: {', '.join(file_ids)}. Use the 'analyze_medical_image' or 'analyze_dicom_image' tools to examine them if they are images."
            current_context += file_context
            
        if self.user_id:
            user_context = f"\n\n[SYSTEM CONTEXT]: current_user_id = \"{self.user_id}\". Use this ID for any tool calls that require a 'user_id' or 'patient_id' parameter."
            current_context += user_context
        
        for round_num in range(self.max_thinking_rounds):
            if self.verbose:
                logger.info(f"Thinking round {round_num + 1}/{self.max_thinking_rounds}")
            
            # Generate response
            response = await self._generate_response(query, current_context)
            
            # Parse thinking and actions
            thinking, action = self._parse_response(response)
            
            if thinking:
                block = ThinkingBlock(
                    content=thinking,
                    reasoning_type=self._classify_thinking(thinking),
                    key_insights=self._extract_insights(thinking)
                )
                self.thinking_history.append(block)
                
                if self.verbose:
                    logger.info(f"💭 Thinking: {thinking[:100]}...")
                
                # P1.2: Early exit check - if we're confident, skip remaining rounds
                confidence = self._extract_confidence(thinking)
                if confidence > self.early_exit_confidence and round_num > 0:
                    logger.info(f"P1.2: Early exit at round {round_num + 1} (confidence={confidence:.2f})")
                    
                    # Run self-verification before early exit if enabled
                    if self.enable_self_verification:
                        verified = await self._self_verify(query, current_context, thinking)
                        if not verified:
                            logger.info("Self-verification failed, continuing reasoning...")
                            continue
                    
                    # Force a final answer generation
                    return await self._generate_final_answer(query, current_context, start_time)
            
            # Check for final answer
            if action.get("type") == "answer":
                return ThinkingResult(
                    answer=action["content"],
                    thinking_blocks=self.thinking_history,
                    tool_calls=self.tool_call_history,
                    total_thinking_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
                )
            
            # Execute tool call
            if action.get("type") == "tool_call":
                tool_name = action["tool_name"]
                tool_args = action["args"]
                
                if self.verbose:
                    logger.info(f"🔧 Tool call: {tool_name}")
                
                result = await self._execute_tool(tool_name, tool_args)
                
                self.tool_call_history.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result
                })
                
                # Add result to context (with context window management)
                new_content = f"\n\nAssistant Step {round_num+1}:\n{response}\n\nTool Result:\n{result}"
                current_context = self._manage_context_window(current_context, new_content)
        
        # Max rounds reached - run verification then generate final answer
        if self.enable_self_verification and self.thinking_history:
            last_thinking = self.thinking_history[-1].content if self.thinking_history else ""
            await self._self_verify(query, current_context, last_thinking)
        
        return await self._generate_final_answer(query, current_context, start_time)
    
    async def _generate_response(self, query: str, context: str) -> str:
        """Generate next response from LLM with token budget awareness."""
        # P2.3: Use token budget to determine safe context size
        if self._token_budget:
            budget = self._token_budget.calculate(
                prompt=query,
                num_documents=len(self.tool_call_history),
                avg_doc_tokens=200,
                system_prompt_tokens=300  # THINKING_PROMPT overhead
            )
            if not budget["safe"]:
                logger.warning(f"⚠️ Token budget tight: {budget['total_used']}/{budget['total_limit']} used")
                # Trim context to fit within budget
                max_context_tokens = budget["total_limit"] - budget["prompt_tokens"] - budget["system_prompt_tokens"] - budget["output_budget"]
                max_chars = max(2000, max_context_tokens * 4)  # ~4 chars per token
                if len(context) > max_chars:
                    context = context[:max_chars] + "\n...[context trimmed for token budget]"
        
        prompt = self.THINKING_PROMPT.format(
            tool_descriptions=self._get_tool_descriptions(),
            query=query,
            context=context or "No previous context."
        )
        
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"<think>Error calling LLM: {e}</think><answer>I encountered an error.</answer>"
    
    def _parse_response(self, response: str) -> tuple:
        """
        Parse thinking and action from response.
        Supports multi-line JSON arguments and nested structures.
        
        Returns:
            Tuple of (thinking_content, action_dict)
        """
        thinking = None
        action = {}
        
        # Extract thinking (support multiple think blocks)
        think_matches = re.findall(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
        if think_matches:
            thinking = "\n---\n".join(m.strip() for m in think_matches)
        
        # Check for answer
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            action = {
                "type": "answer",
                "content": answer_match.group(1).strip()
            }
            return thinking, action
        
        # Check for tool call - enhanced regex to handle multi-line JSON
        # Pattern 1: Standard format <tool_call>name({...})</tool_call>
        tool_match = re.search(
            r'<tool_call>\s*(\w+)\s*\(\s*(\{.*?\})\s*\)\s*</tool_call>',
            response, re.DOTALL | re.IGNORECASE
        )
        
        # Pattern 2: tool_call with just name and args without parens
        if not tool_match:
            tool_match = re.search(
                r'<tool_call>\s*(\w+)\s*\n\s*(\{.*?\})\s*</tool_call>',
                response, re.DOTALL | re.IGNORECASE
            )
        
        # Pattern 3: Less strict - tool name followed by JSON-like content
        if not tool_match:
            tool_match = re.search(
                r'<tool_call>\s*(\w+)\s*\(([^)]*?)\)\s*</tool_call>',
                response, re.DOTALL | re.IGNORECASE
            )
        
        if tool_match:
            tool_name = tool_match.group(1).strip()
            args_str = tool_match.group(2).strip()
            
            # Try multiple JSON parsing strategies
            tool_args = self._parse_tool_args(args_str)
            
            action = {
                "type": "tool_call",
                "tool_name": tool_name,
                "args": tool_args
            }
        
        return thinking, action
    
    def _parse_tool_args(self, args_str: str) -> Dict[str, Any]:
        """
        Parse tool arguments with multiple fallback strategies.
        
        Strategies:
        1. Direct JSON parse
        2. Fix common JSON errors (single quotes, trailing commas)
        3. Extract from markdown code block
        4. Simple key=value parsing
        """
        if not args_str:
            return {}
        
        # Strategy 1: Direct JSON parse
        try:
            return json.loads(args_str)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Fix common JSON errors
        try:
            fixed = args_str.replace("'", '"')  # Single to double quotes
            fixed = re.sub(r',\s*}', '}', fixed)  # Remove trailing commas
            fixed = re.sub(r',\s*]', ']', fixed)  # Remove trailing commas in arrays
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Extract JSON from code block
        code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', args_str, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Simple key=value parsing
        return self._parse_simple_args(args_str)
    
    def _parse_simple_args(self, args_str: str) -> Dict[str, Any]:
        """Parse simple key=value arguments."""
        args = {}
        for match in re.finditer(r'(\w+)\s*[=:]\s*["\']?([^"\',:}]+)["\']?', args_str):
            args[match.group(1)] = match.group(2).strip()
        return args
    
    def _classify_thinking(self, thinking: str) -> ReasoningType:
        """Classify the type of thinking."""
        thinking_lower = thinking.lower()
        
        if any(word in thinking_lower for word in ["plan", "steps", "first", "then", "next"]):
            return ReasoningType.PLANNING
        elif any(word in thinking_lower for word in ["evaluate", "compare", "option", "better"]):
            return ReasoningType.EVALUATION
        elif any(word in thinking_lower for word in ["reflect", "review", "consider", "mistake"]):
            return ReasoningType.REFLECTION
        elif any(word in thinking_lower for word in ["conclude", "therefore", "answer is", "final"]):
            return ReasoningType.CONCLUSION
        else:
            return ReasoningType.ANALYSIS
    
    def _extract_insights(self, thinking: str) -> List[str]:
        """Extract key insights from thinking."""
        insights = []
        
        # Look for bullet points or numbered items
        for match in re.finditer(r'[-•*]\s+(.+?)(?:\n|$)', thinking):
            insight = match.group(1).strip()
            if len(insight) > 10:
                insights.append(insight)
        
        # Look for "I should", "I need to", "Important:" patterns
        for pattern in [
            r'I (?:should|need to|must) (.+?)(?:\.|$)',
            r'Important:?\s*(.+?)(?:\.|$)',
            r'Key (?:point|insight):?\s*(.+?)(?:\.|$)'
        ]:
            for match in re.finditer(pattern, thinking, re.IGNORECASE):
                insight = match.group(1).strip()
                if len(insight) > 10:
                    insights.append(insight)
        
        return insights[:5]  # Max 5 insights
    
    def _extract_confidence(self, thinking: str) -> float:
        """Extract confidence level from thinking text.
        
        P1.2: Used to determine if we should exit early based on
        the agent's perceived confidence in its reasoning.
        
        Enhanced: Considers negation context ("I'm NOT certain"),
        proximity of confidence words to negation words, and
        structured confidence statements.
        
        Returns:
            Float between 0.0 and 1.0 indicating confidence level
        """
        thinking_lower = thinking.lower()
        
        # Check for explicit confidence statements first
        # e.g., "confidence: 0.85" or "confidence level: high"
        explicit_match = re.search(
            r'confidence[:\s]+([\d.]+)',
            thinking_lower
        )
        if explicit_match:
            try:
                val = float(explicit_match.group(1))
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                pass
        
        # Structured confidence levels
        level_match = re.search(
            r'confidence[:\s]+(very high|high|medium|moderate|low|very low)',
            thinking_lower
        )
        if level_match:
            level_map = {
                "very high": 0.95, "high": 0.85,
                "medium": 0.65, "moderate": 0.65,
                "low": 0.35, "very low": 0.15
            }
            return level_map.get(level_match.group(1), 0.60)
        
        # Negation-aware keyword detection
        negation_words = {"not", "don't", "doesn't", "isn't", "aren't", "no", "never", "hardly"}
        
        # Confidence keywords with context-aware scoring
        confidence_patterns = [
            (r"\b(certain|definitely|absolutely)\b", 0.95),
            (r"\b(confident|sure|clear(?:ly)?)\b", 0.88),
            (r"\bstrongly suggest\b", 0.85),
            (r"\b(likely|probably|suggests?)\b", 0.75),
            (r"\b(appears?|seems?)\b", 0.65),
            (r"\b(think|believe)\b", 0.60),
            (r"\b(possibly|might|could)\b", 0.50),
            (r"\b(maybe|perhaps)\b", 0.45),
            (r"\b(unsure|uncertain|unclear)\b", 0.30),
            (r"\b(don't know|no idea|cannot determine)\b", 0.20),
            (r"\b(impossible|cannot|no evidence)\b", 0.10),
        ]
        
        best_confidence = 0.60  # default
        found_any = False
        
        for pattern, conf in confidence_patterns:
            matches = list(re.finditer(pattern, thinking_lower))
            for match in matches:
                found_any = True
                # Check for negation within 3 words before the match
                start = max(0, match.start() - 30)
                prefix = thinking_lower[start:match.start()]
                prefix_words = prefix.split()
                
                has_negation = any(w in negation_words for w in prefix_words[-3:])
                
                if has_negation:
                    # Invert confidence: "not certain" -> low confidence
                    adjusted = max(1.0 - conf, 0.15)
                    if adjusted > best_confidence or not found_any:
                        best_confidence = adjusted
                else:
                    if conf > best_confidence:
                        best_confidence = conf
                
                break  # One match per pattern is sufficient
        
        return best_confidence
    
    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool and return result."""
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"
        
        tool = self.tools[tool_name]
        
        # Inject user_id if missing but required by tool signature
        if self.user_id and "user_id" not in args:
             import inspect
             try:
                 sig = inspect.signature(tool if not hasattr(tool, 'forward') else tool.forward)
                 if "user_id" in sig.parameters:
                     args["user_id"] = self.user_id
                     logger.debug(f"💉 Injected user_id={self.user_id} into tool {tool_name}")
             except Exception:
                 pass
        
        try:
            if hasattr(tool, 'aforward'):
                result = await tool.aforward(**args)
            elif hasattr(tool, 'forward'):
                result = tool.forward(**args)
            elif hasattr(tool, 'execute'):
                result = await tool.execute(**args)
            elif callable(tool):
                import inspect
                if inspect.iscoroutinefunction(tool):
                    result = await tool(**args)
                else:
                    result = tool(**args)
                    if inspect.isawaitable(result):
                        result = await result
            else:
                return f"Error: Tool '{tool_name}' not callable"
            
            if hasattr(result, 'data'):
                return json.dumps(result.data, default=str)
            return str(result)
            
        except Exception as e:
            return f"Error executing {tool_name}: {e}"
    
    async def _generate_final_answer(
        self,
        query: str,
        context: str,
        start_time: datetime
    ) -> ThinkingResult:
        """Generate final answer after max rounds with structured synthesis."""
        # Summarize context if it's too long
        summarized_context = self._summarize_context(context) if len(context) > self.max_context_chars else context
        
        prompt = f"""Based on all your reasoning and tool results, provide your final comprehensive answer.

Original Query: {query}

Context & Evidence Gathered:
{summarized_context}

Before answering, briefly verify:
1. Have I answered the original question?
2. Is my answer supported by the evidence gathered?
3. Have I included appropriate safety disclaimers for medical content?
4. Are my sources cited?

<answer>Your complete, well-structured final answer here. Include:
- Direct answer to the question
- Supporting evidence and reasoning
- Relevant caveats or limitations
- Safety disclaimers if medical in nature
- Source citations where applicable</answer>"""
        
        response = await self.llm.ainvoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else response_text
        
        return ThinkingResult(
            answer=answer,
            thinking_blocks=self.thinking_history,
            tool_calls=self.tool_call_history,
            total_thinking_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
        )
    
    def _get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions."""
        if not self.tools:
            return "No tools available."
        
        descriptions = []
        for tool in self.tools.values():
            name = getattr(tool, 'name', None) or getattr(tool, '__name__', str(tool))
            desc = getattr(tool, 'description', "") or getattr(tool, '__doc__', "")
            descriptions.append(f"- {name}: {desc}")
        
        return "\n".join(descriptions)
    
    def get_thinking_trace(self) -> str:
        """Get formatted thinking trace from last run."""
        if not self.thinking_history:
            return "No thinking recorded."
        
        lines = []
        for i, block in enumerate(self.thinking_history, 1):
            lines.append(f"## Step {i}: {block.reasoning_type.value.title()}")
            lines.append(f"```\n{block.content}\n```\n")
        
        return "\n".join(lines)

    def _manage_context_window(self, current_context: str, new_content: str) -> str:
        """
        Manage context window to prevent overflow.
        
        When context exceeds max_context_chars, summarize older entries
        while preserving recent tool results and thinking.
        
        Args:
            current_context: Existing context
            new_content: New content to add
            
        Returns:
            Managed context string within limits
        """
        combined = current_context + new_content
        
        if len(combined) <= self.max_context_chars:
            return combined
        
        # Split into sections (by "Assistant Step" markers)
        sections = re.split(r'(\n\nAssistant Step \d+:)', combined)
        
        if len(sections) <= 3:
            # Too few sections to summarize, just truncate
            return combined[-self.max_context_chars:]
        
        # Keep last 2 sections in full, summarize earlier ones
        preserved_sections = sections[-4:]  # Last 2 step markers + content
        older_sections = sections[:-4]
        
        # Summarize older sections
        older_text = "".join(older_sections)
        summary = self._quick_summarize(older_text)
        
        return f"[Earlier context summary]: {summary}\n\n{''.join(preserved_sections)}"
    
    def _quick_summarize(self, text: str, max_length: int = 500) -> str:
        """
        Quick local summarization without LLM call.
        Extracts key sentences based on information density.
        """
        sentences = re.split(r'[.!?]+\s+', text)
        if not sentences:
            return text[:max_length]
        
        # Score sentences by keyword density
        scored = []
        important_words = {
            "found", "result", "error", "success", "failed", "extracted",
            "insight", "important", "key", "critical", "note", "warning",
            "conclusion", "source", "url", "data", "evidence",
        }
        
        for sent in sentences:
            if len(sent.strip()) < 20:
                continue
            words = set(sent.lower().split())
            score = len(words & important_words)
            scored.append((score, sent.strip()))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        result = []
        current_len = 0
        for _, sent in scored:
            if current_len + len(sent) > max_length:
                break
            result.append(sent)
            current_len += len(sent)
        
        return ". ".join(result) if result else text[:max_length]
    
    def _summarize_context(self, context: str) -> str:
        """
        Summarize long context for final answer generation.
        Preserves tool results and key findings.
        """
        if len(context) <= self.max_context_chars:
            return context
        
        # Extract tool results (they are the most valuable)
        tool_results = re.findall(
            r'Tool Result:\s*(.*?)(?=\n\nAssistant Step|$)',
            context, re.DOTALL
        )
        
        # Keep tool results, summarize the rest
        preserved = "\n\n".join(
            f"Tool Result: {tr[:300]}" for tr in tool_results[-5:]  # Last 5 results
        )
        
        # Summarize non-tool-result content
        non_tool = re.sub(r'Tool Result:.*?(?=\n\nAssistant Step|$)', '', context, flags=re.DOTALL)
        summary = self._quick_summarize(non_tool, max_length=800)
        
        return f"[Context Summary]: {summary}\n\n[Key Tool Results]:\n{preserved}"
    
    async def _self_verify(self, query: str, context: str, thinking: str) -> bool:
        """
        Chain-of-Verification: Verify the thinking before committing to an answer.
        
        Asks the model to critique its own reasoning for:
        - Logical consistency
        - Evidence support
        - Safety concerns (medical context)
        - Potential biases
        
        Args:
            query: Original query
            context: Current context
            thinking: The thinking to verify
            
        Returns:
            True if verification passes, False if issues found
        """
        verification_prompt = f"""Review this reasoning for issues. Be critical but fair.

Original Query: {query}

Reasoning to Verify:
{thinking[:1500]}

Check for:
1. LOGICAL ERRORS: Are there contradictions or unsupported leaps?
2. EVIDENCE GAPS: Are conclusions supported by the gathered evidence?
3. SAFETY RISKS: Could this advice be harmful if followed? (CRITICAL for medical)
4. BIAS: Is the reasoning balanced, or does it overlook alternatives?
5. COMPLETENESS: Does the reasoning address the full query?

Respond with:
<verification>
PASSED or ISSUES_FOUND
Brief explanation (1-2 sentences)
</verification>"""

        try:
            response = await self.llm.ainvoke(verification_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            match = re.search(r'<verification>\s*(.*?)\s*</verification>', response_text, re.DOTALL)
            if match:
                verdict = match.group(1).strip()
                passed = "PASSED" in verdict.upper()
                
                if not passed:
                    logger.info(f"\u26a0\ufe0f Self-verification found issues: {verdict[:200]}")
                    # Add verification result as a reflection block
                    self.thinking_history.append(ThinkingBlock(
                        content=f"Self-verification: {verdict}",
                        reasoning_type=ReasoningType.REFLECTION,
                        key_insights=["Verification flagged potential issues"]
                    ))
                else:
                    logger.info("\u2705 Self-verification passed")
                
                return passed
            
            # If can't parse response, assume passed (don't block on verification)
            return True
            
        except Exception as e:
            logger.warning(f"Self-verification failed: {e}")
            return True  # Don't block on verification failures


# Factory function
def create_thinking_agent(llm, tools: List[Any] = None) -> ThinkingAgent:
    """Create a configured thinking agent."""
    return ThinkingAgent(llm=llm, tools=tools or [], verbose=True)
