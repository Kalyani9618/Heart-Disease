"""
Medical Planner Component.
Adapts the 'Plan-and-Execute' pattern from the RAG pipeline.
"""
from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from core.llm.llm_gateway import get_llm_gateway


# --- Data Models ---
class Plan(BaseModel):
    steps: List[str] = Field(description="List of steps to follow, in logical order.")

class ReplanningResult(BaseModel):
    plan: Plan = Field(description="The updated plan to follow.")
    explanation: str = Field(description="Why the plan was updated.")

# --- Prompts (Adapted from Notebook) ---
PLANNER_PROMPT = """
For the given medical query: {question}
Create a step-by-step plan to answer it accurately.
The plan must involve retrieving information or performing analysis.
Do not skip steps. The final step must be generating the answer.

Output format: JSON with a 'steps' list.
"""

REPLANNER_PROMPT = """
Your original plan was: {plan}
You have completed: {past_steps}
Current context gathered: {aggregated_context}

Update the plan to finish the task. 
- If you have enough information, the next step should be "Final Answer".
- If information is missing, add specific retrieval steps.
- Remove completed steps.

Output format: JSON with 'plan' (object with 'steps') and 'explanation'.
"""

class MedicalPlanner:
    def __init__(self):
        self.llm = get_llm_gateway()
        self.parser = JsonOutputParser(pydantic_object=Plan)
        self.replan_parser = JsonOutputParser(pydantic_object=ReplanningResult)

    async def create_initial_plan(self, question: str) -> List[str]:
        """Generates the initial execution plan."""
        prompt = PromptTemplate(
            template=PLANNER_PROMPT,
            input_variables=["question"]
        ).format(question=question)
        
        # Call LLM (Assuming LLM Gateway supports text generation)
        response = await self.llm.generate(prompt)
        
        try:
            parsed = self.parser.parse(response)
            return parsed['steps']
        except Exception:
            # Fallback for parsing errors
            return ["Retrieve information about " + question, "Generate answer"]

    async def update_plan(
        self, 
        question: str, 
        current_plan: List[str], 
        past_steps: List[str], 
        context: str
    ) -> List[str]:
        """Updates the plan based on new information (Replanning)."""
        prompt = PromptTemplate(
            template=REPLANNER_PROMPT,
            input_variables=["plan", "past_steps", "aggregated_context"]
        ).format(
            plan=str(current_plan),
            past_steps=str(past_steps),
            aggregated_context=context[:2000] # Truncate context to save tokens
        )
        
        response = await self.llm.generate(prompt)
        
        try:
            parsed = self.replan_parser.parse(response)
            return parsed['plan']['steps']
        except Exception:
            return ["Generate answer based on gathered info"]
