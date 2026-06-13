"""
Medical Prompt Registry

Centralized storage for all medical-related prompts used in the RAG system.
This separates presentation logic from orchestration logic as per the architectural plan.
"""

from typing import Dict, Optional


class PromptBuilder:
    """
    Builder class for handling prompt construction with variable injection
    and PII placeholder management.
    """
    
    # System prompt for healthcare context
    SYSTEM_PROMPT = """You are a knowledgeable healthcare assistant for Cardio AI.
Guidelines:
1. Provide evidence-based information using the context provided.
2. Always cite sources (e.g., [Source: AHA Guidelines]).
3. If the Context mentions "Knowledge Graph", treat it as high-confidence structured data.
4. Never diagnose; advise consulting a professional.
5. If the user shares symptoms, check the "Triage" section of the context."""

    # Template for building augmented prompts
    RAG_PROMPT_TEMPLATE = """{system_prompt}

=== RELEVANT CONTEXT ===
{context}

=== END CONTEXT ===

CONVERSATION HISTORY:
{history}

USER QUERY: {query}
Response:"""

    # Template for medical context formatting
    MEDICAL_CONTEXT_TEMPLATE = """Medical Knowledge:
{medical_docs}

Drug Information:
{drug_info}

Patient History:
{patient_memories}

Knowledge Graph:
{graph_context}

Drug Interactions:
{drug_interactions}"""

    # Template for triage context
    TRIAGE_TEMPLATE = """Based on the provided symptoms and context:
- Severity level: {severity}
- Recommended action: {action}
- Timeframe: {timeframe}
- Additional notes: {notes}"""

    def build_rag_prompt(
        self,
        query: str,
        context: str = "",
        history: str = "",
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Build a complete RAG prompt with all components.
        
        Args:
            query: User's query
            context: Retrieved context from RAG
            history: Conversation history
            system_prompt: Custom system prompt (uses default if None)
            
        Returns:
            Complete prompt string ready for LLM
        """
        system = system_prompt or self.SYSTEM_PROMPT
        
        return self.RAG_PROMPT_TEMPLATE.format(
            system_prompt=system,
            context=context,
            history=history,
            query=query
        )

    def format_medical_context(
        self,
        medical_docs: str = "",
        drug_info: str = "",
        patient_memories: str = "",
        graph_context: str = "",
        drug_interactions: str = ""
    ) -> str:
        """
        Format medical context from various sources.
        
        Args:
            medical_docs: Medical documents retrieved
            drug_info: Drug information retrieved
            patient_memories: Patient history/memory context
            graph_context: Knowledge graph context
            drug_interactions: Drug interaction warnings
            
        Returns:
            Formatted context string
        """
        return self.MEDICAL_CONTEXT_TEMPLATE.format(
            medical_docs=medical_docs or "No medical documents available.",
            drug_info=drug_info or "No drug information available.",
            patient_memories=patient_memories or "No patient history available.",
            graph_context=graph_context or "No knowledge graph data.",
            drug_interactions=drug_interactions or "No drug interactions detected."
        )

    def build_triage_context(
        self,
        severity: str = "Unknown",
        action: str = "Consult a healthcare professional",
        timeframe: str = "As soon as possible",
        notes: str = "No additional notes"
    ) -> str:
        """
        Build triage context based on symptoms.
        
        Args:
            severity: Severity level (e.g., low, medium, high, critical)
            action: Recommended action
            timeframe: Recommended timeframe
            notes: Additional notes
            
        Returns:
            Triage context string
        """
        return self.TRIAGE_TEMPLATE.format(
            severity=severity,
            action=action,
            timeframe=timeframe,
            notes=notes
        )


# Singleton instance for easy access
_prompt_builder_instance: Optional[PromptBuilder] = None


def get_prompt_builder() -> PromptBuilder:
    """
    Get the singleton prompt builder instance.
    
    Returns:
        PromptBuilder instance
    """
    global _prompt_builder_instance
    if _prompt_builder_instance is None:
        _prompt_builder_instance = PromptBuilder()
    return _prompt_builder_instance


# Convenience functions for common prompt operations
def get_system_prompt() -> str:
    """
    Get the default medical system prompt.
    
    Returns:
        System prompt string
    """
    return PromptBuilder.SYSTEM_PROMPT


def build_medical_rag_prompt(
    query: str,
    context: str = "",
    history: str = ""
) -> str:
    """
    Convenience function to build a medical RAG prompt.
    
    Args:
        query: User's query
        context: Retrieved context
        history: Conversation history
        
    Returns:
        Complete RAG prompt
    """
    builder = get_prompt_builder()
    return builder.build_rag_prompt(query, context, history)