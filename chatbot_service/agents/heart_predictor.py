"""
HeartDiseasePredictor - Connects RAG with MedGemma for Risk Assessment

This predictor integrates:
- HeartDiseaseRAG for guideline retrieval
- MemoriRAGBridge for patient history
- LLMGateway for MedGemma generation
- HallucinationGrader for response validation

Architecture:
    1. RETRIEVE: Query HeartDiseaseRAG for relevant guidelines
    2. MEMORY: Get patient history from MemoriRAGBridge
    3. AUGMENT: Build context-rich prompt
    4. GENERATE: Call MedGemma via LLMGateway
    5. VALIDATE: Check response grounding with HallucinationGrader
"""

import logging
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class HeartRiskResult:
    """Result from heart disease risk prediction."""
    risk_level: str = "Unknown"
    confidence: float = 0.0
    response: str = ""
    reasoning: str = ""
    contributing_factors: str = ""
    citations: List[str] = field(default_factory=list)
    is_grounded: bool = False
    needs_medical_attention: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class HeartDiseasePredictor:
    """
    Heart Disease Risk Predictor using RAG-augmented MedGemma.
    
    This predictor:
    - Retrieves relevant heart disease guidelines from ChromaDB vector store
    - Incorporates patient history from memory system
    - Generates grounded, explainable risk assessments
    - Validates responses to prevent hallucinations
    
    Integration points:
    - rag.rag_engines.HeartDiseaseRAG
    - rag.memori_integration.MemoriRAGBridge
    - core.llm.llm_gateway.LLMGateway
    - core.safety.hallucination_grader.HallucinationGrader
    """
    
    def __init__(
        self,
        heart_rag=None,
        llm_gateway=None,
        memori_bridge=None,
        hallucination_grader=None,
        auto_initialize: bool = True
    ):
        """
        Initialize HeartDiseasePredictor.
        
        Args:
            heart_rag: HeartDiseaseRAG instance
            llm_gateway: LLMGateway instance
            memori_bridge: MemoriRAGBridge for patient history
            hallucination_grader: HallucinationGrader for validation
            auto_initialize: Whether to auto-load guidelines
        """
        # Initialize RAG engine
        if heart_rag is None:
            from rag.rag_engines import get_heart_disease_rag
            heart_rag = get_heart_disease_rag()
        self._rag = heart_rag
        
        # Initialize LLM Gateway
        if llm_gateway is None:
            try:
                from core.llm.llm_gateway import get_llm_gateway
                llm_gateway = get_llm_gateway()
            except Exception as e:
                logger.error(f"Failed to get LLM Gateway: {e}")
                raise RuntimeError("LLMGateway is required for HeartDiseasePredictor")
        self._llm = llm_gateway
        
        # Initialize Memori Bridge (optional)
        self._memori = memori_bridge
        if self._memori is None:
            try:
                from core.dependencies import DIContainer
                container = DIContainer.get_instance()
                self._memori = container.get_service('memori_bridge')
            except Exception:
                logger.warning("MemoriRAGBridge not available. Patient history disabled.")
        
        # Initialize HallucinationGrader (optional)
        self._grader = hallucination_grader
        if self._grader is None:
            try:
                from core.safety.hallucination_grader import HallucinationGrader
                self._grader = HallucinationGrader()
            except Exception:
                logger.warning("HallucinationGrader not available. Validation disabled.")
        
        # Auto-load guidelines if requested
        if auto_initialize and not self._rag.is_ready():
            self._rag.load_heart_disease_guidelines()
            logger.info("✅ Heart disease guidelines loaded")
    
    async def predict_risk(
        self,
        patient_symptoms: str,
        user_id: Optional[str] = None,
        include_history: bool = True,
        validate_response: bool = True,
        max_guidelines: int = 5
    ) -> HeartRiskResult:
        """
        Predict heart disease risk for a patient case.
        
        This method implements the full RAG pipeline:
        1. RETRIEVE: Get relevant guidelines from vector store
        2. MEMORY: Get patient history (if available)
        3. AUGMENT: Build structured prompt
        4. GENERATE: Call MedGemma
        5. VALIDATE: Check for hallucinations
        
        Args:
            patient_symptoms: Description of patient symptoms/case
            user_id: User ID for patient history lookup
            include_history: Whether to include patient history
            validate_response: Whether to run hallucination check
            max_guidelines: Maximum guidelines to retrieve
            
        Returns:
            HeartRiskResult with risk assessment and metadata
        """
        start_time = datetime.now()
        
        # Step 1: RETRIEVE - Get relevant heart disease guidelines
        logger.info(f"🔎 Searching knowledge base for: {patient_symptoms[:100]}...")
        retrieval_result = self._rag.retrieve_context(
            query=patient_symptoms,
            top_k=max_guidelines
        )
        
        # Format retrieved guidelines - handle both dict and object return types
        if isinstance(retrieval_result, dict):
            # HeartDiseaseRAG returns dict with 'context' and 'sources'
            guidelines_context = retrieval_result.get("context", "")
            citations = retrieval_result.get("sources", [])
            # Create a simple object-like structure for later use
            class RetrievalResult:
                def __init__(self, ctx, srcs):
                    self.documents = [ctx] if ctx else []
                    self.sources = srcs
                    self.scores = [0.5] * len(srcs) if srcs else []
            retrieval_result = RetrievalResult(guidelines_context, citations)
        elif hasattr(retrieval_result, 'documents') and retrieval_result.documents:
            guidelines_context = "\n".join(
                f"- [{src}] {doc}" 
                for doc, src in zip(retrieval_result.documents, retrieval_result.sources)
            )
            citations = list(set(retrieval_result.sources))
        else:
            guidelines_context = "No specific guidelines found. Use general medical knowledge."
            citations = []
            # Create empty result object
            class RetrievalResult:
                documents = []
                sources = []
                scores = []
            retrieval_result = RetrievalResult()
        
        # Step 2: MEMORY - Get patient history (if available)
        patient_history = ""
        if include_history and user_id and self._memori:
            try:
                memory_context = self._memori.get_context_for_query(
                    query=patient_symptoms,
                    user_id=user_id,
                    max_memories=3
                )
                if memory_context:
                    patient_history = f"\n### Patient History:\n{memory_context}"
            except Exception as e:
                logger.warning(f"Failed to get patient history: {e}")
        
        # Step 3: AUGMENT - Build structured prompt
        prompt = self._build_prediction_prompt(
            symptoms=patient_symptoms,
            guidelines=guidelines_context,
            patient_history=patient_history
        )
        
        # Step 4: GENERATE - Call MedGemma via LLMGateway
        logger.info("🧠 MedGemma is analyzing the case...")
        try:
            response = await self._llm.generate(
                prompt=prompt,
                content_type="medical_analysis",
                user_id=user_id
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return HeartRiskResult(
                risk_level="Error",
                response=f"Analysis failed: {str(e)}",
                metadata={"error": str(e)}
            )
        
        # Step 5: VALIDATE - Check for hallucinations
        is_grounded = True
        if validate_response and self._grader:
            try:
                is_grounded = await self._grader.grade(
                    answer=response,
                    context=guidelines_context + patient_history
                )
                if not is_grounded:
                    logger.warning("⚠️ Response may contain hallucinations")
            except Exception as e:
                logger.warning(f"Hallucination check failed: {e}")
        
        # Parse risk level from response
        risk_level = self._extract_risk_level(response)
        needs_attention = risk_level in ["High", "Critical", "Emergency"]
        
        # Extract contributing factors explanation
        contributing_factors = self._extract_contributing_factors(response)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return HeartRiskResult(
            risk_level=risk_level,
            confidence=max(retrieval_result.scores) if retrieval_result.scores else 0.5,
            response=response,
            reasoning=self._get_reasoning_section(response),
            contributing_factors=contributing_factors,
            citations=citations,
            is_grounded=is_grounded,
            needs_medical_attention=needs_attention,
            metadata={
                "query": patient_symptoms,
                "guidelines_used": len(retrieval_result.documents),
                "patient_history_included": bool(patient_history),
                "processing_time_seconds": processing_time,
                "retrieval_scores": retrieval_result.scores
            }
        )
    
    def _build_prediction_prompt(
        self,
        symptoms: str,
        guidelines: str,
        patient_history: str
    ) -> str:
        """Build the structured prompt for MedGemma."""
        return f"""You are a medical AI assistant specializing in cardiovascular health.
Analyze the following patient case using ONLY the provided medical guidelines.

### Medical Guidelines (Verified Sources):
{guidelines}
{patient_history}

### Patient Case:
{symptoms}

### Instructions:
1. Assess the risk of heart disease based STRICTLY on the guidelines above
2. Explain your reasoning step-by-step
3. Cite specific guidelines that support your assessment
4. Classify risk as: Low, Moderate, High, or Critical
5. If symptoms suggest emergency (STEMI, cardiogenic shock), clearly state this
6. For EACH patient result, lab value, symptom, and risk factor present in the case, explain WHY it contributes to or protects against heart disease risk
7. Connect each contributing factor to its clinical significance using the guidelines

### Response Format:
**Risk Level:** [Low/Moderate/High/Critical]

**Assessment:**
[Your medical assessment]

**Contributing Factors Analysis:**
For each result/value/symptom found in the patient case, explain its role:
- [Result/Value/Symptom 1]: Why this contributes to heart disease risk and what it indicates
- [Result/Value/Symptom 2]: Why this contributes to heart disease risk and what it indicates
- [Continue for all relevant factors...]
Include both risk-increasing AND protective factors if present.

**Reasoning:**
[Step-by-step reasoning citing guidelines, synthesizing how the contributing factors combine to determine the overall risk level]

**Recommendations:**
[Any recommended actions or follow-up based on the identified contributing factors]

⚠️ **Disclaimer:** This is an AI-assisted analysis. Always consult a qualified healthcare provider for medical decisions.
"""
    
    def _extract_risk_level(self, response: str) -> str:
        """Extract risk level from response."""
        response_lower = response.lower()
        
        # Try to find explicit "Risk Level:" pattern first (most reliable)
        match = re.search(r"risk level[:\s]*(low|moderate|high|critical)", response_lower)
        if match:
            return match.group(1).capitalize()
        
        if "critical" in response_lower or "emergency" in response_lower:
            return "Critical"
        elif "high risk" in response_lower or "high-risk" in response_lower:
            return "High"
        elif "moderate" in response_lower:
            return "Moderate"
        elif "low risk" in response_lower or "low-risk" in response_lower:
            return "Low"
        
        return "Undetermined"
    
    def _extract_contributing_factors(self, response: str) -> str:
        """Extract contributing factors analysis from response."""
        patterns = [
            r"\*\*Contributing Factors Analysis:\*\*\s*(.*?)(?=\*\*Reasoning|\*\*Recommendations|$)",
            r"Contributing Factors Analysis:\s*(.*?)(?=Reasoning:|Recommendations:|$)",
            r"\*\*Contributing Factors:\*\*\s*(.*?)(?=\*\*|$)",
            r"Contributing Factors:\s*(.*?)(?=\n\n\*\*|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fallback: try to extract bullet points that discuss individual factors
        factor_lines = []
        in_factors = False
        for line in response.split('\n'):
            if 'contributing factor' in line.lower() or ('why' in line.lower() and 'risk' in line.lower()):
                in_factors = True
                continue
            if in_factors:
                if line.strip().startswith('-') or line.strip().startswith('•'):
                    factor_lines.append(line.strip())
                elif line.strip().startswith('**') and factor_lines:
                    break
        
        return '\n'.join(factor_lines) if factor_lines else ""
    
    def _get_reasoning_section(self, response: str) -> str:
        """Extract reasoning section from response."""
        # Look for reasoning section
        patterns = [
            r"\*\*Reasoning:\*\*\s*(.*?)(?=\*\*|$)",
            r"Reasoning:\s*(.*?)(?=\n\n|$)",
            r"(?:because|since|as)\s+(.+?)(?=\.|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return ""


# Factory function for easy access
async def predict_heart_risk(
    symptoms: str,
    user_id: Optional[str] = None,
    **kwargs
) -> HeartRiskResult:
    """
    Convenience function to predict heart disease risk.
    
    Example:
        result = await predict_heart_risk(
            "55-year-old female with fatigue, nausea, and jaw pain"
        )
        print(f"Risk: {result.risk_level}")
        print(result.response)
    """
    predictor = HeartDiseasePredictor()
    return await predictor.predict_risk(symptoms, user_id, **kwargs)


# --- Test Runner ---
async def _test_predictor():
    """Test the HeartDiseasePredictor."""
    predictor = HeartDiseasePredictor()
    
    # Test case: Atypical female presentation
    test_case = """
    55-year-old female presenting with:
    - Extreme fatigue for 3 days
    - Persistent nausea
    - Intermittent jaw pain
    - No crushing chest pain
    - History of hypertension
    - LDL cholesterol: 175 mg/dL
    """
    
    print("📋 *** HEART DISEASE RISK ANALYSIS ***")
    print("-" * 50)
    
    result = await predictor.predict_risk(test_case)
    
    print(f"Risk Level: {result.risk_level}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Grounded: {'✅' if result.is_grounded else '⚠️'}")
    print(f"Needs Attention: {'🚨 YES' if result.needs_medical_attention else 'No'}")
    print(f"\nCitations: {', '.join(result.citations)}")
    print(f"\n{result.response}")
    
    return result


if __name__ == "__main__":
    import asyncio
    asyncio.run(_test_predictor())
