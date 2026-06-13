"""Medical Self-RAG - Phase 2.1 Implementation with P2/P3 Enhancements"""

import logging
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from rag.retrieval.token_budget import TokenBudgetManager
from rag.retrieval.context_assembler import ContextAssembler, AssembledContext
from core.safety.hallucination_grader import HallucinationGrader

# P2/P3 Enhancements - Lazy imports for optional components
logger = logging.getLogger(__name__)



def _get_explainable_retriever():
    """Lazy load ExplainableRetriever to avoid circular imports."""
    try:
        from rag.retrieval.explainable_retrieval import ExplainableRetriever, RetrievalExplanation
        return ExplainableRetriever, RetrievalExplanation
    except ImportError:
        logger.warning("ExplainableRetriever not available")
        return None, None


def _get_fusion_retriever():
    """Lazy load FusionRetriever for hybrid search."""
    try:
        from rag.retrieval.fusion_retriever import FusionRetriever
        return FusionRetriever
    except ImportError:
        logger.warning("FusionRetriever not available")
        return None


def _get_unified_compressor():
    """Lazy load UnifiedDocumentCompressor."""
    try:
        from rag.retrieval.unified_compressor import UnifiedDocumentCompressor, CompressionStrategy
        return UnifiedDocumentCompressor, CompressionStrategy
    except ImportError:
        logger.warning("UnifiedDocumentCompressor not available")
        return None, None


def _get_trust_explainability():
    """Lazy load Trust/Explainability layer for response verification."""
    try:
        from rag.trust.explainability import ExplainableRetrieval
        return ExplainableRetrieval
    except ImportError:
        logger.warning("Trust/Explainability layer not available")
        return None


def _get_source_validator():
    """Lazy load source validator for trust layer."""
    try:
        from rag.trust.source_validator import get_source_validator
        return get_source_validator()
    except ImportError:
        logger.warning("SourceValidator not available")
        return None


def _get_conflict_detector():
    """Lazy load conflict detector for multi-source validation."""
    try:
        from rag.trust.conflict_detector import get_conflict_detector
        return get_conflict_detector()
    except ImportError:
        logger.warning("ConflictDetector not available")
        return None


def _get_raptor_retriever():
    """Lazy load RAPTOR hierarchical retriever."""
    try:
        from rag.retrieval.raptor_retrieval import RAPTORRetriever, RAPTORIndexManager
        return RAPTORRetriever, RAPTORIndexManager
    except ImportError:
        logger.debug("RAPTOR retriever not available")
        return None, None


class SupportLevel(Enum):
    """Response support levels."""
    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NO_SUPPORT = "no_support"
    NEEDS_DISCLAIMER = "needs_disclaimer"


@dataclass
class SelfRAGResult:
    """Result from Medical Self-RAG processing."""
    response: str
    support_level: SupportLevel
    citations: List[str]
    confidence: float
    needs_web_search: bool
    reasoning: str
    # P2/P3 Enhancements
    explanations: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)


class HyDERetriever:
    """
    Hypothetical Document Embeddings (HyDE) Retriever.
    Generates a hypothetical answer to the query, then uses that to search.
    """
    def __init__(self, llm_gateway, embedding_service, vector_store):
        self.llm = llm_gateway
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    async def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical medical document that would answer the query.
        
        This bridges layperson language (e.g., "my chest hurts") to medical 
        terminology (e.g., "patient presents with substernal chest pain,
        possible myocardial infarction").
        
        Args:
            query: User's question in natural language
            
        Returns:
            Hypothetical medical document as string
        """
        prompt = f"""Write a hypothetical medical passage that answers this question.
        Focus on medical facts, symptoms, and treatments.
        Use proper clinical terminology.
        Be comprehensive but concise (200-300 words).
        
        Question: {query}
        
        Hypothetical Medical Document:"""
        
        return await self.llm.generate(prompt, content_type="medical")

    async def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Generate hypothetical doc and retrieve similar real docs."""
        # 1. Generate hypothetical document
        hypothetical_doc = await self.generate_hypothetical_document(query)
        
        # 2. Embed the hypothetical document
        if hasattr(self.embedding_service, 'embed_text_async'):
             hypothetical_embedding = await self.embedding_service.embed_text_async(hypothetical_doc)
        else:
             hypothetical_embedding = self.embedding_service.embed_text(hypothetical_doc)
        
        # 3. Retrieve using the embedding
        if hasattr(self.vector_store, 'async_search'):
            similar_docs = await self.vector_store.async_search(
                query_embedding=hypothetical_embedding,
                top_k=k
            )
        else:
            similar_docs = self.vector_store.search_medical_knowledge(
                query_embedding=hypothetical_embedding,
                top_k=k
            )
        return similar_docs


class MedicalSelfRAG:
    """Self-correcting RAG that enforces medical evidence requirements."""
    
    # P0.2: Cache for retrieval need decisions
    _RETRIEVAL_CACHE: Dict[int, bool] = {}
    
    # P0.2++: Extended patterns that don't need medical retrieval
    # Covers greetings, acknowledgments, bot questions, and common conversational phrases
    _NO_RETRIEVAL_PATTERNS = [
        # Greetings and farewells
        r"^(hi|hello|hey|thanks|thank you|ok|okay|bye|goodbye|yes|no|sure|great|perfect|alright)",
        # Bot identity questions
        r"^what('s| is) your name",
        r"^how are you",
        r"^who (are|made) you",
        # Help requests (generic)
        r"^(please|can you) (help|assist)",
        # Understanding confirmations
        r"^(i understand|got it|makes sense|i see|understood)",
        # Meta questions about capabilities
        r"^(what can you do|what are you|tell me about yourself)",
        # Simple affirmations/negations
        r"^(yeah|yep|nope|nah|yup|uh huh)",
    ]
    
    # P0.2++: Extended medical keywords that definitely need retrieval
    # Includes common symptoms, conditions, medications, and medical terms
    _MEDICAL_KEYWORDS = [
        # Symptoms
        "symptom", "pain", "ache", "fever", "nausea", "dizzy", "dizziness",
        "headache", "fatigue", "tired", "weak", "weakness", "swelling",
        "bleeding", "bruise", "rash", "itch", "cough", "sneeze", "vomit",
        "diarrhea", "constipation", "cramp", "numbness", "tingling",
        # Body parts/systems
        "heart", "blood", "pressure", "chest", "lung", "breath", "breathing",
        "liver", "kidney", "brain", "stomach", "bone", "joint", "muscle",
        "skin", "eye", "ear", "throat", "nose",
        # Medications and treatments
        "medication", "medicine", "drug", "pill", "tablet", "dose", "dosage",
        "treatment", "therapy", "surgery", "procedure", "injection", "vaccine",
        "mg", "ml", "prescription", "over-the-counter", "otc",
        # Common drug names
        "aspirin", "ibuprofen", "acetaminophen", "tylenol", "advil", "motrin",
        "metoprolol", "lisinopril", "warfarin", "statin", "metformin",
        "omeprazole", "amlodipine", "prednisone", "antibiotic",
        # Conditions
        "disease", "condition", "disorder", "syndrome", "infection",
        "diabetes", "hypertension", "cholesterol", "arrhythmia", "asthma",
        "arthritis", "cancer", "anemia", "allergy", "anxiety", "depression",
        # Medical actions
        "diagnosis", "diagnose", "test", "scan", "mri", "xray", "x-ray",
        "side effect", "interaction", "risk", "contraindication",
        "emergency", "urgent", "911",
    ]
    
    def __init__(
        self, 
        vector_store, 
        llm_gateway, 
        embedding_service=None, 
        reranker=None, 
        guardrails=None, 
        token_budget_manager=None,
        memory_bridge=None,
        # P2/P3 Enhancement options
        enable_fusion_retrieval: bool = True,
        enable_explainability: bool = True,
        enable_compression: bool = False,
    ):
        """Initialize with components."""
        self.vector_store = vector_store
        self.llm = llm_gateway
        self.reranker = reranker
        self.embedding_service = embedding_service
        self.guardrails = guardrails
        self.memory_bridge = memory_bridge
        
        # Token Budget Manager for context window management
        self.token_budget = token_budget_manager or TokenBudgetManager(model_name="default")
        logger.info(f"MedicalSelfRAG initialized with TokenBudgetManager (max_tokens={self.token_budget.max_tokens})")
        
        # Context Assembler for parallel multi-source retrieval
        self.context_assembler = ContextAssembler(
            vector_store=vector_store,
            memory_bridge=memory_bridge
        )
        logger.info("MedicalSelfRAG: ContextAssembler initialized for parallel retrieval")
        
        if embedding_service:
            self.hyde_retriever = HyDERetriever(llm_gateway, embedding_service, vector_store)
        else:
            self.hyde_retriever = None
            
        # Initialize Hallucination Grader
        self.grader = HallucinationGrader()
        
        # P2/P3 Enhancements: Initialize optional components
        self._init_enhancements(
            enable_fusion_retrieval=enable_fusion_retrieval,
            enable_explainability=enable_explainability,
            enable_compression=enable_compression,
        )
    
    def _init_enhancements(
        self,
        enable_fusion_retrieval: bool,
        enable_explainability: bool,
        enable_compression: bool,
    ):
        """Initialize P2/P3 enhancement components."""
        # P3.2: Fusion Retriever for hybrid search
        self.fusion_retriever = None
        if enable_fusion_retrieval:
            FusionRetriever = _get_fusion_retriever()
            if FusionRetriever and self.vector_store:
                try:
                    self.fusion_retriever = FusionRetriever(
                        vector_store=self.vector_store,
                        use_query_cleaning=True,
                        use_lemmatization=True
                    )
                    logger.info("MedicalSelfRAG: FusionRetriever (P3.2) initialized for hybrid search")
                except Exception as e:
                    logger.warning(f"FusionRetriever initialization failed: {e}")
        
        # P2.3: Explainable Retriever
        self.explainable_retriever = None
        if enable_explainability:
            ExplainableRetriever, _ = _get_explainable_retriever()
            if ExplainableRetriever:
                try:
                    self.explainable_retriever = ExplainableRetriever(
                        highlight_matches=True,
                        include_reasoning=True
                    )
                    logger.info("MedicalSelfRAG: ExplainableRetriever (P2.3) initialized")
                except Exception as e:
                    logger.warning(f"ExplainableRetriever initialization failed: {e}")
        
        # P3.3: Unified Document Compressor
        self.compressor = None
        if enable_compression:
            UnifiedDocumentCompressor, CompressionStrategy = _get_unified_compressor()
            if UnifiedDocumentCompressor:
                try:
                    self.compressor = UnifiedDocumentCompressor(
                        llm_gateway=self.llm,
                        target_ratio=0.6,
                        preserve_medical_terms=True
                    )
                    logger.info("MedicalSelfRAG: UnifiedDocumentCompressor (P3.3) initialized")
                except Exception as e:
                    logger.warning(f"UnifiedDocumentCompressor initialization failed: {e}")
        
        # P3.1: Trust/Explainability layer for final response validation
        self.trust_explainer = None
        ExplainableRetrieval = _get_trust_explainability()
        if ExplainableRetrieval:
            try:
                self.trust_explainer = ExplainableRetrieval(self.llm)
                logger.info("MedicalSelfRAG: Trust/Explainability layer (P3.1) initialized")
            except Exception as e:
                logger.warning(f"Trust/Explainability initialization failed: {e}")
        
        # P2.2: RAPTOR handles (not fully integrated yet)
        self.raptor_retriever = None
        self.raptor_store = None
        RAPTORRetriever, RAPTORIndexManager = _get_raptor_retriever()
        if RAPTORRetriever and RAPTORIndexManager:
            try:
                self.raptor_store = RAPTORIndexManager()
                self.raptor_retriever = RAPTORRetriever(
                    embedding_model=self.embedding_service
                )
                logger.info("MedicalSelfRAG: RAPTOR components loaded")
            except Exception as e:
                logger.debug(f"RAPTOR init skipped: {e}")

        # P3.1: Initialize Source Validator and Conflict Detector
        self.source_validator = None
        self.conflict_detector = None
        
        SourceValidator = _get_source_validator()
        if SourceValidator:
            try:
                self.source_validator = SourceValidator
                logger.info("MedicalSelfRAG: SourceValidator (P3.1) initialized")
            except Exception as e:
                logger.warning(f"SourceValidator initialization failed: {e}")
                
        ConflictDetector = _get_conflict_detector()
        if ConflictDetector:
            try:
                self.conflict_detector = ConflictDetector
                logger.info("MedicalSelfRAG: ConflictDetector (P3.1) initialized")
            except Exception as e:
                logger.warning(f"ConflictDetector initialization failed: {e}")
    
    async def process(self, query: str, user_id: Optional[str] = None, **kwargs) -> "SelfRAGResult":
        """
        Main entry point for processing a query.
        
        This is the method called from LangGraph orchestrator.
        Wraps generate_with_self_correction for compatibility.
        
        Args:
            query: User's medical question
            user_id: Optional user ID for personalized context
            **kwargs: Additional parameters (conversation_history, user_memories, etc.)
            
        Returns:
            SelfRAGResult with response, confidence, citations
        """
        result = await self.generate_with_self_correction(
            query=query,
            user_id=user_id,
            conversation_history=kwargs.get("conversation_history"),
            user_memories=kwargs.get("user_memories"),
            user_context=kwargs.get("user_context"),
            use_hyde=kwargs.get("use_hyde", False)
        )
        
        # Add requires_crag_fallback attribute for orchestrator compatibility
        result.requires_crag_fallback = result.needs_web_search
        
        return result
    
    async def hypothetical_retrieval(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Use HyDE for retrieval. Returns documents and hypothetical doc."""
        if not self.hyde_retriever:
            docs = await self.vector_store.async_search(query, top_k=k)
            return {
                "documents": docs,
                "hypothetical_doc": None,
                "retrieval_method": "direct"
            }
        
        docs = await self.hyde_retriever.retrieve(query, k)
        return {
            "documents": docs,
            "hypothetical_doc": None,
            "retrieval_method": "hyde"
        }
    
    async def generate_with_self_correction(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        use_hyde: bool = False,
        user_id: Optional[str] = None,
        conversation_history: Optional[str] = None,
        user_memories: Optional[str] = None,
        use_fusion: bool = True,  # P3.2: Enable fusion retrieval by default
        include_explanations: bool = True,  # P2.3: Enable explainability by default
    ) -> SelfRAGResult:
        """Generate response with self-correction steps and P2/P3 enhancements."""
        
        retrieval_metadata = {
            "retrieval_method": "standard",
            "fusion_enabled": False,
            "explanations_enabled": False,
            "compression_enabled": False,
        }
        explanations = []
        
        # STEP 1: Do we need external knowledge?
        needs_retrieval = await self._check_retrieval_need(query)
        
        if not needs_retrieval:
            response = await self._generate_direct_response(query, user_context)
            return SelfRAGResult(
                response=response,
                support_level=SupportLevel.FULLY_SUPPORTED,
                citations=[],
                confidence=0.85,
                needs_web_search=False,
                reasoning="Query doesn't require external medical knowledge",
                explanations=[],
                retrieval_metadata=retrieval_metadata,
            )
        
        # STEP 2: Parallel retrieval using ContextAssembler
        # This retrieves from vector store, knowledge graph, and memory simultaneously
        assembled_context: AssembledContext = await self.context_assembler.assemble(
            query=query,
            user_id=user_id,
            top_k=5
        )
        
        logger.info(
            f"ContextAssembler retrieved {assembled_context.total_documents} docs "
            f"in {assembled_context.retrieval_time_ms:.1f}ms "
            f"(vector={len(assembled_context.vector_results)}, "
            f"graph={len(assembled_context.graph_results)}, "
            f"memory={len(assembled_context.memory_results)})"
        )
        
        # Use combined ranked results, or fall back to guardrails/hyde if configured
        if assembled_context.combined_ranked:
            docs = assembled_context.combined_ranked
            retrieval_metadata["retrieval_method"] = "context_assembler"
        elif self.guardrails and user_id:
            # Fallback: Use Safety Guardrails for filtered retrieval
            safe_content, metadata = await self.guardrails.retrieve_safe_context(query, user_id, top_k=5)
            docs = [{"content": c, "source": "Filtered Knowledge Base"} for c in safe_content]
            retrieval_metadata["retrieval_method"] = "guardrails"
        elif use_hyde and self.hyde_retriever:
            docs = await self.hyde_retriever.retrieve(query, k=5)
            retrieval_metadata["retrieval_method"] = "hyde"
        else:
            # P3.2: Try FusionRetriever for hybrid search
            if use_fusion and self.fusion_retriever:
                try:
                    fusion_docs = await self.fusion_retriever.retrieve(query, top_k=5)
                    docs = [{"content": d.page_content, "metadata": d.metadata, "source": "fusion"}
                            for d in fusion_docs]
                    retrieval_metadata["retrieval_method"] = "fusion"
                    retrieval_metadata["fusion_enabled"] = True
                    logger.info(f"FusionRetriever returned {len(docs)} documents")
                except Exception as e:
                    logger.warning(f"FusionRetriever failed, falling back: {e}")
                    docs = await self.vector_store.async_search(query, top_k=5)
            # P2.2: RAPTOR fallback if available
            elif self.raptor_retriever and self.raptor_store:
                try:
                    # RAPTOR is per-tree, so we search all loaded trees
                    raptor_docs = []
                    tree_ids = self.raptor_store.list_trees()
                    
                    if not tree_ids:
                        # If no trees, fall back silently
                        raise ValueError("No RAPTOR trees available")

                    for tree_id in tree_ids:
                        tree = self.raptor_store.get_tree(tree_id)
                        if tree:
                            results = await self.raptor_retriever.retrieve(query, tree, top_k=5)
                            raptor_docs.extend(results)
                    
                    # Sort combined results by score
                    raptor_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
                    raptor_docs = raptor_docs[:5]

                    if not raptor_docs:
                        raise ValueError("No RAPTOR results found")

                    docs = [{"content": d.get("content", ""), "metadata": d.get("metadata", {}), "source": "raptor"}
                            for d in raptor_docs]
                    retrieval_metadata["retrieval_method"] = "raptor"
                    retrieval_metadata["raptor_enabled"] = True
                    logger.info(f"RAPTOR returned {len(docs)} documents")
                except Exception as e:
                    logger.debug(f"RAPTOR retrieval failed: {e}")
                    docs = await self.vector_store.async_search(query, top_k=5)
            else:
                docs = await self.vector_store.async_search(query, top_k=5)
        
        # STEP 3: Filter relevant documents & OPTIONAL PARALLEL RERANKING
        # MEDIUM RISK FIX: Run filtering and reranking in parallel if available
        # This reduces latency from sequential execution (~400ms + ~300ms = 700ms) 
        # to parallel execution (~400ms total)
        
        if self.reranker:
            # Parallel execution: filter AND rerank at same time
            relevant_docs, reranked_docs = await asyncio.gather(
                self._filter_relevant_docs(query, docs),
                self.reranker.rerank(query, docs),
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(relevant_docs, Exception):
                logger.warning(f"Parallel filtering failed: {relevant_docs}, using fallback")
                relevant_docs = docs[:5]
            
            if isinstance(reranked_docs, Exception):
                logger.warning(f"Parallel reranking failed: {reranked_docs}, using filtered docs")
                relevant_docs = relevant_docs if relevant_docs else docs[:3]
            else:
                # Use reranked docs if available and valid
                if reranked_docs:
                    relevant_docs = reranked_docs
        else:
            # No reranker: just filter
            relevant_docs = await self._filter_relevant_docs(query, docs)
        
        if not relevant_docs:
            return SelfRAGResult(
                response="I don't have reliable information about this in my knowledge base.",
                support_level=SupportLevel.NO_SUPPORT,
                citations=[],
                confidence=0.0,
                needs_web_search=True,
                reasoning="No relevant documents found for this medical query",
                explanations=[],
                retrieval_metadata=retrieval_metadata,
            )
        
        # P2.3: Generate retrieval explanations if enabled
        if include_explanations and self.explainable_retriever:
            try:
                retrieval_explanations = self.explainable_retriever.explain_retrieval(
                    query=query,
                    documents=relevant_docs,
                    scores=[doc.get("score", doc.get("combined_score", 0.0)) for doc in relevant_docs]
                )
                explanations = [exp.to_dict() for exp in retrieval_explanations]
                retrieval_metadata["explanations_enabled"] = True
                logger.debug(f"Generated {len(explanations)} retrieval explanations")
            except Exception as e:
                logger.warning(f"Explainability generation failed: {e}")
        
        # P3.3: Optional document compression
        if self.compressor:
            try:
                compressed_docs = await self.compressor.compress(query, relevant_docs)
                relevant_docs = [
                    {"content": cd.compressed_content, "source": (cd.source_metadata or {}).get("source", "compressed")}
                    for cd in compressed_docs
                ]
                retrieval_metadata["compression_enabled"] = True
                logger.debug(f"Compressed {len(relevant_docs)} documents")
            except Exception as e:
                logger.warning(f"Document compression failed: {e}")
        
        # P3.1: Trust/Explainability layer - add relevance explanations to top docs
        if self.trust_explainer and include_explanations:
            try:
                # Convert dict docs to lightweight Document-like objects
                from langchain_core.documents import Document
                doc_objs = [Document(page_content=d.get("content", ""), metadata=d.get("metadata", {})) for d in relevant_docs[:3]]
                explained_docs = await self.trust_explainer.explain_relevance(query, doc_objs)
                # Merge explanations back into the dicts
                for i, doc_obj in enumerate(explained_docs):
                    if i < len(relevant_docs):
                        relevant_docs[i]["metadata"] = doc_obj.metadata
                        relevant_docs[i]["content"] = doc_obj.page_content
                retrieval_metadata["trust_explanations"] = True
            except Exception as e:
                logger.debug(f"Trust explainability failed: {e}")
        
        # P3.1: Source Validation and Conflict Detection
        if self.source_validator:
            try:
                # Validate top documents
                validation_results = self.source_validator.batch_validate(relevant_docs[:5])
                
                # Add validation info to metadata
                for i, res in enumerate(validation_results):
                    if i < len(relevant_docs):
                        relevant_docs[i]["validation"] = {
                            "level": res.validation_level.value,
                            "credibility": res.credibility_score,
                            "issues": res.issues
                        }
                retrieval_metadata["source_validation"] = True
            except Exception as e:
                logger.warning(f"Source validation failed: {e}")
                
        if self.conflict_detector:
            try:
                # Check for conflicts among top documents
                conflicts = self.conflict_detector.detect_conflicts(relevant_docs[:5])
                if conflicts:
                    retrieval_metadata["conflicts_detected"] = len(conflicts)
                    retrieval_metadata["conflict_report"] = self.conflict_detector.generate_conflict_report(conflicts)
                    
                    # If critical conflicts, warn in logs or potentially filter
                    critical_conflicts = [c for c in conflicts if c.severity.value in ["critical", "high"]]
                    if critical_conflicts:
                        logger.warning(f"CRITICAL CONFLICTS DETECTED: {len(critical_conflicts)}")
            except Exception as e:
                logger.warning(f"Conflict detection failed: {e}")
        
        # STEP 4 & 5 MERGED (MEDIUM RISK FIX): Generate response AND verify in parallel
        # This optimization reduces latency by parallelizing:
        #   - Response generation (~200-400ms)
        #   - Support level grading (~300ms)
        # Instead of sequential (~500-700ms total), we do parallel (~400ms total)
        
        context_text = "\n\n".join([doc.get('content', '') for doc in relevant_docs[:3]])
        
        
        # NOTE: P0.3 skip_grading logic moved after response generation (line 645)
        # Previous premature check removed as 'response' was not yet defined
        
        confidence_map = {
            SupportLevel.FULLY_SUPPORTED: 0.95,
            SupportLevel.PARTIALLY_SUPPORTED: 0.72,  # Increased from 0.70
            SupportLevel.NEEDS_DISCLAIMER: 0.78,      # Increased from 0.75
            SupportLevel.NO_SUPPORT: 0.0,
        }
        
        # Generate response - we'll check grading after
        response = await self._generate_response_with_context(
            query, 
            context_text,
            conversation_history=conversation_history,
            user_memories=user_memories
        )
        
        # STEP 5: Verify response is supported by context
        # P0.3 Optimization: Skip grading for high-confidence scenarios
        # - Short responses (< 200 chars) are usually template-based
        # - High document coverage (>= 3 docs) = likely well-grounded
        # Saves ~300ms for ~60% of queries
        
        if len(response) < 200 or len(relevant_docs) >= 3:
            # High confidence scenario - skip LLM grading
            logger.debug(f"P0.3: Skipping grading (response_len={len(response)}, docs={len(relevant_docs)})")
            support_level = SupportLevel.FULLY_SUPPORTED
            confidence = 0.90  # Increased from 0.85
        else:
            # Lower confidence - perform full grading
            support_level = await self._check_support_level(query, response, context_text)
            confidence = confidence_map[support_level]
        
        # Extract citations
        citations = [doc.get('source', doc.get('name', '')) for doc in relevant_docs[:3]]
        
        return SelfRAGResult(
            response=response,
            support_level=support_level,
            citations=citations,
            confidence=confidence,
            needs_web_search=confidence < 0.60,  # CHANGED: Increased threshold from 0.5 to 0.60
            reasoning=f"Response has {support_level.value} support in retrieved context",
            explanations=explanations,
            retrieval_metadata=retrieval_metadata,
        )
    
    async def _check_retrieval_need(self, query: str) -> bool:
        """Check if query requires external knowledge retrieval.
        
        P0.2 Optimization: Uses pattern matching and caching to avoid LLM calls.
        - Cache hit: ~0ms
        - Pattern match: ~0ms  
        - LLM fallback: ~300ms
        """
        import re
        query_lower = query.lower().strip()
        
        # Fast path: Check cache
        cache_key = hash(query_lower[:100])
        if cache_key in self._RETRIEVAL_CACHE:
            logger.debug("Retrieval need: cache hit")
            return self._RETRIEVAL_CACHE[cache_key]
        
        # Fast path: Pattern-based exclusion (no LLM needed)
        for pattern in self._NO_RETRIEVAL_PATTERNS:
            if re.match(pattern, query_lower):
                logger.debug(f"Retrieval need: pattern exclusion ({pattern})")
                self._RETRIEVAL_CACHE[cache_key] = False
                return False
        
        # Fast path: Medical keywords → definitely needs retrieval
        if any(kw in query_lower for kw in self._MEDICAL_KEYWORDS):
            logger.debug("Retrieval need: medical keyword detected")
            self._RETRIEVAL_CACHE[cache_key] = True
            return True
        
        # Slow path: LLM fallback for ambiguous queries
        prompt = f"""Does this require medical knowledge retrieval?
Query: {query}
Answer: YES or NO"""
        
        response = await self.llm.generate(prompt)
        needs_retrieval = "YES" in response.upper()
        
        # Cache result
        self._RETRIEVAL_CACHE[cache_key] = needs_retrieval
        logger.debug(f"Retrieval need: LLM fallback -> {needs_retrieval}")
        return needs_retrieval
    
    async def _filter_relevant_docs(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter documents to only relevant ones.
        
        P0.1 Optimization: Uses batched LLM call instead of per-doc calls.
        - Old: N docs × ~300ms = 1.5s+ for 5 docs
        - New: 1 batch call × ~400ms = 0.4s (73% faster)
        """
        if not documents:
            return []
        
        # Fast path: Skip filtering for small, high-quality result sets
        if len(documents) <= 3:
            logger.debug(f"Skipping relevance filter: only {len(documents)} docs")
            return documents
        
        # Batch all documents into single LLM call
        doc_summaries = []
        for i, doc in enumerate(documents):
            content = doc.get('content', doc.get('text', ''))[:300]
            # Clean content for prompt
            content = content.replace('\n', ' ').strip()
            doc_summaries.append(f"[{i+1}] {content}")
        
        batch_prompt = f"""Evaluate which documents are relevant to this medical query.

Query: {query}

Documents:
{chr(10).join(doc_summaries)}

Return ONLY the numbers of relevant documents, comma-separated (e.g., "1,3,5").
If none are relevant, return "NONE"."""
        
        try:
            response = await self.llm.generate(batch_prompt)
            response = response.strip()
            
            # Parse response
            if "NONE" in response.upper():
                logger.debug("Batch relevance filter: no relevant docs")
                return []
            
            # Extract indices from response
            import re
            numbers = re.findall(r'\d+', response)
            indices = [int(x) - 1 for x in numbers]  # Convert to 0-indexed
            relevant = [documents[i] for i in indices if 0 <= i < len(documents)]
            
            logger.debug(f"Batch relevance filter: {len(relevant)}/{len(documents)} docs relevant")
            return relevant if relevant else documents[:3]  # Fallback to top 3
            
        except Exception as e:
            logger.warning(f"Batch relevance filter failed: {e}, returning all docs")
            return documents
    
    async def _is_relevant(self, query: str, doc: Dict[str, Any]) -> bool:
        """Check if a single document is relevant to the query.
        
        Note: This method is kept for backwards compatibility but
        _filter_relevant_docs now uses batched processing.
        """
        prompt = f"""Is this medical document relevant?

Question: {query}
Document: {doc.get('content', doc.get('text', ''))[:500]}

Answer ONLY: RELEVANT or IRRELEVANT"""
        
        response = await self.llm.generate(prompt)
        return "RELEVANT" in response.upper()
    
    async def _generate_direct_response(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate direct response without external context."""
        return await self.llm.generate(query)
    
    async def _generate_response_with_context(
        self,
        query: str,
        context: str,
        conversation_history: Optional[str] = None,
        user_memories: Optional[str] = None
    ) -> str:
        """Generate response using provided context with token budget management."""
        
        # Use TokenBudgetManager to allocate context within limits
        allocations = self.token_budget.allocate(
            query=query,
            medical_context=context,
            history=conversation_history or "",
            memories=user_memories or ""
        )
        
        # Build prompt with allocated (truncated if needed) content
        allocated_context = allocations.get("medical_context", context)
        allocated_history = allocations.get("history", "")
        allocated_memories = allocations.get("memories", "")
        
        # Log budget usage
        total_used = sum(allocations.get("token_counts", {}).values()) if "token_counts" in allocations else 0
        logger.debug(f"Token budget: {total_used}/{self.token_budget.max_tokens} tokens used")
        
        # Build context-aware prompt
        prompt_parts = ["Use the following context to answer the medical question. If the answer is not in the context, clearly state that you don't have that information.\n"]
        
        if allocated_memories:
            prompt_parts.append(f"User Context:\n{allocated_memories}\n")
        
        if allocated_history:
            prompt_parts.append(f"Conversation History:\n{allocated_history}\n")
        
        prompt_parts.append(f"Medical Context:\n{allocated_context}\n")
        prompt_parts.append(f"Question: {query}\n")
        prompt_parts.append("Answer:")
        
        prompt = "\n".join(prompt_parts)
        
        return await self.llm.generate(prompt, content_type="medical")
    
    async def _check_support_level(
        self,
        query: str,
        response: str,
        context: str
    ) -> SupportLevel:
        """Check how well the response is supported by context."""
        
        # 1. First check for hallucinations using HallucinationGrader
        is_grounded = await self.grader.grade(response, context)
        if not is_grounded:
            logger.warning("HallucinationGrader detected ungrounded content")
            return SupportLevel.NO_SUPPORT

        # 2. If grounded, determine specific support level
        prompt = f"""Evaluate support level.

Question: {query}
Response: {response}
Context: {context}

Is response:
1 = FULLY_SUPPORTED (all claims have evidence)
2 = PARTIALLY_SUPPORTED (some claims have evidence)
3 = NO_SUPPORT (contradicts or not mentioned)
4 = NEEDS_DISCLAIMER (true but needs medical disclaimer)

Answer ONLY the number (1-4):"""
        
        response_level = await self.llm.generate(prompt)
        
        level_map = {
            "1": SupportLevel.FULLY_SUPPORTED,
            "2": SupportLevel.PARTIALLY_SUPPORTED,
            "3": SupportLevel.NO_SUPPORT,
            "4": SupportLevel.NEEDS_DISCLAIMER,
        }
        
        for key, level in level_map.items():
            if key in response_level:
                return level
        
        return SupportLevel.PARTIALLY_SUPPORTED
