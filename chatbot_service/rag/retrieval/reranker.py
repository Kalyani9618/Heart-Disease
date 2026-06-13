"""
Cross-Encoder Reranker for RAG Pipeline - Production-Ready with Safety Checks

Rerank retrieved documents using cross-encoder for higher accuracy
than bi-encoder similarity search. Implements RAGFlow's robustness patterns:
- Explicit max_length truncation to prevent OOM errors
- Batch size handling for large document sets
- Graceful degradation on model failures

Key Features:
- Max length: 512 tokens (standard for MS-MARCO)
- Batch size: 32 (prevents GPU memory issues)
- Token truncation: Applied before model input
- Fallback: Returns original ranking if reranker unavailable
"""


import logging
import time
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

from core.services.performance_monitor import record_rerank_operation

# Try to import sentence-transformers for cross-encoder
try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CrossEncoder = None
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not available for cross-encoder reranking.")


class MedicalReranker:
    """
    Rerank retrieved documents using cross-encoder.

    Cross-encoders are more accurate than bi-encoders for ranking
    but slower, so we use them only on top-k results.
    
    This implementation includes safety measures:
    - Token truncation (max_length=512)
    - Batch processing to prevent OOM
    - Graceful fallback on errors
    """

    # RAGFlow-inspired configuration constants
    MAX_LENGTH = 512  # Token limit for cross-encoder input
    BATCH_SIZE = 32   # Process documents in batches
    QUERY_MAX_LENGTH = 256  # Separate limit for query
    DOC_MAX_LENGTH = 512    # Document truncation

    def __init__(
        self,
        # CHANGE: Point to your local folder containing model.safetensors and config.json
        model_name: str = "models/cross-encoder_model",
        max_length: int = MAX_LENGTH,
        batch_size: int = BATCH_SIZE,
        device: str = "cpu"
    ):
        """
        Initialize medical reranker with configurable parameters.

        Args:
            model_name: Path to local model folder or HuggingFace model name
            max_length: Maximum token length for model input (default 512)
            batch_size: Batch size for processing documents (default 32)
            device: Device to use for inference - "cpu" or "cuda" (default "cpu")
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.model = None

        if not CROSS_ENCODER_AVAILABLE:
            logger.warning(
                "Cross-encoder reranking disabled due to missing dependencies. "
                "Install with: pip install sentence-transformers"
            )
            return

        try:
            # CHANGE: Added explicit local_files_only=True check if you want to force offline mode,
            # but usually passing the path is enough for sentence-transformers to detect it.
            self.model = CrossEncoder(model_name, device=device)
            logger.info(
                f"✓ Medical reranker initialized\n"
                f"  - Model: {model_name}\n"
                f"  - Max Length: {max_length}\n"
                f"  - Batch Size: {batch_size}\n"
                f"  - Device: {device}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {e}")
            self.model = None

    def _truncate_text(self, text: str, max_chars: int = 1200) -> str:
        """
        Truncate text to prevent token explosion in the model.
        
        Args:
            text: Input text to truncate
            max_chars: Maximum characters (rough proxy for tokens)
        
        Returns:
            Truncated text
        """
        if not text:
            return ""
        
        text = str(text).strip()
        
        if len(text) > max_chars:
            # Truncate and try to break at sentence boundary
            truncated = text[:max_chars]
            
            # Find last period before truncation point
            last_period = truncated.rfind('.')
            if last_period > max_chars * 0.8:  # Only if reasonably close
                truncated = truncated[:last_period + 1]
            
            return truncated
        
        return text

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
        show_progress: bool = False,
        bidirectional: bool = True,
        diversity_penalty: float = 0.1,
        temperature: float = 1.0
    ) -> List[Dict]:
        """
        Rerank documents by relevance to query using cross-encoder.

        Implements RAGFlow's safety patterns plus advanced enhancements:
        - Token truncation before model input
        - Batch processing to prevent OOM
        - Graceful fallback on errors
        - Bidirectional scoring (query→doc AND doc→query)
        - Temperature-scaled confidence calibration
        - MMR-based diversity penalty

        Args:
            query: User query string
            documents: List of retrieved docs with 'content' or 'page_content' field
            top_k: Number of top results to return
            show_progress: Whether to show progress bar during inference
            bidirectional: If True, score both (query,doc) and (doc,query) and average
            diversity_penalty: MMR lambda for penalizing similar documents (0=no penalty, 1=max)
            temperature: Temperature for score calibration (lower = more confident)

        Returns:
            List of top-k reranked documents with 'rerank_score' added
        """
        start_time = time.time()

        try:
            # Early return if model not available or no documents
            if not self.model or not documents:
                logger.warning("Reranker unavailable or no documents provided")
                return documents[:top_k] if documents else []

            # LATENCY OPTIMIZATION: Skip reranking for small result sets (negligible benefit)
            if len(documents) <= 3:
                logger.debug(f"Skipping reranking: only {len(documents)} documents (threshold=3)")
                return documents[:top_k]

            # Step 1: Clean and truncate query (RAGFlow practice)
            clean_query = self._truncate_text(query, self.QUERY_MAX_LENGTH)
            if not clean_query:
                return documents[:top_k]

            # Step 2: Extract and truncate document content
            pairs_forward = []  # (query, doc) pairs
            pairs_reverse = []  # (doc, query) pairs for bidirectional
            valid_docs = []
            doc_contents = []  # Store for diversity calculation

            for doc in documents:
                # Handle different document formats
                if isinstance(doc, dict):
                    content = doc.get("content") or doc.get("page_content") or doc.get("text", "")
                elif hasattr(doc, "page_content"):
                    content = doc.page_content
                elif hasattr(doc, "content"):
                    content = doc.content
                else:
                    content = str(doc)

                # Truncate document content (RAGFlow safety check)
                clean_content = self._truncate_text(content, self.DOC_MAX_LENGTH)

                if clean_content:  # Only include non-empty documents
                    pairs_forward.append([clean_query, clean_content])
                    if bidirectional:
                        pairs_reverse.append([clean_content, clean_query])
                    valid_docs.append(doc)
                    doc_contents.append(clean_content)

            if not pairs_forward:
                logger.warning("No valid document-query pairs for reranking")
                return documents[:top_k]

            # Step 3: Process in batches to prevent OOM (RAGFlow pattern)
            all_scores_forward = []
            all_scores_reverse = []

            for batch_idx in range(0, len(pairs_forward), self.batch_size):
                batch_end = min(batch_idx + self.batch_size, len(pairs_forward))
                batch_pairs = pairs_forward[batch_idx:batch_end]

                try:
                    # Score forward batch (query → doc)
                    batch_scores = self.model.predict(
                        batch_pairs,
                        batch_size=self.batch_size,
                        show_progress_bar=show_progress,
                        convert_to_numpy=True,
                        max_length=self.max_length
                    )

                    # Ensure scores are numpy array
                    if not isinstance(batch_scores, np.ndarray):
                        batch_scores = np.array(batch_scores)

                    all_scores_forward.extend(batch_scores.tolist())

                    # Score reverse batch if bidirectional (doc → query)
                    if bidirectional:
                        reverse_batch = pairs_reverse[batch_idx:batch_end]
                        reverse_scores = self.model.predict(
                            reverse_batch,
                            batch_size=self.batch_size,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            max_length=self.max_length
                        )
                        if not isinstance(reverse_scores, np.ndarray):
                            reverse_scores = np.array(reverse_scores)
                        all_scores_reverse.extend(reverse_scores.tolist())

                    logger.debug(f"Processed batch {batch_idx // self.batch_size + 1}/{(len(pairs_forward) + self.batch_size - 1) // self.batch_size}")

                except RuntimeError as e:
                    # Handle OOM or other runtime errors
                    if "cuda" in str(e).lower() or "memory" in str(e).lower():
                        logger.error(f"GPU memory error during reranking: {e}. Falling back to original ranking.")
                        elapsed_ms = (time.time() - start_time) * 1000
                        record_rerank_operation(elapsed_ms, error=True)
                        return documents[:top_k]
                    raise

            # Step 4: Combine scores with calibration
            final_scores = []
            for i, (forward_score) in enumerate(all_scores_forward):
                if bidirectional and all_scores_reverse:
                    # Average forward and reverse scores
                    combined = (forward_score + all_scores_reverse[i]) / 2.0
                else:
                    combined = forward_score
                
                # Apply temperature calibration (softmax-style normalization later)
                calibrated = combined / temperature
                final_scores.append(calibrated)
            
            # Normalize scores to 0-1 range for MMR calculation
            scores_array = np.array(final_scores)
            if scores_array.max() != scores_array.min():
                normalized_scores = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())
            else:
                normalized_scores = np.ones_like(scores_array)
            
            # Step 5: MMR-style diversity-aware selection
            if diversity_penalty > 0 and top_k < len(valid_docs):
                selected_indices = []
                selected_contents = []
                remaining_indices = list(range(len(valid_docs)))
                
                for _ in range(min(top_k, len(valid_docs))):
                    best_idx = None
                    best_mmr_score = float('-inf')
                    
                    for idx in remaining_indices:
                        relevance = normalized_scores[idx]
                        
                        # Calculate max similarity to already selected docs
                        max_sim = 0.0
                        if selected_contents:
                            doc_content = doc_contents[idx]
                            for sel_content in selected_contents:
                                # Simple Jaccard similarity as proxy
                                words1 = set(doc_content.lower().split())
                                words2 = set(sel_content.lower().split())
                                if words1 or words2:
                                    sim = len(words1 & words2) / len(words1 | words2) if (words1 | words2) else 0
                                    max_sim = max(max_sim, sim)
                        
                        # MMR score: λ * relevance - (1-λ) * max_similarity
                        mmr_score = (1 - diversity_penalty) * relevance - diversity_penalty * max_sim
                        
                        if mmr_score > best_mmr_score:
                            best_mmr_score = mmr_score
                            best_idx = idx
                    
                    if best_idx is not None:
                        selected_indices.append(best_idx)
                        selected_contents.append(doc_contents[best_idx])
                        remaining_indices.remove(best_idx)
                
                # Assign final scores to selected docs
                result = []
                for rank, idx in enumerate(selected_indices):
                    doc = valid_docs[idx]
                    if isinstance(doc, dict):
                        doc["rerank_score"] = float(final_scores[idx])
                        doc["diversity_rank"] = rank + 1
                    else:
                        if not hasattr(doc, "metadata"):
                            doc.metadata = {}
                        doc.metadata["rerank_score"] = float(final_scores[idx])
                        doc.metadata["diversity_rank"] = rank + 1
                    result.append(doc)
            else:
                # Standard sorting without diversity
                for doc, score in zip(valid_docs, final_scores):
                    if isinstance(doc, dict):
                        doc["rerank_score"] = float(score)
                    else:
                        if not hasattr(doc, "metadata"):
                            doc.metadata = {}
                        doc.metadata["rerank_score"] = float(score)

                sorted_docs = sorted(
                    valid_docs,
                    key=lambda x: (
                        x.get("rerank_score") if isinstance(x, dict)
                        else x.metadata.get("rerank_score", 0)
                    ),
                    reverse=True
                )
                result = sorted_docs[:top_k]

            # Log statistics
            if result:
                scores_in_result = [
                    d.get("rerank_score") if isinstance(d, dict) else d.metadata.get("rerank_score")
                    for d in result
                ]
                min_score = min(scores_in_result)
                max_score = max(scores_in_result)
                logger.info(
                    f"✓ Reranked {len(documents)} docs → {len(result)} returned "
                    f"(bidirectional={bidirectional}, diversity={diversity_penalty:.2f}, "
                    f"score range: {min_score:.3f}-{max_score:.3f})"
                )

            return result

        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            # Graceful fallback: return original ranking
            elapsed_ms = (time.time() - start_time) * 1000
            record_rerank_operation(elapsed_ms, error=True)
            return documents[:top_k]

        finally:
            # Record performance metrics (RAGFlow practice)
            elapsed_ms = (time.time() - start_time) * 1000
            record_rerank_operation(elapsed_ms)

    def is_available(self) -> bool:
        """Check if reranker model is available and ready."""
        return self.model is not None

    def get_config(self) -> Dict:
        """Get current reranker configuration."""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "device": self.device,
            "available": self.is_available()
        }


class LLMReranker:
    """LLM-based document reranking for Phase 1.3."""
    
    def __init__(self, llm_gateway, k: int = 3, rerank_threshold: int = 5):
        """Initialize LLM reranker."""
        self.llm = llm_gateway
        self.k = k
        self.rerank_threshold = rerank_threshold
    
    async def rerank(self, query: str, documents: List[Dict], k: int = None) -> List[Dict]:
        """Rerank documents by LLM relevance judgment."""
        
        k = k or self.k
        
        if len(documents) <= self.rerank_threshold:
            return documents[:k]
        
        docs_text = "\n".join([
            f"[{i}] {doc.get('content', doc.get('text', ''))[:300]}"
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""Medical librarian. Rank these by relevance to query.

Query: {query}

Documents:
{docs_text}

Return {k} most relevant indices (comma-separated):
INDICES:"""
        
        response = await self.llm.generate(prompt)
        
        try:
            lines = response.split('\n')
            for line in lines:
                if 'INDICES:' in line:
                    indices_str = line.split('INDICES:')[1].strip()
                    indices = [int(x.strip()) for x in indices_str.split(',') if x.strip().isdigit()]
                    return [documents[i] for i in indices if i < len(documents)][:k]
        except Exception as e:
            logger.error(f"Reranking parse error: {e}")
        
        return documents[:k]
