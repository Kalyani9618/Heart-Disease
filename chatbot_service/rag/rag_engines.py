"""
HeartDiseaseRAG Engine — Production Inference Mode

Connects the Router directly to the populated ChromaDB vector store
containing 125,000+ medical textbook documents.

This module operates in INFERENCE-ONLY mode:
  1. Embeds user queries via EmbeddingService
  2. Searches the existing ChromaDB (read-only)
  3. Optionally reranks results via MedicalReranker
"""

import logging
import os

from rag.store.chromadb_store import ChromaDBVectorStore
from rag.embedding.remote import RemoteEmbeddingService

logger = logging.getLogger(__name__)


class HeartDiseaseRAG:
    """
    Singleton engine that connects the Router to the Vector Database.
    Provides semantic search over medical textbooks.

    Operates in inference-only mode — no ingestion or data loading.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HeartDiseaseRAG, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize connection to ChromaDB in inference (read-only) mode."""
        logger.info("Initializing RAG Engine in INFERENCE mode...")
        try:
            # 1. Initialize Embedding Service (768-dim MedCPT via remote Colab)
            self.embedding_service = RemoteEmbeddingService.get_instance()
            logger.info(f"✅ Embedding service loaded (dim={self.embedding_service.get_dimension()})")

            # 2. Connect to EXISTING ChromaDB (read-only, no ingestion)
            persist_dir = os.getenv(
                "CHROMADB_DIR",
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "Chromadb",
                ),
            )
            self.vector_store = ChromaDBVectorStore(
                persist_directory=persist_dir,
            )

            # 3. Register the ACTUAL data collections used during training
            #    Training created: medical_text_768 (text) and medical_images_1152 (images)
            #    These differ from the default collection names in ChromaDBVectorStore
            self._text_collection_name = os.getenv(
                "CHROMADB_TEXT_COLLECTION", "medical_text_768"
            )
            self._image_collection_name = os.getenv(
                "CHROMADB_IMAGE_COLLECTION", "medical_images_1152"
            )
            # Ensure these collections are accessible via the store
            self._text_collection = self.vector_store._get_collection(
                self._text_collection_name
            )
            self._image_collection = self.vector_store._get_collection(
                self._image_collection_name
            )

            # Log collection stats
            text_count = self._text_collection.count()
            image_count = self._image_collection.count()
            logger.info(f"✅ Connected to ChromaDB at {persist_dir}")
            logger.info(f"   📚 {self._text_collection_name}: {text_count:,} documents")
            logger.info(f"   🖼️  {self._image_collection_name}: {image_count:,} documents")
            logger.info(f"   Total: {text_count + image_count:,} documents")

            # 4. Load Reranker (optional, for better accuracy)
            self.reranker = None
            try:
                from rag.retrieval.reranker import MedicalReranker
                self.reranker = MedicalReranker()
                logger.info("✅ MedicalReranker loaded")
            except Exception as e:
                logger.warning(f"⚠️ Reranker not available (results will be unranked): {e}")

            logger.info("✅ HeartGuard RAG initialized in INFERENCE mode.")

        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG Engine: {e}")
            self.vector_store = None
            self.reranker = None
        finally:
            self._initialized = True

    @classmethod
    def get_instance(cls):
        """Get singleton instance of RAG engine."""
        return cls()

    def is_ready(self) -> bool:
        return hasattr(self, '_initialized') and self._initialized and self.vector_store is not None

    def _search_text_collection(self, query: str, top_k: int = 5):
        """
        Search the actual medical text collection (medical_text_768).

        Uses the embedding service to embed the query, then queries
        the trained collection directly.

        Returns:
            List of result dicts with id, content, metadata, score
        """
        import numpy as np

        # Embed the query
        query_embedding = self.embedding_service.embed_text(query)
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        # Query the actual trained collection
        raw = self._text_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        if raw and raw["ids"] and raw["ids"][0]:
            for i, doc_id in enumerate(raw["ids"][0]):
                distance = raw["distances"][0][i] if raw["distances"] else 0.0
                results.append({
                    "id": doc_id,
                    "content": raw["documents"][0][i] if raw["documents"] else "",
                    "metadata": raw["metadatas"][0][i] if raw["metadatas"] else {},
                    "score": 1.0 - distance,
                })

        return results

    def retrieve_context(self, query: str, top_k: int = 3):
        """
        Retrieves relevant text from the vector database using semantic search.

        Args:
            query: Medical question or symptom description
            top_k: Number of top results to retrieve (default: 3)

        Returns:
            Dictionary with summary, context, and sources
        """
        if not self.vector_store:
            return {
                "summary": "System Error",
                "context": "Vector store not connected.",
                "sources": []
            }

        try:
            from core.monitoring.prometheus_metrics import get_metrics
            import time
            start_time = time.perf_counter()
            
            # Search the actual trained collection (medical_text_768)
            results = self._search_text_collection(query, top_k=top_k)
            
            # Record metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            get_metrics().record_vector_search(duration_ms, len(results) if results else 0)

            if not results:
                return {
                    "summary": "No results found in medical textbooks.",
                    "context": "The knowledge base did not return any matches.",
                    "sources": []
                }

            # Optionally rerank for better quality
            if self.reranker and self.reranker.is_available() and len(results) > 1:
                try:
                    results = self.reranker.rerank(query, results, top_k=top_k)
                except Exception as e:
                    logger.warning(f"Reranking failed, using raw results: {e}")

            # Format the results for the chatbot
            context_str = ""
            sources = []

            for res in results:
                metadata = res.get('metadata', {})
                title = metadata.get('title', metadata.get('source', 'Unknown Source'))
                text = res.get('content', res.get('text', ''))
                score = res.get('score', 0.0)

                context_str += f"[SOURCE: {title}] (relevance: {score:.2f})\n{text}\n{'-'*40}\n"
                if title not in sources:
                    sources.append(title)

            return {
                "summary": f"Found {len(results)} relevant textbook excerpts.",
                "context": context_str,
                "sources": sources
            }

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return {
                "summary": "Error retrieving context",
                "context": f"An error occurred: {str(e)}",
                "sources": []
            }

    def retrieve_and_rerank(self, query: str, top_k: int = 5, rerank_top_k: int = 3):
        """
        Retrieve documents and rerank for highest quality results.

        Fetches more candidates (top_k) then reranks down to rerank_top_k.

        Args:
            query: Medical question
            top_k: Number of candidates to retrieve from vector DB
            rerank_top_k: Number of final results after reranking

        Returns:
            Dictionary with summary, context, and sources
        """
        if not self.vector_store:
            return {
                "summary": "System Error",
                "context": "Vector store not connected.",
                "sources": []
            }

        try:
            # Retrieve more candidates from trained collection
            results = self._search_text_collection(query, top_k=top_k)

            if not results:
                return {
                    "summary": "No results found.",
                    "context": "",
                    "sources": []
                }

            # Rerank if available
            if self.reranker and self.reranker.is_available():
                results = self.reranker.rerank(query, results, top_k=rerank_top_k)
            else:
                results = results[:rerank_top_k]

            # Format
            context_str = ""
            sources = []
            for res in results:
                metadata = res.get('metadata', {})
                title = metadata.get('title', metadata.get('source', 'Unknown Source'))
                text = res.get('content', res.get('text', ''))
                score = res.get('score', 0.0)

                context_str += f"[SOURCE: {title}] (relevance: {score:.2f})\n{text}\n{'-'*40}\n"
                if title not in sources:
                    sources.append(title)

            return {
                "summary": f"Found {len(results)} reranked excerpts.",
                "context": context_str,
                "sources": sources
            }

        except Exception as e:
            logger.error(f"Error during retrieve_and_rerank: {e}")
            return {
                "summary": "Error",
                "context": str(e),
                "sources": []
            }

    def get_stats(self) -> dict:
        """Get ChromaDB collection statistics."""
        if not self.vector_store:
            return {"error": "Vector store not connected"}
        try:
            return {
                self._text_collection_name: self._text_collection.count(),
                self._image_collection_name: self._image_collection.count(),
            }
        except Exception:
            return self.vector_store.get_collection_stats()


def get_heart_disease_rag():
    """Factory function to get the HeartDiseaseRAG instance."""
    return HeartDiseaseRAG.get_instance()

