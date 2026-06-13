"""
RAG Pipeline — End-to-End Query → Embed → Search → Generate

Orchestrates the full retrieval-augmented generation flow:
1. Embeds user query with MedCPT (768-dim)
2. Searches dual ChromaDB indexes (text + image)
3. Assembles context from top-k results
4. Generates response via MedGemma

Can run as:
  - In-process within Colab (direct model calls)
  - Flask endpoint exposed via colab_server.py

Usage in Colab:
    from rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    answer = pipeline.query("What are the symptoms of heart failure?")
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class RetrievedChunk:
    """A single retrieved result from vector search."""
    text: str
    score: float
    source: str          # file name or dataset
    chunk_type: str      # "text" or "image"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    """Complete RAG pipeline result."""
    query: str
    answer: str
    chunks: List[RetrievedChunk]
    text_results: int
    image_results: int
    generation_tokens: int
    latency_ms: float


class RAGPipeline:
    """
    Multi-modal RAG pipeline using Colab-hosted models.

    Supports:
      - Text-only retrieval (MedCPT embeddings)
      - Multi-modal retrieval (MedCPT + SigLIP embeddings)
      - Generation with RAG context (MedGemma)
    """

    def __init__(
        self,
        chromadb_dir: str = "/content/drive/MyDrive/IDP/Medical/Chromadb",
        text_collection: str = "medical_text_768",
        image_collection: str = "medical_images_1152",
        top_k_text: int = 5,
        top_k_image: int = 3,
        max_context_tokens: int = 3000,
    ):
        self.chromadb_dir = chromadb_dir
        self.text_collection_name = text_collection
        self.image_collection_name = image_collection
        self.top_k_text = top_k_text
        self.top_k_image = top_k_image
        self.max_context_tokens = max_context_tokens

        self._chroma_client = None
        self._text_collection = None
        self._image_collection = None

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._chroma_client is None:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=self.chromadb_dir)
        return self._chroma_client

    def _get_text_collection(self):
        if self._text_collection is None:
            client = self._get_client()
            self._text_collection = client.get_or_create_collection(
                name=self.text_collection_name,
                metadata={"hnsw:space": "cosine", "dimension": 768},
            )
        return self._text_collection

    def _get_image_collection(self):
        if self._image_collection is None:
            client = self._get_client()
            self._image_collection = client.get_or_create_collection(
                name=self.image_collection_name,
                metadata={"hnsw:space": "cosine", "dimension": 1152},
            )
        return self._image_collection

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_query(self, text: str) -> List[float]:
        """Embed query text with MedCPT query encoder (768-dim)."""
        try:
            from colab_server import _embed_texts_medcpt
            return _embed_texts_medcpt([text], mode="query")[0]
        except ImportError:
            from data_ingestion import embed_texts_medcpt
            return embed_texts_medcpt([text])[0]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _search_text(self, query_embedding: List[float], top_k: int) -> List[RetrievedChunk]:
        """Search text collection by vector similarity."""
        collection = self._get_text_collection()

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

        chunks = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append(RetrievedChunk(
                    text=doc,
                    score=1 - dist,  # cosine distance → similarity
                    source=meta.get("source", "unknown"),
                    chunk_type="text",
                    metadata=meta,
                ))
        return chunks

    def _search_images(self, query_embedding: List[float], top_k: int) -> List[RetrievedChunk]:
        """
        Search image collection.

        Note: MedCPT query embeddings (768-dim) cannot directly query
        a SigLIP index (1152-dim). This method is a placeholder that
        returns image metadata stored with text descriptions.
        For cross-modal search, a projection layer or shared embedding
        space would be needed.
        """
        collection = self._get_image_collection()

        try:
            # Use text-based search on image descriptions
            results = collection.query(
                query_texts=[" "],  # Placeholder — returns top by insertion order
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.debug(f"Image search skipped: {e}")
            return []

        chunks = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append(RetrievedChunk(
                    text=f"[Image from {meta.get('source', 'unknown')}, page {meta.get('page', '?')}]: {doc}",
                    score=1 - dist,
                    source=meta.get("source", "unknown"),
                    chunk_type="image",
                    metadata=meta,
                ))
        return chunks

    # ------------------------------------------------------------------
    # Context Assembly
    # ------------------------------------------------------------------

    def _assemble_context(
        self, text_chunks: List[RetrievedChunk], image_chunks: List[RetrievedChunk],
    ) -> str:
        """Combine retrieved chunks into a context string for MedGemma."""
        sections = []

        if text_chunks:
            sections.append("=== Relevant Medical Text ===")
            for i, chunk in enumerate(text_chunks, 1):
                sections.append(
                    f"[{i}] (Source: {chunk.source}, Score: {chunk.score:.3f})\n{chunk.text}"
                )

        if image_chunks:
            sections.append("\n=== Related Medical Images ===")
            for i, chunk in enumerate(image_chunks, 1):
                sections.append(
                    f"[IMG-{i}] (Source: {chunk.source})\n{chunk.text}"
                )

        context = "\n\n".join(sections)

        # Truncate if too long (rough token estimate: 4 chars per token)
        max_chars = self.max_context_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n[Context truncated...]"

        return context

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self, query: str, context: str) -> Tuple[str, int]:
        """Generate answer with MedGemma using assembled context."""
        try:
            from colab_server import _load_medgemma, _models
            import torch

            _load_medgemma()
            tokenizer = _models["medgemma_tokenizer"]
            model = _models["medgemma_model"]

            prompt = (
                f"You are a knowledgeable medical AI assistant. "
                f"Based on the following medical context, provide an accurate, "
                f"evidence-based answer to the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                )

            input_len = inputs["input_ids"].shape[1]
            response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            tokens_used = len(outputs[0]) - input_len

            return response.strip(), tokens_used

        except ImportError:
            # Fallback: return context as-is if MedGemma not available
            logger.warning("MedGemma not available, returning raw context")
            return f"[No generation model available. Raw context:]\n\n{context}", 0

    # ------------------------------------------------------------------
    # Main Query Entry Point
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        include_images: bool = True,
        top_k_text: Optional[int] = None,
        top_k_image: Optional[int] = None,
        generate: bool = True,
    ) -> RAGResult:
        """
        Execute full RAG pipeline: embed → search → assemble → generate.

        Args:
            query_text: User query string.
            include_images: Whether to search image index too.
            top_k_text: Override default top-k for text search.
            top_k_image: Override default top-k for image search.
            generate: Whether to generate answer with MedGemma.

        Returns:
            RAGResult with answer, retrieved chunks, and metadata.
        """
        t0 = time.time()

        k_text = top_k_text or self.top_k_text
        k_image = top_k_image or self.top_k_image

        # 1. Embed query
        query_embedding = self._embed_query(query_text)

        # 2. Search text index
        text_chunks = self._search_text(query_embedding, k_text)
        logger.info(f"Text search: {len(text_chunks)} results")

        # 3. Search image index (optional)
        image_chunks = []
        if include_images:
            image_chunks = self._search_images(query_embedding, k_image)
            logger.info(f"Image search: {len(image_chunks)} results")

        # 4. Assemble context
        context = self._assemble_context(text_chunks, image_chunks)

        # 5. Generate answer
        answer = ""
        tokens_used = 0
        if generate and context:
            answer, tokens_used = self._generate(query_text, context)
        elif not generate:
            answer = context  # Return raw context

        latency = (time.time() - t0) * 1000

        return RAGResult(
            query=query_text,
            answer=answer,
            chunks=text_chunks + image_chunks,
            text_results=len(text_chunks),
            image_results=len(image_chunks),
            generation_tokens=tokens_used,
            latency_ms=round(latency, 1),
        )

    def search_only(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """Search without generation — retrieval only."""
        result = self.query(
            query_text,
            include_images=False,
            top_k_text=top_k,
            generate=False,
        )
        return result.chunks


# ---------------------------------------------------------------------------
# Flask endpoint (registered in colab_server.py)
# ---------------------------------------------------------------------------

def register_query_endpoint(app, pipeline: Optional[RAGPipeline] = None):
    """Register /query endpoint on a Flask app."""
    _pipeline = pipeline or RAGPipeline()

    @app.route("/query", methods=["POST"])
    def query_endpoint():
        from flask import request, jsonify

        data = request.get_json(force=True)
        q = data.get("query") or data.get("question")
        if not q:
            return jsonify({"error": "Missing 'query' field"}), 400

        result = _pipeline.query(
            query_text=q,
            include_images=data.get("include_images", True),
            top_k_text=data.get("top_k_text"),
            top_k_image=data.get("top_k_image"),
            generate=data.get("generate", True),
        )

        return jsonify({
            "answer": result.answer,
            "text_results": result.text_results,
            "image_results": result.image_results,
            "generation_tokens": result.generation_tokens,
            "latency_ms": result.latency_ms,
            "sources": [
                {"source": c.source, "type": c.chunk_type, "score": c.score}
                for c in result.chunks
            ],
        })

    return _pipeline


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = RAGPipeline()
    print("Multi-Modal RAG Pipeline ready. Enter a query (Ctrl+C to exit):")
    while True:
        try:
            q = input("\n> ").strip()
            if not q:
                continue
            result = pipeline.query(q)
            print(f"\n📝 Answer ({result.latency_ms}ms, {result.text_results} text + "
                  f"{result.image_results} image results):\n")
            print(result.answer)
        except KeyboardInterrupt:
            print("\nBye!")
            break
