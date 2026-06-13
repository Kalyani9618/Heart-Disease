"""
Remote Embedding Service — Colab-hosted MedCPT / SigLIP via ngrok

Implements BaseEmbeddingService so it can be used as a drop-in replacement
for the ONNX and PyTorch backends throughout the RAG pipeline.

Wraps the RemoteColabEmbeddings client from colab/remote_embeddings.py
to conform to the rag/interfaces/embedding_base.py contract.
"""


import os
import time
import logging
import hashlib
import asyncio
import importlib.util
import numpy as np
from typing import List, Optional
from collections import OrderedDict
from pathlib import Path

from rag.embedding.base import BaseEmbeddingService

logger = logging.getLogger(__name__)


class RemoteEmbeddingService(BaseEmbeddingService):
    """
    Embedding service that delegates to a remote Colab instance via ngrok.

    Text:  MedCPT  → 768-dim
    Image: SigLIP  → 1152-dim

    The service defaults to text mode; use ``embed_image`` / ``embed_batch_images``
    for the SigLIP pathway.
    """

    _instance: Optional["RemoteEmbeddingService"] = None

    def __init__(
        self,
        base_url: Optional[str] = None,
        dimension: int = 768,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache_size: int = 5000,
    ):
        """
        Args:
            base_url:    ngrok URL of the Colab server.
                         Falls back to COLAB_API_URL env var.
            dimension:   Embedding dimension for *text* (768 for MedCPT).
            timeout:     HTTP request timeout in seconds.
            max_retries: Number of retry attempts on failure.
            retry_delay: Seconds between retries.
            cache_size:  Max entries in the local embedding cache.
        """
        self.base_url = (base_url or os.getenv("COLAB_API_URL", "")).rstrip("/")
        if not self.base_url:
            raise ValueError(
                "Remote embedding URL required. Pass base_url or set COLAB_API_URL env var."
            )

        self.dimension = dimension
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Local L1 cache (LRU)
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._cache_size = cache_size

        # Lazy-loaded RemoteColabEmbeddings client
        self._client = None

        # Validate the remote server is reachable on startup
        self._validate_connection()

        logger.info(
            f"RemoteEmbeddingService initialized (url={self.base_url}, dim={self.dimension})"
        )

    def _validate_connection(self) -> None:
        """Validate that the remote server is reachable at init time.

        Logs a warning instead of raising so the service can still be
        constructed (the server may come online later).
        """
        try:
            import requests

            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                logger.info(f"Remote embedding server reachable at {self.base_url}")
            else:
                logger.warning(
                    f"Remote embedding server returned status {resp.status_code} "
                    f"at {self.base_url}/health"
                )
        except Exception as exc:
            logger.warning(
                f"Remote embedding server not reachable at {self.base_url}: {exc}. "
                "Embedding calls will be attempted lazily."
            )

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, base_url: Optional[str] = None, **kwargs) -> "RemoteEmbeddingService":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(base_url=base_url, **kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Client
    # ------------------------------------------------------------------

    def _import_remote_colab_embeddings(self):
        """Resolve RemoteColabEmbeddings across supported project layouts."""
        try:
            from colab.remote_embeddings import RemoteColabEmbeddings
            return RemoteColabEmbeddings
        except Exception:
            pass

        current_file = Path(__file__).resolve()
        chatbot_service_root = current_file.parents[2]
        workspace_root = chatbot_service_root.parent

        candidates = [
            chatbot_service_root / "colab" / "remote_embeddings.py",
            workspace_root / "colab" / "remote_embeddings.py",
            workspace_root / "colab Rag_Training" / "remote_embeddings.py",
        ]

        for path in candidates:
            if not path.exists():
                continue
            spec = importlib.util.spec_from_file_location("remote_embeddings_dynamic", str(path))
            if not spec or not spec.loader:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls = getattr(module, "RemoteColabEmbeddings", None)
            if cls is not None:
                logger.info(f"Loaded RemoteColabEmbeddings from: {path}")
                return cls

        searched = ", ".join(str(p) for p in candidates)
        raise RuntimeError(
            "RemoteColabEmbeddings could not be imported. "
            f"Checked: {searched}. Ensure remote_embeddings.py exists "
            "or disable USE_REMOTE_EMBEDDINGS."
        )

    def _get_client(self):
        """Lazy-load the RemoteColabEmbeddings LangChain wrapper.

        Raises ``RuntimeError`` with an actionable message when the
        dependency is missing so callers get a clear error instead of
        a raw ``ImportError``.
        """
        if self._client is None:
            RemoteColabEmbeddings = self._import_remote_colab_embeddings()

            self._client = RemoteColabEmbeddings(
                base_url=self.base_url,
                dimension=self.dimension,
                timeout=self.timeout,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
            )
        return self._client

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _get_cached(self, key: str) -> Optional[List[float]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _set_cached(self, key: str, embedding: List[float]):
        self._cache[key] = embedding
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    # ------------------------------------------------------------------
    # BaseEmbeddingService interface
    # ------------------------------------------------------------------

    def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Generate embedding for a single text (768-dim via MedCPT).

        Implements BaseEmbeddingService.embed_text.
        Retries up to ``max_retries`` times with exponential back-off.
        """
        if use_cache:
            key = self._cache_key(text)
            cached = self._get_cached(key)
            if cached is not None:
                return cached

        client = self._get_client()

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                embedding = client.embed_query(text)
                if use_cache:
                    self._set_cached(key, embedding)
                return embedding
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"embed_text attempt {attempt}/{self.max_retries} failed: {exc}. "
                        f"Retrying in {wait:.1f}s …"
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"embed_text failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        use_cache: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Implements BaseEmbeddingService.embed_batch.
        """
        if not texts:
            return []

        # Check cache first
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []

        if use_cache:
            for i, text in enumerate(texts):
                key = self._cache_key(text)
                cached = self._get_cached(key)
                if cached is not None:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
        else:
            uncached_indices = list(range(len(texts)))

        # Embed uncached texts in batches of `batch_size`
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            client = self._get_client()

            for batch_start in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[batch_start : batch_start + batch_size]
                batch_indices = uncached_indices[batch_start : batch_start + batch_size]
                new_embeddings = client.embed_documents(batch_texts)

                for idx, emb in zip(batch_indices, new_embeddings):
                    results[idx] = emb
                    if use_cache:
                        self._set_cached(self._cache_key(texts[idx]), emb)

        return results  # type: ignore[return-value]

    # Backward-compatible aliases used by existing vector store code.
    def embed(self, text: str) -> List[float]:
        return self.embed_text(text)

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        return self.embed_batch(texts=texts, batch_size=batch_size)

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.

        Implements BaseEmbeddingService.similarity.
        """
        emb1 = np.array(self.embed_text(text1))
        emb2 = np.array(self.embed_text(text2))

        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)

        if norm == 0:
            return 0.0
        return float(dot / norm)

    def get_dimension(self) -> int:
        """
        Get the embedding dimension (768 for MedCPT text).

        Implements BaseEmbeddingService.get_dimension.
        """
        return self.dimension

    # ------------------------------------------------------------------
    # Extended: image embeddings (SigLIP, 1152-dim)
    # ------------------------------------------------------------------

    def embed_image(self, image_path: str) -> List[float]:
        """Embed a single image via SigLIP (1152-dim)."""
        client = self._get_client()
        return client.embed_image(image_path)

    def embed_batch_images(self, image_paths: List[str]) -> List[List[float]]:
        """Embed multiple images via SigLIP (1152-dim)."""
        client = self._get_client()
        return client.embed_batch_images(image_paths)

    # ------------------------------------------------------------------
    # Cache warming
    # ------------------------------------------------------------------

    def warm_cache(self, texts: List[str]) -> int:
        """Pre-populate the embedding cache with common terms.

        Args:
            texts: List of texts to pre-embed and cache.

        Returns:
            Number of embeddings warmed (newly cached).
        """
        warmed = 0
        for text in texts:
            key = self._cache_key(text)
            if self._get_cached(key) is None:
                try:
                    self.embed_text(text, use_cache=True)
                    warmed += 1
                except Exception as exc:
                    logger.debug(f"warm_cache: failed to embed '{text[:30]}…': {exc}")
        return warmed

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict:
        """Return model information."""
        return {
            "backend": "remote_colab",
            "base_url": self.base_url,
            "text_model": "ncbi/MedCPT-Query-Encoder",
            "image_model": "google/siglip-base-patch16-256",
            "text_dimension": 768,
            "image_dimension": 1152,
            "cache_entries": len(self._cache),
        }

    def health_check(self) -> bool:
        """Check if the remote server is reachable (synchronous)."""
        try:
            import requests
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    async def ahealth_check(self) -> bool:
        """Async health check — non-blocking version of ``health_check``."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/health", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False
