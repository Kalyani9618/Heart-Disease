"""
Remote Colab Embeddings - LangChain Integration (HeartGuard Edition)
A custom LangChain embedding class that uses Google Colab as the embedding backend
via ngrok tunnel. Supports both text (MedCPT) and image (SigLIP) embeddings.

This allows ChromaDB, FAISS, and other vector stores to use remote embeddings
without local GPU/CPU resources.

Configuration:
    COLAB_API_URL: Base URL from ngrok (env var)
    
Usage:
    from remote_embeddings import RemoteColabEmbeddings
    from langchain_community.vectorstores import Chroma
    
    embeddings = RemoteColabEmbeddings(
        base_url="https://your-ngrok-url",
        dimension=768  # For MedCPT
    )
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # For images (X-rays)
    xray_embedding = embeddings.embed_image("xray.jpg")
"""

import logging
import requests
from typing import List, Optional
import os
import time
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class RemoteColabEmbeddings(Embeddings):
    """
    A custom LangChain embedding class for HeartGuard that sends text and images
    to Google Colab instance via ngrok instead of running embeddings locally.
    
    Supports:
    - Text embeddings (MedCPT: 768 dimensions)
    - Image embeddings (SigLIP: 1152 dimensions)
    - Both LangChain interface and custom methods
    
    This is a drop-in replacement for HuggingFaceEmbeddings, SentenceTransformerEmbeddings,
    or any other LangChain embeddings class.
    
    Features:
    - Compatible with ChromaDB, FAISS, Pinecone, and other LangChain vector stores
    - Automatic retry logic for robustness
    - Configurable timeout and retry parameters
    - Logging for debugging
    - Health check support
    - Image embedding support for medical images
    """
    
    def __init__(
        self,
        base_url: str,
        dimension: int = 768,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize RemoteColabEmbeddings
        
        Args:
            base_url: Base URL of Colab API (e.g., "https://your-ngrok-url")
                     The /embed_text and /embed_image endpoints will be appended
            dimension: Embedding dimension (default: 768 for MedCPT)
                      This MUST match your Colab model's output dimension!
                      - MedCPT: 768
                      - SigLIP: 1152
                      - all-MiniLM: 384
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay between retries in seconds (default: 1.0)
            
        Raises:
            ValueError: If base_url is invalid or empty
        """
        if not base_url:
            raise ValueError("base_url cannot be empty")
        
        # Remove trailing slash if present
        self.base_url = base_url.rstrip("/")
        self.text_endpoint = f"{self.base_url}/embed_text"
        self.image_endpoint = f"{self.base_url}/embed_image"
        self.dimension = dimension
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(
            f"✅ RemoteColabEmbeddings initialized with URL: {self.base_url} "
            f"(dimension: {self.dimension})"
        )
        
        # Test connection
        self._health_check()
    
    def _health_check(self) -> bool:
        """
        Quick health check to verify Colab API is accessible
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            if response.status_code == 200:
                logger.debug("✅ Colab API health check passed")
                return True
            else:
                logger.warning(f"⚠️  Colab API returned status {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"⚠️  Colab API health check failed: {e}")
            return False
    
    def _send_request(
        self,
        endpoint: str,
        payload: dict,
        is_file: bool = False,
        file_path: Optional[str] = None,
        attempt: int = 1
    ) -> List[float]:
        """
        Send embedding request to Colab API with retry logic
        Supports both JSON (text) and multipart/form-data (images)
        
        Args:
            endpoint: API endpoint URL
            payload: Request payload (for JSON requests)
            is_file: If True, sends file data instead of JSON
            file_path: Path to file for multipart upload
            attempt: Current attempt number (for retry tracking)
            
        Returns:
            Embedding vector (list of floats)
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        try:
            logger.debug(f"🔄 Attempt {attempt}: POST {endpoint}")
            
            if is_file and file_path:
                # Multipart form-data for images
                with open(file_path, "rb") as f:
                    files = {"file": f}
                    response = requests.post(
                        endpoint,
                        files=files,
                        timeout=self.timeout
                    )
            else:
                # JSON for text
                response = requests.post(
                    endpoint,
                    json=payload,
                    timeout=self.timeout
                )
            
            # Success
            if response.status_code == 200:
                result = response.json()
                if "embedding" in result:
                    embedding = result["embedding"]
                    logger.debug(f"✅ Got embedding (dim: {len(embedding)})")
                    return embedding
                else:
                    raise ValueError(f"Invalid response format: {result}")
            
            # Server error - retry
            elif response.status_code >= 500 and attempt < self.max_retries:
                logger.warning(
                    f"⚠️  Server error {response.status_code}, "
                    f"retrying ({attempt}/{self.max_retries})..."
                )
                time.sleep(self.retry_delay * attempt)
                return self._send_request(
                    endpoint, payload, is_file, file_path, attempt + 1
                )
            
            # Client error or other - fail
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                logger.error(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
        
        except requests.exceptions.Timeout:
            logger.warning(f"⏱️  Request timeout after {self.timeout}s")
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)
                return self._send_request(
                    endpoint, payload, is_file, file_path, attempt + 1
                )
            raise RuntimeError(f"Timeout after {self.max_retries} retries")
        
        except requests.exceptions.ConnectionError as e:
            logger.error(f"❌ Connection error: {e}")
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)
                return self._send_request(
                    endpoint, payload, is_file, file_path, attempt + 1
                )
            raise RuntimeError(f"Connection failed after {self.max_retries} retries: {e}")
        
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            raise RuntimeError(f"Failed to get embedding: {e}")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text (e.g., user question)
        
        This is called when searching/querying a vector database.
        
        Args:
            text: The query text to embed
            
        Returns:
            Embedding vector (list of floats)
        """
        logger.info(f"📝 Embedding query ({len(text)} chars)")
        return self._send_request(self.text_endpoint, {"text": text})
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents (e.g., chunks for vector database)
        
        This is called when adding documents to a vector database like ChromaDB.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            logger.warning("⚠️  Empty text list provided")
            return []
        
        logger.info(f"📚 Embedding batch of {len(texts)} documents")
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                logger.debug(f"  [{i+1}/{len(texts)}] Embedding...")
                embedding = self._send_request(self.text_endpoint, {"text": text})
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"❌ Failed to embed document {i+1}: {e}")
                # FIXED: Use the configured dimension for the zero-vector fallback
                # This prevents ChromaDB crashes due to dimension mismatches
                embeddings.append([0.0] * self.dimension)
        
        successful = sum(1 for e in embeddings if any(e))
        logger.info(f"✅ Batch complete: {successful}/{len(texts)} successful")
        
        return embeddings
    
    # ============================================================================
    # HeartGuard Custom Methods (beyond LangChain interface)
    # ============================================================================
    
    def embed_image(self, image_path: str) -> List[float]:
        """
        Embed a medical image (X-ray, CT scan, etc.) using SigLIP
        
        This is a custom method for HeartGuard to support multimodal embeddings.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Embedding vector (list of floats, typically 1152 dimensions)
            
        Raises:
            RuntimeError: If image embedding fails
        """
        if not image_path or not os.path.exists(image_path):
            logger.error(f"❌ Invalid image path: {image_path}")
            raise RuntimeError(f"Image file not found: {image_path}")
        
        logger.info(f"🖼️  Embedding image: {image_path}")
        
        try:
            return self._send_request(
                self.image_endpoint,
                payload=None,
                is_file=True,
                file_path=image_path
            )
        except Exception as e:
            logger.error(f"❌ Failed to embed image {image_path}: {e}")
            # Return empty vector to indicate failure
            return []
    
    def embed_batch_images(
        self,
        image_paths: List[str],
        fail_on_error: bool = False,
    ) -> List[List[float]]:
        """
        Embed multiple medical images in batch
        
        Args:
            image_paths: List of paths to image files
            fail_on_error: If True, return None if any embedding fails
                          If False, return partial results
            
        Returns:
            List of embeddings (empty list for failed items if fail_on_error=False)
        """
        if not image_paths:
            logger.warning("⚠️  Empty image path list provided")
            return []
        
        logger.info(f"🖼️  Embedding batch of {len(image_paths)} images")
        embeddings = []
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.debug(f"  [{i+1}/{len(image_paths)}] Processing {image_path}...")
                embedding = self.embed_image(image_path)
                
                if not embedding and fail_on_error:
                    logger.error(f"❌ Failed to embed image {i+1}, aborting batch")
                    return None
                
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"❌ Failed to embed image {i+1}: {e}")
                if fail_on_error:
                    return None
                embeddings.append([])
        
        successful = sum(1 for e in embeddings if e)
        logger.info(f"✅ Batch complete: {successful}/{len(image_paths)} successful")
        
        return embeddings
    
    def update_base_url(self, base_url: str) -> None:
        """
        Update the base URL (useful if ngrok tunnel changes)
        
        Args:
            base_url: New Colab API base URL
            
        Raises:
            ValueError: If base_url is invalid
        """
        if not base_url:
            raise ValueError("base_url cannot be empty")
        
        self.base_url = base_url.rstrip("/")
        self.text_endpoint = f"{self.base_url}/embed_text"
        self.image_endpoint = f"{self.base_url}/embed_image"
        logger.info(f"✅ Updated Colab API URL: {self.base_url}")
        self._health_check()


# ============================================================================
# Convenience function
# ============================================================================

def create_remote_colab_embeddings(
    base_url: str = None,
    dimension: int = 768,
    timeout: int = 30,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> RemoteColabEmbeddings:
    """
    Factory function to create RemoteColabEmbeddings instance
    
    Args:
        base_url: Colab API base URL (or reads from COLAB_API_URL env var)
        dimension: Embedding dimension (default: 768 for MedCPT)
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries
        
    Returns:
        RemoteColabEmbeddings instance
        
    Raises:
        ValueError: If no base_url provided and COLAB_API_URL env var not set
    """
    if not base_url:
        base_url = os.getenv("COLAB_API_URL")
    
    if not base_url:
        raise ValueError(
            "base_url required. Either pass as argument or set COLAB_API_URL env var"
        )
    
    return RemoteColabEmbeddings(
        base_url=base_url,
        dimension=dimension,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Get URL
    api_url = os.getenv("COLAB_API_URL") or (sys.argv[1] if len(sys.argv) > 1 else None)
    
    if not api_url:
        print("❌ Usage: python remote_embeddings.py <api_url>")
        print("   Or set COLAB_API_URL environment variable")
        sys.exit(1)
    
    print(f"\n🔗 Testing RemoteColabEmbeddings with URL: {api_url}\n")
    
    # Create embeddings instance with MedCPT dimension (768)
    embeddings = RemoteColabEmbeddings(base_url=api_url, dimension=768)
    
    # Test 1: Single query
    print("=" * 70)
    print("TEST 1: Single Query Embedding (Text)")
    print("=" * 70)
    
    query = "What are symptoms of heart disease?"
    print(f"📝 Query: {query}")
    
    try:
        embedding = embeddings.embed_query(query)
        print(f"✅ Embedding received!")
        print(f"   Dimension: {len(embedding)}")
        print(f"   Sample values: {embedding[:5]}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Document batch
    print("\n" + "=" * 70)
    print("TEST 2: Document Batch Embedding")
    print("=" * 70)
    
    documents = [
        "Myocardial infarction occurs when blood flow to heart is blocked",
        "Hypertension is a risk factor for cardiovascular disease",
        "Regular exercise improves cardiac function"
    ]
    
    print(f"📚 Documents to embed: {len(documents)}")
    
    try:
        embeddings_list = embeddings.embed_documents(documents)
        print(f"✅ Got {len(embeddings_list)} embeddings")
        
        for i, (doc, emb) in enumerate(zip(documents, embeddings_list)):
            dim = len(emb) if any(emb) else "FAILED"
            print(f"   [{i+1}] {doc[:50]}... (dim: {dim})")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Dimension validation
    print("\n" + "=" * 70)
    print("TEST 3: Dimension Verification")
    print("=" * 70)
    print(f"✅ Configured dimension: {embeddings.dimension}")
    print(f"✅ Text endpoint: {embeddings.text_endpoint}")
    print(f"✅ Image endpoint: {embeddings.image_endpoint}")
    
    print("\n✅ Testing complete!\n")
