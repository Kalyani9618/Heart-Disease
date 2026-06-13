"""
Colab Embedding Service - Text Embedding Integration
Uses ngrok tunnel to connect to Colab-hosted embedding API
Provides text embedding functionality with error handling and retries

Configuration:
    COLAB_API_URL: Base URL from ngrok (env var) - e.g., "https://your-ngrok-url"
"""

import logging
import requests
from typing import Optional, List, Union
import os
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ColabEmbeddingService:
    """Text embedding service using Colab API via ngrok tunnel"""
    
    _instance: Optional['ColabEmbeddingService'] = None
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize Colab Embedding Service
        
        Args:
            api_url: Base URL for Colab API (e.g., "https://your-ngrok-url")
                    If None, reads from COLAB_API_URL env var
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum retry attempts on failure (default: 3)
            retry_delay: Delay between retries in seconds (default: 1.0)
        """
        self.api_url = api_url or os.getenv("COLAB_API_URL")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Health check tracking
        self._last_health_check: Optional[datetime] = None
        self._health_check_interval: timedelta = timedelta(minutes=5)
        self._is_healthy: bool = False
        
        if not self.api_url:
            logger.warning(
                "⚠️  COLAB_API_URL not configured. "
                "Set it before making embedding requests."
            )
        else:
            logger.info(f"✅ Colab Embedding Service initialized with URL: {self.api_url}")
            # Run initial health check
            self._health_check()
    
    @classmethod
    def get_instance(
        cls,
        api_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> 'ColabEmbeddingService':
        """
        Get or create singleton instance
        
        Args:
            api_url: Base URL for Colab API
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            
        Returns:
            Singleton instance of ColabEmbeddingService
        """
        if cls._instance is None:
            cls._instance = cls(
                api_url=api_url,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
        return cls._instance
    
    def _health_check(self) -> bool:
        """
        Check if Colab API is accessible
        
        Returns:
            True if healthy, False otherwise
        """
        if not self.api_url:
            logger.error("❌ COLAB_API_URL not configured")
            self._is_healthy = False
            return False
        
        # Skip if recently checked
        now = datetime.now()
        if (self._last_health_check and 
            now - self._last_health_check < self._health_check_interval):
            return self._is_healthy
        
        try:
            response = requests.get(
                f"{self.api_url}/health",
                timeout=5
            )
            self._is_healthy = response.status_code == 200
            self._last_health_check = now
            
            if self._is_healthy:
                logger.debug("✅ Colab API health check passed")
            else:
                logger.warning(f"⚠️  Colab API health check failed: {response.status_code}")
            
            return self._is_healthy
        except Exception as e:
            logger.error(f"❌ Colab API health check error: {e}")
            self._is_healthy = False
            self._last_health_check = now
            return False
    
    def _make_request(self, endpoint: str, payload: dict, attempt: int = 1) -> Optional[dict]:
        """
        Make request to Colab API with retry logic
        
        Args:
            endpoint: API endpoint (e.g., "/embed_text")
            payload: Request payload
            attempt: Current attempt number
            
        Returns:
            Response JSON or None if failed
        """
        if not self.api_url:
            logger.error("❌ COLAB_API_URL not configured")
            return None
        
        full_url = f"{self.api_url}{endpoint}"
        
        try:
            logger.debug(f"🔄 Attempt {attempt}: POST {full_url}")
            response = requests.post(
                full_url,
                json=payload,
                timeout=self.timeout,
            )
            
            if response.status_code == 200:
                logger.debug(f"✅ Request successful: {endpoint}")
                return response.json()
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:100]}"
                logger.warning(f"⚠️  {error_msg}")
                
                # Retry on server errors
                if response.status_code >= 500 and attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
                    return self._make_request(endpoint, payload, attempt + 1)
                
                return None
        
        except requests.exceptions.Timeout:
            logger.warning(f"⏱️  Request timeout after {self.timeout}s")
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)
                return self._make_request(endpoint, payload, attempt + 1)
            return None
        
        except requests.exceptions.ConnectionError as e:
            logger.error(f"❌ Connection error: {e}")
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)
                return self._make_request(endpoint, payload, attempt + 1)
            return None
        
        except Exception as e:
            logger.error(f"❌ Request error: {e}")
            return None
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Get text embedding from Colab API
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (list of floats) or None if failed
            
        Raises:
            ValueError: If text is empty or invalid
        """
        if not text or not isinstance(text, str):
            logger.error(f"❌ Invalid text input: {text}")
            return None
        
        if not text.strip():
            logger.error("❌ Text cannot be empty")
            return None
        
        logger.info(f"📝 Embedding text ({len(text)} chars)")
        
        response = self._make_request(
            "/embed_text",
            {"text": text}
        )
        
        if response and "embedding" in response:
            embedding = response["embedding"]
            logger.debug(f"✅ Got embedding of dimension {len(embedding)}")
            return embedding
        
        logger.error("❌ Failed to get text embedding")
        return None
    
    def embed_batch(
        self,
        texts: List[str],
        fail_on_error: bool = False,
    ) -> List[Optional[List[float]]]:
        """
        Embed multiple texts
        
        Args:
            texts: List of texts to embed
            fail_on_error: If True, return None if any embedding fails
                          If False, return partial results
            
        Returns:
            List of embeddings (None for failed items)
        """
        if not texts:
            logger.warning("⚠️  Empty text list provided")
            return []
        
        logger.info(f"📚 Embedding batch of {len(texts)} texts")
        embeddings = []
        
        for i, text in enumerate(texts):
            logger.debug(f"  [{i+1}/{len(texts)}] Embedding...")
            embedding = self.embed_text(text)
            
            if embedding is None and fail_on_error:
                logger.error(f"❌ Failed to embed text {i+1}, aborting batch")
                return None
            
            embeddings.append(embedding)
        
        successful = sum(1 for e in embeddings if e is not None)
        logger.info(f"✅ Batch complete: {successful}/{len(texts)} successful")
        
        return embeddings
    
    def is_available(self) -> bool:
        """
        Check if embedding service is available and configured
        
        Returns:
            True if service is ready to use
        """
        if not self.api_url:
            return False
        return self._health_check()
    
    def set_api_url(self, api_url: str) -> None:
        """
        Update the API URL (e.g., when ngrok tunnel changes)
        
        Args:
            api_url: New API URL from ngrok
        """
        if api_url:
            self.api_url = api_url
            logger.info(f"✅ Updated Colab API URL: {api_url}")
            # Reset health check
            self._last_health_check = None
            self._health_check()
        else:
            logger.error("❌ Invalid API URL provided")


# ============================================================================
# Utility functions for easy integration
# ============================================================================

def get_colab_embedding_service(
    api_url: Optional[str] = None,
) -> ColabEmbeddingService:
    """
    Get the Colab embedding service instance
    
    Args:
        api_url: Optional API URL (overrides env var)
        
    Returns:
        ColabEmbeddingService instance
    """
    return ColabEmbeddingService.get_instance(api_url=api_url)


def embed_text(
    text: str,
    api_url: Optional[str] = None,
) -> Optional[List[float]]:
    """
    Quick function to embed text using Colab API
    
    Args:
        text: Text to embed
        api_url: Optional API URL (overrides env var)
        
    Returns:
        Embedding vector or None if failed
    """
    service = get_colab_embedding_service(api_url=api_url)
    return service.embed_text(text)


def embed_batch(
    texts: List[str],
    api_url: Optional[str] = None,
    fail_on_error: bool = False,
) -> List[Optional[List[float]]]:
    """
    Quick function to embed multiple texts using Colab API
    
    Args:
        texts: List of texts to embed
        api_url: Optional API URL (overrides env var)
        fail_on_error: If True, return None if any embedding fails
        
    Returns:
        List of embeddings
    """
    service = get_colab_embedding_service(api_url=api_url)
    return service.embed_batch(texts, fail_on_error=fail_on_error)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Configure logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Get API URL from environment or command line
    api_url = os.getenv("COLAB_API_URL") or (sys.argv[1] if len(sys.argv) > 1 else None)
    
    if not api_url:
        print("❌ Usage: python embedding_colab.py <api_url>")
        print("   Or set COLAB_API_URL environment variable")
        print("\n   Example: python embedding_colab.py https://your-ngrok-url")
        sys.exit(1)
    
    print(f"\n🔗 Testing Colab Embedding Service with URL: {api_url}\n")
    
    # Initialize service
    service = ColabEmbeddingService.get_instance(api_url=api_url)
    
    # Test 1: Single text embedding
    print("=" * 70)
    print("TEST 1: Single Text Embedding")
    print("=" * 70)
    
    test_text = "Patient has severe chest pain and shortness of breath"
    print(f"📝 Text: {test_text}")
    
    embedding = service.embed_text(test_text)
    if embedding:
        print(f"✅ Embedding received!")
        print(f"   Dimension: {len(embedding)}")
        print(f"   Sample values: {embedding[:5]}")
    else:
        print("❌ Failed to get embedding")
    
    # Test 2: Batch embedding
    print("\n" + "=" * 70)
    print("TEST 2: Batch Text Embedding")
    print("=" * 70)
    
    test_texts = [
        "Patient reports high blood pressure",
        "Cardiac imaging shows abnormal patterns",
        "Treatment plan includes medication adjustment"
    ]
    
    print(f"📚 Embedding {len(test_texts)} texts...")
    embeddings = service.embed_batch(test_texts)
    
    successful = sum(1 for e in embeddings if e is not None)
    print(f"✅ Results: {successful}/{len(test_texts)} successful")
    
    for i, (text, emb) in enumerate(zip(test_texts, embeddings)):
        status = "✅" if emb else "❌"
        dim = len(emb) if emb else "N/A"
        print(f"   {status} [{i+1}] {text[:40]}... (dim: {dim})")
    
    # Test 3: Service availability
    print("\n" + "=" * 70)
    print("TEST 3: Service Status")
    print("=" * 70)
    
    if service.is_available():
        print("✅ Service is available and healthy")
    else:
        print("⚠️  Service may be unavailable")
    
    print("\n✅ Testing complete!\n")
