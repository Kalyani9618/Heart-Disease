"""
Base Embedding Service Interface

Defines the abstract base class for all embedding services to ensure
consistent API across different implementations (ONNX, PyTorch, etc.)
"""


from abc import ABC, abstractmethod
from typing import List, Optional


class BaseEmbeddingService(ABC):
    """
    Abstract base class for embedding services.
    
    All embedding implementations must inherit from this class to ensure
    consistent interface across the application.
    """
    
    @abstractmethod
    def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            use_cache: Whether to use caching if available
            
        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 32, use_cache: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts to embed
            batch_size: Size of processing batches
            use_cache: Whether to use caching if available
            
        Returns:
            List of embedding vectors (one per input text)
        """
        pass

    @abstractmethod
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (typically cosine similarity, 0-1 range)
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the embedding dimension.
        
        Returns:
            Dimension of the embedding vectors
        """
        pass