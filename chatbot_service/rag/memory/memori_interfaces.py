"""
Memori Integration Interfaces

Defines contracts for Memori integration with proper type safety
instead of duck typing.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime



class AbstractMemoriInterface(ABC):
    """
    Abstract interface defining the contract for Memori integration.

    Any Memori-compatible object must implement these methods for
    proper integration with the MemoriRAGBridge.

    This replaces duck typing (hasattr checks) with explicit contracts.
    """

    @abstractmethod
    def get_all_memories(self, user_id: Optional[str] = None, limit: int = None) -> List[Dict]:
        """
        Retrieve all memories for a user.

        Args:
            user_id: User identifier (optional)
            limit: Maximum number of memories to return

        Returns:
            List of memory dictionaries with 'id', 'content', 'metadata' keys
        """
        pass

    @abstractmethod
    def search(self, query: str, user_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """
        Search memories by keyword.

        Args:
            query: Search query string
            user_id: Limit search to specific user (optional)
            limit: Maximum number of results

        Returns:
            List of matching memory dictionaries
        """
        pass

    @abstractmethod
    def add_memory(self, user_id: str, content: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Add a new memory.

        Args:
            user_id: User identifier
            content: Memory content
            metadata: Optional metadata dictionary

        Returns:
            Created memory dictionary with 'id' key
        """
        pass

    @abstractmethod
    def update_memory(self, memory_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """
        Update an existing memory.

        Args:
            memory_id: Memory identifier
            content: Updated content
            metadata: Updated metadata (optional)

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory identifier

        Returns:
            True if successful, False otherwise
        """
        pass


class MemoriValidator:
    """
    Validates that an object implements the Memori interface properly.

    Replaces duck typing with explicit validation.
    """

    @staticmethod
    def validate(memori: Any) -> bool:
        """
        Validate that an object implements AbstractMemoriInterface.

        Args:
            memori: Object to validate

        Returns:
            True if valid, False otherwise

        Raises:
            TypeError: If memori is None
            AttributeError: Detailed information about missing methods
        """
        if memori is None:
            raise TypeError("Memori instance cannot be None")

        if isinstance(memori, AbstractMemoriInterface):
            return True

        # Check for required methods (backward compatibility)
        required_methods = [
            "get_all_memories",
            "search",
            "add_memory",
            "update_memory",
            "delete_memory",
        ]

        missing_methods = [
            method for method in required_methods
            if not hasattr(memori, method) or not callable(getattr(memori, method))
        ]

        if missing_methods:
            raise AttributeError(
                f"Memori object is missing required methods: {', '.join(missing_methods)}. "
                f"Object should implement AbstractMemoriInterface."
            )

        return True

    @staticmethod
    def validate_method(memori: Any, method_name: str) -> bool:
        """
        Validate that a specific method exists and is callable.

        Args:
            memori: Object to check
            method_name: Name of method to validate

        Returns:
            True if method exists and is callable

        Raises:
            AttributeError: If method is missing or not callable
        """
        if not hasattr(memori, method_name):
            raise AttributeError(
                f"Memori object is missing method: '{method_name}'"
            )

        method = getattr(memori, method_name)
        if not callable(method):
            raise AttributeError(
                f"Memori.{method_name} exists but is not callable"
            )

        return True
