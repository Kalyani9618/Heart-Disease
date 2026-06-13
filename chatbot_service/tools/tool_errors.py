"""
Tool Error Handling - Structured error handling for agentic tools.

Provides:
- ToolError: Structured error dataclass with context
- ToolErrorHandler: Factory for creating standardized errors
- Error codes with recovery suggestions

This ensures agents receive actionable error information with
suggestions for recovery or alternative approaches.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
import traceback
import logging
from datetime import datetime

if TYPE_CHECKING:
    from tools.tool_registry import ToolResult

logger = logging.getLogger(__name__)



@dataclass
class ToolError:
    """
    Structured tool error with context and recovery information.
    
    Attributes:
        code: Error code for categorization (e.g., "TIMEOUT", "NOT_FOUND")
        message: Human-readable error message
        recoverable: Whether the error can potentially be recovered from
        context: Additional context about the error
        suggestions: List of suggested recovery actions
        stack_trace: Optional stack trace for debugging
        timestamp: When the error occurred
    """
    code: str
    message: str
    recoverable: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    stack_trace: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
            "context": self.context,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp,
        }
    
    def __str__(self) -> str:
        """String representation for logging."""
        return f"[{self.code}] {self.message}"


class ToolErrorHandler:
    """
    Factory for creating structured tool errors.
    
    Provides standardized error codes with predefined messages and
    recovery suggestions for common error scenarios.
    
    Example:
        ```python
        try:
            result = await some_tool()
        except TimeoutError as e:
            error = ToolErrorHandler.create_error("TIMEOUT", e, {"query": query})
            return ToolErrorHandler.to_tool_result(error, "my_tool")
        ```
    """
    
    # Error code definitions: (message_template, recoverable, suggestions)
    ERROR_CODES: Dict[str, Tuple[str, bool, List[str]]] = {
        # Initialization errors
        "NOT_INITIALIZED": (
            "Tool not initialized",
            True,
            ["Call initialize_agent_tools() first", "Check service startup logs"]
        ),
        
        # Input validation errors
        "INVALID_INPUT": (
            "Invalid input parameters",
            True,
            ["Check parameter types", "Review tool schema", "Verify required fields"]
        ),
        "MISSING_PARAMETER": (
            "Required parameter missing",
            True,
            ["Provide all required parameters", "Check tool documentation"]
        ),
        
        # External service errors
        "EXTERNAL_SERVICE": (
            "External service error",
            True,
            ["Retry in a few seconds", "Check service status", "Use fallback if available"]
        ),
        "CONNECTION_FAILED": (
            "Failed to connect to service",
            True,
            ["Check network connectivity", "Verify service URL", "Retry after delay"]
        ),
        
        # Rate limiting and timeouts
        "TIMEOUT": (
            "Operation timed out",
            True,
            ["Retry with simpler query", "Increase timeout", "Check service health"]
        ),
        "RATE_LIMIT": (
            "Rate limit exceeded",
            True,
            ["Wait before retrying", "Use cached results", "Reduce request frequency"]
        ),
        
        # Authentication and authorization
        "AUTH_FAILED": (
            "Authentication failed",
            False,
            ["Check API keys", "Verify credentials", "Refresh authentication token"]
        ),
        "PERMISSION_DENIED": (
            "Permission denied",
            False,
            ["Verify user permissions", "Check access rights", "Contact administrator"]
        ),
        
        # Resource errors
        "NOT_FOUND": (
            "Resource not found",
            False,
            ["Verify resource exists", "Check ID/path", "Search for alternatives"]
        ),
        "ALREADY_EXISTS": (
            "Resource already exists",
            True,
            ["Use existing resource", "Choose different identifier"]
        ),
        
        # Data errors
        "EMPTY_RESULT": (
            "No results found",
            True,
            ["Try broader search terms", "Check data availability", "Verify filters"]
        ),
        "PARSE_ERROR": (
            "Failed to parse response",
            True,
            ["Check data format", "Verify encoding", "Try alternative parser"]
        ),
        
        # Execution errors
        "EXECUTION_ERROR": (
            "Tool execution failed",
            True,
            ["Check input parameters", "Review tool logs", "Try alternative approach"]
        ),
        "CALCULATION_ERROR": (
            "Calculation error",
            True,
            ["Check math expression syntax", "Verify numeric values", "Simplify expression"]
        ),
        
        # Safety and security
        "SECURITY_VIOLATION": (
            "Security check failed",
            False,
            ["Remove unsafe content", "Check for injection attacks", "Use sanitized input"]
        ),
        "CONTENT_FILTERED": (
            "Content was filtered for safety",
            True,
            ["Rephrase the request", "Remove sensitive content"]
        ),
        
        # Unknown/generic
        "UNKNOWN": (
            "An unknown error occurred",
            True,
            ["Check logs for details", "Retry the operation", "Contact support"]
        ),
    }
    
    @classmethod
    def create_error(
        cls,
        code: str,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        include_trace: bool = True
    ) -> ToolError:
        """
        Create a structured error from an error code and optional exception.
        
        Args:
            code: Error code (from ERROR_CODES or custom)
            exception: Optional exception that caused the error
            context: Additional context dictionary
            include_trace: Whether to include stack trace (default True)
            
        Returns:
            ToolError instance with all relevant information
        """
        # Get predefined error info or use defaults
        message_template, recoverable, suggestions = cls.ERROR_CODES.get(
            code, ("Unknown error", True, ["Check logs for details"])
        )
        
        # Build message
        if exception:
            message = f"{message_template}: {str(exception)}"
            stack_trace = traceback.format_exc() if include_trace else None
        else:
            message = message_template
            stack_trace = None
        
        # Log the error
        log_level = logging.WARNING if recoverable else logging.ERROR
        logger.log(log_level, f"ToolError [{code}]: {message}")
        
        return ToolError(
            code=code,
            message=message,
            recoverable=recoverable,
            context=context or {},
            suggestions=list(suggestions),  # Copy to avoid mutation
            stack_trace=stack_trace
        )
    
    @classmethod
    def to_tool_result(cls, error: ToolError, tool_name: str) -> 'ToolResult':
        """
        Convert a ToolError to a ToolResult for tool return.
        
        Args:
            error: The ToolError to convert
            tool_name: Name of the tool that failed
            
        Returns:
            ToolResult with success=False and error details
        """
        # Import here to avoid circular imports
        from tools.tool_registry import ToolResult
        
        return ToolResult(
            success=False,
            error=error.message,
            data={
                "error_code": error.code,
                "recoverable": error.recoverable,
                "suggestions": error.suggestions,
                "context": error.context,
                "timestamp": error.timestamp
            },
            tool_name=tool_name
        )
    
    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        tool_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> 'ToolResult':
        """
        Convenience method to create ToolResult directly from an exception.
        
        Automatically maps exception types to appropriate error codes.
        
        Args:
            exception: The exception to convert
            tool_name: Name of the tool that failed
            context: Optional context dictionary
            
        Returns:
            ToolResult with appropriate error details
        """
        # Map exception types to error codes
        exception_map = {
            TimeoutError: "TIMEOUT",
            ConnectionError: "CONNECTION_FAILED",
            PermissionError: "PERMISSION_DENIED",
            FileNotFoundError: "NOT_FOUND",
            ValueError: "INVALID_INPUT",
            TypeError: "INVALID_INPUT",
            KeyError: "NOT_FOUND",
            ZeroDivisionError: "CALCULATION_ERROR",
            OverflowError: "CALCULATION_ERROR",
        }
        
        # Find matching error code
        error_code = "EXECUTION_ERROR"  # Default
        for exc_type, code in exception_map.items():
            if isinstance(exception, exc_type):
                error_code = code
                break
        
        error = cls.create_error(error_code, exception, context)
        return cls.to_tool_result(error, tool_name)


# Convenience function for quick error creation
def create_tool_error(
    code: str,
    tool_name: str,
    exception: Optional[Exception] = None,
    context: Optional[Dict[str, Any]] = None
) -> 'ToolResult':
    """
    Quick function to create a ToolResult with error details.
    
    Example:
        ```python
        if not _initialized:
            return create_tool_error("NOT_INITIALIZED", "my_tool")
        ```
    """
    error = ToolErrorHandler.create_error(code, exception, context)
    return ToolErrorHandler.to_tool_result(error, tool_name)
