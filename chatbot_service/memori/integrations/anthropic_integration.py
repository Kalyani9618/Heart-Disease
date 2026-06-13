"""
Anthropic Integration - Clean wrapper without monkey-patching

RECOMMENDED: Use LiteLLM instead for unified API and native callback support.
This integration is provided for direct Anthropic SDK usage.

Updated: Added streaming support, retry logic, and token counting

Usage:
    from memori.integrations.anthropic_integration import MemoriAnthropic

    # Initialize with your memori instance
    client = MemoriAnthropic(memori_instance, api_key="your-key")

    # Use exactly like Anthropic client
    response = client.messages.create(...)
    
    # Streaming support
    with client.messages.stream(...) as stream:
        for text in stream.text_stream:
            print(text)
"""

import time
from typing import Optional, Generator, Any, Dict
from functools import wraps

from loguru import logger


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_errors: tuple = None
):
    """
    Decorator for retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        retryable_errors: Tuple of exception types to retry on
    """
    if retryable_errors is None:
        retryable_errors = (Exception,)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_errors as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}: {e}")
                        raise
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
            
            raise last_exception
        return wrapper
    return decorator


class TokenCounter:
    """
    Token counting utility for Anthropic models.
    Estimates token count based on character length (approximate).
    """
    
    # Average characters per token (varies by model)
    CHARS_PER_TOKEN = 4.0
    
    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """Estimate token count from text."""
        if not text:
            return 0
        return int(len(text) / cls.CHARS_PER_TOKEN)
    
    @classmethod
    def count_message_tokens(cls, messages: list) -> int:
        """Count tokens in a list of messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total += cls.estimate_tokens(block.get("text", ""))
            else:
                total += cls.estimate_tokens(str(content))
        return total


class MemoriAnthropic:
    """
    Clean Anthropic wrapper that automatically records conversations
    without monkey-patching. Drop-in replacement for Anthropic client.
    """

    def __init__(self, memori_instance, api_key: str | None = None, **kwargs):
        """
        Initialize MemoriAnthropic wrapper

        Args:
            memori_instance: Memori instance for recording conversations
            api_key: Anthropic API key
            **kwargs: Additional arguments passed to Anthropic client
        """
        try:
            import anthropic

            self._anthropic = anthropic.Anthropic(api_key=api_key, **kwargs)
            self._memori = memori_instance

            # Create wrapped messages
            self.messages = self._create_messages_wrapper()

            # Pass through other attributes
            for attr in dir(self._anthropic):
                if not attr.startswith("_") and attr not in ["messages"]:
                    setattr(self, attr, getattr(self._anthropic, attr))

        except ImportError as err:
            raise ImportError(
                "Anthropic package required: pip install anthropic"
            ) from err

    def _create_messages_wrapper(self):
        """Create wrapped messages"""

        class MessagesWrapper:
            def __init__(self, anthropic_client, memori_instance):
                self._anthropic = anthropic_client
                self._memori = memori_instance

            def create(self, **kwargs):
                # Inject context if conscious ingestion is enabled
                if self._memori.is_enabled and self._memori.conscious_ingest:
                    kwargs = self._inject_context(kwargs)

                # Make the actual API call
                response = self._anthropic.messages.create(**kwargs)

                # Record conversation if memori is enabled
                if self._memori.is_enabled:
                    self._record_conversation(kwargs, response)

                return response

            def _inject_context(self, kwargs):
                """Inject relevant context into messages"""
                try:
                    # Extract user input from messages
                    user_input = ""
                    for msg in reversed(kwargs.get("messages", [])):
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                # Handle content blocks
                                user_input = " ".join(
                                    [
                                        block.get("text", "")
                                        for block in content
                                        if isinstance(block, dict)
                                        and block.get("type") == "text"
                                    ]
                                )
                            else:
                                user_input = content
                            break

                    if user_input:
                        # Fetch relevant context
                        context = self._memori.retrieve_context(user_input, limit=3)

                        if context:
                            # Create a context prompt
                            context_prompt = "--- Relevant Memories ---\n"
                            for mem in context:
                                if isinstance(mem, dict):
                                    summary = mem.get("summary", "") or mem.get(
                                        "content", ""
                                    )
                                    context_prompt += f"- {summary}\n"
                                else:
                                    context_prompt += f"- {str(mem)}\n"
                            context_prompt += "-------------------------\n"

                            # Inject context into the system parameter
                            if kwargs.get("system"):
                                # Prepend to existing system message
                                kwargs["system"] = context_prompt + kwargs["system"]
                            else:
                                # Add as system message
                                kwargs["system"] = context_prompt

                            logger.debug(f"Injected context: {len(context)} memories")
                except Exception as e:
                    logger.error(f"Context injection failed: {e}")

                return kwargs

            def _record_conversation(self, kwargs, response):
                """Record the conversation"""
                try:
                    # Extract details
                    messages = kwargs.get("messages", [])
                    model = kwargs.get("model", "claude-unknown")

                    # Find user input (last user message)
                    user_input = ""
                    for message in reversed(messages):
                        if message.get("role") == "user":
                            content = message.get("content", "")
                            if isinstance(content, list):
                                # Handle content blocks
                                user_input = " ".join(
                                    [
                                        block.get("text", "")
                                        for block in content
                                        if isinstance(block, dict)
                                        and block.get("type") == "text"
                                    ]
                                )
                            else:
                                user_input = content
                            break

                    # Extract AI response
                    ai_output = ""
                    if hasattr(response, "content") and response.content:
                        if isinstance(response.content, list):
                            # Handle content blocks
                            ai_output = " ".join(
                                [
                                    block.text
                                    for block in response.content
                                    if hasattr(block, "text")
                                ]
                            )
                        else:
                            ai_output = str(response.content)

                    # Calculate tokens used
                    tokens_used = 0
                    if hasattr(response, "usage") and response.usage:
                        input_tokens = getattr(response.usage, "input_tokens", 0)
                        output_tokens = getattr(response.usage, "output_tokens", 0)
                        tokens_used = input_tokens + output_tokens

                    # Record conversation
                    self._memori.record_conversation(
                        user_input=user_input,
                        ai_output=ai_output,
                        model=model,
                        metadata={
                            "integration": "anthropic_wrapper",
                            "api_type": "messages",
                            "tokens_used": tokens_used,
                            "auto_recorded": True,
                        },
                    )
                except Exception as e:
                    logger.error(f"Failed to record Anthropic conversation: {e}")
            
            def stream(self, **kwargs):
                """
                Stream messages from Anthropic API.
                
                Yields text chunks as they arrive and records the full
                conversation when streaming completes.
                
                Usage:
                    with client.messages.stream(...) as stream:
                        for text in stream.text_stream:
                            print(text)
                
                Returns:
                    StreamingContext manager with text_stream generator
                """
                # Inject context if conscious ingestion is enabled
                if self._memori.is_enabled and self._memori.conscious_ingest:
                    kwargs = self._inject_context(kwargs)
                
                return StreamingContext(
                    self._anthropic,
                    self._memori,
                    kwargs
                )
            
            @retry_with_backoff(max_retries=3, initial_delay=1.0)
            def create_with_retry(self, **kwargs):
                """
                Create message with automatic retry on transient failures.
                
                Uses exponential backoff for rate limits and temporary errors.
                """
                return self.create(**kwargs)

        return MessagesWrapper(self._anthropic, self._memori)


class StreamingContext:
    """
    Context manager for streaming Anthropic responses.
    
    Collects the full response for recording while yielding chunks.
    """
    
    def __init__(self, anthropic_client, memori_instance, kwargs: Dict[str, Any]):
        self._anthropic = anthropic_client
        self._memori = memori_instance
        self._kwargs = kwargs
        self._stream = None
        self._collected_text = []
        self._input_tokens = 0
        self._output_tokens = 0
    
    def __enter__(self):
        """Start the stream."""
        # Ensure stream=True is set
        self._kwargs["stream"] = True
        self._stream = self._anthropic.messages.create(**self._kwargs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up and record the conversation."""
        if exc_type is None and self._memori.is_enabled:
            self._record_streamed_conversation()
        return False
    
    @property
    def text_stream(self) -> Generator[str, None, None]:
        """
        Generator that yields text chunks from the stream.
        
        Yields:
            Text chunks as they arrive from the API
        """
        if self._stream is None:
            return
        
        for event in self._stream:
            # Handle different event types
            if hasattr(event, "type"):
                if event.type == "content_block_delta":
                    if hasattr(event, "delta") and hasattr(event.delta, "text"):
                        text = event.delta.text
                        self._collected_text.append(text)
                        yield text
                elif event.type == "message_delta":
                    if hasattr(event, "usage"):
                        self._output_tokens = getattr(event.usage, "output_tokens", 0)
                elif event.type == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        self._input_tokens = getattr(event.message.usage, "input_tokens", 0)
    
    def get_full_text(self) -> str:
        """Get the full collected text from streaming."""
        return "".join(self._collected_text)
    
    def _record_streamed_conversation(self):
        """Record the streamed conversation to Memori."""
        try:
            messages = self._kwargs.get("messages", [])
            model = self._kwargs.get("model", "claude-unknown")
            
            # Find user input
            user_input = ""
            for message in reversed(messages):
                if message.get("role") == "user":
                    content = message.get("content", "")
                    if isinstance(content, list):
                        user_input = " ".join(
                            [
                                block.get("text", "")
                                for block in content
                                if isinstance(block, dict) and block.get("type") == "text"
                            ]
                        )
                    else:
                        user_input = content
                    break
            
            # Get collected response
            ai_output = self.get_full_text()
            tokens_used = self._input_tokens + self._output_tokens
            
            # Record conversation
            self._memori.record_conversation(
                user_input=user_input,
                ai_output=ai_output,
                model=model,
                metadata={
                    "integration": "anthropic_wrapper",
                    "api_type": "messages_stream",
                    "tokens_used": tokens_used,
                    "input_tokens": self._input_tokens,
                    "output_tokens": self._output_tokens,
                    "auto_recorded": True,
                },
            )
            logger.debug(f"Recorded streamed conversation: {tokens_used} tokens")
        except Exception as e:
            logger.error(f"Failed to record streamed conversation: {e}")
