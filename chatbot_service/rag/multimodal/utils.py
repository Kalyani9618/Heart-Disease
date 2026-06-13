"""
Multimodal Utilities for Cardio AI RAG System

Provides:
- Robust JSON parsing with fallback strategies
- Image encoding utilities
- VLM message building
- Cache key generation

Adapted from RAG-Anything utilities.
"""


import re
import os
import json
import hashlib
import base64
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# ROBUST JSON PARSING (from RAG-Anything modalprocessors.py)
# =============================================================================

class RobustJSONParser:
    """
    Robust JSON parser with multiple fallback strategies.
    Handles malformed LLM responses that may contain reasoning tags,
    improperly escaped quotes, or other common issues.
    """
    
    @staticmethod
    def parse(response: str) -> dict:
        """
        Parse JSON with multiple fallback strategies.
        
        Args:
            response: String that may contain JSON
            
        Returns:
            Parsed dictionary or fallback with extracted fields
        """
        parser = RobustJSONParser()
        
        # Strategy 1: Try direct parsing first
        for json_candidate in parser._extract_all_json_candidates(response):
            result = parser._try_parse_json(json_candidate)
            if result:
                return result

        # Strategy 2: Try with basic cleanup
        for json_candidate in parser._extract_all_json_candidates(response):
            cleaned = parser._basic_json_cleanup(json_candidate)
            result = parser._try_parse_json(cleaned)
            if result:
                return result

        # Strategy 3: Try progressive quote fixing
        for json_candidate in parser._extract_all_json_candidates(response):
            fixed = parser._progressive_quote_fix(json_candidate)
            result = parser._try_parse_json(fixed)
            if result:
                return result

        # Strategy 4: Fallback to regex field extraction
        return parser._extract_fields_with_regex(response)
    
    def _extract_all_json_candidates(self, response: str) -> list:
        """Extract all possible JSON candidates from response"""
        candidates = []

        # Pre-process: Remove thinking/reasoning tags that some models use
        # This handles models like qwen2.5-think, deepseek-r1 that wrap reasoning in tags
        cleaned_response = re.sub(
            r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE
        )
        cleaned_response = re.sub(
            r"<thinking>.*?</thinking>",
            "",
            cleaned_response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned_response = re.sub(
            r"<reasoning>.*?</reasoning>",
            "",
            cleaned_response,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Method 1: JSON in code blocks
        json_blocks = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_response, re.DOTALL
        )
        candidates.extend(json_blocks)

        # Method 2: Balanced braces extraction
        brace_count = 0
        start_pos = -1

        for i, char in enumerate(cleaned_response):
            if char == "{":
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    candidates.append(cleaned_response[start_pos : i + 1])

        # Method 3: Simple regex fallback
        simple_match = re.search(r"\{.*\}", cleaned_response, re.DOTALL)
        if simple_match:
            candidates.append(simple_match.group(0))

        return candidates

    def _try_parse_json(self, json_str: str) -> Optional[dict]:
        """Try to parse JSON string, return None if failed"""
        if not json_str or not json_str.strip():
            return None

        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return None

    def _basic_json_cleanup(self, json_str: str) -> str:
        """Basic cleanup for common JSON issues"""
        # Remove extra whitespace
        json_str = json_str.strip()

        # Fix common quote issues (smart quotes to regular quotes)
        json_str = json_str.replace('"', '"').replace('"', '"')
        json_str = json_str.replace(''', "'").replace(''', "'")

        # Fix trailing commas (common LLM mistake)
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

        return json_str

    def _progressive_quote_fix(self, json_str: str) -> str:
        """Progressive fixing of quote and escape issues"""
        # Only escape unescaped backslashes before quotes
        json_str = re.sub(r'(?<!\\)\\(?=")', r"\\\\", json_str)

        # Fix unescaped backslashes in string values (more conservative)
        def fix_string_content(match):
            content = match.group(1)
            # Only escape obvious problematic patterns
            content = re.sub(r"\\(?=[a-zA-Z])", r"\\\\", content)
            return f'"{content}"'

        json_str = re.sub(r'"([^"]*(?:\\.[^"]*)*)"', fix_string_content, json_str)
        return json_str

    def _extract_fields_with_regex(self, response: str) -> dict:
        """Extract required fields using regex as last resort"""
        logger.warning("Using regex fallback for JSON parsing")

        # Extract common fields with flexible patterns
        def extract_field(pattern: str, default: str = "") -> str:
            match = re.search(pattern, response, re.DOTALL)
            return match.group(1) if match else default

        # Medical/healthcare specific fields
        result = {
            "description": extract_field(
                r'"(?:detailed_description|description)":\s*"([^"]*(?:\\.[^"]*)*)"'
            ),
            "entity_name": extract_field(
                r'"entity_name":\s*"([^"]*(?:\\.[^"]*)*)"', "unknown_entity"
            ),
            "entity_type": extract_field(
                r'"entity_type":\s*"([^"]*(?:\\.[^"]*)*)"', "medical_content"
            ),
            "summary": extract_field(
                r'"summary":\s*"([^"]*(?:\\.[^"]*)*)"'
            ),
            "findings": extract_field(
                r'"findings":\s*"([^"]*(?:\\.[^"]*)*)"'
            ),
            "analysis": extract_field(
                r'"analysis":\s*"([^"]*(?:\\.[^"]*)*)"'
            ),
        }

        # If no summary, use description
        if not result["summary"] and result["description"]:
            result["summary"] = result["description"][:200]

        return result


# =============================================================================
# IMAGE UTILITIES
# =============================================================================

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"
}


def validate_image_file(image_path: str) -> bool:
    """
    Validate that a file exists and is a supported image format.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        bool: True if valid image file
    """
    if not image_path:
        return False
    
    path = Path(image_path)
    
    # Check extension
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return False
    
    # Check file exists
    if not path.exists():
        logger.debug(f"Image file not found: {image_path}")
        return False
    
    # Check file is not empty
    if path.stat().st_size == 0:
        logger.warning(f"Image file is empty: {image_path}")
        return False
    
    return True


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string or None if failed
    """
    try:
        if not validate_image_file(image_path):
            return None
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        return base64.b64encode(image_data).decode("utf-8")
    
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None


def get_image_mime_type(image_path: str) -> str:
    """Get MIME type for image file"""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    return mime_types.get(ext, "image/png")


# =============================================================================
# CACHE KEY GENERATION (from RAG-Anything query.py)
# =============================================================================

def generate_multimodal_cache_key(
    query: str,
    multimodal_content: List[Dict[str, Any]],
    mode: str,
    **kwargs
) -> str:
    """
    Generate cache key for multimodal query.
    
    Creates a stable hash that can be used to cache query results.
    
    Args:
        query: Base query text
        multimodal_content: List of multimodal content items
        mode: Query mode (hybrid, local, global, etc.)
        **kwargs: Additional query parameters
        
    Returns:
        str: Cache key hash
    """
    # Create a normalized representation of the query parameters
    cache_data = {
        "query": query.strip(),
        "mode": mode,
    }

    # Normalize multimodal content for stable caching
    normalized_content = []
    if multimodal_content:
        for item in multimodal_content:
            if isinstance(item, dict):
                normalized_item = {}
                for key, value in item.items():
                    # For file paths, use basename to make cache more portable
                    if key in ["img_path", "image_path", "file_path"] and isinstance(value, str):
                        normalized_item[key] = Path(value).name
                    # For large content, create a hash instead of storing directly
                    elif key in ["table_data", "table_body"] and isinstance(value, str) and len(value) > 200:
                        normalized_item[f"{key}_hash"] = hashlib.md5(value.encode()).hexdigest()
                    else:
                        normalized_item[key] = value
                normalized_content.append(normalized_item)
            else:
                normalized_content.append(item)

    cache_data["multimodal_content"] = normalized_content

    # Add relevant kwargs to cache data
    relevant_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in ["stream", "response_type", "top_k", "max_tokens", "temperature"]
    }
    cache_data.update(relevant_kwargs)

    # Generate hash from the cache data
    cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
    cache_hash = hashlib.md5(cache_str.encode()).hexdigest()

    return f"multimodal_query:{cache_hash}"


# =============================================================================
# VLM MESSAGE BUILDING (from RAG-Anything query.py)
# =============================================================================

def extract_image_paths_from_context(context: str) -> Tuple[List[str], str]:
    """
    Extract image paths from context and return processed context with markers.
    
    Args:
        context: Context text that may contain image path references
        
    Returns:
        Tuple of (list of image paths, context with VLM markers)
    """
    image_paths = []
    
    # Pattern to match image paths
    # Handles: "Image Path: /path/to/image.jpg" or "Image: /path/image.png"
    image_path_pattern = (
        r"(?:Image Path|Image|image_path):\s*([^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))"
    )
    
    matches = re.findall(image_path_pattern, context, re.IGNORECASE)
    
    enhanced_context = context
    image_counter = 0
    
    def replace_image_path(match):
        nonlocal image_counter
        image_path = match.group(1).strip()
        
        if validate_image_file(image_path):
            image_counter += 1
            image_paths.append(image_path)
            return f"Image Path: {image_path}\n[VLM_IMAGE_{image_counter}]"
        
        return match.group(0)  # Keep original if invalid
    
    enhanced_context = re.sub(
        image_path_pattern, replace_image_path, context, flags=re.IGNORECASE
    )
    
    return image_paths, enhanced_context


def build_vlm_messages(
    context: str,
    query: str,
    images_base64: List[str],
    system_prompt: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Build VLM-compatible message format with images.
    
    Creates message structure suitable for vision-language models like GPT-4V.
    
    Args:
        context: Text context (may contain VLM_IMAGE markers)
        query: User query
        images_base64: List of base64-encoded images
        system_prompt: Optional system prompt
        
    Returns:
        List of message dictionaries for VLM
    """
    messages = []
    
    # Add system message if provided
    base_system = "You are a helpful medical AI assistant that can analyze both text and image content to provide comprehensive, accurate answers about cardiovascular health."
    
    if system_prompt:
        full_system = f"{base_system}\n\n{system_prompt}"
    else:
        full_system = base_system
    
    messages.append({
        "role": "system",
        "content": full_system
    })
    
    # Build user message with multimodal content
    if not images_base64:
        # Pure text mode
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        })
    else:
        # Multimodal mode - interleave text and images
        content_parts = []
        
        # Split text at image markers and insert images
        text_parts = context.split("[VLM_IMAGE_")
        
        for i, text_part in enumerate(text_parts):
            if i == 0:
                # First text part
                if text_part.strip():
                    content_parts.append({"type": "text", "text": text_part})
            else:
                # Find marker number and insert corresponding image
                marker_match = re.match(r"(\d+)\](.*)", text_part, re.DOTALL)
                if marker_match:
                    image_num = int(marker_match.group(1)) - 1  # 0-based index
                    remaining_text = marker_match.group(2)
                    
                    # Insert image if available
                    if 0 <= image_num < len(images_base64):
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{images_base64[image_num]}"
                            }
                        })
                    
                    # Add remaining text
                    if remaining_text.strip():
                        content_parts.append({"type": "text", "text": remaining_text})
                else:
                    # No valid marker, just add as text
                    if text_part.strip():
                        content_parts.append({"type": "text", "text": text_part})
        
        # Add user question at the end
        content_parts.append({
            "type": "text",
            "text": f"\n\nQuestion: {query}\n\nPlease provide a comprehensive answer based on the context and any images provided."
        })
        
        messages.append({
            "role": "user",
            "content": content_parts
        })
    
    return messages


# =============================================================================
# PROCESSOR UTILITIES
# =============================================================================

def get_processor_for_type(
    processors: Dict[str, Any],
    content_type: str
) -> Optional[Any]:
    """
    Get the appropriate processor for a content type.
    
    Args:
        processors: Dictionary of available processors
        content_type: Type of content (image, table, equation, etc.)
        
    Returns:
        Processor instance or None
    """
    # Direct match
    if content_type in processors:
        return processors[content_type]
    
    # Fuzzy matching for common variations
    type_mapping = {
        "img": "image",
        "picture": "image",
        "photo": "image",
        "diagram": "image",
        "chart": "table",
        "graph": "table",
        "formula": "equation",
        "math": "equation",
        "latex": "equation",
    }
    
    mapped_type = type_mapping.get(content_type.lower())
    if mapped_type and mapped_type in processors:
        return processors[mapped_type]
    
    # Fall back to generic processor
    return processors.get("generic")


def compute_content_hash(content: str, prefix: str = "chunk-") -> str:
    """
    Compute MD5 hash for content with prefix.
    
    Args:
        content: Content to hash
        prefix: Prefix for the hash ID
        
    Returns:
        str: Prefixed hash ID
    """
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return f"{prefix}{content_hash[:16]}"
