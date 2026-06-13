"""
Multimodal Query Mixin for Cardio AI RAG System

Provides query functionality for multimodal content:
- Text queries with RAG context
- Multimodal queries with images, tables, equations
- VLM-enhanced queries that analyze images in context
- Query caching for performance

Adapted from RAG-Anything query.py.
"""


import os
import re
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple

from .config import MultimodalConfig
from .utils import (
    generate_multimodal_cache_key,
    extract_image_paths_from_context,
    build_vlm_messages,
    encode_image_to_base64,
    get_processor_for_type,
    RobustJSONParser,
)
from .prompts import MEDICAL_PROMPTS

logger = logging.getLogger(__name__)


class MultimodalQueryMixin:
    """
    Mixin class providing multimodal query functionality.
    
    Designed to be mixed into service classes that have:
    - self.llm_func: LLM function for text generation
    - self.vision_func: Vision function for image analysis
    - self.retriever: Retriever for context fetching
    - self.config: MultimodalConfig instance
    - self.processors: Dict of modal processors
    
    Provides:
    - aquery(): Pure text query
    - aquery_with_multimodal(): Query with multimodal content
    - aquery_vlm_enhanced(): VLM-enhanced query with image analysis
    """
    
    # Required attributes (should be set by the class using this mixin)
    llm_func: Optional[Callable] = None
    vision_func: Optional[Callable] = None
    retriever: Optional[Any] = None
    config: Optional[MultimodalConfig] = None
    processors: Optional[Dict[str, Any]] = None
    
    # Optional: cache for query results
    _query_cache: Dict[str, str] = {}
    _cache_enabled: bool = True
    _max_cache_size: int = 100
    
    async def aquery(
        self,
        query: str,
        mode: str = "hybrid",
        system_prompt: Optional[str] = None,
        top_k: int = 5,
        vlm_enhanced: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        Execute a text query with optional VLM enhancement.
        
        Args:
            query: User query text
            mode: Query mode (hybrid, local, global, naive)
            system_prompt: Optional system prompt
            top_k: Number of context chunks to retrieve
            vlm_enhanced: Whether to analyze images with VLM (auto-detected if None)
            **kwargs: Additional query parameters
            
        Returns:
            str: Query result
        """
        logger.info(f"Executing query: {query[:100]}...")
        
        # Auto-determine VLM enhancement based on availability
        if vlm_enhanced is None:
            vlm_enhanced = self.vision_func is not None
        
        # Use VLM enhanced query if enabled and available
        if vlm_enhanced and self.vision_func:
            return await self.aquery_vlm_enhanced(
                query, mode=mode, system_prompt=system_prompt, top_k=top_k, **kwargs
            )
        elif vlm_enhanced and not self.vision_func:
            logger.warning(
                "VLM enhanced query requested but vision_func is not available, "
                "falling back to normal query"
            )
        
        # Standard retrieval-augmented query
        context = await self._retrieve_context(query, top_k, mode)
        
        # Generate answer
        answer = await self._generate_answer(query, context, system_prompt)
        
        logger.info("Query completed")
        return answer
    
    async def aquery_with_multimodal(
        self,
        query: str,
        multimodal_content: Optional[List[Dict[str, Any]]] = None,
        mode: str = "hybrid",
        top_k: int = 5,
        **kwargs
    ) -> str:
        """
        Execute a query with multimodal content (images, tables, equations).
        
        Args:
            query: Base query text
            multimodal_content: List of multimodal content, each containing:
                - type: Content type ("image", "table", "equation")
                - img_path/image_path: Path to image file (for images)
                - table_data: Table content as string (for tables)
                - latex: LaTeX equation (for equations)
            mode: Query mode
            top_k: Number of context chunks to retrieve
            **kwargs: Additional parameters
            
        Returns:
            str: Query result
            
        Example:
            result = await service.aquery_with_multimodal(
                "What does this ECG show?",
                multimodal_content=[{
                    "type": "image",
                    "img_path": "./ecg_reading.jpg"
                }]
            )
        """
        logger.info(f"Executing multimodal query: {query[:100]}...")
        
        # If no multimodal content, fall back to text query
        if not multimodal_content:
            logger.info("No multimodal content provided, executing text query")
            return await self.aquery(query, mode=mode, top_k=top_k, **kwargs)
        
        # Check cache
        cache_key = generate_multimodal_cache_key(query, multimodal_content, mode, **kwargs)
        if self._cache_enabled and cache_key in self._query_cache:
            logger.info("Returning cached multimodal query result")
            return self._query_cache[cache_key]
        
        # Process multimodal content to generate enhanced query
        enhanced_query = await self._process_multimodal_content(query, multimodal_content)
        
        # Retrieve context using enhanced query
        context = await self._retrieve_context(enhanced_query, top_k, mode)
        
        # Combine everything for final answer
        full_context = f"{enhanced_query}\n\n{context}"
        answer = await self._generate_answer(query, full_context)
        
        # Cache result
        if self._cache_enabled:
            self._cache_result(cache_key, answer)
        
        logger.info("Multimodal query completed")
        return answer
    
    async def aquery_vlm_enhanced(
        self,
        query: str,
        mode: str = "hybrid",
        system_prompt: Optional[str] = None,
        top_k: int = 5,
        **kwargs
    ) -> str:
        """
        VLM-enhanced query that analyzes images found in retrieved context.
        
        This method:
        1. Retrieves relevant context
        2. Detects image paths in the context
        3. Encodes images as base64
        4. Sends combined text + images to VLM for analysis
        
        Args:
            query: User query
            mode: Query mode
            system_prompt: Optional system prompt
            top_k: Number of context chunks
            **kwargs: Additional parameters
            
        Returns:
            str: VLM-generated answer
        """
        if not self.vision_func:
            raise ValueError(
                "VLM enhanced query requires vision_func. "
                "Please provide a vision model function."
            )
        
        logger.info(f"Executing VLM enhanced query: {query[:100]}...")
        
        # Retrieve context
        context = await self._retrieve_context(query, top_k, mode)
        
        # Extract and process image paths
        image_paths, enhanced_context = extract_image_paths_from_context(context)
        
        if not image_paths:
            logger.info("No valid images found in context, falling back to text query")
            return await self._generate_answer(query, context, system_prompt)
        
        logger.info(f"Found {len(image_paths)} images to analyze")
        
        # Encode images to base64
        images_base64 = []
        for path in image_paths[:5]:  # Limit to 5 images
            encoded = encode_image_to_base64(path)
            if encoded:
                images_base64.append(encoded)
        
        if not images_base64:
            logger.warning("Failed to encode any images, falling back to text query")
            return await self._generate_answer(query, context, system_prompt)
        
        # Build VLM messages
        messages = build_vlm_messages(
            enhanced_context, query, images_base64, system_prompt
        )
        
        # Call VLM
        result = await self._call_vlm(messages)
        
        logger.info("VLM enhanced query completed")
        return result
    
    async def _retrieve_context(
        self,
        query: str,
        top_k: int,
        mode: str
    ) -> str:
        """Retrieve relevant context for query"""
        if not self.retriever:
            logger.warning("No retriever available, returning empty context")
            return ""
        
        try:
            if hasattr(self.retriever, 'retrieve'):
                if asyncio.iscoroutinefunction(self.retriever.retrieve):
                    results = await self.retriever.retrieve(query, top_k)
                else:
                    results = self.retriever.retrieve(query, top_k)
            else:
                return ""
            
            # Format results into context string
            context_parts = []
            for i, result in enumerate(results):
                if isinstance(result, dict):
                    text = result.get("text", result.get("content", ""))
                    chunk_type = result.get("chunk_type", "text")
                    
                    if chunk_type == "image":
                        image_path = result.get("metadata", {}).get("image_path", "")
                        context_parts.append(f"[Image content]\n{text}\nImage Path: {image_path}")
                    elif chunk_type == "table":
                        context_parts.append(f"[Table]\n{text}")
                    elif chunk_type == "equation":
                        context_parts.append(f"[Equation]\n{text}")
                    else:
                        context_parts.append(text)
                else:
                    context_parts.append(str(result))
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return ""
    
    async def _process_multimodal_content(
        self,
        base_query: str,
        multimodal_content: List[Dict[str, Any]]
    ) -> str:
        """Process multimodal content to generate enhanced query text"""
        enhanced_parts = [f"User query: {base_query}"]
        
        for i, content in enumerate(multimodal_content):
            content_type = content.get("type", "unknown")
            logger.info(f"Processing {i+1}/{len(multimodal_content)} multimodal content: {content_type}")
            
            try:
                description = await self._describe_content(content, content_type)
                enhanced_parts.append(f"\nRelated {content_type} content: {description}")
            except Exception as e:
                logger.error(f"Error processing multimodal content: {e}")
                continue
        
        enhanced_query = "\n".join(enhanced_parts)
        enhanced_query += "\n\nPlease analyze the above information and provide a comprehensive medical assessment."
        
        return enhanced_query
    
    async def _describe_content(
        self,
        content: Dict[str, Any],
        content_type: str
    ) -> str:
        """Generate description for multimodal content"""
        try:
            if content_type == "image":
                return await self._describe_image(content)
            elif content_type == "table":
                return await self._describe_table(content)
            elif content_type == "equation":
                return await self._describe_equation(content)
            else:
                return str(content)[:200]
        except Exception as e:
            logger.error(f"Error describing {content_type}: {e}")
            return f"{content_type} content (description failed)"
    
    async def _describe_image(self, content: Dict[str, Any]) -> str:
        """Generate image description using VLM if available"""
        image_path = content.get("img_path", content.get("image_path", ""))
        caption = content.get("image_caption", content.get("caption", ""))
        
        if self.vision_func and image_path and os.path.exists(image_path):
            try:
                encoded = encode_image_to_base64(image_path)
                if encoded:
                    prompt = MEDICAL_PROMPTS.get(
                        "IMAGE_ANALYSIS",
                        "Describe this medical image in detail, including any findings."
                    )
                    
                    messages = [
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
                        ]}
                    ]
                    
                    return await self._call_vlm(messages)
            except Exception as e:
                logger.warning(f"VLM image description failed: {e}")
        
        # Fallback to caption or path
        parts = []
        if caption:
            parts.append(f"Caption: {caption}")
        if image_path:
            parts.append(f"Image: {os.path.basename(image_path)}")
        
        return "; ".join(parts) if parts else "Image content"
    
    async def _describe_table(self, content: Dict[str, Any]) -> str:
        """Generate table description using LLM"""
        table_data = content.get("table_data", content.get("data", ""))
        caption = content.get("table_caption", content.get("caption", ""))
        
        if self.llm_func and table_data:
            try:
                prompt = f"""Analyze this medical table and summarize the key findings:

Table Caption: {caption or 'None provided'}

Table Data:
{table_data}

Provide a concise summary of the important values and their clinical significance."""

                if asyncio.iscoroutinefunction(self.llm_func):
                    return await self.llm_func(prompt)
                return self.llm_func(prompt)
            except Exception as e:
                logger.warning(f"Table description failed: {e}")
        
        # Fallback
        return f"Table: {caption}" if caption else f"Table data: {table_data[:200]}"
    
    async def _describe_equation(self, content: Dict[str, Any]) -> str:
        """Generate equation description using LLM"""
        latex = content.get("latex", content.get("equation", ""))
        caption = content.get("equation_caption", content.get("caption", ""))
        
        if self.llm_func and latex:
            try:
                prompt = f"""Explain this medical/scientific equation:

Equation: {latex}
Caption: {caption or 'None provided'}

Explain what this equation represents and its clinical relevance."""

                if asyncio.iscoroutinefunction(self.llm_func):
                    return await self.llm_func(prompt)
                return self.llm_func(prompt)
            except Exception as e:
                logger.warning(f"Equation description failed: {e}")
        
        # Fallback
        return f"Equation: {latex}" if latex else "Mathematical expression"
    
    async def _generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate answer using LLM"""
        if not self.llm_func:
            return "LLM function not available for answer generation."
        
        # Build prompt
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Please provide a comprehensive and accurate answer based on the context provided."""

        try:
            if asyncio.iscoroutinefunction(self.llm_func):
                return await self.llm_func(prompt, system_prompt=system_prompt)
            return self.llm_func(prompt, system_prompt=system_prompt)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}"
    
    async def _call_vlm(self, messages: List[Dict[str, Any]]) -> str:
        """Call vision-language model"""
        if not self.vision_func:
            raise ValueError("Vision function not available")
        
        try:
            if asyncio.iscoroutinefunction(self.vision_func):
                return await self.vision_func(messages)
            return self.vision_func(messages)
        except Exception as e:
            logger.error(f"VLM call failed: {e}")
            raise
    
    def _cache_result(self, key: str, result: str):
        """Cache query result with size limit"""
        if len(self._query_cache) >= self._max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        
        self._query_cache[key] = result
    
    def clear_cache(self):
        """Clear the query cache"""
        self._query_cache.clear()
        logger.info("Query cache cleared")
    
    # Synchronous versions
    def query(
        self,
        query: str,
        mode: str = "hybrid",
        **kwargs
    ) -> str:
        """Synchronous version of aquery"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.aquery(query, mode=mode, **kwargs))
    
    def query_with_multimodal(
        self,
        query: str,
        multimodal_content: Optional[List[Dict[str, Any]]] = None,
        mode: str = "hybrid",
        **kwargs
    ) -> str:
        """Synchronous version of aquery_with_multimodal"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.aquery_with_multimodal(query, multimodal_content, mode=mode, **kwargs)
        )


class MultimodalQueryService(MultimodalQueryMixin):
    """
    Standalone query service for multimodal RAG queries.
    
    Implements the MultimodalQueryMixin with all required dependencies.
    """
    
    def __init__(
        self,
        retriever: Optional[Any] = None,
        llm_func: Optional[Callable] = None,
        vision_func: Optional[Callable] = None,
        processors: Optional[Dict[str, Any]] = None,
        config: Optional[MultimodalConfig] = None,
        cache_enabled: bool = True
    ):
        """
        Initialize the query service.
        
        Args:
            retriever: Retriever for context fetching
            llm_func: LLM function for text generation
            vision_func: Vision function for image analysis
            processors: Dict of modal processors
            config: Multimodal configuration
            cache_enabled: Whether to cache query results
        """
        self.retriever = retriever
        self.llm_func = llm_func
        self.vision_func = vision_func
        self.processors = processors or {}
        self.config = config or MultimodalConfig()
        self._cache_enabled = cache_enabled
        self._query_cache = {}
        self._max_cache_size = 100
        
        logger.info(
            f"MultimodalQueryService initialized "
            f"(VLM: {'enabled' if vision_func else 'disabled'}, "
            f"cache: {'enabled' if cache_enabled else 'disabled'})"
        )
