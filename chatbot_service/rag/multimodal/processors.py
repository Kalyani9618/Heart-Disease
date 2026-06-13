"""
Multimodal Processors for Cardio AI RAG System

Adapted from RAG-Anything for medical document processing with:
- Table extraction and understanding
- Image/diagram analysis
- Medical equation processing
- Integration with existing VectorStore

Uses LLMGateway instead of direct LightRAG dependencies.
"""


import os
import json
import hashlib
import base64
import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable

from .config import MultimodalConfig, ContextConfig
from .prompts import MEDICAL_PROMPTS


class DocStatus(Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ContentType(Enum):
    """Types of multimodal content"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    EQUATION = "equation"


@dataclass
class ParsedContent:
    """Represents parsed content from a document"""
    content_type: ContentType
    raw_content: str
    processed_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    position: int = 0  # Position in document
    page_number: int = 1
    confidence: float = 1.0
    

@dataclass 
class ProcessedDocument:
    """Complete processed document with all content types"""
    doc_id: str
    file_path: str
    status: DocStatus
    text_chunks: List[ParsedContent] = field(default_factory=list)
    tables: List[ParsedContent] = field(default_factory=list)
    images: List[ParsedContent] = field(default_factory=list)
    equations: List[ParsedContent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ContextExtractor:
    """
    Extracts surrounding context for multimodal items.
    Helps provide relevant text context when processing tables/images.
    """
    
    def __init__(
        self,
        config: Optional[ContextConfig] = None,
        tokenizer: Optional[Callable] = None
    ):
        self.config = config or ContextConfig()
        self.tokenizer = tokenizer or self._simple_tokenizer
        
    def _simple_tokenizer(self, text: str) -> List[str]:
        """Simple word-based tokenizer fallback"""
        return text.split()
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer(text))
    
    def extract_context(
        self,
        content_list: List[Dict[str, Any]],
        target_index: int,
        content_type: str
    ) -> str:
        """
        Extract surrounding context for a multimodal item.
        
        Args:
            content_list: List of all content items in document
            target_index: Index of the target multimodal item
            content_type: Type of content (table, image, equation)
            
        Returns:
            str: Extracted context text
        """
        context_parts = []
        
        # Look for preceding context
        if self.config.context_mode in ("before", "both"):
            before_context = self._extract_before(content_list, target_index)
            if before_context:
                context_parts.append(f"[Preceding context]\n{before_context}")
        
        # Look for following context
        if self.config.context_mode in ("after", "both"):
            after_context = self._extract_after(content_list, target_index)
            if after_context:
                context_parts.append(f"[Following context]\n{after_context}")
        
        # Include headers if enabled
        if self.config.include_headers:
            headers = self._extract_headers(content_list, target_index)
            if headers:
                context_parts.insert(0, f"[Section headers]\n{headers}")
        
        combined = "\n\n".join(context_parts)
        
        # Truncate if exceeds max tokens
        if self._count_tokens(combined) > self.config.max_context_tokens:
            combined = self._truncate_to_tokens(combined, self.config.max_context_tokens)
        
        return combined
    
    def _extract_before(
        self,
        content_list: List[Dict[str, Any]],
        target_index: int
    ) -> str:
        """Extract text content before the target item"""
        texts = []
        window = self.config.context_window
        
        for i in range(max(0, target_index - window), target_index):
            item = content_list[i]
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        
        return " ".join(texts).strip()
    
    def _extract_after(
        self,
        content_list: List[Dict[str, Any]],
        target_index: int
    ) -> str:
        """Extract text content after the target item"""
        texts = []
        window = self.config.context_window
        
        for i in range(target_index + 1, min(len(content_list), target_index + window + 1)):
            item = content_list[i]
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        
        return " ".join(texts).strip()
    
    def _extract_headers(
        self,
        content_list: List[Dict[str, Any]],
        target_index: int
    ) -> str:
        """Extract section headers before the target item"""
        headers = []
        
        for i in range(target_index):
            item = content_list[i]
            if isinstance(item, dict):
                text = item.get("text", "")
                # Simple header detection (starts with #, is short, or is all caps)
                if text and (
                    text.startswith("#") or 
                    (len(text) < 100 and text.isupper()) or
                    text.startswith("Section") or
                    text.startswith("Chapter")
                ):
                    headers.append(text.strip())
        
        return " > ".join(headers[-3:]) if headers else ""  # Last 3 headers
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        tokens = self.tokenizer(text)
        if len(tokens) <= max_tokens:
            return text
        return " ".join(tokens[:max_tokens]) + "..."


class BaseModalProcessor(ABC):
    """
    Base class for multimodal content processors.
    Subclasses implement specific processing for tables, images, equations.
    """
    
    def __init__(
        self,
        llm_func: Optional[Callable] = None,
        context_extractor: Optional[ContextExtractor] = None,
        config: Optional[MultimodalConfig] = None
    ):
        self.llm_func = llm_func
        self.context_extractor = context_extractor or ContextExtractor()
        self.config = config or MultimodalConfig()
    
    @abstractmethod
    async def process(
        self,
        content: Dict[str, Any],
        context: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParsedContent:
        """Process a single multimodal content item"""
        pass
    
    @abstractmethod
    def get_content_type(self) -> ContentType:
        """Return the content type this processor handles"""
        pass
    
    def _build_prompt(self, template_key: str, **kwargs) -> str:
        """Build prompt from medical prompts template"""
        template = MEDICAL_PROMPTS.get(template_key, {}).get("system", "")
        return template.format(**kwargs) if kwargs else template


class TableProcessor(BaseModalProcessor):
    """
    Processor for extracting and understanding tables.
    Specialized for medical data tables (lab results, medications, vitals).
    """
    
    def get_content_type(self) -> ContentType:
        return ContentType.TABLE
    
    async def process(
        self,
        content: Dict[str, Any],
        context: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParsedContent:
        """
        Process a table and extract structured information.
        
        Args:
            content: Raw table content (table_body, table_caption, etc.)
            context: Surrounding text context
            metadata: Additional metadata
            
        Returns:
            ParsedContent with processed table information
        """
        table_body = content.get("table_body", "")
        table_caption = content.get("table_caption", "")
        table_footnote = content.get("table_footnote", "")
        
        # Build the table analysis prompt
        prompt = self._build_table_prompt(
            table_body=table_body,
            caption=table_caption,
            footnote=table_footnote,
            context=context
        )
        
        # Get LLM analysis if available
        processed_text = table_body
        if self.llm_func:
            try:
                analysis = await self._analyze_with_llm(prompt)
                processed_text = f"{table_caption}\n\n{table_body}\n\n[Analysis]\n{analysis}"
            except Exception as e:
                processed_text = f"{table_caption}\n\n{table_body}\n\n[Analysis Error: {str(e)}]"
        
        return ParsedContent(
            content_type=ContentType.TABLE,
            raw_content=table_body,
            processed_content=processed_text,
            metadata={
                "caption": table_caption,
                "footnote": table_footnote,
                "context": context,
                **(metadata or {})
            },
            confidence=0.9 if self.llm_func else 0.7
        )
    
    def _build_table_prompt(
        self,
        table_body: str,
        caption: str = "",
        footnote: str = "",
        context: str = ""
    ) -> str:
        """Build prompt for table analysis"""
        prompt_parts = [MEDICAL_PROMPTS["LAB_RESULTS_TABLE"]["system"]]
        
        if context:
            prompt_parts.append(f"\n\nDocument Context:\n{context}")
        
        if caption:
            prompt_parts.append(f"\n\nTable Caption: {caption}")
        
        prompt_parts.append(f"\n\nTable Content:\n{table_body}")
        
        if footnote:
            prompt_parts.append(f"\n\nFootnote: {footnote}")
        
        prompt_parts.append("\n\nProvide a structured analysis of this medical table.")
        
        return "".join(prompt_parts)
    
    async def _analyze_with_llm(self, prompt: str) -> str:
        """Analyze table using LLM"""
        if asyncio.iscoroutinefunction(self.llm_func):
            return await self.llm_func(prompt)
        return self.llm_func(prompt)


class ImageProcessor(BaseModalProcessor):
    """
    Processor for medical images and diagrams.
    Uses vision models for analysis.
    """
    
    def __init__(
        self,
        llm_func: Optional[Callable] = None,
        vision_func: Optional[Callable] = None,
        context_extractor: Optional[ContextExtractor] = None,
        config: Optional[MultimodalConfig] = None
    ):
        super().__init__(llm_func, context_extractor, config)
        self.vision_func = vision_func or llm_func
    
    def get_content_type(self) -> ContentType:
        return ContentType.IMAGE
    
    async def process(
        self,
        content: Dict[str, Any],
        context: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParsedContent:
        """
        Process an image and extract information.
        
        Args:
            content: Image content (img_path, img_caption, etc.)
            context: Surrounding text context
            metadata: Additional metadata
            
        Returns:
            ParsedContent with processed image information
        """
        img_path = content.get("img_path", "")
        img_caption = content.get("img_caption", "")
        
        # Try to analyze the image
        description = img_caption or "Image without caption"
        
        if self.vision_func and img_path and os.path.exists(img_path):
            try:
                # Read and encode image
                image_data = self._load_image(img_path)
                if image_data:
                    prompt = self._build_image_prompt(context, img_caption)
                    description = await self._analyze_image(image_data, prompt)
            except Exception as e:
                description = f"{img_caption}\n[Vision Analysis Error: {str(e)}]"
        
        return ParsedContent(
            content_type=ContentType.IMAGE,
            raw_content=img_path,
            processed_content=description,
            metadata={
                "path": img_path,
                "caption": img_caption,
                "context": context,
                **(metadata or {})
            },
            confidence=0.85 if self.vision_func else 0.5
        )
    
    def _load_image(self, path: str) -> Optional[str]:
        """Load and encode image as base64"""
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None
    
    def _build_image_prompt(self, context: str, caption: str) -> str:
        """Build prompt for image analysis"""
        prompt_parts = [MEDICAL_PROMPTS["MEDICAL_IMAGE"]["system"]]
        
        if context:
            prompt_parts.append(f"\n\nDocument Context:\n{context}")
        
        if caption:
            prompt_parts.append(f"\n\nImage Caption: {caption}")
        
        prompt_parts.append("\n\nDescribe this medical image in detail.")
        
        return "".join(prompt_parts)
    
    async def _analyze_image(self, image_data: str, prompt: str) -> str:
        """Analyze image using vision model"""
        if asyncio.iscoroutinefunction(self.vision_func):
            return await self.vision_func(prompt, image_data=image_data)
        return self.vision_func(prompt, image_data=image_data)


class EquationProcessor(BaseModalProcessor):
    """
    Processor for medical equations and formulas.
    Handles LaTeX, mathematical notation, and medical formulas.
    """
    
    def get_content_type(self) -> ContentType:
        return ContentType.EQUATION
    
    async def process(
        self,
        content: Dict[str, Any],
        context: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParsedContent:
        """
        Process an equation and extract information.
        
        Args:
            content: Equation content (text, latex, etc.)
            context: Surrounding text context
            metadata: Additional metadata
            
        Returns:
            ParsedContent with processed equation information
        """
        equation_text = content.get("text", content.get("latex", ""))
        
        # Build explanation
        explanation = equation_text
        
        if self.llm_func:
            try:
                prompt = self._build_equation_prompt(equation_text, context)
                explanation = await self._explain_equation(prompt)
            except Exception as e:
                explanation = f"{equation_text}\n[Equation Explanation Error: {str(e)}]"
        
        return ParsedContent(
            content_type=ContentType.EQUATION,
            raw_content=equation_text,
            processed_content=explanation,
            metadata={
                "latex": content.get("latex", ""),
                "context": context,
                **(metadata or {})
            },
            confidence=0.8 if self.llm_func else 0.6
        )
    
    def _build_equation_prompt(self, equation: str, context: str) -> str:
        """Build prompt for equation explanation"""
        prompt_parts = [MEDICAL_PROMPTS.get("EQUATION_EXPLANATION", {}).get(
            "system", 
            "You are a medical expert. Explain the following medical equation or formula."
        )]
        
        if context:
            prompt_parts.append(f"\n\nContext:\n{context}")
        
        prompt_parts.append(f"\n\nEquation:\n{equation}")
        prompt_parts.append("\n\nExplain this equation's medical significance and how it's used clinically.")
        
        return "".join(prompt_parts)
    
    async def _explain_equation(self, prompt: str) -> str:
        """Explain equation using LLM"""
        if asyncio.iscoroutinefunction(self.llm_func):
            return await self.llm_func(prompt)
        return self.llm_func(prompt)


class DocumentParser:
    """
    Simple document parser for multimodal content extraction.
    Parses documents and extracts text, tables, images, and equations.
    """
    
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".html", ".md"}
    
    def __init__(self, config: Optional[MultimodalConfig] = None):
        self.config = config or MultimodalConfig()
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
    
    def _generate_doc_id(self, file_path: str) -> str:
        """Generate unique document ID based on file path and content hash"""
        path = Path(file_path)
        if path.exists():
            stat = path.stat()
            content = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        else:
            content = file_path
        return f"doc-{hashlib.md5(content.encode()).hexdigest()[:16]}"
    
    def _generate_cache_key(self, file_path: str) -> str:
        """Generate cache key for parsed content"""
        path = Path(file_path)
        if path.exists():
            mtime = path.stat().st_mtime
            return f"{file_path}:{mtime}"
        return file_path
    
    def is_supported(self, file_path: str) -> bool:
        """Check if file type is supported"""
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS
    
    async def parse(self, file_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Parse a document and extract content.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Tuple of (doc_id, content_list)
        """
        cache_key = self._generate_cache_key(file_path)
        
        # Check cache
        if cache_key in self._cache:
            doc_id = self._generate_doc_id(file_path)
            return doc_id, self._cache[cache_key]
        
        # Parse document based on type
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if ext == ".md":
            content_list = await self._parse_markdown(file_path)
        elif ext == ".html":
            content_list = await self._parse_html(file_path)
        elif ext == ".pdf":
            content_list = await self._parse_pdf_simple(file_path)
        else:
            # Default: read as text
            content_list = await self._parse_text(file_path)
        
        # Cache result
        self._cache[cache_key] = content_list
        doc_id = self._generate_doc_id(file_path)
        
        return doc_id, content_list
    
    async def _parse_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse plain text file"""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        return [{"type": "text", "text": text}]
    
    async def _parse_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse markdown file and extract tables"""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        result = []
        lines = content.split("\n")
        current_text = []
        in_table = False
        table_lines = []
        
        for line in lines:
            # Detect table rows (starts with |)
            if line.strip().startswith("|") and "|" in line[1:]:
                if not in_table:
                    # Save accumulated text
                    if current_text:
                        result.append({"type": "text", "text": "\n".join(current_text)})
                        current_text = []
                    in_table = True
                table_lines.append(line)
            else:
                if in_table:
                    # End of table
                    result.append({
                        "type": "table",
                        "table_body": "\n".join(table_lines),
                        "table_caption": ""
                    })
                    table_lines = []
                    in_table = False
                current_text.append(line)
        
        # Handle remaining content
        if in_table and table_lines:
            result.append({
                "type": "table",
                "table_body": "\n".join(table_lines),
                "table_caption": ""
            })
        if current_text:
            result.append({"type": "text", "text": "\n".join(current_text)})
        
        return result
    
    async def _parse_html(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse HTML file (basic extraction)"""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        # Very basic HTML text extraction
        import re
        # Remove script and style tags
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', content)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return [{"type": "text", "text": text}]
    
    async def _parse_pdf_simple(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Simple PDF parsing fallback.
        For full PDF support, use external parser integration.
        """
        try:
            # Try pypdf if available
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            return [{"type": "text", "text": "\n\n".join(text_parts)}]
        except ImportError:
            # pypdf not available
            return [{
                "type": "text",
                "text": f"[PDF content from: {file_path}]\n[Install pypdf for PDF text extraction]"
            }]
        except Exception as e:
            return [{
                "type": "text",
                "text": f"[Error parsing PDF: {str(e)}]"
            }]


def get_processor_for_type(
    content_type: str,
    llm_func: Optional[Callable] = None,
    vision_func: Optional[Callable] = None,
    context_extractor: Optional[ContextExtractor] = None,
    config: Optional[MultimodalConfig] = None
) -> BaseModalProcessor:
    """
    Factory function to get the appropriate processor for a content type.
    
    Args:
        content_type: Type of content (table, image, equation)
        llm_func: LLM function for text analysis
        vision_func: Vision function for image analysis
        context_extractor: Context extractor instance
        config: Multimodal configuration
        
    Returns:
        Appropriate processor instance
    """
    processors = {
        "table": lambda: TableProcessor(llm_func, context_extractor, config),
        "image": lambda: ImageProcessor(llm_func, vision_func, context_extractor, config),
        "equation": lambda: EquationProcessor(llm_func, context_extractor, config),
    }
    
    factory = processors.get(content_type.lower())
    if factory:
        return factory()
    
    raise ValueError(f"Unknown content type: {content_type}")
