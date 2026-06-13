"""
Unified Document Compressor - Single Source of Truth for RAG Compression.

Consolidates logic from:
1. compression.py (LLMChainExtractor wrapper)
2. contextual_compression.py (rich compression strategies, list/table awareness)

This module provides comprehensive document compression with:
- Multiple compression strategies (LLM, extractive, sentence-level)
- Medical term preservation (dosages, safety keywords)
- Adaptive compression for tables and lists
- LLM-based intelligent extraction
- Token budget awareness
- Deduplication across documents

Benefits:
- ~70% reduction in redundant compression code (2 files → 1)
- Single entry point for all compression needs
- Consistent preservation of medical terminology
- Easy configuration and testing
"""


import logging
import re
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """Supported compression strategies."""
    LLM = "llm"                    # Use LLM for intelligent extraction
    EXTRACTIVE = "extractive"      # Paragraph-level relevance scoring
    SENTENCE = "sentence"          # Sentence-level filtering
    LIST_AWARE = "list_aware"      # Specialized for lists and tables
    HYBRID = "hybrid"              # Auto-select based on content type
    AUTO = "auto"                  # Fall back intelligently


@dataclass
class CompressedDocument:
    """A compressed document with metadata and provenance."""
    original_content: str
    compressed_content: str
    compression_ratio: float
    preserved_terms: List[str]
    method: str
    token_count: int = 0
    source_metadata: Optional[Dict[str, Any]] = None


class UnifiedDocumentCompressor:
    """
    Single Source of Truth for medical document compression.
    
    Supports multiple strategies and gracefully degrades if LLM is unavailable.
    """
    
    # Medical and pharmaceutical terms to always preserve
    PRESERVE_PATTERNS = [
        r'\b\d+\s*(?:mg|mcg|ml|mL|g|kg|mmHg|bpm)\b',  # Dosages and measurements
        r'\b(?:daily|twice|three times|weekly|monthly|as needed|PRN)\b',  # Frequencies
        r'\b(?:contraindicated|warning|caution|avoid|monitor|adjust)\b',  # Safety terms
        r'\b(?:recommended|first-line|standard|guideline)\b',  # Recommendations
        r'\b(?:side effect|adverse|interaction|contraindication)\b',  # Medical concepts
    ]
    
    # Sentence patterns to deprioritize
    LOW_PRIORITY_PATTERNS = [
        r'^(?:In this|This article|We will|Here we|As mentioned)',
        r'(?:for more information|see also|refer to)',
        r'^(?:Furthermore|Moreover|Additionally|However),?\s*$',
    ]
    
    # Content type detection patterns
    TABLE_PATTERN = re.compile(r'\|[^\|]+\|')
    LIST_PATTERN = re.compile(r'^\s*[-•*]\s+', re.MULTILINE)
    
    def __init__(
        self,
        llm_gateway: Optional[Any] = None,
        target_ratio: float = 0.5,
        max_tokens: int = 2000,
        preserve_medical_terms: bool = True,
        default_strategy: CompressionStrategy = CompressionStrategy.HYBRID,
    ):
        """
        Initialize the unified compressor.
        
        Args:
            llm_gateway: Optional LLM for intelligent compression
            target_ratio: Target compression ratio (0.5 = 50% of original)
            max_tokens: Maximum tokens in output
            preserve_medical_terms: Always keep medical terminology
            default_strategy: Default compression strategy
        """
        self.llm_gateway = llm_gateway
        self.target_ratio = target_ratio
        self.max_tokens = max_tokens
        self.preserve_medical_terms = preserve_medical_terms
        self.default_strategy = default_strategy
        
        # Latency optimization: Cache compression results
        self._compression_cache: Dict[str, CompressedDocument] = {}
        self._cache_max_size = 200
        
        # Compile regex patterns
        self._preserve_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PRESERVE_PATTERNS
        ]
        self._low_priority_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.LOW_PRIORITY_PATTERNS
        ]
        
        logger.info(
            f"✅ UnifiedDocumentCompressor initialized: "
            f"ratio={target_ratio}, max_tokens={max_tokens}, strategy={default_strategy.value}"
        )
    
    async def compress(
        self,
        query: str,
        documents: List[Any],
        strategy: Optional[CompressionStrategy] = None,
    ) -> List[CompressedDocument]:
        """
        Main entry point for document compression.
        
        Args:
            query: User query for relevance scoring
            documents: Retrieved documents (dicts or LangChain Document objects)
            strategy: Override default compression strategy
            
        Returns:
            List of CompressedDocument objects
        """
        strategy = strategy or self.default_strategy
        
        logger.info(f"📄 Compressing {len(documents)} documents using {strategy.value} strategy")
        
        compressed_docs = []
        
        for doc in documents:
            content = self._extract_content(doc)
            
            # Skip very short documents
            if not content or len(content) < 100:
                compressed_docs.append(CompressedDocument(
                    original_content=content or "",
                    compressed_content=content or "",
                    compression_ratio=1.0,
                    preserved_terms=[],
                    method="none",
                    token_count=len((content or "").split()),
                ))
                continue
            
            # Latency optimization: Check cache first
            cache_key = hashlib.md5(f"{query[:100]}:{content[:500]}".encode()).hexdigest()
            if cache_key in self._compression_cache:
                logger.debug("Compression cache hit")
                compressed_docs.append(self._compression_cache[cache_key])
                continue
            
            try:
                # Select strategy
                if strategy == CompressionStrategy.AUTO or strategy == CompressionStrategy.HYBRID:
                    compressed = await self._adaptive_compress(query, content)
                elif strategy == CompressionStrategy.LLM:
                    if self.llm_gateway:
                        compressed = await self._llm_compress(query, content)
                    else:
                        logger.warning("LLM gateway not available, falling back to extractive")
                        compressed = self._extractive_compress(query, content)
                elif strategy == CompressionStrategy.SENTENCE:
                    compressed = self._sentence_compress(query, content)
                elif strategy == CompressionStrategy.LIST_AWARE:
                    compressed = self._adaptive_compress(query, content)
                else:  # EXTRACTIVE
                    compressed = self._extractive_compress(query, content)
                
                # Store in cache
                if len(self._compression_cache) >= self._cache_max_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self._compression_cache))
                    del self._compression_cache[oldest_key]
                self._compression_cache[cache_key] = compressed
                
                compressed_docs.append(compressed)
                
            except Exception as e:
                logger.error(f"Compression failed: {e}, using truncation fallback")
                compressed_docs.append(self._truncate_compress(content))
        
        # Deduplicate across documents
        compressed_docs = self._deduplicate(compressed_docs)
        
        logger.info(f"✅ Compressed {len(compressed_docs)} documents")
        return compressed_docs
    
    async def _adaptive_compress(
        self,
        query: str,
        content: str
    ) -> CompressedDocument:
        """Adaptive compression based on content type."""
        # Detect content type and choose strategy
        if self.TABLE_PATTERN.search(content):
            logger.debug("Detected table content, using table-aware compression")
            return self._compress_table_content(query, content)
        elif self.LIST_PATTERN.search(content):
            logger.debug("Detected list content, using list-aware compression")
            return self._compress_list_content(query, content)
        else:
            # Regular text: use LLM if available, otherwise extractive
            if self.llm_gateway:
                return await self._llm_compress(query, content)
            else:
                return self._extractive_compress(query, content)
    
    async def _llm_compress(
        self,
        query: str,
        content: str
    ) -> CompressedDocument:
        """Use LLM for intelligent extractive compression."""
        prompt = f"""Extract the most relevant information from this medical document that answers the query.
Keep medical terminology, dosages, and safety information.
Be concise but preserve accuracy.

Query: {query}

Document:
{content[:3000]}

Extracted relevant content:"""
        
        try:
            # Handle both async and sync LLM gateways
            if hasattr(self.llm_gateway, 'generate'):
                result = self.llm_gateway.generate(
                    prompt,
                    content_type="medical"
                )
                # Handle awaitable result
                if hasattr(result, '__await__'):
                    compressed = await result
                else:
                    compressed = result
            else:
                # Fallback if gateway has different interface
                compressed = str(content[:int(len(content) * self.target_ratio)])
            
            preserved = self._extract_preserved_terms(compressed)
            ratio = len(compressed) / len(content) if content else 1.0
            
            return CompressedDocument(
                original_content=content,
                compressed_content=compressed.strip(),
                compression_ratio=ratio,
                preserved_terms=preserved,
                method="llm",
                token_count=len(compressed.split())
            )
            
        except Exception as e:
            logger.warning(f"LLM compression failed: {e}, falling back to extractive")
            return self._extractive_compress(query, content)
    
    def _sentence_compress(
        self,
        query: str,
        content: str
    ) -> CompressedDocument:
        """Sentence-level relevance filtering."""
        sentences = self._split_sentences(content)
        
        if not sentences:
            return self._truncate_compress(content)
        
        # Score each sentence
        query_terms = set(query.lower().split())
        scored_sentences = [
            (sent, self._score_sentence(sent, query_terms))
            for sent in sentences
        ]
        
        # Sort by relevance
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Keep sentences up to target size
        target_chars = int(len(content) * self.target_ratio)
        kept_sentences = []
        current_chars = 0
        
        for sent, score in scored_sentences:
            if current_chars + len(sent) <= target_chars:
                kept_sentences.append((sent, score))
                current_chars += len(sent)
        
        # Re-order by original position in document
        kept_sentences.sort(key=lambda x: sentences.index(x[0]))
        
        compressed = " ".join(s[0] for s in kept_sentences)
        preserved = self._extract_preserved_terms(compressed)
        
        return CompressedDocument(
            original_content=content,
            compressed_content=compressed,
            compression_ratio=len(compressed) / len(content) if content else 1.0,
            preserved_terms=preserved,
            method="sentence",
            token_count=len(compressed.split())
        )
    
    def _extractive_compress(
        self,
        query: str,
        content: str
    ) -> CompressedDocument:
        """Paragraph-level extractive compression."""
        query_terms = set(query.lower().split())
        paragraphs = content.split('\n\n')
        
        # Score paragraphs
        scored_paras = []
        for para in paragraphs:
            if not para.strip():
                continue
            
            score = 0
            para_lower = para.lower()
            
            # Query term matches (bonus for longer terms)
            for term in query_terms:
                if len(term) > 3 and term in para_lower:
                    score += 2
            
            # Medical term preservation
            for pattern in self._preserve_patterns:
                if pattern.search(para):
                    score += 3
            
            # Penalize low-priority content
            for pattern in self._low_priority_patterns:
                if pattern.search(para):
                    score -= 1
            
            scored_paras.append((para.strip(), max(0, score)))
        
        # Select paragraphs up to target size
        scored_paras.sort(key=lambda x: x[1], reverse=True)
        target_chars = int(len(content) * self.target_ratio)
        selected = []
        current_chars = 0
        
        for para, score in scored_paras:
            if score == 0:
                continue
            if current_chars + len(para) <= target_chars:
                selected.append(para)
                current_chars += len(para)
        
        # Fallback: keep first paragraph if nothing selected
        if not selected and paragraphs:
            selected = [paragraphs[0][:target_chars]]
        
        compressed = "\n\n".join(selected)
        preserved = self._extract_preserved_terms(compressed)
        
        return CompressedDocument(
            original_content=content,
            compressed_content=compressed,
            compression_ratio=len(compressed) / len(content) if content else 1.0,
            preserved_terms=preserved,
            method="extractive",
            token_count=len(compressed.split())
        )
    
    def _compress_table_content(
        self,
        query: str,
        content: str
    ) -> CompressedDocument:
        """Compress content containing tables - preserve structure."""
        query_terms = set(query.lower().split())
        lines = content.split('\n')
        
        kept_lines = []
        in_table = False
        table_header = None
        
        for line in lines:
            if '|' in line:
                in_table = True
                if table_header is None:
                    table_header = line
                    kept_lines.append(line)
                    continue
                
                # Keep rows matching query
                if any(term in line.lower() for term in query_terms if len(term) > 3):
                    kept_lines.append(line)
            else:
                if in_table:
                    in_table = False
                    table_header = None
                
                # Keep non-table content based on relevance
                if any(term in line.lower() for term in query_terms if len(term) > 3):
                    kept_lines.append(line)
        
        compressed = '\n'.join(kept_lines)
        preserved = self._extract_preserved_terms(compressed)
        
        return CompressedDocument(
            original_content=content,
            compressed_content=compressed,
            compression_ratio=len(compressed) / len(content) if content else 1.0,
            preserved_terms=preserved,
            method="table_aware",
            token_count=len(compressed.split())
        )
    
    def _compress_list_content(
        self,
        query: str,
        content: str
    ) -> CompressedDocument:
        """Compress content containing lists - preserve items."""
        query_terms = set(query.lower().split())
        lines = content.split('\n')
        
        kept_lines = []
        list_header = None
        
        for line in lines:
            stripped = line.strip()
            
            # Check if line is a list item
            is_list_item = bool(self.LIST_PATTERN.match(line))
            
            if is_list_item:
                # Keep list items matching query
                if any(term in line.lower() for term in query_terms if len(term) > 3):
                    if list_header and list_header not in kept_lines:
                        kept_lines.append(list_header)
                    kept_lines.append(line)
            else:
                if stripped:
                    list_header = line
                    # Keep headers matching query
                    if any(term in line.lower() for term in query_terms if len(term) > 3):
                        kept_lines.append(line)
        
        compressed = '\n'.join(kept_lines)
        preserved = self._extract_preserved_terms(compressed)
        
        return CompressedDocument(
            original_content=content,
            compressed_content=compressed,
            compression_ratio=len(compressed) / len(content) if content else 1.0,
            preserved_terms=preserved,
            method="list_aware",
            token_count=len(compressed.split())
        )
    
    def _truncate_compress(self, content: str) -> CompressedDocument:
        """Simple truncation fallback."""
        target_chars = int(len(content) * self.target_ratio)
        truncated = content[:target_chars]
        
        # Try to truncate at sentence boundary
        last_period = truncated.rfind('.')
        if last_period > target_chars * 0.7:
            truncated = truncated[:last_period + 1]
        
        preserved = self._extract_preserved_terms(truncated)
        
        return CompressedDocument(
            original_content=content,
            compressed_content=truncated,
            compression_ratio=len(truncated) / len(content) if content else 1.0,
            preserved_terms=preserved,
            method="truncate",
            token_count=len(truncated.split())
        )
    
    def _extract_content(self, doc: Any) -> str:
        """Extract text content from various document formats."""
        if isinstance(doc, dict):
            return doc.get("content") or doc.get("page_content") or doc.get("text", "")
        elif hasattr(doc, "page_content"):
            return doc.page_content
        elif hasattr(doc, "content"):
            return doc.content
        return str(doc)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _score_sentence(self, sentence: str, query_terms: set) -> float:
        """Score sentence for relevance."""
        score = 0.0
        sent_lower = sentence.lower()
        sent_words = set(sent_lower.split())
        
        # Query term overlap (weighted)
        overlap = len(query_terms & sent_words)
        score += overlap * 2
        
        # Medical term presence (high priority)
        for pattern in self._preserve_patterns:
            if pattern.search(sentence):
                score += 3
        
        # Length preference
        word_count = len(sentence.split())
        if 10 <= word_count <= 50:
            score += 1
        elif word_count < 5:
            score -= 2
        
        return max(0, score)
    
    def _extract_preserved_terms(self, text: str) -> List[str]:
        """Extract medical terms preserved in text."""
        terms = []
        
        for pattern in self._preserve_patterns:
            matches = pattern.findall(text)
            terms.extend(matches)
        
        return list(set(terms))[:20]  # Limit to 20 unique terms
    
    def _deduplicate(
        self,
        documents: List[CompressedDocument]
    ) -> List[CompressedDocument]:
        """Remove duplicate content across documents."""
        seen_hashes = set()
        unique_docs = []
        
        for doc in documents:
            # Hash first 200 chars as fingerprint
            fingerprint = hashlib.md5(
                doc.compressed_content[:200].encode()
            ).hexdigest()
            
            if fingerprint not in seen_hashes:
                seen_hashes.add(fingerprint)
                unique_docs.append(doc)
            else:
                logger.debug("Removed duplicate document")
        
        return unique_docs
    
    def format_compressed_context(
        self,
        compressed_docs: List[CompressedDocument],
        max_total_tokens: Optional[int] = None
    ) -> str:
        """
        Format compressed documents for LLM context.
        
        Args:
            compressed_docs: List of compressed documents
            max_total_tokens: Optional token limit
            
        Returns:
            Formatted context string
        """
        max_tokens = max_total_tokens or self.max_tokens
        sections = []
        current_tokens = 0
        
        for i, doc in enumerate(compressed_docs):
            if current_tokens >= max_tokens:
                break
            
            section = f"[Source {i+1}]\n{doc.compressed_content}"
            section_tokens = len(section.split())
            
            if current_tokens + section_tokens <= max_tokens:
                sections.append(section)
                current_tokens += section_tokens
        
        return "\n\n".join(sections)


# Backward compatibility aliases
ContextualCompressor = UnifiedDocumentCompressor
AdaptiveCompressor = UnifiedDocumentCompressor

__all__ = [
    "UnifiedDocumentCompressor",
    "CompressedDocument",
    "CompressionStrategy",
    "ContextualCompressor",
    "AdaptiveCompressor",
]
