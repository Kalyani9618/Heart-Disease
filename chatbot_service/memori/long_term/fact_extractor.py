"""
Fact Extractor for Long-term Memory Consolidation

Extracts structured facts from conversations and stores them in long-term memory.
Supports background processing with threading and async operations.

Key Features:
- NLP-based fact extraction from conversation text
- Entity recognition and relationship mapping
- Medical fact categorization (symptoms, diagnoses, medications, etc.)
- Background worker for async processing
- Thread-safe operations with proper synchronization
"""

import asyncio
import logging
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


class FactCategory(str, Enum):
    """Categories for extracted medical facts."""
    
    SYMPTOM = "symptom"
    DIAGNOSIS = "diagnosis"
    MEDICATION = "medication"
    ALLERGY = "allergy"
    VITAL_SIGN = "vital_sign"
    LAB_RESULT = "lab_result"
    PROCEDURE = "procedure"
    FAMILY_HISTORY = "family_history"
    LIFESTYLE = "lifestyle"
    PREFERENCE = "preference"
    APPOINTMENT = "appointment"
    GENERAL = "general"


@dataclass
class ExtractedFact:
    """A fact extracted from conversation text."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: FactCategory = FactCategory.GENERAL
    content: str = ""
    source_text: str = ""
    confidence: float = 0.8
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "category": self.category.value if isinstance(self.category, FactCategory) else self.category,
            "content": self.content,
            "source_text": self.source_text,
            "confidence": self.confidence,
            "entities": self.entities,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedFact":
        """Create from dictionary."""
        category = data.get("category", "general")
        if isinstance(category, str):
            try:
                category = FactCategory(category)
            except ValueError:
                category = FactCategory.GENERAL
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            category=category,
            content=data.get("content", ""),
            source_text=data.get("source_text", ""),
            confidence=data.get("confidence", 0.8),
            entities=data.get("entities", []),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
        )


@dataclass
class ExtractionTask:
    """A task for the extraction worker queue."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable[[List[ExtractedFact]], None]] = None
    priority: int = 0  # Higher = more priority
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Fact Extraction Patterns
# ============================================================================


class MedicalPatternMatcher:
    """Pattern-based extraction for medical facts."""
    
    # Medical entity patterns
    SYMPTOM_PATTERNS = [
        r"(?:i\s+(?:have|feel|experience|am having))\s+(.+?)(?:\.|,|$)",
        r"(?:experiencing|suffering from)\s+(.+?)(?:\.|,|$)",
        r"(?:my|the)\s+(\w+)\s+(?:hurts|aches|is painful)",
        r"(?:pain\s+in\s+(?:my|the))\s+(\w+)",
    ]
    
    MEDICATION_PATTERNS = [
        r"(?:taking|prescribed|on)\s+(\w+(?:\s+\d+\s*(?:mg|ml|g))?)",
        r"(\w+)\s+(?:medication|medicine|drug|pill)",
        r"(?:dose|dosage)\s+of\s+(\w+)",
    ]
    
    ALLERGY_PATTERNS = [
        r"(?:allergic to|allergy to)\s+(.+?)(?:\.|,|$)",
        r"(\w+)\s+allergy",
    ]
    
    DIAGNOSIS_PATTERNS = [
        r"(?:diagnosed with|have been diagnosed)\s+(.+?)(?:\.|,|$)",
        r"(?:doctor said i have|told i have)\s+(.+?)(?:\.|,|$)",
        r"(?:suffering from|condition is)\s+(.+?)(?:\.|,|$)",
    ]
    
    VITAL_PATTERNS = [
        r"(?:blood pressure|bp)\s+(?:is\s+)?(\d+/\d+)",
        r"(?:heart rate|pulse)\s+(?:is\s+)?(\d+)",
        r"(?:temperature)\s+(?:is\s+)?(\d+(?:\.\d+)?)",
        r"(?:weight)\s+(?:is\s+)?(\d+(?:\.\d+)?)\s*(?:kg|lbs?)?",
    ]
    
    def __init__(self):
        """Initialize compiled patterns."""
        self._compiled_patterns = {
            FactCategory.SYMPTOM: [re.compile(p, re.IGNORECASE) for p in self.SYMPTOM_PATTERNS],
            FactCategory.MEDICATION: [re.compile(p, re.IGNORECASE) for p in self.MEDICATION_PATTERNS],
            FactCategory.ALLERGY: [re.compile(p, re.IGNORECASE) for p in self.ALLERGY_PATTERNS],
            FactCategory.DIAGNOSIS: [re.compile(p, re.IGNORECASE) for p in self.DIAGNOSIS_PATTERNS],
            FactCategory.VITAL_SIGN: [re.compile(p, re.IGNORECASE) for p in self.VITAL_PATTERNS],
        }
    
    def extract_patterns(self, text: str) -> List[Tuple[FactCategory, str, float]]:
        """Extract facts using pattern matching.
        
        Returns:
            List of tuples: (category, extracted_text, confidence)
        """
        results = []
        
        for category, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    if match and len(match.strip()) > 2:
                        results.append((category, match.strip(), 0.75))
        
        return results


# ============================================================================
# Fact Extractor
# ============================================================================


class FactExtractor:
    """
    Main fact extraction engine.
    
    Extracts structured facts from conversation text using:
    - Pattern matching for common medical phrases
    - Entity recognition (optional, with spaCy)
    - Keyword-based categorization
    """
    
    def __init__(
        self,
        use_nlp: bool = False,
        min_confidence: float = 0.5,
        max_facts_per_text: int = 20,
    ):
        """
        Initialize the fact extractor.
        
        Args:
            use_nlp: Whether to use spaCy for NLP-based extraction
            min_confidence: Minimum confidence threshold for facts
            max_facts_per_text: Maximum facts to extract per text
        """
        self.use_nlp = use_nlp
        self.min_confidence = min_confidence
        self.max_facts_per_text = max_facts_per_text
        self._pattern_matcher = MedicalPatternMatcher()
        self._nlp = None
        self._nlp_lock = threading.Lock()  # Thread-safe lock for NLP operations
        
        if use_nlp:
            self._init_nlp()
    
    def _init_nlp(self) -> None:
        """Initialize spaCy NLP model via centralized service."""
        try:
            from core.services.spacy_service import get_spacy_service
            self._spacy_service = get_spacy_service()
            self._nlp = self._spacy_service.nlp
            logger.info("Initialized FactExtractor with centralized SpaCyService")
        except Exception as e:
            logger.warning(f"Failed to initialize SpaCyService: {e}")
            self._nlp = None
    
    def extract(
        self,
        text: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractedFact]:
        """
        Extract facts from text.
        
        Args:
            text: The text to extract facts from
            user_id: Optional user identifier
            session_id: Optional session identifier
            context: Optional context for extraction
            
        Returns:
            List of extracted facts
        """
        if not text or not text.strip():
            return []
        
        facts: List[ExtractedFact] = []
        seen_content: Set[str] = set()
        
        # Pattern-based extraction
        pattern_results = self._pattern_matcher.extract_patterns(text)
        for category, content, confidence in pattern_results:
            content_key = content.lower().strip()
            if content_key not in seen_content and confidence >= self.min_confidence:
                seen_content.add(content_key)
                facts.append(ExtractedFact(
                    category=category,
                    content=content,
                    source_text=text[:200],  # Store truncated source
                    confidence=confidence,
                    user_id=user_id,
                    session_id=session_id,
                    metadata=context or {},
                ))
        
        # NLP-based extraction (if available)
        if self._nlp is not None:
            nlp_facts = self._extract_with_nlp(text, user_id, session_id, seen_content)
            facts.extend(nlp_facts)
        
        # Limit facts per text
        if len(facts) > self.max_facts_per_text:
            # Sort by confidence and take top N
            facts.sort(key=lambda f: f.confidence, reverse=True)
            facts = facts[:self.max_facts_per_text]
        
        logger.debug(f"Extracted {len(facts)} facts from text")
        return facts
    
    def _extract_with_nlp(
        self,
        text: str,
        user_id: Optional[str],
        session_id: Optional[str],
        seen_content: Set[str],
    ) -> List[ExtractedFact]:
        """Extract facts using spaCy NLP via SpaCyService (thread-safe)."""
        facts = []
        
        if self._nlp is None:
            return facts
        
        # Thread-safe spaCy access - spaCy models may not be thread-safe
        # depending on pipeline components
        with self._nlp_lock:
            try:
                # Use the service's get_entities method if available for better handling
                if hasattr(self, '_spacy_service'):
                    # 1. Get entities with negation awareness (NER + EntityRuler)
                    entities = self._spacy_service.get_entities(text, include_negated=False)
                    
                    # 2. Get phrase matches (MedicalPhraseMatcher)
                    phrase_matches = self._spacy_service.find_medical_terms(text)
                    
                    # Combine results (phrase matches might duplicate entities, handled by seen_content)
                    all_entities = entities + phrase_matches
                    
                    for ent_data in all_entities:
                        content_key = ent_data["text"].lower().strip()
                        if content_key in seen_content or len(content_key) < 2:
                            continue
                        
                        # Map spaCy entity labels to our categories
                        category = self._map_entity_label(ent_data["label"])
                        if category is not None:
                            seen_content.add(content_key)
                            facts.append(ExtractedFact(
                                category=category,
                                content=ent_data["text"],
                                source_text=text[:200],
                                confidence=0.85,  # Higher confidence with custom pipeline
                                entities=[ent_data["label"]],
                                user_id=user_id,
                                session_id=session_id,
                                metadata={"negated": ent_data.get("is_negated", False)}
                            ))
                    
                    # 3. Get contextual facts (Dependency Parsing)
                    if hasattr(self._spacy_service, 'get_contextual_facts'):
                        context_facts = self._spacy_service.get_contextual_facts(text)
                        for fact in context_facts:
                            content_key = fact["text"].lower().strip()
                            if content_key in seen_content or len(content_key) < 2:
                                continue
                            
                            # Map contextual types to categories
                            category = None
                            if fact["type"] == "SYMPTOM_PRESENT":
                                category = FactCategory.SYMPTOM
                            elif fact["type"] == "DRUG_INTAKE":
                                category = FactCategory.MEDICATION
                            elif fact["type"] == "DIAGNOSIS":
                                category = FactCategory.DIAGNOSIS
                                
                            if category:
                                seen_content.add(content_key)
                                facts.append(ExtractedFact(
                                    category=category,
                                    content=fact["text"],
                                    source_text=fact["sent"][:200],
                                    confidence=0.9,  # High confidence for dependency match
                                    entities=[fact["type"]],
                                    user_id=user_id,
                                    session_id=session_id,
                                    metadata={"extraction_method": "dependency_parsing"}
                                ))
                else:
                    # Fallback to direct doc processing if service not available (shouldn't happen)
                    doc = self._nlp(text)
                    
                    # Extract named entities
                    for ent in doc.ents:
                        content_key = ent.text.lower().strip()
                        if content_key in seen_content or len(content_key) < 2:
                            continue
                        
                        # Map spaCy entity labels to our categories
                        category = self._map_entity_label(ent.label_)
                        if category is not None:
                            seen_content.add(content_key)
                            facts.append(ExtractedFact(
                                category=category,
                                content=ent.text,
                                source_text=text[:200],
                                confidence=0.7,
                                entities=[ent.label_],
                                user_id=user_id,
                                session_id=session_id,
                            ))
            except Exception as e:
                logger.warning(f"NLP extraction failed: {e}")
        
        return facts
    
    def _map_entity_label(self, label: str) -> Optional[FactCategory]:
        """Map spaCy entity labels to fact categories."""
        mapping = {
            "DISEASE": FactCategory.DIAGNOSIS,
            "CHEMICAL": FactCategory.MEDICATION,
            "DRUG": FactCategory.MEDICATION,
            "SYMPTOM": FactCategory.SYMPTOM,
        }
        return mapping.get(label)
    
    async def extract_async(
        self,
        text: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractedFact]:
        """Async wrapper for fact extraction."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.extract,
            text,
            user_id,
            session_id,
            context,
        )

    async def extract_batch(
        self,
        texts: List[str],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_concurrent: int = 5,
    ) -> List[List[ExtractedFact]]:
        """
        Extract facts from multiple texts in parallel.
        
        Uses asyncio to process multiple texts concurrently, enabling
        the AI to process multiple inputs simultaneously.

        Args:
            texts: List of texts to extract facts from
            user_id: Optional user identifier
            session_id: Optional session identifier
            context: Optional context for extraction
            max_concurrent: Maximum concurrent extractions
            
        Returns:
            List of fact lists, one per input text
        """
        if not texts:
            return []

        loop = asyncio.get_running_loop()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _extract_with_limit(text: str) -> List[ExtractedFact]:
            async with semaphore:
                return await loop.run_in_executor(
                    None, self.extract, text, user_id, session_id, context
                )

        tasks = [_extract_with_limit(t) for t in texts if t and t.strip()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions gracefully
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch extraction failed for text {i}: {result}")
                final_results.append([])
            else:
                final_results.append(result)

        logger.info(
            f"Batch extraction complete: {len(texts)} texts, "
            f"{sum(len(r) for r in final_results)} total facts extracted"
        )
        return final_results

    async def extract_from_multiple_sources(
        self,
        sources: List[Dict[str, Any]],
        max_concurrent: int = 10,
    ) -> Dict[str, List[ExtractedFact]]:
        """
        Extract facts from multiple data sources simultaneously.

        Each source can be a different type (conversation, health_data, 
        lab_result, medication_record, etc.) processed in parallel.

        Args:
            sources: List of source dicts with keys:
                - 'type': Source type ('conversation', 'health_data', 'lab_result', etc.) 
                - 'content': Text content to extract from
                - 'user_id': Optional user identifier  
                - 'session_id': Optional session identifier
                - 'metadata': Optional additional metadata
            max_concurrent: Maximum concurrent extractions
            
        Returns:
            Dict mapping source index to extracted facts
        """
        if not sources:
            return {}

        loop = asyncio.get_running_loop()
        semaphore = asyncio.Semaphore(max_concurrent)
        results: Dict[str, List[ExtractedFact]] = {}

        async def _process_source(idx: int, source: Dict[str, Any]) -> tuple:
            async with semaphore:
                source_type = source.get("type", "unknown")
                content = source.get("content", "")
                user_id = source.get("user_id")
                session_id = source.get("session_id")
                metadata = source.get("metadata", {})

                if not content or not content.strip():
                    return idx, []

                # Enrich context with source type information
                context = {
                    "source_type": source_type,
                    "source_index": idx,
                    **metadata,
                }

                # Pre-process based on source type
                processed_content = self._preprocess_source(source_type, content)

                facts = await loop.run_in_executor(
                    None, self.extract, processed_content, user_id, session_id, context
                )

                # Tag facts with source information
                for fact in facts:
                    fact.metadata["source_type"] = source_type
                    fact.metadata["source_index"] = idx

                return idx, facts

        tasks = [_process_source(i, s) for i, s in enumerate(sources)]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in task_results:
            if isinstance(result, Exception):
                logger.error(f"Multi-source extraction error: {result}")
                continue
            idx, facts = result
            results[str(idx)] = facts

        total_facts = sum(len(f) for f in results.values())
        logger.info(
            f"Multi-source extraction complete: {len(sources)} sources, "
            f"{total_facts} total facts extracted"
        )
        return results

    def _preprocess_source(self, source_type: str, content: str) -> str:
        """
        Pre-process content based on source type for better extraction.
        
        Args:
            source_type: Type of the data source
            content: Raw content text
            
        Returns:
            Pre-processed content optimized for extraction
        """
        if source_type == "lab_result":
            # Structure lab results for better pattern matching
            return f"Lab results show: {content}"
        elif source_type == "medication_record":
            return f"Patient is taking {content}"
        elif source_type == "vital_signs":
            return f"Vital signs recorded: {content}"
        elif source_type == "clinical_note":
            return f"Clinical note: {content}"
        elif source_type == "radiology_report":
            return f"Radiology findings: {content}"
        else:
            return content


# ============================================================================
# Memory Extraction Worker
# ============================================================================


class MemoryExtractionWorker:
    """
    Background worker for processing extraction tasks.
    
    Runs in a separate thread and processes tasks from a queue.
    Supports:
    - Priority-based task processing
    - Batch processing for efficiency
    - Callback notifications
    - Persistent storage of extracted facts
    - Retry logic for failed tasks
    - Graceful shutdown
    """
    
    def __init__(
        self,
        extractor: Optional[FactExtractor] = None,
        max_queue_size: int = 1000,
        batch_size: int = 10,
        processing_interval: float = 0.1,
        max_retries: int = 3,
        persistence_callback: Optional[Callable[[List[ExtractedFact]], None]] = None,
    ):
        """
        Initialize the extraction worker.
        
        Args:
            extractor: FactExtractor instance (creates default if None)
            max_queue_size: Maximum queue size
            batch_size: Number of tasks to process per batch
            processing_interval: Sleep between processing cycles (seconds)
            max_retries: Maximum retries for failed tasks
            persistence_callback: Function to persist extracted facts to long-term storage
        """
        self.extractor = extractor or FactExtractor()
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.processing_interval = processing_interval
        self.max_retries = max_retries
        self._persistence_callback = persistence_callback
        
        self._task_queue: Queue[ExtractionTask] = Queue(maxsize=max_queue_size)
        self._retry_queue: Queue[tuple] = Queue()  # (task, retry_count) pairs
        self._results: Dict[str, List[ExtractedFact]] = {}
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False
        self._lock = threading.Lock()
        
        # Metrics
        self._tasks_processed = 0
        self._tasks_failed = 0
        self._tasks_retried = 0
        self._batches_processed = 0
        self._facts_persisted = 0
        self._total_processing_time = 0.0
    
    def start(self) -> None:
        """Start the background worker."""
        if self._is_running:
            logger.warning("Extraction worker already running")
            return
        
        with self._lock:
            self._stop_event.clear()
            self._is_running = True
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="MemoryExtractionWorker",
                daemon=True,
            )
            self._worker_thread.start()
            logger.info("Memory extraction worker started")
    
    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background worker gracefully."""
        if not self._is_running:
            return
        
        logger.info("Stopping memory extraction worker...")
        self._stop_event.set()
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
            if self._worker_thread.is_alive():
                logger.warning("Worker thread did not stop gracefully")
        
        with self._lock:
            self._is_running = False
            self._worker_thread = None
        
        logger.info("Memory extraction worker stopped")
    
    def submit_task(
        self,
        text: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[List[ExtractedFact]], None]] = None,
        priority: int = 0,
    ) -> str:
        """
        Submit a task for background extraction.
        
        Args:
            text: Text to extract facts from
            user_id: Optional user identifier
            session_id: Optional session identifier
            context: Optional context
            callback: Optional callback for results
            priority: Task priority (higher = more urgent)
            
        Returns:
            Task ID
        """
        task = ExtractionTask(
            text=text,
            user_id=user_id,
            session_id=session_id,
            context=context or {},
            callback=callback,
            priority=priority,
        )
        
        try:
            self._task_queue.put_nowait(task)
            logger.debug(f"Submitted extraction task {task.id}")
            return task.id
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise
    
    def get_result(self, task_id: str) -> Optional[List[ExtractedFact]]:
        """Get results for a completed task."""
        with self._lock:
            return self._results.pop(task_id, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        avg_time = (
            self._total_processing_time / self._tasks_processed
            if self._tasks_processed > 0
            else 0.0
        )
        return {
            "is_running": self._is_running,
            "queue_size": self._task_queue.qsize(),
            "tasks_processed": self._tasks_processed,
            "tasks_failed": self._tasks_failed,
            "tasks_retried": self._tasks_retried,
            "batches_processed": self._batches_processed,
            "facts_persisted": self._facts_persisted,
            "avg_processing_time_ms": avg_time * 1000,
        }
    
    def set_persistence_callback(
        self, callback: Callable[[List[ExtractedFact]], None]
    ) -> None:
        """Set or update the persistence callback for storing facts to long-term memory."""
        self._persistence_callback = callback
        logger.info("Persistence callback updated for extraction worker")
    
    def _worker_loop(self) -> None:
        """Main worker loop with batch processing and retry support."""
        while not self._stop_event.is_set():
            try:
                # Collect a batch of tasks
                batch: List[ExtractionTask] = []
                
                # First, check retry queue
                while not self._retry_queue.empty() and len(batch) < self.batch_size:
                    try:
                        task, retry_count = self._retry_queue.get_nowait()
                        task.context["_retry_count"] = retry_count
                        batch.append(task)
                    except Empty:
                        break
                
                # Then, fill from main queue
                while len(batch) < self.batch_size:
                    try:
                        task = self._task_queue.get(
                            timeout=self.processing_interval if not batch else 0.01
                        )
                        batch.append(task)
                    except Empty:
                        break
                
                if not batch:
                    continue
                
                # Process the batch
                start_time = time.time()
                all_batch_facts: List[ExtractedFact] = []
                
                for task in batch:
                    retry_count = task.context.pop("_retry_count", 0)
                    try:
                        facts = self.extractor.extract(
                            text=task.text,
                            user_id=task.user_id,
                            session_id=task.session_id,
                            context=task.context,
                        )
                        
                        # Store results
                        with self._lock:
                            self._results[task.id] = facts
                            self._tasks_processed += 1
                        
                        all_batch_facts.extend(facts)
                        
                        # Call callback if provided
                        if task.callback:
                            try:
                                task.callback(facts)
                            except Exception as e:
                                logger.error(f"Callback failed for task {task.id}: {e}")
                        
                    except Exception as e:
                        if retry_count < self.max_retries:
                            # Schedule for retry
                            self._retry_queue.put((task, retry_count + 1))
                            with self._lock:
                                self._tasks_retried += 1
                            logger.warning(
                                f"Task {task.id} failed (attempt {retry_count + 1}/{self.max_retries}), "
                                f"retrying: {e}"
                            )
                        else:
                            logger.error(
                                f"Task {task.id} permanently failed after {self.max_retries} retries: {e}"
                            )
                            with self._lock:
                                self._tasks_failed += 1
                    
                    finally:
                        self._task_queue.task_done()
                
                batch_time = time.time() - start_time
                with self._lock:
                    self._total_processing_time += batch_time
                    self._batches_processed += 1
                
                # Persist all extracted facts from this batch to long-term storage
                if all_batch_facts and self._persistence_callback:
                    try:
                        self._persistence_callback(all_batch_facts)
                        with self._lock:
                            self._facts_persisted += len(all_batch_facts)
                        logger.debug(
                            f"Persisted {len(all_batch_facts)} facts from batch of {len(batch)} tasks"
                        )
                    except Exception as e:
                        logger.error(f"Failed to persist batch facts: {e}")
                
                logger.debug(
                    f"Batch processed: {len(batch)} tasks, "
                    f"{len(all_batch_facts)} facts, "
                    f"{batch_time*1000:.1f}ms"
                )
                    
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(0.5)  # Prevent tight loop on repeated errors
    
    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._is_running
    
    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._task_queue.qsize()


# ============================================================================
# Singleton Instances and Factory Functions
# ============================================================================


# Global instances (lazy initialization)
_fact_extractor: Optional[FactExtractor] = None
_extraction_worker: Optional[MemoryExtractionWorker] = None
_lock = threading.Lock()


def get_fact_extractor(
    use_nlp: bool = False,
    min_confidence: float = 0.5,
    **kwargs,
) -> FactExtractor:
    """
    Get or create the singleton FactExtractor instance.
    
    Args:
        use_nlp: Whether to use NLP-based extraction
        min_confidence: Minimum confidence threshold
        **kwargs: Additional arguments for FactExtractor
        
    Returns:
        FactExtractor instance
    """
    global _fact_extractor
    
    with _lock:
        if _fact_extractor is None:
            _fact_extractor = FactExtractor(
                use_nlp=use_nlp,
                min_confidence=min_confidence,
                **kwargs,
            )
            logger.info("Created singleton FactExtractor instance")
        return _fact_extractor


def get_extraction_worker(
    extractor: Optional[FactExtractor] = None,
    auto_start: bool = True,
    **kwargs,
) -> MemoryExtractionWorker:
    """
    Get or create the singleton MemoryExtractionWorker instance.
    
    Args:
        extractor: Optional FactExtractor instance
        auto_start: Whether to automatically start the worker
        **kwargs: Additional arguments for MemoryExtractionWorker
        
    Returns:
        MemoryExtractionWorker instance
    """
    global _extraction_worker
    
    with _lock:
        if _extraction_worker is None:
            if extractor is None:
                extractor = get_fact_extractor()
            
            _extraction_worker = MemoryExtractionWorker(
                extractor=extractor,
                **kwargs,
            )
            
            if auto_start:
                _extraction_worker.start()
            
            logger.info("Created singleton MemoryExtractionWorker instance")
        
        return _extraction_worker


def shutdown_extraction_worker() -> None:
    """Shutdown the global extraction worker."""
    global _extraction_worker
    
    with _lock:
        if _extraction_worker is not None:
            _extraction_worker.stop()
            _extraction_worker = None
            logger.info("Shutdown extraction worker")
