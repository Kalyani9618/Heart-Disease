r"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

Implements hierarchical document organization for improved retrieval over
long documents. Creates multi-level summaries that enable:
- Broad conceptual queries → High-level summaries
- Specific detail queries → Leaf-level chunks

Architecture::

                    [Root Summary]
                   /              \
          [Summary L1]        [Summary L1]
         /     |     \        /     |     \
     [Chunk] [Chunk] [Chunk] [Chunk] [Chunk] [Chunk]

Key Features:
- Hierarchical clustering of document chunks
- LLM-generated summaries at each level
- Multi-level retrieval with level-aware scoring
- Medical domain optimization
"""


import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Type of RAPTOR tree node."""
    LEAF = "leaf"        # Original document chunk
    SUMMARY = "summary"  # Summarized cluster
    ROOT = "root"        # Top-level summary


@dataclass
class RAPTORNode:
    """A node in the RAPTOR tree hierarchy."""
    id: str
    content: str
    node_type: NodeType
    level: int  # 0 = leaf, higher = more abstract
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "node_type": self.node_type.value,
            "level": self.level,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata
        }


@dataclass
class RAPTORTree:
    """Complete RAPTOR tree structure."""
    nodes: Dict[str, RAPTORNode] = field(default_factory=dict)
    root_ids: List[str] = field(default_factory=list)
    max_level: int = 0
    document_id: str = ""
    
    def get_nodes_at_level(self, level: int) -> List[RAPTORNode]:
        """Get all nodes at a specific level."""
        return [n for n in self.nodes.values() if n.level == level]
    
    def get_leaves(self) -> List[RAPTORNode]:
        """Get all leaf nodes (original chunks)."""
        return [n for n in self.nodes.values() if n.node_type == NodeType.LEAF]
    
    def get_ancestors(self, node_id: str) -> List[RAPTORNode]:
        """Get all ancestors of a node up to root."""
        ancestors = []
        current = self.nodes.get(node_id)
        
        while current and current.parent_id:
            parent = self.nodes.get(current.parent_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        
        return ancestors


class RAPTORBuilder:
    """
    Builds RAPTOR trees from document chunks.
    
    Process:
    1. Start with leaf chunks
    2. Cluster similar chunks
    3. Generate summary for each cluster
    4. Repeat until single root or max levels
    """
    
    # Medical summarization prompt
    SUMMARY_PROMPT = """Summarize the following medical content into a concise paragraph.
Preserve key medical terms, drug names, dosages, and clinical recommendations.
Focus on the most important clinical information.

Content to summarize:
{content}

Summary:"""

    def __init__(
        self,
        llm_gateway: Any,
        embedding_model: Any,
        max_levels: int = 3,
        cluster_size: int = 4,
        min_cluster_similarity: float = 0.6
    ):
        """
        Initialize RAPTOR builder.
        
        Args:
            llm_gateway: LLM for generating summaries
            embedding_model: Model for computing embeddings
            max_levels: Maximum tree depth
            cluster_size: Target number of chunks per cluster
            min_cluster_similarity: Minimum similarity for clustering
        """
        self.llm_gateway = llm_gateway
        self.embedding_model = embedding_model
        self.max_levels = max_levels
        self.cluster_size = cluster_size
        self.min_cluster_similarity = min_cluster_similarity
        
        logger.info(
            f"✓ RAPTORBuilder initialized "
            f"(max_levels={max_levels}, cluster_size={cluster_size})"
        )
    
    async def build_tree(
        self,
        chunks: List[Dict],
        document_id: str = "doc"
    ) -> RAPTORTree:
        """
        Build RAPTOR tree from document chunks.
        
        Args:
            chunks: List of chunk dicts with 'content' and 'metadata'
            document_id: Identifier for the source document
            
        Returns:
            Complete RAPTORTree structure
        """
        tree = RAPTORTree(document_id=document_id)
        
        if not chunks:
            logger.warning("No chunks provided for RAPTOR tree building")
            return tree
        
        # Step 1: Create leaf nodes
        leaf_nodes = await self._create_leaf_nodes(chunks, document_id)
        for node in leaf_nodes:
            tree.nodes[node.id] = node
        
        # Step 2: Build hierarchy level by level
        current_level_nodes = leaf_nodes
        current_level = 0
        
        while len(current_level_nodes) > 1 and current_level < self.max_levels:
            current_level += 1
            
            # Cluster nodes at current level
            clusters = self._cluster_nodes(current_level_nodes)
            
            if len(clusters) == len(current_level_nodes):
                # No more clustering possible
                break
            
            # Generate summary nodes for each cluster
            summary_nodes = await self._create_summary_nodes(
                clusters, current_level, document_id
            )
            
            # Update parent-child relationships
            for summary_node, cluster in zip(summary_nodes, clusters):
                for child_node in cluster:
                    child_node.parent_id = summary_node.id
                    summary_node.children_ids.append(child_node.id)
                    tree.nodes[child_node.id] = child_node
                tree.nodes[summary_node.id] = summary_node
            
            current_level_nodes = summary_nodes
            tree.max_level = current_level
        
        # Mark top-level nodes as roots
        for node in current_level_nodes:
            node.node_type = NodeType.ROOT
            tree.root_ids.append(node.id)
            tree.nodes[node.id] = node
        
        logger.info(
            f"✓ RAPTOR tree built: {len(tree.nodes)} nodes, "
            f"{len(tree.get_leaves())} leaves, {tree.max_level} levels"
        )
        
        return tree
    
    async def _create_leaf_nodes(
        self,
        chunks: List[Dict],
        doc_id: str
    ) -> List[RAPTORNode]:
        """Create leaf nodes from chunks."""
        nodes = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.get("content") or chunk.get("text", "")
            
            node_id = f"{doc_id}_leaf_{i}"
            
            # Compute embedding
            try:
                embedding = await self._get_embedding(content)
            except Exception as e:
                logger.warning(f"Embedding failed for chunk {i}: {e}")
                embedding = None
            
            node = RAPTORNode(
                id=node_id,
                content=content,
                node_type=NodeType.LEAF,
                level=0,
                embedding=embedding,
                metadata=chunk.get("metadata", {})
            )
            nodes.append(node)
        
        return nodes
    
    def _cluster_nodes(
        self,
        nodes: List[RAPTORNode]
    ) -> List[List[RAPTORNode]]:
        """Cluster nodes by embedding similarity."""
        if len(nodes) <= self.cluster_size:
            return [nodes]
        
        # Simple greedy clustering based on embedding similarity
        clusters = []
        remaining = list(nodes)
        
        while remaining:
            # Start new cluster with first remaining node
            cluster = [remaining.pop(0)]
            
            # Add similar nodes to cluster
            while len(cluster) < self.cluster_size and remaining:
                # Find most similar remaining node
                best_idx = -1
                best_sim = -1
                
                for i, node in enumerate(remaining):
                    sim = self._compute_similarity(cluster[0], node)
                    if sim > best_sim and sim >= self.min_cluster_similarity:
                        best_sim = sim
                        best_idx = i
                
                if best_idx >= 0:
                    cluster.append(remaining.pop(best_idx))
                else:
                    break
            
            clusters.append(cluster)
        
        return clusters
    
    def _compute_similarity(
        self,
        node1: RAPTORNode,
        node2: RAPTORNode
    ) -> float:
        """Compute cosine similarity between node embeddings."""
        if node1.embedding is None or node2.embedding is None:
            # Fallback to simple text overlap
            words1 = set(node1.content.lower().split())
            words2 = set(node2.content.lower().split())
            if not words1 or not words2:
                return 0.0
            return len(words1 & words2) / len(words1 | words2)
        
        # Cosine similarity
        dot = np.dot(node1.embedding, node2.embedding)
        norm1 = np.linalg.norm(node1.embedding)
        norm2 = np.linalg.norm(node2.embedding)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    
    async def _create_summary_nodes(
        self,
        clusters: List[List[RAPTORNode]],
        level: int,
        doc_id: str
    ) -> List[RAPTORNode]:
        """Create summary nodes for clusters."""
        summary_nodes = []
        
        for i, cluster in enumerate(clusters):
            # Combine cluster content
            combined_content = "\n\n".join(node.content for node in cluster)
            
            # Generate summary
            try:
                summary = await self._generate_summary(combined_content)
            except Exception as e:
                logger.warning(f"Summary generation failed: {e}")
                # Fallback to truncation
                summary = combined_content[:500] + "..."
            
            node_id = f"{doc_id}_L{level}_{i}"
            
            # Compute embedding for summary
            try:
                embedding = await self._get_embedding(summary)
            except Exception:
                embedding = None
            
            node = RAPTORNode(
                id=node_id,
                content=summary,
                node_type=NodeType.SUMMARY,
                level=level,
                embedding=embedding,
                metadata={"cluster_size": len(cluster)}
            )
            summary_nodes.append(node)
        
        return summary_nodes
    
    async def _generate_summary(self, content: str) -> str:
        """Generate summary using LLM."""
        prompt = self.SUMMARY_PROMPT.format(content=content[:3000])
        
        response = await self.llm_gateway.generate(
            prompt,
            max_tokens=300,
            temperature=0.3
        )
        
        return response.strip()
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        if hasattr(self.embedding_model, "embed"):
            return await self.embedding_model.embed(text)
        elif hasattr(self.embedding_model, "encode"):
            return self.embedding_model.encode(text)
        elif callable(self.embedding_model):
            return await self.embedding_model(text)
        else:
            raise ValueError("Embedding model has no compatible method")


class RAPTORRetriever:
    """
    Retrieves from RAPTOR trees using multi-level search.
    
    Strategy:
    1. Search all levels for query matches
    2. Weight results by level (higher levels for broad queries)
    3. Optionally expand to children for more detail
    """
    
    def __init__(
        self,
        embedding_model: Any,
        level_weights: Optional[Dict[int, float]] = None,
        expand_to_leaves: bool = True
    ):
        """
        Initialize RAPTOR retriever.
        
        Args:
            embedding_model: Model for query embeddings
            level_weights: Weights for each level (higher = more important)
            expand_to_leaves: Whether to expand summaries to leaf content
        """
        self.embedding_model = embedding_model
        self.level_weights = level_weights or {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4}
        self.expand_to_leaves = expand_to_leaves
        
        logger.info(f"✓ RAPTORRetriever initialized")
    
    async def retrieve(
        self,
        query: str,
        tree: RAPTORTree,
        top_k: int = 5,
        level_filter: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant nodes from RAPTOR tree.
        
        Args:
            query: User query
            tree: RAPTOR tree to search
            top_k: Number of results to return
            level_filter: Only search specific level (None = all levels)
            
        Returns:
            List of result dicts with content, score, level, etc.
        """
        if not tree.nodes:
            return []
        
        # Get query embedding
        try:
            query_embedding = await self._get_embedding(query)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return []
        
        # Score all nodes
        scored_nodes = []
        
        for node in tree.nodes.values():
            if level_filter is not None and node.level != level_filter:
                continue
            
            if node.embedding is None:
                continue
            
            # Compute similarity
            similarity = self._cosine_similarity(query_embedding, node.embedding)
            
            # Apply level weight
            level_weight = self.level_weights.get(node.level, 0.5)
            weighted_score = similarity * level_weight
            
            scored_nodes.append((node, weighted_score, similarity))
        
        # Sort by score
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Take top-k
        top_nodes = scored_nodes[:top_k]
        
        # Format results
        results = []
        for node, weighted_score, raw_score in top_nodes:
            result = {
                "content": node.content,
                "score": weighted_score,
                "raw_score": raw_score,
                "level": node.level,
                "node_type": node.node_type.value,
                "node_id": node.id,
                "metadata": node.metadata
            }
            
            # Optionally expand summaries to include leaf content
            if self.expand_to_leaves and node.node_type != NodeType.LEAF:
                leaves = self._get_descendant_leaves(node, tree)
                result["expanded_content"] = "\n\n".join(
                    leaf.content for leaf in leaves[:5]
                )
            
            results.append(result)
        
        logger.debug(f"RAPTOR retrieved {len(results)} nodes from tree")
        
        return results
    
    def _get_descendant_leaves(
        self,
        node: RAPTORNode,
        tree: RAPTORTree
    ) -> List[RAPTORNode]:
        """Get all leaf descendants of a node."""
        if node.node_type == NodeType.LEAF:
            return [node]
        
        leaves = []
        
        def collect_leaves(node_id: str):
            n = tree.nodes.get(node_id)
            if not n:
                return
            if n.node_type == NodeType.LEAF:
                leaves.append(n)
            else:
                for child_id in n.children_ids:
                    collect_leaves(child_id)
        
        for child_id in node.children_ids:
            collect_leaves(child_id)
        
        return leaves
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot / (norm_a * norm_b))
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        if hasattr(self.embedding_model, "embed"):
            return await self.embedding_model.embed(text)
        elif hasattr(self.embedding_model, "encode"):
            return self.embedding_model.encode(text)
        elif callable(self.embedding_model):
            return await self.embedding_model(text)
        else:
            raise ValueError("Embedding model has no compatible method")
    
    async def adaptive_retrieve(
        self,
        query: str,
        tree: RAPTORTree,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Adaptive retrieval that chooses level based on query type.
        
        - Broad/conceptual queries → Higher levels (summaries)
        - Specific/detailed queries → Lower levels (leaves)
        """
        # Detect query type heuristically
        query_lower = query.lower()
        
        # Indicators of conceptual queries
        conceptual_keywords = [
            "overview", "summary", "explain", "what is", "how does",
            "general", "broadly", "overall", "main"
        ]
        
        # Indicators of specific queries
        specific_keywords = [
            "dosage", "mg", "specific", "exactly", "precise",
            "when", "how much", "what dose", "which drug"
        ]
        
        is_conceptual = any(kw in query_lower for kw in conceptual_keywords)
        is_specific = any(kw in query_lower for kw in specific_keywords)
        
        # Adjust level weights based on query type
        if is_conceptual and not is_specific:
            # Prefer higher levels
            adjusted_weights = {0: 0.4, 1: 0.7, 2: 0.9, 3: 1.0}
        elif is_specific and not is_conceptual:
            # Prefer lower levels
            adjusted_weights = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.2}
        else:
            # Balanced
            adjusted_weights = self.level_weights
        
        # Temporarily apply adjusted weights
        original_weights = self.level_weights
        self.level_weights = adjusted_weights
        
        results = await self.retrieve(query, tree, top_k)
        
        # Restore original weights
        self.level_weights = original_weights
        
        return results


class RAPTORIndexManager:
    """
    Manages RAPTOR tree storage and retrieval.
    
    Provides persistence and caching for RAPTOR trees.
    """
    
    def __init__(
        self,
        storage_path: str = "raptor_index",
        cache_size: int = 10
    ):
        """
        Initialize index manager.
        
        Args:
            storage_path: Directory for storing RAPTOR trees
            cache_size: Number of trees to keep in memory
        """
        self.storage_path = storage_path
        self.cache_size = cache_size
        self._cache: Dict[str, RAPTORTree] = {}
        
        logger.info(f"✓ RAPTORIndexManager initialized at {storage_path}")
    
    def store_tree(self, tree: RAPTORTree) -> None:
        """Store a RAPTOR tree."""
        self._cache[tree.document_id] = tree
        
        # Evict old entries if cache full
        while len(self._cache) > self.cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        logger.debug(f"Stored RAPTOR tree for document: {tree.document_id}")
    
    def get_tree(self, document_id: str) -> Optional[RAPTORTree]:
        """Retrieve a RAPTOR tree by document ID."""
        return self._cache.get(document_id)
    
    def list_trees(self) -> List[str]:
        """List all stored tree document IDs."""
        return list(self._cache.keys())
    
    def clear(self) -> None:
        """Clear all cached trees."""
        self._cache.clear()
        logger.info("Cleared RAPTOR tree cache")
