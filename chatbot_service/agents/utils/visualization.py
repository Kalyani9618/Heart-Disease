"""
Graph Visualization - Visualize agent workflow graphs.

Provides:
- GraphVisualizer: Generate Mermaid diagrams from LangGraph
- to_mermaid: Convert graph to Mermaid syntax
- save_diagram: Export to PNG or markdown

Based on LangGraph visualization patterns.
"""
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)



@dataclass
class GraphNode:
    """A node in the graph."""
    name: str
    node_type: str = "default"  # "start", "end", "worker", "supervisor", "conditional"
    description: Optional[str] = None
    
    def to_mermaid(self) -> str:
        """Convert to Mermaid node syntax."""
        if self.node_type == "start":
            return f"    {self.name}(({self.name}))"
        elif self.node_type == "end":
            return f"    {self.name}[/{self.name}/]"
        elif self.node_type == "supervisor":
            return f"    {self.name}{{{{{self.name}}}}}"
        elif self.node_type == "conditional":
            return f"    {self.name}{{{self.name}}}"
        else:
            return f"    {self.name}[{self.name}]"


@dataclass
class GraphEdge:
    """An edge in the graph."""
    source: str
    target: str
    label: Optional[str] = None
    conditional: bool = False
    
    def to_mermaid(self) -> str:
        """Convert to Mermaid edge syntax."""
        arrow = "-.->" if self.conditional else "-->"
        if self.label:
            return f"    {self.source} {arrow}|{self.label}| {self.target}"
        return f"    {self.source} {arrow} {self.target}"


class GraphVisualizer:
    """
    Visualize agent workflow graphs.
    
    Supports:
    - LangGraph StateGraph
    - Custom graph definitions
    - Mermaid diagram generation
    - PNG export (with mermaid-cli)
    
    Usage:
        viz = GraphVisualizer()
        
        # From LangGraph
        mermaid = viz.from_langgraph(compiled_graph)
        
        # Save as PNG (requires mermaid-cli)
        viz.save_diagram(mermaid, "agent_graph.png")
        
        # Or save as markdown
        viz.save_markdown(mermaid, "agent_graph.md")
    """
    
    # Styles for different node types
    NODE_STYLES = {
        "supervisor": "fill:#f9f,stroke:#333,stroke-width:2px",
        "worker": "fill:#bbf,stroke:#333,stroke-width:1px",
        "router": "fill:#fbb,stroke:#333,stroke-width:1px",
        "start": "fill:#bfb,stroke:#333,stroke-width:2px",
        "end": "fill:#fbf,stroke:#333,stroke-width:2px",
    }
    
    def __init__(self, theme: str = "default"):
        """
        Initialize visualizer.
        
        Args:
            theme: Mermaid theme ("default", "dark", "neutral")
        """
        self.theme = theme
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
    
    def from_langgraph(self, graph) -> str:
        """
        Generate Mermaid diagram from LangGraph StateGraph.
        
        Args:
            graph: Compiled LangGraph StateGraph
            
        Returns:
            Mermaid diagram string
        """
        self.nodes = {}
        self.edges = []
        
        try:
            # Try to use built-in method first
            if hasattr(graph, 'get_graph'):
                internal_graph = graph.get_graph()
                
                # Extract nodes
                for node_name in internal_graph.nodes:
                    node_type = self._infer_node_type(node_name)
                    self.nodes[node_name] = GraphNode(
                        name=node_name,
                        node_type=node_type
                    )
                
                # Extract edges
                if hasattr(internal_graph, 'edges'):
                    for edge in internal_graph.edges:
                        if isinstance(edge, tuple) and len(edge) >= 2:
                            source, target = edge[0], edge[1]
                            self.edges.append(GraphEdge(
                                source=source,
                                target=target
                            ))
            else:
                # Fallback: extract from graph attributes
                if hasattr(graph, 'nodes'):
                    for node_name in graph.nodes:
                        self.nodes[node_name] = GraphNode(name=node_name)
                
                if hasattr(graph, '_edges'):
                    for source, targets in graph._edges.items():
                        for target in (targets if isinstance(targets, list) else [targets]):
                            self.edges.append(GraphEdge(source=source, target=target))
        
        except Exception as e:
            logger.error(f"Could not extract graph structure: {e}")
        
        return self.to_mermaid()
    
    def from_dict(self, graph_dict: Dict[str, Any]) -> str:
        """
        Generate Mermaid from dictionary definition.
        
        Args:
            graph_dict: {"nodes": [{"name": "...", "type": "..."}], 
                        "edges": [{"from": "...", "to": "...", "label": "..."}]}
        
        Returns:
            Mermaid diagram string
        """
        self.nodes = {}
        self.edges = []
        
        for node in graph_dict.get("nodes", []):
            self.nodes[node["name"]] = GraphNode(
                name=node["name"],
                node_type=node.get("type", "default"),
                description=node.get("description")
            )
        
        for edge in graph_dict.get("edges", []):
            self.edges.append(GraphEdge(
                source=edge["from"],
                target=edge["to"],
                label=edge.get("label"),
                conditional=edge.get("conditional", False)
            ))
        
        return self.to_mermaid()
    
    def to_mermaid(self, direction: str = "TD") -> str:
        """
        Generate Mermaid diagram syntax.
        
        Args:
            direction: Graph direction ("TD", "LR", "BT", "RL")
            
        Returns:
            Mermaid diagram string
        """
        lines = [f"graph {direction}"]
        
        # Add nodes
        for node in self.nodes.values():
            lines.append(node.to_mermaid())
        
        lines.append("")  # Blank line
        
        # Add edges
        for edge in self.edges:
            lines.append(edge.to_mermaid())
        
        lines.append("")  # Blank line
        
        # Add styles
        lines.append("    %% Styles")
        for node_name, node in self.nodes.items():
            if node.node_type in self.NODE_STYLES:
                style_name = f"style_{node.node_type}"
                lines.append(f"    style {node_name} {self.NODE_STYLES[node.node_type]}")
        
        return "\n".join(lines)
    
    def _infer_node_type(self, name: str) -> str:
        """Infer node type from name."""
        name_lower = name.lower()
        
        if name_lower in ["__start__", "start"]:
            return "start"
        elif name_lower in ["__end__", "end"]:
            return "end"
        elif "supervisor" in name_lower:
            return "supervisor"
        elif "router" in name_lower:
            return "conditional"
        else:
            return "worker"
    
    def save_diagram(self, mermaid: str, filepath: str = "graph.png") -> Optional[str]:
        """
        Save diagram to file (PNG or markdown).
        
        Args:
            mermaid: Mermaid diagram string
            filepath: Output file path
            
        Returns:
            Filepath if successful, None otherwise
        """
        if filepath.endswith('.png'):
            return self._save_png(mermaid, filepath)
        else:
            return self._save_markdown(mermaid, filepath)
    
    def _save_png(self, mermaid: str, filepath: str) -> Optional[str]:
        """Save as PNG using mermaid-cli."""
        try:
            import subprocess
            import tempfile
            import os
            
            # Write mermaid to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
                f.write(mermaid)
                temp_path = f.name
            
            try:
                # Use mmdc (mermaid-cli) if available
                result = subprocess.run(
                    ['mmdc', '-i', temp_path, '-o', filepath],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info(f"Saved diagram to {filepath}")
                    return filepath
                else:
                    logger.warning(f"mmdc failed: {result.stderr}")
                    # Fallback to markdown
                    return self._save_markdown(mermaid, filepath.replace('.png', '.md'))
            finally:
                os.unlink(temp_path)
                
        except FileNotFoundError:
            logger.warning("mermaid-cli not installed. Install with: npm install -g @mermaid-js/mermaid-cli")
            return self._save_markdown(mermaid, filepath.replace('.png', '.md'))
        except Exception as e:
            logger.error(f"Failed to save PNG: {e}")
            return None
    
    def _save_markdown(self, mermaid: str, filepath: str) -> str:
        """Save as markdown with mermaid code block."""
        if not filepath.endswith('.md'):
            filepath = filepath.rsplit('.', 1)[0] + '.md'
        
        content = f"""# Agent Workflow Graph

```mermaid
{mermaid}
```

## Nodes

{self._generate_node_table()}

## Legend

- **Supervisor** (diamond): Decision-making node
- **Worker** (rectangle): Task execution node
- **Router** (hexagon): Conditional routing
- **Start/End** (rounded): Entry/exit points
"""
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        logger.info(f"Saved diagram to {filepath}")
        return filepath
    
    def _generate_node_table(self) -> str:
        """Generate markdown table of nodes."""
        lines = ["| Node | Type | Description |", "|------|------|-------------|"]
        for node in self.nodes.values():
            desc = node.description or "-"
            lines.append(f"| {node.name} | {node.node_type} | {desc} |")
        return "\n".join(lines)


# Pre-defined graph for our orchestrator
def get_orchestrator_graph() -> str:
    """Get Mermaid diagram for our LangGraph orchestrator."""
    viz = GraphVisualizer()
    
    graph_def = {
        "nodes": [
            {"name": "START", "type": "start"},
            {"name": "router", "type": "conditional", "description": "SemanticRouterV2 intent classification"},
            {"name": "supervisor", "type": "supervisor", "description": "LLM-based task delegation"},
            {"name": "medical_analyst", "type": "worker", "description": "RAG for medical queries"},
            {"name": "researcher", "type": "worker", "description": "Web search for information"},
            {"name": "data_analyst", "type": "worker", "description": "SQL queries for vitals"},
            {"name": "drug_expert", "type": "worker", "description": "Drug interactions via GraphRAG"},
            {"name": "profile_manager", "type": "worker", "description": "User profile management"},
            {"name": "END", "type": "end"},
        ],
        "edges": [
            {"from": "START", "to": "router"},
            {"from": "router", "to": "supervisor", "label": "complex", "conditional": True},
            {"from": "router", "to": "medical_analyst", "label": "medical", "conditional": True},
            {"from": "router", "to": "data_analyst", "label": "vitals", "conditional": True},
            {"from": "supervisor", "to": "medical_analyst"},
            {"from": "supervisor", "to": "researcher"},
            {"from": "supervisor", "to": "data_analyst"},
            {"from": "supervisor", "to": "drug_expert"},
            {"from": "supervisor", "to": "profile_manager"},
            {"from": "medical_analyst", "to": "supervisor"},
            {"from": "researcher", "to": "supervisor"},
            {"from": "data_analyst", "to": "supervisor"},
            {"from": "drug_expert", "to": "supervisor"},
            {"from": "profile_manager", "to": "supervisor"},
            {"from": "supervisor", "to": "END", "label": "FINISH", "conditional": True},
        ]
    }
    
    return viz.from_dict(graph_def)
