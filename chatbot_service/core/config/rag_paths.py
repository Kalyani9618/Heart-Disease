"""
RAG Paths Configuration

Centralized filesystem path management for the RAG system.
All file paths go through this module to ensure:
- Consistency across all services
- Environment-based overrides
- Automatic directory creation
- Path validation


The path hierarchy:
BASE_DIR (project root, typically where main.py is)
├── data/                      (JSON data files, dictionaries)
│   ├── drugs.json
│   ├── guidelines.json
│   ├── symptoms.json
│   ├── dictionaries/          (text-based dictionaries)
│   │   └── common_drugs.txt
│   └── fixtures/              (test fixtures, mock data)
├── chroma_db/                 (Vector database)
├── models/                    (ONNX, PyTorch models)
├── logs/                      (Application logs)
└── memori/                    (Legacy memory system)

Environment Variables:
- RAG_BASE_DIR: Override project root (default: project root from __file__)
- RAG_DATA_DIR: Override data directory
- RAG_MODELS_DIR: Override models directory
- RAG_LOGS_DIR: Override logs directory
"""

import os
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class PathConfig:
    """
    Manages all filesystem paths for the RAG system.
    
    Provides:
    - Centralized path resolution
    - Environment-based overrides
    - Automatic directory creation
    - Path validation
    """
    
    def __init__(self, base_dir: Optional[str] = None, auto_create: bool = True):
        """
        Initialize path configuration.
        
        Args:
            base_dir: Base directory for all data. If None, uses project root.
                      Can also be set via RAG_BASE_DIR environment variable.
            auto_create: Automatically create directories if they don't exist.
                         Default True.
        
        Example:
            # Use default (project root)
            paths = PathConfig()
            
            # Use custom base directory
            paths = PathConfig(base_dir="/custom/path")
            
            # Use environment variable
            os.environ["RAG_BASE_DIR"] = "/custom/path"
            paths = PathConfig()
        """
        self.auto_create = auto_create
        
        # Determine base directory with priority:
        # 1. Explicit parameter
        # 2. RAG_BASE_DIR environment variable
        # 3. Project root (parent of core/ directory)
        if base_dir:
            self.base_dir = Path(base_dir).resolve()
        elif os.environ.get("RAG_BASE_DIR"):
            self.base_dir = Path(os.environ["RAG_BASE_DIR"]).resolve()
        else:
            # Default: parent of core directory
            # __file__ = core/config/rag_paths.py
            # parent = core/config
            # parent.parent = core
            # parent.parent.parent = project root
            self.base_dir = Path(__file__).parent.parent.parent.resolve()
        
        logger.info(f"PathConfig initialized with base_dir: {self.base_dir}")
        
        # Validate base directory exists
        if not self.base_dir.exists():
            if self.auto_create:
                self.base_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created base directory: {self.base_dir}")
            else:
                logger.warning(f"Base directory does not exist: {self.base_dir}")
    
    # ========== DIRECTORY PROPERTIES ==========
    
    @property
    def data_dir(self) -> Path:
        """Get data directory (JSON files, dictionaries)."""
        return self._ensure_dir(
            os.environ.get("RAG_DATA_DIR") or self.base_dir / "data"
        )
    
    @property
    def dictionaries_dir(self) -> Path:
        """Get dictionaries directory (common_drugs.txt, etc.)."""
        return self._ensure_dir(self.data_dir / "dictionaries")
    
    @property
    def fixtures_dir(self) -> Path:
        """Get fixtures directory (test data, mock data)."""
        return self._ensure_dir(self.data_dir / "fixtures")
    
    @property
    def chroma_db_dir(self) -> Path:
        """Get ChromaDB persistence directory."""
        return self._ensure_dir(
            os.environ.get("CHROMADB_DIR") or self.base_dir / "Chromadb"
        )
    
    @property
    def models_dir(self) -> Path:
        """Get models directory (ONNX, PyTorch models)."""
        return self._ensure_dir(
            os.environ.get("RAG_MODELS_DIR") or self.base_dir / "models"
        )
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory (application logs)."""
        return self._ensure_dir(
            os.environ.get("RAG_LOGS_DIR") or self.base_dir / "logs"
        )
    
    @property
    def memori_dir(self) -> Path:
        """Get Memori legacy memory system directory."""
        return self._ensure_dir(self.base_dir / "memori")
    
    # ========== FILE PATH METHODS ==========
    
    def get_drugs_file(self) -> Path:
        """Get path to drugs.json."""
        return self.data_dir / "drugs.json"
    
    def get_guidelines_file(self) -> Path:
        """Get path to guidelines.json."""
        return self.data_dir / "guidelines.json"
    
    def get_symptoms_file(self) -> Path:
        """Get path to symptoms.json."""
        return self.data_dir / "symptoms.json"
    
    def get_drug_dictionary_file(self) -> Path:
        """Get path to drug dictionary text file."""
        return self.dictionaries_dir / "common_drugs.txt"
    
    def get_chroma_db_path(self) -> Path:
        """Get path to ChromaDB persistence directory."""
        return self.chroma_db_dir
    
    def get_onnx_models_dir(self) -> Path:
        """Get directory for ONNX models."""
        return self._ensure_dir(self.models_dir / "onnx")
    
    def get_pytorch_models_dir(self) -> Path:
        """Get directory for PyTorch models."""
        return self._ensure_dir(self.models_dir / "pytorch")
    
    # ========== VALIDATION & UTILITY ==========
    
    def validate_all_paths(self) -> Dict[str, bool]:
        """
        Validate all required directories exist.
        
        Returns:
            Dictionary mapping directory names to existence status.
            
        Example:
            paths = PathConfig()
            results = paths.validate_all_paths()
            if not all(results.values()):
                logger.error("Some required paths are missing!")
        """
        results = {
            "base_dir": self.base_dir.exists(),
            "data_dir": self.data_dir.exists(),
            "dictionaries_dir": self.dictionaries_dir.exists(),
            "chroma_db_dir": self.chroma_db_dir.exists(),
            "models_dir": self.models_dir.exists(),
            "logs_dir": self.logs_dir.exists(),
            "memori_dir": self.memori_dir.exists(),
        }
        
        # Log validation results
        for path_name, exists in results.items():
            status = "✓" if exists else "✗"
            logger.debug(f"{status} {path_name}: {getattr(self, path_name.replace('_dir', '_dir'), 'N/A')}")
        
        return results
    
    def validate_required_files(self) -> Dict[str, bool]:
        """
        Validate all required data files exist.
        
        Returns:
            Dictionary mapping file names to existence status.
        """
        results = {
            "drugs.json": self.get_drugs_file().exists(),
            "guidelines.json": self.get_guidelines_file().exists(),
            "symptoms.json": self.get_symptoms_file().exists(),
            "common_drugs.txt": self.get_drug_dictionary_file().exists(),
        }
        
        for filename, exists in results.items():
            status = "✓" if exists else "✗"
            logger.debug(f"{status} {filename}")
        
        return results
    
    def _ensure_dir(self, path: Path) -> Path:
        """
        Ensure directory exists, creating if necessary.
        
        Args:
            path: Path to ensure
            
        Returns:
            Resolved Path object
        """
        path = Path(path).resolve()
        if not path.exists() and self.auto_create:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {path}")
        return path
    
    def __repr__(self) -> str:
        """String representation of path configuration."""
        return (
            f"PathConfig(base_dir={self.base_dir}, "
            f"data_dir={self.data_dir}, "
            f"models_dir={self.models_dir})"
        )
    
    def to_dict(self) -> Dict[str, str]:
        """
        Convert all paths to dictionary (useful for logging/debugging).
        
        Returns:
            Dictionary mapping path names to string paths.
        """
        return {
            "base_dir": str(self.base_dir),
            "data_dir": str(self.data_dir),
            "dictionaries_dir": str(self.dictionaries_dir),
            "fixtures_dir": str(self.fixtures_dir),
            "chroma_db_dir": str(self.chroma_db_dir),
            "models_dir": str(self.models_dir),
            "onnx_models_dir": str(self.get_onnx_models_dir()),
            "pytorch_models_dir": str(self.get_pytorch_models_dir()),
            "logs_dir": str(self.logs_dir),
            "memori_dir": str(self.memori_dir),
            "drugs_file": str(self.get_drugs_file()),
            "guidelines_file": str(self.get_guidelines_file()),
            "symptoms_file": str(self.get_symptoms_file()),
            "drug_dictionary_file": str(self.get_drug_dictionary_file()),
        }


# Singleton instance (thread-safe)
_global_path_config: Optional[PathConfig] = None
_path_config_lock = __import__("threading").Lock()


def get_path_config(base_dir: Optional[str] = None) -> PathConfig:
    """
    Get or create the global PathConfig singleton.
    
    Args:
        base_dir: Base directory for first initialization (ignored on subsequent calls).
                  Use RAG_BASE_DIR environment variable for subsequent calls.
    
    Returns:
        Global PathConfig instance
        
    Example:
        # First call creates singleton
        paths = get_path_config()
        
        # Subsequent calls return same instance
        paths2 = get_path_config()
        assert paths is paths2
    """
    global _global_path_config
    
    if _global_path_config is None:
        with _path_config_lock:
            if _global_path_config is None:
                _global_path_config = PathConfig(base_dir=base_dir)
                logger.info("PathConfig singleton created")
    
    return _global_path_config


def reset_path_config() -> None:
    """
    Reset the global PathConfig singleton (for testing only).
    
    Example:
        reset_path_config()
        paths = get_path_config(base_dir="/test/path")
    """
    global _global_path_config
    _global_path_config = None
    logger.debug("PathConfig singleton reset")
