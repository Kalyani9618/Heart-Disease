"""
Centralized Configuration for Colab-based Multi-Modal RAG System

All settings for models, dimensions, paths, server, and ngrok integration.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def _env(key: str, default, cast=str):
    """Read env var with type casting."""
    val = os.getenv(key, default)
    if cast == bool:
        return str(val).lower() in ("true", "1", "yes")
    return cast(val)


@dataclass
class ModelConfig:
    """Model names and embedding dimensions."""

    # ---------- Drive root for local model copies ----------
    drive_root: str = "/content/drive/MyDrive/IDP/Medical"

    # Text embedding model (MedCPT) — local copy on Drive
    text_model_hf: str = "ncbi/MedCPT-Query-Encoder"      # HuggingFace fallback
    text_article_model_hf: str = "ncbi/MedCPT-Article-Encoder"
    text_model_local: str = "MedCPT-Query-Encoder"         # folder name on Drive
    text_embedding_dim: int = 768

    # Image embedding model (SigLIP) — local copy on Drive
    image_model_hf: str = "google/siglip-so400m-patch14-384"  # HuggingFace fallback
    image_model_local: str = "siglip-so400m"                  # folder name on Drive
    image_embedding_dim: int = 1152

    # Generation model (MedGemma 1.5)
    generation_model_hf: str = "google/medgemma-1.5-4b-it"  # HuggingFace ID for download
    generation_model_local: str = "medgemma-1.5-4b-it"      # folder name on Drive
    generation_max_tokens: int = 2000
    generation_temperature: float = 0.3

    @property
    def text_model_path(self) -> str:
        """Resolved path: local Drive copy first, then HuggingFace ID."""
        local = os.path.join(self.drive_root, self.text_model_local)
        return local if os.path.isdir(local) else self.text_model_hf

    @property
    def text_article_model_path(self) -> str:
        # Article encoder may not be stored locally; use HF
        return self.text_article_model_hf

    @property
    def image_model_path(self) -> str:
        local = os.path.join(self.drive_root, self.image_model_local)
        return local if os.path.isdir(local) else self.image_model_hf

    @property
    def generation_model_path(self) -> str:
        local = os.path.join(self.drive_root, self.generation_model_local)
        return local if os.path.isdir(local) else self.generation_model_hf


@dataclass
class DriveConfig:
    """Google Drive paths for data sources."""

    # Root data folder on Google Drive
    drive_root: str = "/content/drive/MyDrive/IDP/Medical"

    # Sub-paths (relative to drive_root)
    books_dir: str = "Books"               # PDF medical textbooks
    json_dir: str = "Books/JsonData"       # JSON datasets
    code_dir: str = "Code"                 # Code directory
    output_dir: str = "output"             # Parsed output cache
    chromadb_dir: str = "Chromadb"         # Persisted ChromaDB

    @property
    def books_path(self) -> str:
        return os.path.join(self.drive_root, self.books_dir)

    @property
    def json_path(self) -> str:
        return os.path.join(self.drive_root, self.json_dir)

    @property
    def code_path(self) -> str:
        return os.path.join(self.drive_root, self.code_dir)

    @property
    def output_path(self) -> str:
        return os.path.join(self.drive_root, self.output_dir)

    @property
    def chromadb_path(self) -> str:
        return os.path.join(self.drive_root, self.chromadb_dir)


@dataclass
class ServerConfig:
    """Flask server and ngrok settings."""

    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False

    # Ngrok
    ngrok_auth_token: str = field(
        default_factory=lambda: _env("NGROK_AUTH_TOKEN", "")
    )
    ngrok_region: str = "us"

    # Request limits
    max_batch_size: int = 32
    request_timeout: int = 120  # seconds


@dataclass
class ChunkingConfig:
    """Text chunking parameters for ingestion."""

    chunk_size: int = 1500
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    separator: str = "\n\n"


@dataclass
class ColabConfig:
    """Top-level configuration aggregating all sub-configs."""

    models: ModelConfig = field(default_factory=ModelConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    # Collection names for dual-index vector store
    text_collection: str = "medical_text_768"
    image_collection: str = "medical_images_1152"

    # Logging
    log_level: str = field(
        default_factory=lambda: _env("LOG_LEVEL", "INFO")
    )

    def validate(self) -> bool:
        """Check required settings are present."""
        issues = []
        if not self.server.ngrok_auth_token:
            issues.append("NGROK_AUTH_TOKEN not set (ngrok may not work)")
        if issues:
            import logging
            for issue in issues:
                logging.getLogger(__name__).warning(issue)
        return len(issues) == 0


# Singleton default config
_default_config: Optional[ColabConfig] = None


def get_config() -> ColabConfig:
    """Get or create the default ColabConfig singleton."""
    global _default_config
    if _default_config is None:
        _default_config = ColabConfig()
    return _default_config
