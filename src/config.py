"""Centralized configuration for the Legixo Q&A service.

All environment variables are read exactly once, here, and exposed as a
single `settings` object. Every other module imports `settings` instead of
calling `os.getenv` directly, so there is one place to see what the app
depends on and one place to validate it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_REPO_ROOT / ".env")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value or value.strip() == "":
        raise ConfigError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a float, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    
    anthropic_api_key: str
    anthropic_model: str

    openai_api_key: str
    openai_embedding_model: str

    pinecone_api_key: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_index_name: str
    pinecone_namespace: str

    corpus_dir: Path
    chunk_size_tokens: int
    chunk_overlap_tokens: int

    top_k: int
    max_loops: int
    min_similarity: float

    # API server
    api_host: str
    api_port: int

    def as_safe_dict(self) -> dict:
        """Config snapshot for logs/health checks with secrets redacted."""

        def redact(v: str) -> str:
            return v[:4] + "…redacted" if v else v

        return {
            "anthropic_model": self.anthropic_model,
            "anthropic_api_key": redact(self.anthropic_api_key),
            "openai_embedding_model": self.openai_embedding_model,
            "openai_api_key": redact(self.openai_api_key),
            "pinecone_api_key": redact(self.pinecone_api_key),
            "pinecone_cloud": self.pinecone_cloud,
            "pinecone_region": self.pinecone_region,
            "pinecone_index_name": self.pinecone_index_name,
            "pinecone_namespace": self.pinecone_namespace,
            "corpus_dir": str(self.corpus_dir),
            "chunk_size_tokens": self.chunk_size_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "top_k": self.top_k,
            "max_loops": self.max_loops,
            "min_similarity": self.min_similarity,
            "api_host": self.api_host,
            "api_port": self.api_port,
        }


def load_settings() -> Settings:
    """Read and validate all configuration. Raises ConfigError on problems."""

    chunk_size = _int("CHUNK_SIZE_TOKENS", 500)
    chunk_overlap = _int("CHUNK_OVERLAP_TOKENS", 50)
    if chunk_overlap >= chunk_size:
        raise ConfigError(
            f"CHUNK_OVERLAP_TOKENS ({chunk_overlap}) must be smaller than "
            f"CHUNK_SIZE_TOKENS ({chunk_size})"
        )

    max_loops = _int("MAX_LOOPS", 2)
    if max_loops < 1:
        raise ConfigError("MAX_LOOPS must be >= 1")

    corpus_dir = Path(os.getenv("CORPUS_DIR", "corpus"))
    if not corpus_dir.is_absolute():
        corpus_dir = _REPO_ROOT / corpus_dir

    return Settings(
        anthropic_api_key=_require("ANTHROPIC_API_KEY"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        openai_api_key=_require("OPENAI_API_KEY"),
        openai_embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        pinecone_api_key=_require("PINECONE_API_KEY"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "legixo-qa-docs"),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "default"),
        corpus_dir=corpus_dir,
        chunk_size_tokens=chunk_size,
        chunk_overlap_tokens=chunk_overlap,
        top_k=_int("TOP_K", 5),
        max_loops=max_loops,
        min_similarity=_float("MIN_SIMILARITY", 0.15),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=_int("API_PORT", 8000),
    )


# Import-time singleton. Modules do `from src.config import settings`.
# If env vars are missing, this fails fast and loudly at process start
# rather than deep inside a request handler.
settings = load_settings()