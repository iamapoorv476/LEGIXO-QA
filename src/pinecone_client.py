"""Thin wrapper around the Pinecone Python client.

Keeps all Pinecone-specific calls (index creation, upsert, query) in one
place so the rest of the codebase (ingest, graph nodes) talks to a small,
typed interface instead of the raw SDK. Makes it easy to see exactly what
"talking to Pinecone" means for this project, and easy to mock in tests.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from pinecone import Pinecone, ServerlessSpec

from src.config import Settings, settings

logger = logging.getLogger(__name__)

# Must match the embedding model's output dimension.
# text-embedding-3-small -> 1536, text-embedding-3-large -> 3072
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


@dataclass
class ScoredChunk:
    """A single retrieved chunk with its similarity score and metadata."""

    chunk_id: str
    text: str
    source_file: str
    chunk_index: int
    score: float


class PineconeIndexError(RuntimeError):
    """Raised when a Pinecone operation fails in a way callers should see."""


class PineconeClient:
    """Wraps index creation, upsert, and query for the QA corpus index."""

    def __init__(self, cfg: Settings = settings):
        self._cfg = cfg
        self._pc = Pinecone(api_key=cfg.pinecone_api_key)
        self._index = None  

    def _embedding_dim(self) -> int:
        dim = EMBEDDING_DIMENSIONS.get(self._cfg.openai_embedding_model)
        if dim is None:
            raise PineconeIndexError(
                f"Unknown embedding model {self._cfg.openai_embedding_model!r}; "
                f"add its dimension to EMBEDDING_DIMENSIONS in pinecone_client.py"
            )
        return dim

    def ensure_index_exists(self, wait_seconds: float = 1.0, max_wait: float = 60.0) -> None:
        """Create the index if it doesn't exist yet. Idempotent — safe to call
        every time the app starts."""

        existing = {idx["name"] for idx in self._pc.list_indexes()}
        if self._cfg.pinecone_index_name in existing:
            logger.info("Pinecone index %r already exists.", self._cfg.pinecone_index_name)
            return

        logger.info(
            "Creating Pinecone index %r (dim=%d, cloud=%s, region=%s)...",
            self._cfg.pinecone_index_name,
            self._embedding_dim(),
            self._cfg.pinecone_cloud,
            self._cfg.pinecone_region,
        )
        self._pc.create_index(
            name=self._cfg.pinecone_index_name,
            dimension=self._embedding_dim(),
            metric="cosine",
            spec=ServerlessSpec(
                cloud=self._cfg.pinecone_cloud,
                region=self._cfg.pinecone_region,
            ),
        )

        # Serverless index creation is async; poll until ready or time out.
        waited = 0.0
        while waited < max_wait:
            desc = self._pc.describe_index(self._cfg.pinecone_index_name)
            if desc.status.ready:
                logger.info("Index %r is ready.", self._cfg.pinecone_index_name)
                return
            time.sleep(wait_seconds)
            waited += wait_seconds

        raise PineconeIndexError(
            f"Index {self._cfg.pinecone_index_name!r} did not become ready "
            f"within {max_wait}s."
        )

    @property
    def index(self):
        if self._index is None:
            self._index = self._pc.Index(self._cfg.pinecone_index_name)
        return self._index

    def upsert_chunks(
        self,
        vectors: list[tuple[str, list[float], dict]],
        batch_size: int = 100,
    ) -> int:
        """Upsert (id, embedding, metadata) tuples in batches.

        Returns the total number of vectors upserted. Deterministic IDs
        (see ingest.py) mean re-running this with the same input is a clean
        overwrite, not a duplicate — see README "Idempotent ingest".
        """
        total = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            try:
                self.index.upsert(vectors=batch, namespace=self._cfg.pinecone_namespace)
            except Exception as exc:  
                raise PineconeIndexError(f"Upsert failed for batch starting at {i}: {exc}") from exc
            total += len(batch)
        return total

    def query(self, embedding: list[float], top_k: int | None = None) -> list[ScoredChunk]:
        """Query the index and return scored chunks, best match first."""
        k = top_k or self._cfg.top_k
        try:
            result = self.index.query(
                vector=embedding,
                top_k=k,
                include_metadata=True,
                namespace=self._cfg.pinecone_namespace,
            )
        except Exception as exc:
            raise PineconeIndexError(f"Query failed: {exc}") from exc

        chunks: list[ScoredChunk] = []
        for match in result.get("matches", []):
            meta = match.get("metadata", {}) or {}
            chunks.append(
                ScoredChunk(
                    chunk_id=match["id"],
                    text=meta.get("text", ""),
                    source_file=meta.get("source_file", "unknown"),
                    chunk_index=meta.get("chunk_index", -1),
                    score=match.get("score", 0.0),
                )
            )
        return chunks

    def stats(self) -> dict:
        """Vector count etc. Used by ingest's idempotency check and /health."""
        try:
            return self.index.describe_index_stats()
        except Exception as exc:
            raise PineconeIndexError(f"describe_index_stats failed: {exc}") from exc