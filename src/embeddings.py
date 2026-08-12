"""OpenAI embeddings client, isolated behind a small function.

Kept separate from ingest.py and graph.py so both can call `embed_texts`
without knowing which provider or batching strategy is behind it.
"""

from __future__ import annotations

import logging

from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=settings.openai_api_key)

# OpenAI's embeddings endpoint accepts up to 2048 inputs per call, but we
# batch conservatively to keep request payloads and retry blast-radius small.
_BATCH_SIZE = 64


class EmbeddingError(RuntimeError):
    """Raised when the embeddings API call fails or returns unexpected data."""


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings, preserving input order.

    Raises EmbeddingError on API failure rather than letting a raw SDK
    exception propagate, so callers get a consistent error type.
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        try:
            response = _client.embeddings.create(
                model=settings.openai_embedding_model,
                input=batch,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"OpenAI embeddings call failed for batch starting at index {i}: {exc}"
            ) from exc

        if len(response.data) != len(batch):
            raise EmbeddingError(
                f"Expected {len(batch)} embeddings, got {len(response.data)}"
            )
        # response.data is returned in the same order as the input batch.
        all_embeddings.extend(item.embedding for item in response.data)

    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Convenience wrapper for embedding a single query string."""
    return embed_texts([text])[0]