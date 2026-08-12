"""Token-aware text chunking.

Splits document text into overlapping chunks measured in tokens (not
characters), using tiktoken so chunk sizes are meaningful regardless of
how dense the source markdown is.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

# cl100k_base is the tokenizer used by embedding and chat models in this
# project's family; good enough as a size estimate even if the exact
# embedding model's tokenizer differs slightly.
_ENCODING_NAME = "cl100k_base"


class TokenEncoding(Protocol):
    """Minimal interface chunk_text needs. Lets tests inject a fake
    tokenizer instead of hitting tiktoken's network-fetched BPE vocab."""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


@lru_cache(maxsize=1)
def _default_encoding() -> TokenEncoding:
    # Imported lazily (not at module load) so importing this module never
    # requires network access -- only calling chunk_text() without an
    # explicit `encoding=` does, on its first call.
    import tiktoken

    return tiktoken.get_encoding(_ENCODING_NAME)


@dataclass
class Chunk:
    text: str
    chunk_index: int


def chunk_text(
    text: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    encoding: TokenEncoding | None = None,
) -> list[Chunk]:
    """Split `text` into overlapping chunks of ~chunk_size_tokens tokens.

    Raises ValueError if parameters are nonsensical, since a silent bad
    split is worse than a loud failure at ingest time.

    `encoding` defaults to tiktoken's cl100k_base; tests may inject a
    different TokenEncoding to avoid network calls.
    """
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if chunk_overlap_tokens < 0 or chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be >= 0 and < chunk_size_tokens")

    enc = encoding or _default_encoding()
    tokens = enc.encode(text)
    if not tokens:
        return []

    stride = chunk_size_tokens - chunk_overlap_tokens
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(tokens):
        window = tokens[start : start + chunk_size_tokens]
        chunk_text_str = enc.decode(window).strip()
        if chunk_text_str:
            chunks.append(Chunk(text=chunk_text_str, chunk_index=index))
            index += 1
        if start + chunk_size_tokens >= len(tokens):
            break
        start += stride

    return chunks