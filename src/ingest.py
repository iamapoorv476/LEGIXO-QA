"""Ingest pipeline: corpus files -> chunks -> embeddings -> Pinecone.

Idempotency strategy (see README "Pinecone checklist" for the write-up):
point IDs are deterministic, derived from `sha256(source_file:chunk_index)`.
Re-running ingest on an unchanged corpus therefore overwrites the same
vector IDs rather than creating duplicates. If a source file's content
changes, its chunk boundaries may shift, which can leave stale chunks
behind under old IDs — see README for how this is handled.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from src.chunking import chunk_text
from src.config import Settings, settings
from src.embeddings import embed_texts
from src.pinecone_client import PineconeClient

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".txt"}


class IngestError(RuntimeError):
    """Raised when the ingest pipeline cannot complete."""


@dataclass
class IngestResult:
    files_processed: int
    chunks_created: int
    vectors_upserted: int
    skipped_empty_files: list[str]


def _deterministic_chunk_id(source_file: str, chunk_index: int) -> str:
    """Stable ID for a chunk, so re-ingesting overwrites cleanly.

    Uses the file's path relative to the corpus dir (not absolute path)
    so ingest is reproducible across machines/checkouts.
    """
    key = f"{source_file}:{chunk_index}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return f"chunk_{digest}"


def _discover_files(corpus_dir: Path) -> list[Path]:
    if not corpus_dir.exists():
        raise IngestError(f"Corpus directory does not exist: {corpus_dir}")
    files = sorted(
        p for p in corpus_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise IngestError(
            f"No files with extensions {SUPPORTED_EXTENSIONS} found in {corpus_dir}"
        )
    return files


def run_ingest(cfg: Settings = settings, pinecone_client: PineconeClient | None = None) -> IngestResult:
    """Run the full ingest pipeline. Safe to call repeatedly (idempotent)."""

    client = pinecone_client or PineconeClient(cfg)
    client.ensure_index_exists()

    files = _discover_files(cfg.corpus_dir)
    logger.info("Found %d source file(s) in %s", len(files), cfg.corpus_dir)

    all_ids: list[str] = []
    all_texts: list[str] = []
    all_metadata: list[dict] = []
    skipped_empty: list[str] = []

    for path in files:
        relative_path = str(path.relative_to(cfg.corpus_dir))
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        if not raw_text.strip():
            logger.warning("Skipping empty file: %s", relative_path)
            skipped_empty.append(relative_path)
            continue

        chunks = chunk_text(raw_text, cfg.chunk_size_tokens, cfg.chunk_overlap_tokens)
        logger.info("  %s -> %d chunk(s)", relative_path, len(chunks))

        for chunk in chunks:
            chunk_id = _deterministic_chunk_id(relative_path, chunk.chunk_index)
            all_ids.append(chunk_id)
            all_texts.append(chunk.text)
            all_metadata.append(
                {
                    "text": chunk.text,
                    "source_file": relative_path,
                    "chunk_index": chunk.chunk_index,
                }
            )

    if not all_texts:
        raise IngestError(
            "All discovered files were empty; nothing to ingest. "
            f"Skipped: {skipped_empty}"
        )

    logger.info("Embedding %d chunk(s) via %s...", len(all_texts), cfg.openai_embedding_model)
    embeddings = embed_texts(all_texts)

    vectors = list(zip(all_ids, embeddings, all_metadata))
    logger.info("Upserting %d vector(s) into Pinecone index %r...", len(vectors), cfg.pinecone_index_name)
    upserted = client.upsert_chunks(vectors)

    return IngestResult(
        files_processed=len(files) - len(skipped_empty),
        chunks_created=len(all_texts),
        vectors_upserted=upserted,
        skipped_empty_files=skipped_empty,
    )