"""CLI entry point for running ingest.

Usage:
    python -m scripts.ingest_cli
    python -m scripts.ingest_cli --verbose

This is intentionally separate from the FastAPI app because the assignment
requires Q&A to be HTTP-only, while ingestion may be a separate documented
command.
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import ConfigError
from src.ingest import IngestError, run_ingest
from src.pinecone_client import PineconeIndexError


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest the document corpus into Pinecone.")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug-level logging."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    logger = logging.getLogger("ingest_cli")

    try:
        result = run_ingest()
    except ConfigError as exc:
        logger.error("Configuration problem: %s", exc)
        return 1
    except (IngestError, PineconeIndexError) as exc:
        logger.error("Ingest failed: %s", exc)
        return 1

    logger.info("Ingest complete.")
    logger.info("  Files processed:   %d", result.files_processed)
    logger.info("  Chunks created:    %d", result.chunks_created)
    logger.info("  Vectors upserted:  %d", result.vectors_upserted)
    if result.skipped_empty_files:
        logger.info("  Skipped (empty):   %s", result.skipped_empty_files)
    return 0


if __name__ == "__main__":
    sys.exit(main())