"""Run the full ingestion pipeline: fetch -> parse -> chunk -> embed & store.

Usage: python -m ingest [--force]
"""

import argparse
import logging
from pathlib import Path

from config import settings
from ingest.fetch import PARTS, TITLE, fetch_all_parts
from ingest.parse import parse_all
from ingest.chunk import chunk_all
from ingest.embed_store import store_chunks

logger = logging.getLogger(__name__)


def run(force: bool = False) -> None:
    raw_paths = fetch_all_parts(Path("data/raw"), parts=PARTS, force=force)
    logger.info("Fetched %d parts of Title %d", len(raw_paths), TITLE)

    sections = parse_all(raw_paths)
    logger.info("Parsed %d sections", len(sections))

    chunks = chunk_all(sections)
    logger.info("Built %d chunks", len(chunks))

    store_chunks(
        chunks,
        persist_dir=Path(settings.chroma_dir),
        collection_name=settings.chroma_collection,
        model=settings.embedding_model,
    )
    logger.info("Ingestion complete: %d chunks in collection %r", len(chunks), settings.chroma_collection)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest HIPAA regulation text into Chroma")
    parser.add_argument("--force", action="store_true", help="Re-fetch parts even if already cached")
    args = parser.parse_args()
    run(force=args.force)
