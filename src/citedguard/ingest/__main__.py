"""Run the full ingestion pipeline: fetch -> parse -> chunk -> embed & store.

Usage: python -m citedguard.ingest [--force]
"""

import argparse
import logging
from pathlib import Path

from citedguard.config import settings
from citedguard.ingest.fetch import PARTS, TITLE, fetch_all_parts
from citedguard.ingest.parse import parse_all
from citedguard.ingest.chunk import chunk_all
from citedguard.ingest.embed_store import store_chunks
from citedguard.locks import INGESTION_LOCK, try_lock

logger = logging.getLogger(__name__)


def run(force: bool = False) -> bool:
    """Ingest the corpus, unless another run already holds the ingestion lock.

    Returns True if ingestion actually ran, False if it was skipped because a
    run was already in flight (see locks.py for why skipping beats waiting).
    """
    with try_lock(INGESTION_LOCK) as acquired:
        if not acquired:
            logger.warning("Another ingestion is already in progress -- skipping this run")
            return False

        issue_date, raw_paths = fetch_all_parts(Path("data/raw"), parts=PARTS, force=force)
        logger.info("Fetched %d parts of Title %d as of %s", len(raw_paths), TITLE, issue_date)

        sections = parse_all(raw_paths, issue_date)
        logger.info("Parsed %d sections", len(sections))

        chunks = chunk_all(sections)
        logger.info("Built %d chunks", len(chunks))

        store_chunks(
            chunks,
            persist_dir=Path(settings.chroma_dir),
            collection_name=settings.chroma_collection,
            model=settings.embedding_model,
            history_db_path=Path(settings.history_db),
        )
        logger.info(
            "Ingestion complete: %d chunks in collection %r (issue date %s)",
            len(chunks), settings.chroma_collection, issue_date,
        )
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest HIPAA regulation text into Chroma")
    parser.add_argument("--force", action="store_true", help="Re-fetch parts even if already cached")
    parser.add_argument(
        "--queue", action="store_true",
        help="Enqueue as a background RQ job instead of running inline (requires a worker: see jobs.py)",
    )
    args = parser.parse_args()

    if args.queue:
        from citedguard.jobs import enqueue_ingestion

        job = enqueue_ingestion(force=args.force)
        print(f"Enqueued ingestion job {job.id}")
    elif not run(force=args.force):
        print("Skipped: another ingestion is already in progress.")
