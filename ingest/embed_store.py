"""Embed chunks and store them in the local Chroma collection.

Chunks are embedded via the OpenAI embeddings API (batched, since one call per
chunk would be slow and wasteful) and written with deterministic IDs so
re-running ingestion upserts existing rows instead of duplicating them.
"""

import logging
from pathlib import Path

import chromadb
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from ingest import history
from ingest.chunk import Chunk

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


def chunk_id_for(citation: str, chunk_index: int) -> str:
    slug = citation.replace("45 CFR ", "").replace(" ", "_")
    return f"{slug}_{chunk_index}"


def chunk_id(chunk: Chunk) -> str:
    return chunk_id_for(chunk.citation, chunk.chunk_index)


def chunk_metadata(chunk: Chunk) -> dict:
    return {
        "citation": chunk.citation,
        "heading": chunk.heading,
        "part": chunk.part,
        "subpart": chunk.subpart or "",
        "chunk_index": chunk.chunk_index,
        "total_chunks": chunk.total_chunks,
        "issue_date": chunk.issue_date,
    }


def _archived_from_existing(chunk_id_: str, doc: str, meta: dict, reason: str) -> history.ArchivedChunk:
    """Build an ArchivedChunk from what's *currently* in Chroma, before it's overwritten."""
    return history.ArchivedChunk(
        chunk_id=chunk_id_,
        citation=meta["citation"],
        heading=meta["heading"],
        part=meta["part"],
        subpart=meta.get("subpart") or None,
        chunk_index=meta["chunk_index"],
        total_chunks=meta["total_chunks"],
        issue_date=meta.get("issue_date", ""),
        text=doc,
        reason=reason,
    )


def current_issue_date(persist_dir: Path, collection_name: str) -> str | None:
    """The issue_date currently stored in Chroma, or None if the collection is empty/missing."""
    db = chromadb.PersistentClient(path=str(persist_dir))
    try:
        collection = db.get_collection(collection_name)
    except (ValueError, chromadb.errors.NotFoundError):
        return None
    existing = collection.get(limit=1, include=["metadatas"])
    metas = existing.get("metadatas") or []
    return metas[0].get("issue_date") if metas else None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def embed_batch(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    resp = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in resp.data]


def store_chunks(
    chunks: list[Chunk],
    persist_dir: Path,
    collection_name: str,
    model: str,
    client: OpenAI | None = None,
    history_db_path: Path | None = None,
) -> None:
    """Upsert chunks into Chroma, archiving whatever version they replace first.

    Chroma's upsert (and delete, for chunks/sections dropped from this run's
    corpus) is destructive -- it does not keep the row it overwrites. So every
    chunk id already present in the collection is diffed against the
    incoming/outgoing set *before* any Chroma write, and any chunk whose text
    or issue_date actually changed (or that's gone entirely) is archived to
    history first. history.archive_batch() commits synchronously, so by the
    time the Chroma call runs, the prior version is already durable on disk --
    a crash between the two loses nothing but a re-fetch, never a version.
    """
    client = client or OpenAI(api_key=settings.openai_api_key)
    db = chromadb.PersistentClient(path=str(persist_dir))
    collection = db.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )
    history_conn = history.connect(history_db_path or Path(settings.history_db))
    try:
        existing = collection.get(include=["documents", "metadatas"])
        existing_map = {
            id_: (doc, meta)
            for id_, doc, meta in zip(existing["ids"], existing["documents"], existing["metadatas"])
        }

        new_ids = {chunk_id(c) for c in chunks}
        removed_ids = [id_ for id_ in existing_map if id_ not in new_ids]
        if removed_ids:
            removed_entries = [
                _archived_from_existing(id_, *existing_map[id_], reason="removed")
                for id_ in removed_ids
            ]
            history.archive_batch(history_conn, removed_entries)  # durable before delete
            collection.delete(ids=removed_ids)
            logger.info("Archived and removed %d chunk(s) no longer in the corpus", len(removed_ids))

        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            logger.info("Embedding chunks %d-%d of %d", start + 1, start + len(batch), len(chunks))

            superseded_entries = []
            for c in batch:
                cid = chunk_id(c)
                prior = existing_map.get(cid)
                if prior is None:
                    continue
                old_doc, old_meta = prior
                old_date = old_meta.get("issue_date", "")
                # A row stored before issue dates were tracked has no date at all.
                # If its text is also unchanged, nothing was actually superseded --
                # archiving it would write a fake supersession into the audit trail
                # for every chunk on the first run after upgrading.
                if old_doc == c.text and (not old_date or old_date == c.issue_date):
                    continue
                superseded_entries.append(_archived_from_existing(cid, old_doc, old_meta, reason="superseded"))
            history.archive_batch(history_conn, superseded_entries)  # durable before upsert

            embeddings = embed_batch(client, [c.text for c in batch], model)
            collection.upsert(
                ids=[chunk_id(c) for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[chunk_metadata(c) for c in batch],
            )
    finally:
        history_conn.close()

    logger.info("Stored %d chunks in collection %r at %s", len(chunks), collection_name, persist_dir)


if __name__ == "__main__":
    from ingest.fetch import PARTS, fetch_all_parts
    from ingest.parse import parse_all
    from ingest.chunk import chunk_all

    logging.basicConfig(level=logging.INFO)

    issue_date, paths = fetch_all_parts(Path("data/raw"), parts=PARTS)
    sections = parse_all(paths, issue_date)
    chunks = chunk_all(sections)

    store_chunks(
        chunks,
        persist_dir=Path(settings.chroma_dir),
        collection_name=settings.chroma_collection,
        model=settings.embedding_model,
    )
