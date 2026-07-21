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
from ingest.chunk import Chunk

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


def chunk_id(chunk: Chunk) -> str:
    slug = chunk.citation.replace("45 CFR ", "").replace(" ", "_")
    return f"{slug}_{chunk.chunk_index}"


def chunk_metadata(chunk: Chunk) -> dict:
    return {
        "citation": chunk.citation,
        "heading": chunk.heading,
        "part": chunk.part,
        "subpart": chunk.subpart or "",
        "chunk_index": chunk.chunk_index,
        "total_chunks": chunk.total_chunks,
    }


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
) -> None:
    client = client or OpenAI(api_key=settings.openai_api_key)
    db = chromadb.PersistentClient(path=str(persist_dir))
    collection = db.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        logger.info("Embedding chunks %d-%d of %d", start + 1, start + len(batch), len(chunks))
        embeddings = embed_batch(client, [c.text for c in batch], model)
        collection.upsert(
            ids=[chunk_id(c) for c in batch],
            embeddings=embeddings,
            documents=[c.text for c in batch],
            metadatas=[chunk_metadata(c) for c in batch],
        )

    logger.info("Stored %d chunks in collection %r at %s", len(chunks), collection_name, persist_dir)


if __name__ == "__main__":
    from ingest.fetch import PARTS, TITLE
    from ingest.parse import parse_all
    from ingest.chunk import chunk_all

    logging.basicConfig(level=logging.INFO)

    paths = {p: Path(f"data/raw/title-{TITLE}-part-{p}.xml") for p in PARTS}
    sections = parse_all(paths)
    chunks = chunk_all(sections)

    store_chunks(
        chunks,
        persist_dir=Path(settings.chroma_dir),
        collection_name=settings.chroma_collection,
        model=settings.embedding_model,
    )
