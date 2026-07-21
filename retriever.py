"""Retrieve relevant HIPAA regulation chunks for a user question.

Hybrid retrieval, but not naive rank fusion. Dense (cosine similarity between
embeddings) is the primary, calibrated signal here -- it's what
settings.similarity_threshold is tuned against, and it's generally the better
judge of "is this substantively relevant" for regulatory text. Sparse (BM25
keyword overlap) is used only to *rescue* sections dense missed entirely
(e.g. an exact citation number or defined term dense embeds weakly), appended
after the dense results rather than reordering them.

Equal-weight rank fusion (e.g. RRF) was tried first and rejected: a keyword-
dense but shallow section like 164.304 (Definitions, which briefly touches
dozens of terms) can out-rank the single best dense match on almost any
query, since BM25 rank 0 there counted as strong a signal as dense rank 0.
Rescue-only avoids that failure mode by construction -- BM25 can only add
results, never bump a strong dense match down.

Results are also deduplicated by citation (best-scoring chunk per section),
so the top-k spans more distinct sections instead of several slots going to
different chunks of the same long section.

The corpus here is small (a few hundred chunks), so rather than relying on
Chroma's approximate nearest-neighbor index, we pull every chunk's embedding
once and score it exactly against the query.
"""

import re
from dataclasses import dataclass

import chromadb
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

from config import settings
from ingest.embed_store import embed_batch

RESCUE_CANDIDATES = 5  # how many top BM25 matches are eligible to be rescued

# A rescued chunk still needs *some* semantic relationship to the query, just
# not enough to clear the primary bar -- otherwise raw BM25 magnitude alone
# lets completely off-topic queries "rescue" chunks that merely share common
# regulatory words (e.g. "change") with the query, even after stopword
# removal. A genuine rescue case (an exact citation/term dense underweights)
# should still clear this relaxed bar; a truly unrelated topic won't.
RESCUE_SIMILARITY_THRESHOLD = settings.similarity_threshold * 0.5


@dataclass
class RetrievedChunk:
    citation: str
    heading: str
    part: int
    subpart: str
    text: str
    similarity: float


class _Corpus:
    """Every chunk's text/metadata/embedding, loaded once and reused."""

    def __init__(self):
        db = chromadb.PersistentClient(path=settings.chroma_dir)
        collection = db.get_collection(settings.chroma_collection)
        data = collection.get(include=["documents", "metadatas", "embeddings"])

        self.documents: list[str] = data["documents"]
        self.metadatas: list[dict] = data["metadatas"]

        embeddings = np.array(data["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.normalized_embeddings = embeddings / norms

        tokenized_docs = [_tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)


_corpus: _Corpus | None = None


def _get_corpus() -> _Corpus:
    global _corpus
    if _corpus is None:
        _corpus = _Corpus()
    return _corpus


_STOPWORDS = frozenset(
    """
    a an the and or but if then else when how what why where who whom which
    at by for with about against between into through during before after
    above below to from up down in out on off over under again further once
    here there all any both each few more most other some such no nor not
    only own same so than too very s t can will just don should now is are
    was were be been being have has had do does did doing this that these
    those i you he she it we they it's its as of
    """.split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS]


def _to_chunk(corpus: _Corpus, index: int, similarity: float) -> RetrievedChunk:
    meta = corpus.metadatas[index]
    return RetrievedChunk(
        citation=meta["citation"],
        heading=meta["heading"],
        part=meta["part"],
        subpart=meta["subpart"],
        text=corpus.documents[index],
        similarity=similarity,
    )


def retrieve(
    query: str,
    top_k: int = 5,
    client: OpenAI | None = None,
) -> list[RetrievedChunk]:
    corpus = _get_corpus()

    client = client or OpenAI(api_key=settings.openai_api_key)
    query_embedding = np.array(embed_batch(client, [query], settings.embedding_model)[0])
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    dense_similarities = corpus.normalized_embeddings @ query_embedding
    bm25_scores = np.array(corpus.bm25.get_scores(_tokenize(query)))

    seen_citations: set[str] = set()
    results: list[RetrievedChunk] = []

    # Primary: dense matches above threshold, sorted best-first, one per section.
    for i in np.argsort(-dense_similarities):
        similarity = float(dense_similarities[i])
        if similarity < settings.similarity_threshold:
            break  # descending order, so nothing further clears the bar either
        citation = corpus.metadatas[i]["citation"]
        if citation in seen_citations:
            continue
        seen_citations.add(citation)
        results.append(_to_chunk(corpus, i, similarity))
        if len(results) >= top_k:
            return results

    # Rescue: strong keyword matches for sections dense missed, appended after.
    # Still gated on dense similarity (relaxed) -- see RESCUE_SIMILARITY_THRESHOLD.
    for i in np.argsort(-bm25_scores)[:RESCUE_CANDIDATES]:
        if bm25_scores[i] <= 0:
            break  # descending order, so nothing further has any keyword overlap
        if float(dense_similarities[i]) < RESCUE_SIMILARITY_THRESHOLD:
            continue
        citation = corpus.metadatas[i]["citation"]
        if citation in seen_citations:
            continue
        seen_citations.add(citation)
        results.append(_to_chunk(corpus, i, float(dense_similarities[i])))
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    for q in [
        "What are the technical safeguards for encryption?",
        "How does climate change affect coral reefs?",
        "business associate",
    ]:
        results = retrieve(q)
        print(f"\nQuery: {q!r} -> {len(results)} chunks")
        for r in results:
            print(f"  {r.citation} | similarity={r.similarity:.3f} | {r.heading}")
