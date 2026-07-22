"""Tests for the hybrid dense+BM25-rescue logic in retriever.py.

The real corpus lives in Chroma, but retrieve() only ever reads a few plain
attributes off the module-level _corpus singleton (documents, metadatas,
normalized_embeddings, bm25), so tests swap in a hand-built fake corpus via
monkeypatch rather than touching Chroma or the embeddings API at all.

Embeddings are 2D unit vectors of the form [sim, sqrt(1-sim^2)], chosen so the
dot product with a query vector of [1, 0] is exactly `sim` -- this lets each
fake document's cosine similarity to the query be picked directly rather than
computed.
"""

import types

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

import retriever
from retriever import _tokenize, retrieve


def _unit_vector(sim: float) -> list[float]:
    return [sim, (1 - sim**2) ** 0.5]


# (citation, heading, text, similarity-to-query)
_DOCS = [
    ("45 CFR 1", "Best chunk of section 1", "alpha beta gamma technical safeguards", 0.9),
    ("45 CFR 1", "Worse chunk, same section", "alpha duplicate chunk of same section", 0.5),
    ("45 CFR 2", "Access control", "beta encryption access control", 0.6),
    ("45 CFR 3", "Rescued via keyword", "delta keyword only, no embedding overlap", 0.2),
    ("45 CFR 4", "Too dissimilar to rescue", "delta another keyword match but too dissimilar", 0.1),
    ("45 CFR 5", "Unrelated, no keyword match", "unrelated filler text with no shared terms", 0.05),
]


def _build_fake_corpus():
    metadatas = [{"citation": c, "heading": h, "part": 45, "subpart": ""} for c, h, _, _ in _DOCS]
    documents = [t for _, _, t, _ in _DOCS]
    embeddings = np.array([_unit_vector(s) for *_, s in _DOCS], dtype=np.float32)
    bm25 = BM25Okapi([_tokenize(t) for t in documents])
    return types.SimpleNamespace(
        documents=documents, metadatas=metadatas, normalized_embeddings=embeddings, bm25=bm25
    )


@pytest.fixture
def fake_corpus(monkeypatch):
    corpus = _build_fake_corpus()
    monkeypatch.setattr(retriever, "_corpus", corpus)
    return corpus


class _QueryEmbeddingClient:
    """Returns a fixed, not-yet-normalized query embedding regardless of input."""

    def __init__(self, vector):
        self.embeddings = types.SimpleNamespace(create=self._create)
        self._vector = vector

    def _create(self, input, model):
        return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=self._vector)])


def test_dense_results_sorted_best_first_and_deduped_by_citation(fake_corpus):
    client = _QueryEmbeddingClient([5.0, 0.0])  # normalizes to [1, 0]

    results = retrieve("delta epsilon", top_k=5, client=client)

    citations = [r.citation for r in results]
    assert citations == ["45 CFR 1", "45 CFR 2", "45 CFR 3"]
    assert results[0].similarity == pytest.approx(0.9)
    assert results[1].similarity == pytest.approx(0.6)


def test_dedup_keeps_best_chunk_per_citation(fake_corpus):
    """The worse '45 CFR 1' chunk (sim=0.5) must never appear once the better one is taken."""
    client = _QueryEmbeddingClient([5.0, 0.0])

    results = retrieve("delta epsilon", top_k=5, client=client)

    matches = [r for r in results if r.citation == "45 CFR 1"]
    assert len(matches) == 1
    assert matches[0].similarity == pytest.approx(0.9)


def test_rescue_adds_keyword_match_below_primary_threshold(fake_corpus):
    client = _QueryEmbeddingClient([5.0, 0.0])

    results = retrieve("delta epsilon", top_k=5, client=client)

    rescued = [r for r in results if r.citation == "45 CFR 3"]
    assert len(rescued) == 1
    assert rescued[0].similarity == pytest.approx(0.2)


def test_rescue_floor_excludes_keyword_matches_too_dissimilar(fake_corpus):
    client = _QueryEmbeddingClient([5.0, 0.0])

    results = retrieve("delta epsilon", top_k=5, client=client)

    assert "45 CFR 4" not in [r.citation for r in results]


def test_no_keyword_overlap_is_never_rescued(fake_corpus):
    client = _QueryEmbeddingClient([5.0, 0.0])

    results = retrieve("delta epsilon", top_k=5, client=client)

    assert "45 CFR 5" not in [r.citation for r in results]


def test_top_k_truncates_before_rescue_runs(fake_corpus):
    client = _QueryEmbeddingClient([5.0, 0.0])

    results = retrieve("delta epsilon", top_k=1, client=client)

    assert len(results) == 1
    assert results[0].citation == "45 CFR 1"


def test_tokenize_lowercases_and_strips_stopwords():
    assert _tokenize("The Encryption of ePHI is required") == ["encryption", "ephi", "required"]
