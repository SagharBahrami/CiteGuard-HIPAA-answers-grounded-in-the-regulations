import chromadb

from conftest import FakeOpenAIClient
from ingest.chunk import Chunk
from ingest.embed_store import chunk_id, chunk_metadata, store_chunks


def _chunk(citation="45 CFR 164.312", index=1, total=1, subpart="Subpart C"):
    return Chunk(
        citation=citation, heading="Technical safeguards", part=164, subpart=subpart, text="Encrypt ePHI.",
        chunk_index=index, total_chunks=total,
    )


def test_chunk_id_slugifies_citation_and_appends_index():
    assert chunk_id(_chunk(citation="45 CFR 164.312", index=2)) == "164.312_2"


def test_chunk_metadata_fields():
    assert chunk_metadata(_chunk()) == {
        "citation": "45 CFR 164.312",
        "heading": "Technical safeguards",
        "part": 164,
        "subpart": "Subpart C",
        "chunk_index": 1,
        "total_chunks": 1,
    }


def test_chunk_metadata_defaults_missing_subpart_to_empty_string():
    assert chunk_metadata(_chunk(subpart=None))["subpart"] == ""


def test_store_chunks_upserts_all_chunks_into_chroma(tmp_path):
    chunks = [_chunk(citation="45 CFR 1", index=1), _chunk(citation="45 CFR 2", index=1)]
    client = FakeOpenAIClient(embedding_vectors=lambda texts: [[float(len(t)), 0.0] for t in texts])

    store_chunks(chunks, persist_dir=tmp_path, collection_name="test_collection", model="fake-model", client=client)

    collection = chromadb.PersistentClient(path=str(tmp_path)).get_collection("test_collection")
    assert collection.count() == 2
    assert set(collection.get(ids=["1_1", "2_1"])["ids"]) == {"1_1", "2_1"}


def test_store_chunks_upserts_rather_than_duplicates_on_rerun(tmp_path):
    chunk = _chunk(citation="45 CFR 1", index=1)
    client = FakeOpenAIClient(embedding_vectors=lambda texts: [[float(len(t)), 0.0] for t in texts])

    store_chunks([chunk], persist_dir=tmp_path, collection_name="test_collection", model="fake-model", client=client)
    store_chunks([chunk], persist_dir=tmp_path, collection_name="test_collection", model="fake-model", client=client)

    collection = chromadb.PersistentClient(path=str(tmp_path)).get_collection("test_collection")
    assert collection.count() == 1
