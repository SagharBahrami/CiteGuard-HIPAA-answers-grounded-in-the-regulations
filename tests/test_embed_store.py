import chromadb

from conftest import FakeOpenAIClient
from citedguard.ingest import history
from citedguard.ingest.chunk import Chunk
from citedguard.ingest.embed_store import chunk_id, chunk_metadata, current_issue_date, store_chunks


def _chunk(citation="45 CFR 164.312", index=1, total=1, subpart="Subpart C", text="Encrypt ePHI.", issue_date="2026-01-01"):
    return Chunk(
        citation=citation, heading="Technical safeguards", part=164, subpart=subpart, text=text,
        chunk_index=index, total_chunks=total, issue_date=issue_date,
    )


def _store(chunks, tmp_path, collection_name="test_collection"):
    client = FakeOpenAIClient(embedding_vectors=lambda texts: [[float(len(t)), 0.0] for t in texts])
    store_chunks(
        chunks, persist_dir=tmp_path / "chroma", collection_name=collection_name, model="fake-model",
        client=client, history_db_path=tmp_path / "history.sqlite3",
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
        "issue_date": "2026-01-01",
    }


def test_chunk_metadata_defaults_missing_subpart_to_empty_string():
    assert chunk_metadata(_chunk(subpart=None))["subpart"] == ""


def test_store_chunks_upserts_all_chunks_into_chroma(tmp_path):
    chunks = [_chunk(citation="45 CFR 1", index=1), _chunk(citation="45 CFR 2", index=1)]

    _store(chunks, tmp_path)

    collection = chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_collection("test_collection")
    assert collection.count() == 2
    assert set(collection.get(ids=["1_1", "2_1"])["ids"]) == {"1_1", "2_1"}


def test_store_chunks_upserts_rather_than_duplicates_on_rerun(tmp_path):
    chunk = _chunk(citation="45 CFR 1", index=1)

    _store([chunk], tmp_path)
    _store([chunk], tmp_path)

    collection = chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_collection("test_collection")
    assert collection.count() == 1


def test_current_issue_date_reflects_what_was_last_stored(tmp_path):
    _store([_chunk(citation="45 CFR 1", issue_date="2026-01-01")], tmp_path)

    assert current_issue_date(tmp_path / "chroma", "test_collection") == "2026-01-01"


def test_current_issue_date_is_none_for_a_missing_collection(tmp_path):
    assert current_issue_date(tmp_path / "chroma", "nonexistent") is None


def test_rerun_with_changed_text_archives_the_prior_version_to_history_before_upserting(tmp_path):
    old = _chunk(citation="45 CFR 1", text="Old text.", issue_date="2026-01-01")
    new = _chunk(citation="45 CFR 1", text="New text.", issue_date="2026-07-02")

    _store([old], tmp_path)
    _store([new], tmp_path)

    collection = chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_collection("test_collection")
    assert collection.get(ids=["1_1"])["documents"] == ["New text."]

    conn = history.connect(tmp_path / "history.sqlite3")
    archived = history.history_for_citation(conn, "45 CFR 1")
    assert len(archived) == 1
    assert archived[0]["text"] == "Old text."
    assert archived[0]["issue_date"] == "2026-01-01"
    assert archived[0]["reason"] == "superseded"


def test_rerun_with_unchanged_content_archives_nothing(tmp_path):
    chunk = _chunk(citation="45 CFR 1", text="Same text.", issue_date="2026-01-01")

    _store([chunk], tmp_path)
    _store([chunk], tmp_path)

    conn = history.connect(tmp_path / "history.sqlite3")
    assert history.history_for_citation(conn, "45 CFR 1") == []


def test_upgrading_an_undated_collection_does_not_fake_a_supersession(tmp_path):
    """Rows written before issue dates were tracked have no date. Gaining one
    isn't a regulatory change, so unchanged text must not be archived."""
    chroma_dir = tmp_path / "chroma"
    collection = chromadb.PersistentClient(path=str(chroma_dir)).get_or_create_collection(
        name="test_collection", metadata={"hnsw:space": "cosine"}
    )
    collection.upsert(  # a pre-upgrade row: no issue_date in metadata
        ids=["1_1"], embeddings=[[1.0, 0.0]], documents=["Unchanged text."],
        metadatas=[{"citation": "45 CFR 1", "heading": "H", "part": 1, "subpart": "",
                    "chunk_index": 1, "total_chunks": 1}],
    )

    _store([_chunk(citation="45 CFR 1", text="Unchanged text.", issue_date="2026-07-02")], tmp_path)

    conn = history.connect(tmp_path / "history.sqlite3")
    assert history.history_for_citation(conn, "45 CFR 1") == []
    assert collection.get(ids=["1_1"])["metadatas"][0]["issue_date"] == "2026-07-02"


def test_undated_row_whose_text_changed_is_still_archived(tmp_path):
    chroma_dir = tmp_path / "chroma"
    collection = chromadb.PersistentClient(path=str(chroma_dir)).get_or_create_collection(
        name="test_collection", metadata={"hnsw:space": "cosine"}
    )
    collection.upsert(
        ids=["1_1"], embeddings=[[1.0, 0.0]], documents=["Genuinely old text."],
        metadatas=[{"citation": "45 CFR 1", "heading": "H", "part": 1, "subpart": "",
                    "chunk_index": 1, "total_chunks": 1}],
    )

    _store([_chunk(citation="45 CFR 1", text="New text.", issue_date="2026-07-02")], tmp_path)

    conn = history.connect(tmp_path / "history.sqlite3")
    archived = history.history_for_citation(conn, "45 CFR 1")
    assert len(archived) == 1
    assert archived[0]["text"] == "Genuinely old text."


def test_chunk_dropped_from_a_rerun_is_archived_as_removed_and_deleted_from_chroma(tmp_path):
    first_run = [_chunk(citation="45 CFR 1", index=1), _chunk(citation="45 CFR 2", index=1)]
    _store(first_run, tmp_path)

    second_run = [_chunk(citation="45 CFR 1", index=1)]  # "45 CFR 2" repealed/merged away
    _store(second_run, tmp_path)

    collection = chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_collection("test_collection")
    assert collection.count() == 1
    assert collection.get(ids=["2_1"])["ids"] == []

    conn = history.connect(tmp_path / "history.sqlite3")
    archived = history.history_for_citation(conn, "45 CFR 2")
    assert len(archived) == 1
    assert archived[0]["reason"] == "removed"
