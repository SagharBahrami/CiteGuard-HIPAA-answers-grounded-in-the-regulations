import pytest

from conftest import FakeOpenAIClient
from citedguard.ingest import history
from citedguard.ingest.chunk import Chunk
from citedguard.ingest.embed_store import chunk_id_for, store_chunks
from citedguard.retriever import RetrievedChunk
from citedguard.version_query import extract_target_date, is_historical_query, resolve_as_of


# --- is_historical_query -------------------------------------------------

@pytest.mark.parametrize("query", [
    "What is the previous version of the encryption standard?",
    "What did this section used to require?",
    "The rule no longer requires annual audits, correct?",
    "How has the breach notification rule changed over time?",
    "What is the history of the minimum necessary standard?",
    "Is 45 CFR 164.306 historically different from today's rule?",
    "Was this requirement superseded by a later revision?",
    "Which sections were repealed?",
    "What did 164.312 originally require before later amendments?",
    "What applied before the 2023 amendment?",
    "What did 45 CFR 164.312 require before 2024?",
    "What applied prior to the 2022 revision?",
    "What did the rule say as it used to stand?",
    "How has the breach notification rule changed?",
    "Show me the rule as it stood in 2023",
    "What was the wording prior to the 2013 amendment?",
])
def test_is_historical_query_true_for_versioning_language(query):
    assert is_historical_query(query) is True


@pytest.mark.parametrize("query", [
    "What are the technical safeguards for encryption?",
    "What must a covered entity do before disclosing PHI to a business associate?",
    "What changes must occur when a breach is discovered?",
    "What is required before granting workforce access to ePHI?",
    "What is the minimum necessary standard?",
    "How does a covered entity report a breach?",
    # These carry historical-sounding words in a present-tense sense. Each one
    # is about a record, an organization's own duties, or a process -- not
    # about the regulation text having changed.
    "How has my organization's obligation changed since we became a business associate?",
    "How has our risk analysis changed our obligations?",
    "Explain the history of PHI",
    "Can we retain historical records of access?",
    "What are the prior authorization requirements?",
    "Is a former employee's access considered a breach?",
    "Must I notify individuals before a change in privacy practices?",
    "What is the old-age exception for minors?",
])
def test_is_historical_query_false_for_ordinary_compliance_questions(query):
    """These use words like "before" or "change" in a non-versioning sense --
    a false positive here would silently answer a present-tense compliance
    question from a possibly outdated rule, which is the expensive failure."""
    assert is_historical_query(query) is False


# --- extract_target_date ---------------------------------------------------

def test_extract_target_date_finds_iso_date():
    assert extract_target_date("What applied as of 2023-06-15?") == "2023-06-15"


def test_extract_target_date_treats_before_year_as_end_of_prior_year():
    assert extract_target_date("What applied before 2024?") == "2023-12-31"
    assert extract_target_date("What applied prior to 2022?") == "2021-12-31"


def test_extract_target_date_treats_bare_year_as_start_of_year():
    assert extract_target_date("What applied in 2022?") == "2022-01-01"


def test_extract_target_date_returns_none_when_no_date_present():
    assert extract_target_date("What is the previous version of this rule?") is None


# --- resolve_as_of -----------------------------------------------------------

def _chunk(citation="45 CFR 164.312", chunk_index=1, text="Current text.", issue_date="2026-07-02"):
    return RetrievedChunk(
        citation=citation, heading="Technical safeguards", part=164, subpart="", text=text,
        similarity=0.9, issue_date=issue_date, chunk_index=chunk_index,
    )


def _seed_history(db_path, citation="45 CFR 164.312", chunk_index=1, versions=()):
    conn = history.connect(db_path)
    entries = [
        history.ArchivedChunk(
            chunk_id=chunk_id_for(citation, chunk_index), citation=citation, heading="Technical safeguards",
            part=164, subpart=None, chunk_index=chunk_index, total_chunks=1, issue_date=issue_date, text=text,
            reason="superseded",
        )
        for issue_date, text in versions
    ]
    history.archive_batch(conn, entries)
    conn.close()


def test_resolve_as_of_with_no_target_date_returns_most_recently_superseded_version(tmp_path):
    db_path = tmp_path / "history.sqlite3"
    _seed_history(db_path, versions=[("2025-01-01", "Old text."), ("2026-01-01", "Newer old text.")])

    resolved = resolve_as_of([_chunk()], target_date=None, history_db_path=db_path)

    assert resolved[0].text == "Newer old text."
    assert resolved[0].issue_date == "2026-01-01"


def test_resolve_as_of_with_no_history_and_no_target_date_keeps_current_version(tmp_path):
    db_path = tmp_path / "history.sqlite3"
    history.connect(db_path)  # empty db

    resolved = resolve_as_of([_chunk()], target_date=None, history_db_path=db_path)

    assert resolved[0].text == "Current text."
    assert resolved[0].issue_date == "2026-07-02"


def test_resolve_as_of_with_target_date_picks_the_latest_version_at_or_before_it(tmp_path):
    db_path = tmp_path / "history.sqlite3"
    _seed_history(db_path, versions=[("2024-01-01", "v1"), ("2025-01-01", "v2")])

    resolved = resolve_as_of([_chunk()], target_date="2024-06-01", history_db_path=db_path)

    assert resolved[0].text == "v1"
    assert resolved[0].issue_date == "2024-01-01"


def test_resolve_as_of_with_target_date_can_select_the_current_version(tmp_path):
    db_path = tmp_path / "history.sqlite3"
    _seed_history(db_path, versions=[("2024-01-01", "v1")])

    resolved = resolve_as_of([_chunk(issue_date="2026-07-02")], target_date="2026-12-31", history_db_path=db_path)

    assert resolved[0].text == "Current text."
    assert resolved[0].issue_date == "2026-07-02"


def test_resolve_as_of_falls_back_to_earliest_known_version_when_target_predates_everything(tmp_path):
    db_path = tmp_path / "history.sqlite3"
    _seed_history(db_path, versions=[("2024-01-01", "v1"), ("2025-01-01", "v2")])

    resolved = resolve_as_of([_chunk()], target_date="2020-01-01", history_db_path=db_path)

    assert resolved[0].text == "v1"
    assert resolved[0].issue_date == "2024-01-01"


def test_resolve_as_of_matches_by_exact_chunk_index_not_just_citation(tmp_path):
    """A section split into multiple chunks must not mix up another chunk's history."""
    db_path = tmp_path / "history.sqlite3"
    _seed_history(db_path, chunk_index=2, versions=[("2024-01-01", "chunk 2's old text")])

    resolved = resolve_as_of([_chunk(chunk_index=1)], target_date=None, history_db_path=db_path)

    assert resolved[0].text == "Current text."  # chunk 1 has no history of its own


def test_versions_archived_by_ingestion_are_readable_back_by_resolution(tmp_path):
    """End-to-end contract between the two halves: what store_chunks archives on
    an overwrite is exactly what resolve_as_of can find again, chunk ids included."""
    db_path = tmp_path / "history.sqlite3"
    client = FakeOpenAIClient(embedding_vectors=lambda texts: [[float(len(t)), 0.0] for t in texts])

    def ingest(text, issue_date):
        store_chunks(
            [Chunk(
                citation="45 CFR 164.312", heading="Technical safeguards", part=164, subpart="Subpart C",
                text=text, chunk_index=1, total_chunks=1, issue_date=issue_date,
            )],
            persist_dir=tmp_path / "chroma", collection_name="test_collection", model="fake-model",
            client=client, history_db_path=db_path,
        )

    ingest("The 2024 requirement.", "2024-01-01")
    ingest("The current requirement.", "2026-07-02")

    current = RetrievedChunk(
        citation="45 CFR 164.312", heading="Technical safeguards", part=164, subpart="Subpart C",
        text="The current requirement.", similarity=0.9, issue_date="2026-07-02", chunk_index=1,
    )

    resolved = resolve_as_of([current], target_date="2025-01-01", history_db_path=db_path)

    assert resolved[0].text == "The 2024 requirement."
    assert resolved[0].issue_date == "2024-01-01"
