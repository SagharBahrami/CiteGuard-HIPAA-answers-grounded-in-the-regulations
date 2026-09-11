from ingest.history import ArchivedChunk, archive_batch, connect, history_for_citation


def _entry(chunk_id="1_1", citation="45 CFR 1", text="Old text.", issue_date="2026-01-01", reason="superseded"):
    return ArchivedChunk(
        chunk_id=chunk_id, citation=citation, heading="Heading", part=1, subpart=None,
        chunk_index=1, total_chunks=1, issue_date=issue_date, text=text, reason=reason,
    )


def test_archive_batch_persists_entries_queryable_by_citation(tmp_path):
    conn = connect(tmp_path / "history.sqlite3")

    archive_batch(conn, [_entry(citation="45 CFR 1", text="v1")])

    rows = history_for_citation(conn, "45 CFR 1")
    assert len(rows) == 1
    assert rows[0]["text"] == "v1"
    assert rows[0]["reason"] == "superseded"


def test_archive_batch_is_a_noop_for_an_empty_list(tmp_path):
    conn = connect(tmp_path / "history.sqlite3")

    archive_batch(conn, [])

    assert history_for_citation(conn, "45 CFR 1") == []


def test_archived_entries_survive_a_reconnect(tmp_path):
    db_path = tmp_path / "history.sqlite3"
    conn = connect(db_path)
    archive_batch(conn, [_entry()])
    conn.close()

    reconnected = connect(db_path)
    assert len(history_for_citation(reconnected, "45 CFR 1")) == 1


def test_history_for_citation_only_returns_matching_rows(tmp_path):
    conn = connect(tmp_path / "history.sqlite3")

    archive_batch(conn, [_entry(citation="45 CFR 1"), _entry(citation="45 CFR 2")])

    assert len(history_for_citation(conn, "45 CFR 1")) == 1
    assert len(history_for_citation(conn, "45 CFR 2")) == 1


def test_history_for_citation_orders_oldest_first(tmp_path):
    conn = connect(tmp_path / "history.sqlite3")

    archive_batch(conn, [_entry(citation="45 CFR 1", issue_date="2026-01-01")])
    archive_batch(conn, [_entry(citation="45 CFR 1", issue_date="2026-07-02")])

    rows = history_for_citation(conn, "45 CFR 1")
    assert [r["issue_date"] for r in rows] == ["2026-01-01", "2026-07-02"]
