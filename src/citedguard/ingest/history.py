"""Append-only audit trail of superseded and removed chunk versions.

Chroma only ever holds the *current* version of a chunk -- retrieval wants one
row per section, not one per historical revision (see embed_store.py). So the
version a Chroma upsert/delete is about to overwrite has to be captured
somewhere else first, or it's gone the moment eCFR republishes a part.

That "somewhere else" is this SQLite table. archive_batch() commits before
returning, and callers MUST call it -- and let it finish -- before performing
the corresponding Chroma upsert or delete. That ordering is the entire point:
if the process dies between the two calls, the old version is already durable
on disk and the worst outcome next run is a harmless duplicate archive row,
never a silently lost regulatory version.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunk_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT NOT NULL,
    citation TEXT NOT NULL,
    heading TEXT NOT NULL,
    part INTEGER NOT NULL,
    subpart TEXT,
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    issue_date TEXT NOT NULL,
    text TEXT NOT NULL,
    reason TEXT NOT NULL,
    archived_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunk_history_chunk_id ON chunk_history(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunk_history_citation ON chunk_history(citation);
"""


@dataclass
class ArchivedChunk:
    chunk_id: str
    citation: str
    heading: str
    part: int
    subpart: str | None
    chunk_index: int
    total_chunks: int
    issue_date: str
    text: str
    reason: str  # "superseded" (content/issue_date changed) or "removed" (no longer in the corpus)


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def archive_batch(conn: sqlite3.Connection, entries: list[ArchivedChunk]) -> None:
    """Durably record superseded/removed chunk versions. Commits before returning."""
    if not entries:
        return
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.executemany(
            """
            INSERT INTO chunk_history
                (chunk_id, citation, heading, part, subpart, chunk_index, total_chunks, issue_date, text, reason, archived_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    e.chunk_id, e.citation, e.heading, e.part, e.subpart,
                    e.chunk_index, e.total_chunks, e.issue_date, e.text, e.reason, now,
                )
                for e in entries
            ],
        )


def history_for_citation(conn: sqlite3.Connection, citation: str) -> list[sqlite3.Row]:
    """All archived versions of a section's chunks, oldest first."""
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT * FROM chunk_history WHERE citation = ? ORDER BY archived_at ASC", (citation,)
    )
    try:
        return cur.fetchall()
    finally:
        cur.close()
