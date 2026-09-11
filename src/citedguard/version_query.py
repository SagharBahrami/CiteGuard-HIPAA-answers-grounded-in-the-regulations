"""Detect questions asking about a past/superseded version of a regulation,
and resolve retrieved chunks to the version that actually applied then.

Detection is a deterministic pattern match, not another LLM call -- this
project already tracks token cost closely (see usage.py/audit.py, and the
"Token usage dropped 35%" history), and the two ways this classification can
fail aren't symmetric. Missing a genuinely historical question just means the
user gets today's rule and can re-ask more explicitly; a false positive would
silently answer an ordinary present-tense compliance question (e.g. "what
must happen before disclosing PHI") from a possibly no-longer-binding rule,
which is the worse failure for a compliance tool. So the patterns below are
deliberately narrow -- generic words like "before" or "changed" alone don't
trigger it, only phrasing that specifically names a prior version, a repeal,
or a past date/year used in a versioning sense.

Resolution then swaps each retrieved chunk's text for whichever version --
current (Chroma) or archived (ingest/history.py's SQLite table) -- applied at
the requested point in time, or for the most recently superseded version if
no date could be parsed from the question at all.
"""

import re
from dataclasses import replace
from pathlib import Path

from citedguard.config import settings
from citedguard.ingest import history
from citedguard.ingest.embed_store import chunk_id_for
from citedguard.retriever import RetrievedChunk

# Nouns that mark a temporal phrase as being about the *regulation text*
# rather than about a patient record, an organization's own duties, or a
# business process. "history of PHI" and "how has my obligation changed" are
# present-tense questions; "history of the standard" and "how has the rule
# changed" are not, and only the noun tells them apart.
_REG_NOUN = (
    r"(?:rule|section|standard|regulation|requirement|provision|subpart|part"
    r"|version|text|wording|language|\d{3}\.\d+)"
)
_GAP = r"[^.?!]{0,40}?"  # stay within one sentence

_HISTORICAL_PATTERNS = [
    re.compile(r"\b(previous|prior|former|old|older)\s+(version|rule|standard|requirement|regulation|text|wording|language)\b", re.I),
    re.compile(r"\bused to (require|say|state|mandate)\b", re.I),
    re.compile(r"\bno longer (require|requires|say|says|state|states)\b", re.I),
    re.compile(r"\bhow (has|have|did|does)\b" + _GAP + r"\b" + _REG_NOUN + r"\b" + _GAP + r"\bchang(e|ed|ing)\b", re.I),
    re.compile(r"\bchang(e|ed|es)\b[^.?!]{0,15}\bover (time|the years)\b", re.I),
    re.compile(r"\bhistory of\b" + _GAP + r"\b" + _REG_NOUN + r"\b", re.I),
    re.compile(r"\bhistorical(ly)?\b" + _GAP + r"\b" + _REG_NOUN + r"\b", re.I),
    re.compile(r"\bsuperseded\b", re.I),
    re.compile(r"\brepealed\b", re.I),
    re.compile(r"\boriginally (require[ds]?|state[ds]?|says?|said|mandate[ds]?)\b", re.I),
    re.compile(r"\bbefore (the )?(\d{4}|amendment|revision|update)\b", re.I),
    re.compile(r"\bprior to (the )?(\d{4}|amendment|revision|update)\b", re.I),
    re.compile(r"\bas (it|they) (used to|stood|were)\b", re.I),
]

_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_BEFORE_YEAR = re.compile(r"\b(?:before|prior to)\b[^.?!]{0,15}?((?:19|20)\d{2})\b", re.I)
_ANY_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")


def is_historical_query(query: str) -> bool:
    """True only for phrasing that specifically implies a prior/repealed version."""
    return any(p.search(query) for p in _HISTORICAL_PATTERNS)


def extract_target_date(query: str) -> str | None:
    """Best-effort point-in-time cutoff implied by the question, if any.

    "before/prior to <year>" is treated as the end of the prior year, so the
    version resolved is the one actually in effect right before that year
    started. A bare year or an ISO date is treated as "as of" that point.
    """
    iso = _ISO_DATE.search(query)
    if iso:
        return iso.group(0)

    before = _BEFORE_YEAR.search(query)
    if before:
        return f"{int(before.group(1)) - 1}-12-31"

    year = _ANY_YEAR.search(query)
    if year:
        return f"{year.group(1)}-01-01"

    return None


def _versions_for(chunk: RetrievedChunk, conn) -> list[tuple[str, str]]:
    """(issue_date, text) for every known version of this chunk, oldest first, current last."""
    cid = chunk_id_for(chunk.citation, chunk.chunk_index)
    archived = [
        (row["issue_date"], row["text"])
        for row in history.history_for_citation(conn, chunk.citation)
        if row["chunk_id"] == cid
    ]
    archived.append((chunk.issue_date, chunk.text))
    # Sort the current version in with the rest rather than assuming it's the
    # newest: resolution below indexes by position, so the ordering has to be
    # true, not just usually true. Ties keep the archived row first (stable).
    archived.sort(key=lambda v: v[0])
    return archived


def resolve_as_of(
    chunks: list[RetrievedChunk],
    target_date: str | None,
    history_db_path: Path | None = None,
) -> list[RetrievedChunk]:
    """Swap each chunk's text for the version effective at target_date, or for
    its most recently superseded version if no target_date could be parsed.

    Falls back to the earliest version on record when target_date predates
    everything known -- the resulting issue_date still ends up on the chunk,
    so the answer's own citation makes clear which date it's actually from.
    """
    conn = history.connect(history_db_path or Path(settings.history_db))
    try:
        resolved = []
        for chunk in chunks:
            versions = _versions_for(chunk, conn)
            if target_date:
                applicable = [v for v in versions if v[0] <= target_date]
                issue_date, text = applicable[-1] if applicable else versions[0]
            elif len(versions) > 1:
                issue_date, text = versions[-2]  # most recently superseded
            else:
                issue_date, text = versions[-1]  # nothing superseded on record
            resolved.append(replace(chunk, text=text, issue_date=issue_date))
        return resolved
    finally:
        conn.close()
