"""Append-only audit log for guardrail triggers.

Records every time a guardrail actually changes what the caller sees: an
empty-context decline (generate.py's NO_CONTEXT_MESSAGE path) or a faithfulness
failure (guardrails.py flagging unsupported claims). Logs the query and answer
text verbatim -- a deliberate choice for reviewability, not a default to
assume elsewhere; treat logs/ as containing the same sensitive content as the
questions users ask.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from guardrails import FaithfulnessCheck

LOG_PATH = Path("logs/guardrail_audit.jsonl")


def _write(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def log_decline(query: str) -> None:
    _write(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger": "empty_context_decline",
            "query": query,
        }
    )


def log_unfaithful(
    query: str,
    answer: str,
    faithfulness: FaithfulnessCheck,
    citations: list[str],
) -> None:
    _write(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger": "faithfulness_failure",
            "query": query,
            "answer": answer,
            "unsupported_claims": faithfulness.unsupported_claims,
            "explanation": faithfulness.explanation,
            "citations": citations,
        }
    )
