"""Append-only logs for guardrail triggers and per-question token usage.

Two separate logs, two separate purposes:
- logs/guardrail_audit.jsonl: every empty-context decline and faithfulness
  failure, for reviewing what the guardrails actually caught. Logs the query
  and answer text verbatim -- a deliberate choice for reviewability, not a
  default to assume elsewhere; treat logs/ as containing the same sensitive
  content as the questions users ask.
- logs/token_usage.jsonl: prompt/completion token counts for *every*
  question, triggered or not -- so token cost can be measured before tuning
  anything (e.g. top_k) to reduce it.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from guardrails import FaithfulnessCheck
from usage import TokenUsage

LOG_PATH = Path("logs/guardrail_audit.jsonl")
USAGE_LOG_PATH = Path("logs/token_usage.jsonl")


def _append(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def log_decline(query: str) -> None:
    _append(
        LOG_PATH,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger": "empty_context_decline",
            "query": query,
        },
    )


def log_unfaithful(
    query: str,
    answer: str,
    faithfulness: FaithfulnessCheck,
    citations: list[str],
) -> None:
    _append(
        LOG_PATH,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger": "faithfulness_failure",
            "query": query,
            "answer": answer,
            "unsupported_claims": faithfulness.unsupported_claims,
            "explanation": faithfulness.explanation,
            "citations": citations,
        },
    )


def log_usage(query: str, generation: TokenUsage, guardrail: TokenUsage) -> None:
    _append(
        USAGE_LOG_PATH,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "generation_prompt_tokens": generation.prompt_tokens,
            "generation_completion_tokens": generation.completion_tokens,
            "guardrail_prompt_tokens": guardrail.prompt_tokens,
            "guardrail_completion_tokens": guardrail.completion_tokens,
            "total_tokens": generation.total_tokens + guardrail.total_tokens,
        },
    )
