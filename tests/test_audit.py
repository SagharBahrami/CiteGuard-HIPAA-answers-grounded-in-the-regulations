import json

import audit
from guardrails import FaithfulnessCheck
from usage import TokenUsage


def test_log_decline_writes_expected_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "LOG_PATH", tmp_path / "audit.jsonl")

    audit.log_decline("What about coral reefs?")

    entry = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[0])
    assert entry["trigger"] == "empty_context_decline"
    assert entry["query"] == "What about coral reefs?"
    assert "timestamp" in entry


def test_log_unfaithful_writes_expected_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "LOG_PATH", tmp_path / "audit.jsonl")
    check = FaithfulnessCheck(is_faithful=False, unsupported_claims=["fake claim"], explanation="why")

    audit.log_unfaithful("q", "a", check, ["45 CFR 164.312"])

    entry = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[0])
    assert entry["trigger"] == "faithfulness_failure"
    assert entry["query"] == "q"
    assert entry["answer"] == "a"
    assert entry["unsupported_claims"] == ["fake claim"]
    assert entry["explanation"] == "why"
    assert entry["citations"] == ["45 CFR 164.312"]


def test_log_appends_rather_than_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "LOG_PATH", tmp_path / "audit.jsonl")

    audit.log_decline("first")
    audit.log_decline("second")

    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["query"] == "first"
    assert json.loads(lines[1])["query"] == "second"


def test_log_creates_parent_directory_if_missing(tmp_path, monkeypatch):
    log_path = tmp_path / "nested" / "logs" / "audit.jsonl"
    monkeypatch.setattr(audit, "LOG_PATH", log_path)

    audit.log_decline("q")

    assert log_path.exists()


def test_log_usage_writes_token_counts_for_both_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "USAGE_LOG_PATH", tmp_path / "usage.jsonl")

    audit.log_usage("q", TokenUsage(100, 50, 150), TokenUsage(120, 20, 140))

    entry = json.loads((tmp_path / "usage.jsonl").read_text().splitlines()[0])
    assert entry["query"] == "q"
    assert entry["generation_prompt_tokens"] == 100
    assert entry["generation_completion_tokens"] == 50
    assert entry["guardrail_prompt_tokens"] == 120
    assert entry["guardrail_completion_tokens"] == 20
    assert entry["total_tokens"] == 290
    assert "timestamp" in entry


def test_log_usage_writes_to_a_separate_file_from_guardrail_triggers(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "LOG_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setattr(audit, "USAGE_LOG_PATH", tmp_path / "usage.jsonl")

    audit.log_usage("q", TokenUsage.zero(), TokenUsage.zero())

    assert not (tmp_path / "audit.jsonl").exists()
    assert (tmp_path / "usage.jsonl").exists()
