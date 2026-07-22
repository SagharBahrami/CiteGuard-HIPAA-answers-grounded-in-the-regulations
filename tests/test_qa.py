import pytest

import qa
from guardrails import FaithfulnessCheck
from retriever import RetrievedChunk
from usage import TokenUsage


def _chunk():
    return RetrievedChunk(citation="45 CFR 164.312", heading="H", part=164, subpart="", text="t", similarity=0.9)


def test_answer_question_wires_retrieve_generate_and_check_together(monkeypatch):
    chunks = [_chunk()]
    faithfulness = FaithfulnessCheck(is_faithful=True, unsupported_claims=[], explanation="ok")
    generation_usage = TokenUsage(100, 50, 150)
    guardrail_usage = TokenUsage(120, 20, 140)

    monkeypatch.setattr(qa, "retrieve", lambda query, top_k: chunks)
    monkeypatch.setattr(qa, "generate_answer", lambda query, c: ("the answer", generation_usage))
    monkeypatch.setattr(qa, "check_faithfulness", lambda text, c: (faithfulness, guardrail_usage))
    monkeypatch.setattr(qa, "log_decline", lambda q: pytest.fail("should not decline"))
    monkeypatch.setattr(qa, "log_unfaithful", lambda *a: pytest.fail("should not flag unfaithful"))
    logged_usage = []
    monkeypatch.setattr(qa, "log_usage", lambda *a: logged_usage.append(a))

    result = qa.answer_question("What are the technical safeguards?", top_k=5)

    assert result.text == "the answer"
    assert result.sources == chunks
    assert result.faithfulness is faithfulness
    assert logged_usage == [("What are the technical safeguards?", generation_usage, guardrail_usage)]


def test_answer_question_logs_decline_on_empty_context(monkeypatch):
    monkeypatch.setattr(qa, "retrieve", lambda query, top_k: [])
    monkeypatch.setattr(qa, "generate_answer", lambda query, c: ("decline message", TokenUsage.zero()))
    monkeypatch.setattr(
        qa, "check_faithfulness",
        lambda text, c: (FaithfulnessCheck(is_faithful=True, unsupported_claims=[], explanation="no context"), TokenUsage.zero()),
    )
    monkeypatch.setattr(qa, "log_usage", lambda *a: None)
    logged = []
    monkeypatch.setattr(qa, "log_decline", lambda q: logged.append(q))
    monkeypatch.setattr(qa, "log_unfaithful", lambda *a: pytest.fail("should not be called"))

    qa.answer_question("off topic question")

    assert logged == ["off topic question"]


def test_answer_question_logs_unfaithful_result(monkeypatch):
    chunks = [_chunk()]
    faithfulness = FaithfulnessCheck(is_faithful=False, unsupported_claims=["bad claim"], explanation="nope")

    monkeypatch.setattr(qa, "retrieve", lambda query, top_k: chunks)
    monkeypatch.setattr(qa, "generate_answer", lambda query, c: ("fabricated answer", TokenUsage.zero()))
    monkeypatch.setattr(qa, "check_faithfulness", lambda text, c: (faithfulness, TokenUsage.zero()))
    monkeypatch.setattr(qa, "log_usage", lambda *a: None)
    logged = []
    monkeypatch.setattr(qa, "log_unfaithful", lambda query, text, f, citations: logged.append((query, text, f, citations)))
    monkeypatch.setattr(qa, "log_decline", lambda q: pytest.fail("should not be called"))

    qa.answer_question("question")

    assert len(logged) == 1
    query, text, f, citations = logged[0]
    assert query == "question"
    assert text == "fabricated answer"
    assert citations == ["45 CFR 164.312"]
