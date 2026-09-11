import pytest

import qa
from guardrails import FaithfulnessCheck
from retriever import RetrievedChunk
from usage import TokenUsage


def _chunk():
    return RetrievedChunk(citation="45 CFR 164.312", heading="H", part=164, subpart="", text="t", similarity=0.9)


def _no_retry(monkeypatch):
    """Fail the test loudly if regenerate_answer gets called when it shouldn't."""
    monkeypatch.setattr(qa, "regenerate_answer", lambda *a, **kw: pytest.fail("should not retry"))


def test_answer_question_wires_retrieve_generate_and_check_together(monkeypatch):
    chunks = [_chunk()]
    faithfulness = FaithfulnessCheck(is_faithful=True, unsupported_claims=[], explanation="ok")
    generation_usage = TokenUsage(100, 50, 150)
    guardrail_usage = TokenUsage(120, 20, 140)

    monkeypatch.setattr(qa, "retrieve", lambda query, top_k: chunks)
    monkeypatch.setattr(qa, "generate_answer", lambda query, c, **kw: ("the answer", generation_usage))
    monkeypatch.setattr(qa, "check_faithfulness", lambda text, c, **kw: (faithfulness, guardrail_usage))
    monkeypatch.setattr(qa, "log_decline", lambda q: pytest.fail("should not decline"))
    monkeypatch.setattr(qa, "log_unfaithful", lambda *a: pytest.fail("should not flag unfaithful"))
    _no_retry(monkeypatch)
    logged_usage = []
    monkeypatch.setattr(qa, "log_usage", lambda *a, **kw: logged_usage.append((a, kw)))

    result = qa.answer_question("What are the technical safeguards?", top_k=5)

    assert result.text == "the answer"
    assert result.sources == chunks
    assert result.faithfulness is faithfulness
    assert logged_usage == [
        (("What are the technical safeguards?", generation_usage, guardrail_usage), {"retried": False})
    ]


def test_historical_question_resolves_chunks_from_history_and_flags_the_answer(monkeypatch):
    current = _chunk()
    historical_version = RetrievedChunk(
        citation="45 CFR 164.312", heading="H", part=164, subpart="", text="old text",
        similarity=0.9, issue_date="2023-01-01",
    )
    faithfulness = FaithfulnessCheck(is_faithful=True, unsupported_claims=[], explanation="ok")

    resolve_calls = []
    monkeypatch.setattr(qa, "retrieve", lambda query, top_k: [current])
    monkeypatch.setattr(
        qa, "resolve_as_of",
        lambda chunks, target_date: resolve_calls.append((chunks, target_date)) or [historical_version],
    )
    generate_calls = []
    monkeypatch.setattr(
        qa, "generate_answer",
        lambda query, c, **kw: generate_calls.append((c, kw)) or ("the answer", TokenUsage.zero()),
    )
    monkeypatch.setattr(qa, "check_faithfulness", lambda text, c, **kw: (faithfulness, TokenUsage.zero()))
    monkeypatch.setattr(qa, "log_usage", lambda *a, **kw: None)
    monkeypatch.setattr(qa, "log_decline", lambda q: pytest.fail("should not decline"))
    _no_retry(monkeypatch)

    result = qa.answer_question("What did 45 CFR 164.312 require before 2024?")

    assert resolve_calls == [([current], "2023-12-31")]
    assert result.historical is True
    assert result.history_unavailable is False
    assert result.sources == [historical_version]
    assert generate_calls == [([historical_version], {"historical": True})]


def test_historical_question_with_nothing_archived_is_not_framed_as_historical(monkeypatch):
    """The archive only reaches back to the first ingestion that superseded
    something. When it can't reach the requested date, resolve_as_of hands back
    today's text -- which must not then be presented as a superseded version."""
    current = _chunk()
    faithfulness = FaithfulnessCheck(is_faithful=True, unsupported_claims=[], explanation="ok")

    monkeypatch.setattr(qa, "retrieve", lambda query, top_k: [current])
    monkeypatch.setattr(qa, "resolve_as_of", lambda chunks, target_date: [current])  # nothing older
    generate_calls = []
    monkeypatch.setattr(
        qa, "generate_answer",
        lambda query, c, **kw: generate_calls.append(kw) or ("the answer", TokenUsage.zero()),
    )
    monkeypatch.setattr(qa, "check_faithfulness", lambda text, c, **kw: (faithfulness, TokenUsage.zero()))
    monkeypatch.setattr(qa, "log_usage", lambda *a, **kw: None)
    monkeypatch.setattr(qa, "log_decline", lambda q: pytest.fail("should not decline"))
    _no_retry(monkeypatch)

    result = qa.answer_question("What did 45 CFR 164.312 require before 2024?")

    assert result.historical is False
    assert result.history_unavailable is True
    assert generate_calls == [{"historical": False}]  # current-version prompt, not the past-version one


def test_ordinary_question_never_touches_history(monkeypatch):
    chunks = [_chunk()]
    faithfulness = FaithfulnessCheck(is_faithful=True, unsupported_claims=[], explanation="ok")

    monkeypatch.setattr(qa, "retrieve", lambda query, top_k: chunks)
    monkeypatch.setattr(qa, "resolve_as_of", lambda *a, **kw: pytest.fail("should not consult history"))
    generate_calls = []
    monkeypatch.setattr(
        qa, "generate_answer",
        lambda query, c, **kw: generate_calls.append(kw) or ("the answer", TokenUsage.zero()),
    )
    monkeypatch.setattr(qa, "check_faithfulness", lambda text, c, **kw: (faithfulness, TokenUsage.zero()))
    monkeypatch.setattr(qa, "log_usage", lambda *a, **kw: None)
    monkeypatch.setattr(qa, "log_decline", lambda q: pytest.fail("should not decline"))
    _no_retry(monkeypatch)

    result = qa.answer_question("What are the technical safeguards for encryption?")

    assert result.historical is False
    assert generate_calls == [{"historical": False}]


def test_answer_question_logs_decline_on_empty_context(monkeypatch):
    monkeypatch.setattr(qa, "retrieve", lambda query, top_k: [])
    monkeypatch.setattr(qa, "generate_answer", lambda query, c, **kw: ("decline message", TokenUsage.zero()))
    monkeypatch.setattr(
        qa, "check_faithfulness",
        lambda text, c, **kw: (FaithfulnessCheck(is_faithful=True, unsupported_claims=[], explanation="no context"), TokenUsage.zero()),
    )
    _no_retry(monkeypatch)
    monkeypatch.setattr(qa, "log_usage", lambda *a, **kw: None)
    logged = []
    monkeypatch.setattr(qa, "log_decline", lambda q: logged.append(q))
    monkeypatch.setattr(qa, "log_unfaithful", lambda *a: pytest.fail("should not be called"))

    qa.answer_question("off topic question")

    assert logged == ["off topic question"]


def test_answer_question_retries_and_logs_correction_when_retry_succeeds(monkeypatch):
    chunks = [_chunk()]
    first_pass = FaithfulnessCheck(is_faithful=False, unsupported_claims=["bad claim"], explanation="nope")
    retried_pass = FaithfulnessCheck(is_faithful=True, unsupported_claims=[], explanation="now supported")

    monkeypatch.setattr(qa, "retrieve", lambda query, top_k: chunks)
    monkeypatch.setattr(qa, "generate_answer", lambda query, c, **kw: ("fabricated answer", TokenUsage(10, 5, 15)))
    monkeypatch.setattr(
        qa, "regenerate_answer",
        lambda query, c, prev_text, claims, **kw: ("corrected answer", TokenUsage(20, 8, 28)),
    )

    check_calls = []

    def fake_check(text, c, **kw):
        check_calls.append(text)
        return (first_pass, TokenUsage(30, 3, 33)) if text == "fabricated answer" else (retried_pass, TokenUsage(31, 4, 35))

    monkeypatch.setattr(qa, "check_faithfulness", fake_check)

    logged_corrected = []
    monkeypatch.setattr(qa, "log_corrected", lambda *a: logged_corrected.append(a))
    monkeypatch.setattr(qa, "log_unfaithful", lambda *a: pytest.fail("should not flag unfaithful after a successful retry"))
    monkeypatch.setattr(qa, "log_decline", lambda q: pytest.fail("should not decline"))
    logged_usage = []
    monkeypatch.setattr(qa, "log_usage", lambda *a, **kw: logged_usage.append((a, kw)))

    result = qa.answer_question("question")

    assert check_calls == ["fabricated answer", "corrected answer"]
    assert result.text == "corrected answer"
    assert result.faithfulness is retried_pass
    assert logged_corrected == [("question", "fabricated answer", "corrected answer", ["bad claim"])]
    (query, gen_usage, guard_usage), kw = logged_usage[0]
    assert kw == {"retried": True}
    assert gen_usage == TokenUsage(30, 13, 43)  # 10+20, 5+8, 15+28
    assert guard_usage == TokenUsage(61, 7, 68)  # 30+31, 3+4, 33+35


def test_answer_question_falls_back_to_warn_when_retry_still_fails(monkeypatch):
    chunks = [_chunk()]
    first_pass = FaithfulnessCheck(is_faithful=False, unsupported_claims=["bad claim"], explanation="nope")
    still_bad = FaithfulnessCheck(is_faithful=False, unsupported_claims=["still bad"], explanation="still nope")

    monkeypatch.setattr(qa, "retrieve", lambda query, top_k: chunks)
    monkeypatch.setattr(qa, "generate_answer", lambda query, c, **kw: ("fabricated answer", TokenUsage.zero()))
    monkeypatch.setattr(
        qa, "regenerate_answer",
        lambda query, c, prev_text, claims, **kw: ("still fabricated answer", TokenUsage.zero()),
    )

    responses = iter([(first_pass, TokenUsage.zero()), (still_bad, TokenUsage.zero())])
    monkeypatch.setattr(qa, "check_faithfulness", lambda text, c, **kw: next(responses))

    monkeypatch.setattr(qa, "log_corrected", lambda *a: pytest.fail("should not log a correction that still failed"))
    monkeypatch.setattr(qa, "log_usage", lambda *a, **kw: None)
    logged_unfaithful = []
    monkeypatch.setattr(qa, "log_unfaithful", lambda query, text, f, citations: logged_unfaithful.append((query, text, f, citations)))
    monkeypatch.setattr(qa, "log_decline", lambda q: pytest.fail("should not decline"))

    result = qa.answer_question("question")

    assert result.text == "still fabricated answer"
    assert result.faithfulness is still_bad
    assert len(logged_unfaithful) == 1
    query, text, f, citations = logged_unfaithful[0]
    assert text == "still fabricated answer"
    assert f is still_bad
    assert citations == ["45 CFR 164.312"]


def test_answer_question_never_retries_on_empty_context_decline(monkeypatch):
    """Empty chunks always come back is_faithful=True from check_faithfulness's own
    short-circuit, but this pins down that the decline path can't reach the retry
    branch even if that ever changed."""
    monkeypatch.setattr(qa, "retrieve", lambda query, top_k: [])
    monkeypatch.setattr(qa, "generate_answer", lambda query, c, **kw: ("decline message", TokenUsage.zero()))
    monkeypatch.setattr(
        qa, "check_faithfulness",
        lambda text, c, **kw: (FaithfulnessCheck(is_faithful=False, unsupported_claims=["x"], explanation="y"), TokenUsage.zero()),
    )
    _no_retry(monkeypatch)
    monkeypatch.setattr(qa, "log_usage", lambda *a, **kw: None)
    monkeypatch.setattr(qa, "log_decline", lambda q: None)
    logged_unfaithful = []
    monkeypatch.setattr(qa, "log_unfaithful", lambda *a: logged_unfaithful.append(a))

    qa.answer_question("off topic question")

    # not chunks -> "elif not faithfulness.is_faithful" branch is unreachable (it's an elif on "if not chunks")
    assert logged_unfaithful == []
