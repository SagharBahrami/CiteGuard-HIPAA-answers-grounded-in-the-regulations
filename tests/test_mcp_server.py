import anyio

import mcp_server
from guardrails import FaithfulnessCheck
from qa import Answer
from retriever import RetrievedChunk


def _chunk(citation="45 CFR 164.312", text="Encrypt ePHI."):
    return RetrievedChunk(citation=citation, heading="Technical safeguards", part=164, subpart="", text=text, similarity=0.876)


class _FakeJob:
    def __init__(self, result=None, status="finished", exc_info=None):
        self.result = result
        self._status = status
        self.exc_info = exc_info

    def get_status(self, refresh=False):
        return self._status


def test_ask_hipaa_question_shapes_the_full_answer(monkeypatch):
    answer = Answer(
        text="Encrypt ePHI in transit and at rest.",
        sources=[_chunk()],
        faithfulness=FaithfulnessCheck(is_faithful=True, unsupported_claims=[], explanation="ok"),
    )
    calls = []
    monkeypatch.setattr(
        mcp_server, "enqueue_question",
        lambda query, top_k: calls.append((query, top_k)) or _FakeJob(result=answer),
    )

    result = mcp_server.ask_hipaa_question("What are the technical safeguards?", top_k=3)

    assert calls == [("What are the technical safeguards?", 3)]
    assert result == {
        "answer": "Encrypt ePHI in transit and at rest.",
        "is_faithful": True,
        "unsupported_claims": [],
        "sources": [{"citation": "45 CFR 164.312", "heading": "Technical safeguards", "similarity": 0.876}],
    }


def test_ask_hipaa_question_surfaces_unfaithful_claims(monkeypatch):
    answer = Answer(
        text="fabricated answer",
        sources=[_chunk()],
        faithfulness=FaithfulnessCheck(is_faithful=False, unsupported_claims=["bad claim"], explanation="nope"),
    )
    monkeypatch.setattr(mcp_server, "enqueue_question", lambda query, top_k: _FakeJob(result=answer))

    result = mcp_server.ask_hipaa_question("q")

    assert result["is_faithful"] is False
    assert result["unsupported_claims"] == ["bad claim"]


def test_ask_hipaa_question_raises_when_job_fails(monkeypatch):
    monkeypatch.setattr(
        mcp_server, "enqueue_question",
        lambda query, top_k: _FakeJob(status="failed", exc_info="boom"),
    )

    try:
        mcp_server.ask_hipaa_question("q")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "boom" in str(e)


def test_search_hipaa_regulations_returns_raw_excerpts(monkeypatch):
    chunks = [_chunk(citation="45 CFR 1", text="alpha"), _chunk(citation="45 CFR 2", text="beta")]
    calls = []
    monkeypatch.setattr(mcp_server, "retrieve", lambda query, top_k: calls.append((query, top_k)) or chunks)

    result = mcp_server.search_hipaa_regulations("access control", top_k=5)

    assert calls == [("access control", 5)]
    assert result == [
        {"citation": "45 CFR 1", "heading": "Technical safeguards", "similarity": 0.876, "text": "alpha"},
        {"citation": "45 CFR 2", "heading": "Technical safeguards", "similarity": 0.876, "text": "beta"},
    ]


def test_tools_are_registered_with_fastmcp():
    tools = anyio.run(mcp_server.mcp.list_tools)
    tool_names = {t.name for t in tools}
    assert {"ask_hipaa_question", "search_hipaa_regulations"} <= tool_names
