from citedguard.config import settings
from conftest import DEFAULT_USAGE, FakeOpenAIClient
from citedguard.generate import (
    CORRECTION_SYSTEM_PROMPT,
    HISTORICAL_CORRECTION_SYSTEM_PROMPT,
    HISTORICAL_SYSTEM_PROMPT,
    NO_CONTEXT_MESSAGE,
    SYSTEM_PROMPT,
    generate_answer,
    regenerate_answer,
)
from citedguard.retriever import RetrievedChunk
from citedguard.usage import TokenUsage


def _chunk(citation="45 CFR 164.312", text="Encrypt ePHI.", issue_date=""):
    return RetrievedChunk(
        citation=citation, heading="Technical safeguards", part=164, subpart="", text=text, similarity=0.9,
        issue_date=issue_date,
    )


def test_empty_chunks_returns_fixed_decline_without_calling_llm():
    client = FakeOpenAIClient(chat_content="should never be returned")

    answer, usage = generate_answer("irrelevant question", [], client=client)

    assert answer == NO_CONTEXT_MESSAGE
    assert client.calls == []
    assert usage == TokenUsage.zero()


def test_generates_answer_from_chunks_via_configured_model():
    client = FakeOpenAIClient(chat_content="Answer citing 45 CFR 164.312.")

    answer, usage = generate_answer("What are the technical safeguards?", [_chunk()], client=client)

    assert answer == "Answer citing 45 CFR 164.312."
    assert len(client.calls) == 1
    _, model, _ = client.calls[0]
    assert model == settings.generation_model
    assert usage == TokenUsage.from_response(DEFAULT_USAGE)


def test_context_message_includes_citation_and_chunk_text():
    client = FakeOpenAIClient(chat_content="answer")

    generate_answer("q", [_chunk(text="Specific encryption text.")], client=client)

    _, _, messages = client.calls[0]
    user_message = messages[1]["content"]
    assert "45 CFR 164.312" in user_message
    assert "Specific encryption text." in user_message
    assert "q" in user_message


def test_context_message_includes_issue_date_when_the_chunk_has_one():
    client = FakeOpenAIClient(chat_content="answer")

    generate_answer("q", [_chunk(issue_date="2026-07-02")], client=client)

    _, _, messages = client.calls[0]
    assert "45 CFR 164.312, as of 2026-07-02" in messages[1]["content"]


def test_context_message_omits_issue_date_when_the_chunk_has_none():
    client = FakeOpenAIClient(chat_content="answer")

    generate_answer("q", [_chunk(issue_date="")], client=client)

    _, _, messages = client.calls[0]
    assert "[45 CFR 164.312]" in messages[1]["content"]


def test_historical_flag_selects_the_past_version_system_prompt():
    client = FakeOpenAIClient(chat_content="answer")

    generate_answer("q", [_chunk(issue_date="2023-01-01")], client=client, historical=True)

    _, _, messages = client.calls[0]
    assert messages[0]["content"] == HISTORICAL_SYSTEM_PROMPT


def test_default_generation_uses_the_current_version_system_prompt():
    client = FakeOpenAIClient(chat_content="answer")

    generate_answer("q", [_chunk()], client=client)

    _, _, messages = client.calls[0]
    assert messages[0]["content"] == SYSTEM_PROMPT


def test_historical_flag_selects_the_past_version_correction_prompt():
    client = FakeOpenAIClient(chat_content="answer")

    regenerate_answer("q", [_chunk()], "prev", ["claim"], client=client, historical=True)

    _, _, messages = client.calls[0]
    assert messages[0]["content"] == HISTORICAL_CORRECTION_SYSTEM_PROMPT


def test_default_correction_uses_the_current_version_correction_prompt():
    client = FakeOpenAIClient(chat_content="answer")

    regenerate_answer("q", [_chunk()], "prev", ["claim"], client=client)

    _, _, messages = client.calls[0]
    assert messages[0]["content"] == CORRECTION_SYSTEM_PROMPT


def test_regenerate_answer_returns_corrected_text_and_usage():
    client = FakeOpenAIClient(chat_content="Corrected answer citing 45 CFR 164.312.")

    answer, usage = regenerate_answer(
        "What are the technical safeguards?", [_chunk()], "Original unsupported answer.",
        ["AES-256 is mandated"], client=client,
    )

    assert answer == "Corrected answer citing 45 CFR 164.312."
    assert usage == TokenUsage.from_response(DEFAULT_USAGE)


def test_regenerate_answer_prompt_includes_previous_answer_and_flagged_claims():
    client = FakeOpenAIClient(chat_content="answer")

    regenerate_answer(
        "q", [_chunk(text="Specific encryption text.")], "The original flawed answer.",
        ["AES-256 is mandated", "keys rotate every 90 days"], client=client,
    )

    _, _, messages = client.calls[0]
    user_message = messages[1]["content"]
    assert "The original flawed answer." in user_message
    assert "AES-256 is mandated" in user_message
    assert "keys rotate every 90 days" in user_message
    assert "Specific encryption text." in user_message
