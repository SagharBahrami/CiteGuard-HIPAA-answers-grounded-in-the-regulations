from config import settings
from conftest import DEFAULT_USAGE, FakeOpenAIClient
from generate import NO_CONTEXT_MESSAGE, generate_answer
from retriever import RetrievedChunk
from usage import TokenUsage


def _chunk(citation="45 CFR 164.312", text="Encrypt ePHI."):
    return RetrievedChunk(
        citation=citation, heading="Technical safeguards", part=164, subpart="", text=text, similarity=0.9
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
