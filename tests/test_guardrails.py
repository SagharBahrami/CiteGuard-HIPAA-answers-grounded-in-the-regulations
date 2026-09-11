from config import settings
from conftest import DEFAULT_USAGE, FakeOpenAIClient
from guardrails import (
    GUARDRAIL_SYSTEM_PROMPT,
    HISTORICAL_GUARDRAIL_SYSTEM_PROMPT,
    FaithfulnessCheck,
    check_faithfulness,
)
from retriever import RetrievedChunk
from usage import TokenUsage


def _chunk():
    return RetrievedChunk(
        citation="45 CFR 164.312", heading="Technical safeguards", part=164, subpart="", text="Encrypt ePHI.",
        similarity=0.9,
    )


def test_no_chunks_short_circuits_to_faithful_without_calling_llm():
    client = FakeOpenAIClient(parsed_result=None)

    result, usage = check_faithfulness("some answer", [], client=client)

    assert result.is_faithful is True
    assert result.unsupported_claims == []
    assert client.calls == []
    assert usage == TokenUsage.zero()


def test_faithful_result_is_passed_through():
    expected = FaithfulnessCheck(is_faithful=True, unsupported_claims=[], explanation="fully supported")
    client = FakeOpenAIClient(parsed_result=expected)

    result, usage = check_faithfulness("answer", [_chunk()], client=client)

    assert result is expected
    _, model, _ = client.calls[0]
    assert model == settings.guardrail_model
    assert usage == TokenUsage.from_response(DEFAULT_USAGE)


def test_historical_flag_selects_the_carve_out_guardrail_prompt():
    """Without this, the required "past version" disclaimer gets flagged as an
    unsupported claim on every historical answer."""
    client = FakeOpenAIClient(
        parsed_result=FaithfulnessCheck(is_faithful=True, unsupported_claims=[], explanation="ok")
    )

    check_faithfulness("answer", [_chunk()], client=client, historical=True)

    _, _, messages = client.calls[0]
    assert messages[0]["content"] == HISTORICAL_GUARDRAIL_SYSTEM_PROMPT


def test_default_check_uses_the_standard_guardrail_prompt():
    client = FakeOpenAIClient(
        parsed_result=FaithfulnessCheck(is_faithful=True, unsupported_claims=[], explanation="ok")
    )

    check_faithfulness("answer", [_chunk()], client=client)

    _, _, messages = client.calls[0]
    assert messages[0]["content"] == GUARDRAIL_SYSTEM_PROMPT


def test_unfaithful_result_flags_unsupported_claims():
    expected = FaithfulnessCheck(
        is_faithful=False, unsupported_claims=["AES-256 is mandated"], explanation="not supported by excerpts"
    )
    client = FakeOpenAIClient(parsed_result=expected)

    result, usage = check_faithfulness("fabricated answer", [_chunk()], client=client)

    assert result.is_faithful is False
    assert result.unsupported_claims == ["AES-256 is mandated"]


def test_context_message_includes_answer_and_source_text():
    client = FakeOpenAIClient(parsed_result=FaithfulnessCheck(is_faithful=True, unsupported_claims=[], explanation=""))

    check_faithfulness("the generated answer", [_chunk()], client=client)

    _, _, messages = client.calls[0]
    user_message = messages[1]["content"]
    assert "the generated answer" in user_message
    assert "Encrypt ePHI." in user_message
