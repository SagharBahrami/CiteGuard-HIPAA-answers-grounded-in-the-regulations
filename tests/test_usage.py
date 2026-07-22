import types

from usage import TokenUsage


def test_zero_has_no_tokens():
    assert TokenUsage.zero() == TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def test_from_response_reads_the_three_fields():
    response_usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)

    usage = TokenUsage.from_response(response_usage)

    assert usage == TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
