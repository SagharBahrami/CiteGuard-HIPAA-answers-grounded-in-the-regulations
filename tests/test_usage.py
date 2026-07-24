import types

from usage import TokenUsage


def test_zero_has_no_tokens():
    assert TokenUsage.zero() == TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def test_from_response_reads_the_three_fields():
    response_usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)

    usage = TokenUsage.from_response(response_usage)

    assert usage == TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)


def test_add_combines_usage_across_calls():
    a = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    b = TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28)

    assert a + b == TokenUsage(prompt_tokens=120, completion_tokens=58, total_tokens=178)
