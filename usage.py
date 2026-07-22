"""Token usage accounting for OpenAI API calls.

Kept separate from audit.py (rather than folding TokenUsage in there) because
generate.py and guardrails.py both need it, and audit.py already imports
FaithfulnessCheck from guardrails.py -- putting TokenUsage in audit.py too
would make audit and guardrails import each other.
"""

from dataclasses import dataclass


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def zero(cls) -> "TokenUsage":
        """No LLM call was made (e.g. an empty-context short-circuit)."""
        return cls(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    @classmethod
    def from_response(cls, usage) -> "TokenUsage":
        """Build from an OpenAI response's `.usage` field."""
        return cls(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
