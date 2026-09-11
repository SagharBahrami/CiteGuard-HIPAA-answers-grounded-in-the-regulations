"""Faithfulness guardrail: check a generated answer against its source chunks.

Runs as a separate LLM call using guardrail_model, independent from the
generation call, so it isn't just the same model defending its own answer.
"""

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from citedguard.config import settings
from citedguard.retriever import RetrievedChunk, format_context
from citedguard.usage import TokenUsage

_RETRYABLE = retry_if_exception_type(
    (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client

GUARDRAIL_SYSTEM_PROMPT = (
    "You are a fact-checker reviewing whether an AI-generated answer is fully "
    "supported by the regulation excerpts it was given. Mark is_faithful=false "
    "if the answer states anything -- a fact, a citation, a requirement -- that "
    "is not directly supported by the excerpts. List each unsupported claim "
    "verbatim in unsupported_claims. If the answer is fully supported, return "
    "is_faithful=true and an empty unsupported_claims list."
)

# The excerpts here are archived text the answer was deliberately asked to
# frame as superseded, so the answer is *required* to carry a disclaimer that
# no excerpt can support on its own. Without this carve-out the guardrail
# flags that disclaimer on every historical answer, which both warns the user
# spuriously and burns a corrective retry that's instructed to keep it anyway.
HISTORICAL_GUARDRAIL_SYSTEM_PROMPT = (
    GUARDRAIL_SYSTEM_PROMPT
    + " These excerpts are a PAST version of the regulation, each labeled with "
    "the issue date it was in effect. Do NOT flag the answer for stating which "
    "issue date its excerpts come from, or for cautioning that this is a "
    "past/superseded version that may not reflect what currently applies -- "
    "that framing is required and is supported by the excerpt labels. Judge "
    "only the regulatory substance against the excerpt text."
)


class FaithfulnessCheck(BaseModel):
    is_faithful: bool
    unsupported_claims: list[str]
    explanation: str


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=_RETRYABLE)
def check_faithfulness(
    answer: str,
    chunks: list[RetrievedChunk],
    client: OpenAI | None = None,
    historical: bool = False,
) -> tuple[FaithfulnessCheck, TokenUsage]:
    if not chunks:
        return (
            FaithfulnessCheck(
                is_faithful=True,
                unsupported_claims=[],
                explanation="No source excerpts were provided; answer declines to answer.",
            ),
            TokenUsage.zero(),
        )

    client = client or _get_client()
    response = client.chat.completions.parse(
        model=settings.guardrail_model,
        response_format=FaithfulnessCheck,
        messages=[
            {
                "role": "system",
                "content": HISTORICAL_GUARDRAIL_SYSTEM_PROMPT if historical else GUARDRAIL_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Regulation excerpts:\n\n{format_context(chunks)}\n\n"
                    f"Generated answer:\n{answer}"
                ),
            },
        ],
    )
    return response.choices[0].message.parsed, TokenUsage.from_response(response.usage)


if __name__ == "__main__":
    from citedguard.qa import answer_question

    result = answer_question("What are the technical safeguards for encryption?")
    print("--- Real answer ---")
    check, usage = check_faithfulness(result.text, result.sources)
    print("is_faithful:", check.is_faithful)
    print("unsupported_claims:", check.unsupported_claims)
    print("explanation:", check.explanation)
    print("tokens:", usage)

    print("\n--- Deliberately fabricated answer ---")
    fake_answer = (
        "Under 45 CFR 164.312, all covered entities must encrypt ePHI using "
        "AES-256 encryption and rotate keys every 90 days, as mandated by the "
        "Security Rule's Technical Safeguards."
    )
    fake_check, fake_usage = check_faithfulness(fake_answer, result.sources)
    print("is_faithful:", fake_check.is_faithful)
    print("unsupported_claims:", fake_check.unsupported_claims)
    print("explanation:", fake_check.explanation)
    print("tokens:", fake_usage)
