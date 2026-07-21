"""Faithfulness guardrail: check a generated answer against its source chunks.

Runs as a separate LLM call using guardrail_model, independent from the
generation call, so it isn't just the same model defending its own answer.
"""

from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from retriever import RetrievedChunk

GUARDRAIL_SYSTEM_PROMPT = (
    "You are a fact-checker reviewing whether an AI-generated answer is fully "
    "supported by the regulation excerpts it was given. Mark is_faithful=false "
    "if the answer states anything -- a fact, a citation, a requirement -- that "
    "is not directly supported by the excerpts. List each unsupported claim "
    "verbatim in unsupported_claims. If the answer is fully supported, return "
    "is_faithful=true and an empty unsupported_claims list."
)


class FaithfulnessCheck(BaseModel):
    is_faithful: bool
    unsupported_claims: list[str]
    explanation: str


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.citation}] {c.heading}\n{c.text}" for c in chunks)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def check_faithfulness(
    answer: str,
    chunks: list[RetrievedChunk],
    client: OpenAI | None = None,
) -> FaithfulnessCheck:
    if not chunks:
        return FaithfulnessCheck(
            is_faithful=True,
            unsupported_claims=[],
            explanation="No source excerpts were provided; answer declines to answer.",
        )

    client = client or OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.parse(
        model=settings.guardrail_model,
        response_format=FaithfulnessCheck,
        messages=[
            {"role": "system", "content": GUARDRAIL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Regulation excerpts:\n\n{_format_context(chunks)}\n\n"
                    f"Generated answer:\n{answer}"
                ),
            },
        ],
    )
    return response.choices[0].message.parsed


if __name__ == "__main__":
    from qa import answer_question

    result = answer_question("What are the technical safeguards for encryption?")
    print("--- Real answer ---")
    check = check_faithfulness(result.text, result.sources)
    print("is_faithful:", check.is_faithful)
    print("unsupported_claims:", check.unsupported_claims)
    print("explanation:", check.explanation)

    print("\n--- Deliberately fabricated answer ---")
    fake_answer = (
        "Under 45 CFR 164.312, all covered entities must encrypt ePHI using "
        "AES-256 encryption and rotate keys every 90 days, as mandated by the "
        "Security Rule's Technical Safeguards."
    )
    fake_check = check_faithfulness(fake_answer, result.sources)
    print("is_faithful:", fake_check.is_faithful)
    print("unsupported_claims:", fake_check.unsupported_claims)
    print("explanation:", fake_check.explanation)
