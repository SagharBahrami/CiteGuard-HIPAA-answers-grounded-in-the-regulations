"""Generate an answer to a HIPAA question from retrieved regulation chunks.

If no chunks were retrieved, this returns a fixed decline-to-answer message
without calling the LLM at all -- there's no basis to answer from, so there's
nothing for the model to usefully do with the question.
"""

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from retriever import RetrievedChunk
from usage import TokenUsage

NO_CONTEXT_MESSAGE = (
    "I don't have enough relevant information in the HIPAA regulations I have "
    "access to (45 CFR Parts 160, 162, 164) to answer that question."
)

SYSTEM_PROMPT = (
    "You are a HIPAA compliance assistant. Answer the user's question using ONLY "
    "the regulation excerpts provided below -- do not use outside knowledge. "
    "For every claim, cite the specific CFR section it comes from (e.g. '45 CFR "
    "164.312'). If the excerpts only partially answer the question, or don't "
    "answer it at all, say so explicitly instead of filling the gap yourself."
)

CORRECTION_SYSTEM_PROMPT = (
    "You are a HIPAA compliance assistant revising a previous answer that an "
    "independent fact-checker flagged as containing claims not supported by the "
    "provided regulation excerpts. Produce a corrected answer using ONLY the "
    "excerpts below -- remove or fix each flagged claim without inventing new "
    "ones. If removing the unsupported claims leaves the question only partially "
    "answered, say so explicitly rather than filling the gap yourself."
)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.citation}] {c.heading}\n{c.text}" for c in chunks)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    client: OpenAI | None = None,
) -> tuple[str, TokenUsage]:
    if not chunks:
        return NO_CONTEXT_MESSAGE, TokenUsage.zero()

    client = client or OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.generation_model,
        # generation_model only supports the default temperature (1), not 0
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Regulation excerpts:\n\n{_format_context(chunks)}\n\nQuestion: {query}",
            },
        ],
    )
    return response.choices[0].message.content, TokenUsage.from_response(response.usage)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def regenerate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    previous_answer: str,
    unsupported_claims: list[str],
    client: OpenAI | None = None,
) -> tuple[str, TokenUsage]:
    client = client or OpenAI(api_key=settings.openai_api_key)
    flagged = "\n".join(f"- {c}" for c in unsupported_claims)
    response = client.chat.completions.create(
        model=settings.generation_model,
        messages=[
            {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Regulation excerpts:\n\n{_format_context(chunks)}\n\n"
                    f"Question: {query}\n\n"
                    f"Previous answer:\n{previous_answer}\n\n"
                    f"Claims flagged as NOT supported by the excerpts:\n{flagged}"
                ),
            },
        ],
    )
    return response.choices[0].message.content, TokenUsage.from_response(response.usage)


if __name__ == "__main__":
    from retriever import retrieve

    for q in [
        "What are the technical safeguards for encryption?",
        "How does climate change affect coral reefs?",
    ]:
        chunks = retrieve(q)
        answer, usage = generate_answer(q, chunks)
        print(f"\nQuery: {q!r}\nAnswer: {answer}\nTokens: {usage}")
