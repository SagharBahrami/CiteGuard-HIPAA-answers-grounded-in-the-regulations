"""Combine retrieval, generation, and the faithfulness guardrail into one call."""

from dataclasses import dataclass

from audit import log_decline, log_unfaithful
from generate import generate_answer
from guardrails import FaithfulnessCheck, check_faithfulness
from retriever import RetrievedChunk, retrieve


@dataclass
class Answer:
    text: str
    sources: list[RetrievedChunk]
    faithfulness: FaithfulnessCheck


def answer_question(query: str, top_k: int = 5) -> Answer:
    chunks = retrieve(query, top_k=top_k)
    text = generate_answer(query, chunks)
    faithfulness = check_faithfulness(text, chunks)

    if not chunks:
        log_decline(query)
    elif not faithfulness.is_faithful:
        log_unfaithful(query, text, faithfulness, [c.citation for c in chunks])

    return Answer(text=text, sources=chunks, faithfulness=faithfulness)


if __name__ == "__main__":
    result = answer_question("What are the technical safeguards for encryption?")
    print(result.text)

    if not result.faithfulness.is_faithful:
        print("\nWARNING: parts of this answer may not be fully supported by the source excerpts.")
        for claim in result.faithfulness.unsupported_claims:
            print(f"  - {claim}")

    print("\nSources:")
    for c in result.sources:
        print(f"  {c.citation} (similarity={c.similarity:.3f})")
