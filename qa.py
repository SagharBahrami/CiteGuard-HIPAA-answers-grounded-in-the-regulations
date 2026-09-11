"""Combine retrieval, generation, and the faithfulness guardrail into one call.

If the first-pass answer is flagged as unfaithful, it gets one corrective
retry -- regenerated with the specific unsupported claims fed back as
feedback, then re-checked. If the retry passes, it's returned with no
warning (the guardrail did its job silently); if it's still flagged, the
retried answer is returned with the faithfulness result attached, same as
before, rather than blocking outright. Blocking was considered and rejected:
evals/eval_guardrail.py measured the guardrail's precision at only 55.6% --
over a third of genuinely faithful answers get flagged too -- so hard-
blocking on any flag would incorrectly refuse a lot of fine answers.
"""

from dataclasses import dataclass

from audit import log_corrected, log_decline, log_unfaithful, log_usage
from generate import generate_answer, regenerate_answer
from guardrails import FaithfulnessCheck, check_faithfulness
from retriever import RetrievedChunk, retrieve
from version_query import extract_target_date, is_historical_query, resolve_as_of


@dataclass
class Answer:
    text: str
    sources: list[RetrievedChunk]
    faithfulness: FaithfulnessCheck
    historical: bool = False  # answered from archived text
    history_unavailable: bool = False  # a past version was asked for, but none is archived


def answer_question(query: str, top_k: int = 3) -> Answer:
    chunks = retrieve(query, top_k=top_k)

    # Asking for a past version doesn't mean one exists: the archive only goes
    # back as far as the first ingestion that superseded something. When
    # resolution can't actually reach the requested date it hands back today's
    # text unchanged, and framing that as "a superseded version" would be
    # exactly backwards for a compliance answer -- so only claim history when
    # a genuinely different version came back.
    historical = False
    history_unavailable = False
    if chunks and is_historical_query(query):
        resolved = resolve_as_of(chunks, extract_target_date(query))
        historical = any(
            r.issue_date != c.issue_date or r.text != c.text
            for r, c in zip(resolved, chunks)
        )
        if historical:
            chunks = resolved
        else:
            history_unavailable = True

    text, generation_usage = generate_answer(query, chunks, historical=historical)
    faithfulness, guardrail_usage = check_faithfulness(text, chunks, historical=historical)
    retried = False

    if chunks and not faithfulness.is_faithful:
        retried = True
        original_text = text
        original_claims = faithfulness.unsupported_claims
        text, retry_generation_usage = regenerate_answer(
            query, chunks, text, original_claims, historical=historical
        )
        faithfulness, retry_guardrail_usage = check_faithfulness(text, chunks, historical=historical)
        generation_usage = generation_usage + retry_generation_usage
        guardrail_usage = guardrail_usage + retry_guardrail_usage
        if faithfulness.is_faithful:
            log_corrected(query, original_text, text, original_claims)

    log_usage(query, generation_usage, guardrail_usage, retried=retried)
    if not chunks:
        log_decline(query)
    elif not faithfulness.is_faithful:
        log_unfaithful(query, text, faithfulness, [c.citation for c in chunks])

    return Answer(
        text=text, sources=chunks, faithfulness=faithfulness,
        historical=historical, history_unavailable=history_unavailable,
    )


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
