"""Streamlit chat UI over qa.answer_question.

Keeps the guardrail result visible rather than hiding it: every answer shows
whether the faithfulness check passed, and flags the specific unsupported
claims when it didn't, right alongside the source excerpts it was checked
against.

Questions are submitted through the Redis/RQ job queue (jobs.py) rather than
calling answer_question directly, so a separate worker process does the
actual OpenAI calls -- this naturally bounds how many questions run
concurrently to however many workers are running, instead of one per
Streamlit session. Requires a worker to be running (see README); if none is,
this will hang in "Queued..." until one picks the job up.
"""

import time

import streamlit as st

from jobs import enqueue_question
from qa import Answer

st.set_page_config(page_title="CitedGuard", page_icon="\U0001f4dc")

st.title("CitedGuard")
st.caption(
    "Ask questions about the HIPAA Administrative Simplification regulations "
    "(45 CFR Parts 160, 162, 164). Answers are grounded in retrieved regulation "
    "text and independently checked for faithfulness before being shown."
)

if "history" not in st.session_state:
    st.session_state.history = []  # list[tuple[str, Answer]]


def render_answer(answer: Answer) -> None:
    st.markdown(answer.text)

    if not answer.faithfulness.is_faithful:
        st.warning(
            "This answer may contain claims not fully supported by the source "
            "excerpts below.",
            icon="⚠️",
        )
        for claim in answer.faithfulness.unsupported_claims:
            st.markdown(f"- {claim}")

    if answer.sources:
        with st.expander(f"Sources ({len(answer.sources)})"):
            for source in answer.sources:
                st.markdown(
                    f"**{source.citation}** — {source.heading} "
                    f"(similarity={source.similarity:.3f})"
                )
                st.text(source.text)
                st.divider()


for question, answer in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        render_answer(answer)

question = st.chat_input("What are the technical safeguards for encryption?")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Queued — retrieving and generating..."):
            job = enqueue_question(question)
            while job.get_status(refresh=True) not in ("finished", "failed"):
                time.sleep(0.5)
            if job.get_status() == "failed":
                st.error(f"Something went wrong: {job.exc_info}")
                st.stop()
            answer = job.result
        render_answer(answer)

    st.session_state.history.append((question, answer))
