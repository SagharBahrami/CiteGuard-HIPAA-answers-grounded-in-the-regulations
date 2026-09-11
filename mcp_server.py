"""MCP server exposing CitedGuard's HIPAA Q&A pipeline as tools.

Two tools, at two different cost/latency points:
- ask_hipaa_question: full pipeline (retrieval + generation + faithfulness
  guardrail), for when the caller wants a synthesized, grounded answer. Goes
  through the same Redis/RQ job queue as the Streamlit UI (jobs.py), so MCP
  callers get the same worker-pool parallelism instead of running inline in
  whatever process is handling the tool call.
- search_hipaa_regulations: retrieval only, for when the caller would rather
  read the source excerpts itself -- no generation or guardrail LLM calls, so
  it stays direct rather than going through the queue; it's cheap and fast
  enough that queuing it would only add latency.

Run with: python mcp_server.py (stdio transport, for local MCP clients like
Claude Desktop or Claude Code to launch as a subprocess). Requires Redis and
a worker to be running for ask_hipaa_question -- see README.
"""

import time

from mcp.server.fastmcp import FastMCP

from jobs import enqueue_question
from retriever import retrieve

mcp = FastMCP("citedguard")


@mcp.tool()
def ask_hipaa_question(query: str, top_k: int = 3) -> dict:
    """Answer a question about the HIPAA Administrative Simplification
    regulations (45 CFR Parts 160, 162, 164), grounded in retrieved
    regulation text and independently checked for faithfulness before
    being returned.

    Args:
        query: The HIPAA-related question to answer.
        top_k: Maximum number of regulation sections to use as context.
    """
    job = enqueue_question(query, top_k=top_k)
    while job.get_status(refresh=True) not in ("finished", "failed"):
        time.sleep(0.5)
    if job.get_status() == "failed":
        raise RuntimeError(f"ask_hipaa_question job failed: {job.exc_info}")

    result = job.result
    return {
        "answer": result.text,
        "is_faithful": result.faithfulness.is_faithful,
        "unsupported_claims": result.faithfulness.unsupported_claims,
        # True when the question was read as asking about a past version, so the
        # sources below are archived text rather than what currently applies.
        "historical": result.historical,
        # True when a past version was asked for but none is archived for that
        # date -- the sources below are current text, not the version requested.
        "history_unavailable": result.history_unavailable,
        "sources": [
            {
                "citation": s.citation, "heading": s.heading, "similarity": round(s.similarity, 3),
                "issue_date": s.issue_date,
            }
            for s in result.sources
        ],
    }


@mcp.tool()
def search_hipaa_regulations(query: str, top_k: int = 5) -> list[dict]:
    """Search the HIPAA Administrative Simplification regulations (45 CFR
    Parts 160, 162, 164) for sections relevant to a query, returning the raw
    excerpts without generating an answer -- cheaper than ask_hipaa_question
    when the caller wants to read or synthesize the source text itself.

    Args:
        query: The search query.
        top_k: Maximum number of regulation sections to return.
    """
    chunks = retrieve(query, top_k=top_k)
    return [
        {
            "citation": c.citation, "heading": c.heading, "similarity": round(c.similarity, 3),
            "issue_date": c.issue_date, "text": c.text,
        }
        for c in chunks
    ]


if __name__ == "__main__":
    mcp.run()
