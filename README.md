# CitedGuard

A retrieval-augmented Q&A system over 45 CFR Parts 160, 162, and 164 (the HIPAA
Administrative Simplification regulations — Privacy Rule, Security Rule,
transaction/code set standards, and identifiers), built as a learning project
covering RAG, hybrid retrieval, LLM guardrails, and (planned) a Redis/RQ task
queue.

## How it works

```
eCFR API  -->  ingest/  -->  Chroma (vectors)
                                  |
question --> retriever.py (hybrid search) --> generate.py --> guardrails.py --> answer
                                  |                                |
                            source chunks                   faithfulness check
```

1. **Ingestion** (`ingest/`) pulls the regulation text, splits it into
   citable chunks, embeds them, and stores them in a local Chroma
   collection. Run once (and again whenever you want to refresh the data).
2. **Retrieval** (`retriever.py`) finds the chunks relevant to a question.
3. **Generation** (`generate.py`) answers the question using only those
   chunks, with citations.
4. **Guardrails** (`guardrails.py`) independently checks the answer against
   its sources before it's returned.
5. **`qa.py`** wires 2-4 together into a single `answer_question(query)` call.

## Setup

```bash
pip install -e .
cp .env.example .env   # then fill in OPENAI_API_KEY
```

Config lives in `config.py` (via `pydantic-settings`, reading from `.env`):

| Setting | Default | Purpose |
|---|---|---|
| `openai_api_key` | *(required)* | OpenAI API key |
| `generation_model` | `gpt-5.6-terra` | Answers the question (only supports default temperature) |
| `guardrail_model` | `gpt-5.6-luna` | Runs the faithfulness check |
| `embedding_model` | `text-embedding-3-small` | Embeds chunks and queries |
| `redis_url` | `redis://localhost:6379/0` | For the planned RQ worker (not yet built) |
| `chroma_dir` | `./chroma_db` | Where the vector store persists |
| `chroma_collection` | `hipaa_regs` | Collection name |
| `similarity_threshold` | `0.35` | Minimum cosine similarity for a chunk to count as relevant |

## Usage

**Ingest the regulation text** (fetches from eCFR's official API, chunks,
embeds, and stores in Chroma — costs a small amount in embedding API calls):

```bash
python -m ingest          # skips parts already cached in data/raw/
python -m ingest --force  # re-fetches everything
```

**Ask a question:**

```bash
python -m qa
```

Or from code:

```python
from qa import answer_question

result = answer_question("What are the technical safeguards for encryption?")
print(result.text)
print(result.faithfulness.is_faithful)
for source in result.sources:
    print(source.citation, source.similarity)
```

## Ingestion pipeline details

- **`ingest/fetch.py`** — Downloads raw XML for Parts 160/162/164 from
  eCFR's Content Versioner API (`api.ecfr.gov`), not by scraping
  `www.ecfr.gov` directly — the rendered site actively blocks scraping with
  a CAPTCHA wall. Caches to `data/raw/`; skips parts already on disk unless
  `force=True`.
- **`ingest/parse.py`** — Parses each part's XML into `Section` records
  (citation, heading, part, subpart, ordered paragraphs). The `(a)/(1)/(i)`
  legal outline within a section is flat text in the XML, not real nesting,
  so it isn't reconstructed here.
- **`ingest/chunk.py`** — Turns each `Section` into one or more `Chunk`s.
  A section under ~1500 characters stays whole; longer ones are split by
  greedily packing paragraphs up to that size, **splitting only at paragraph
  boundaries** (never mid-sentence). Split chunks repeat their previous
  chunk's last paragraph as lead-in context, so a chunk retrieved on its own
  isn't missing the requirement/standard it's an implementation detail of.
- **`ingest/embed_store.py`** — Embeds chunks in batches of 100 via the
  OpenAI embeddings API and upserts them into Chroma with deterministic IDs
  (`<section>_<chunk_index>`), so re-running ingestion updates existing rows
  instead of duplicating them. The Chroma collection is explicitly created
  with `hnsw:space=cosine` so similarity scores have a well-defined meaning
  (`1 - distance`).

Current corpus: 148 sections -> 427 chunks.

## Retrieval details (`retriever.py`)

Hybrid search, but not naive rank fusion:

- **Dense search** (OpenAI embeddings + cosine similarity) is the primary,
  calibrated signal — it's what `similarity_threshold` is tuned against.
  Since the corpus is only ~400 chunks, every chunk's embedding is scored
  exactly against the query rather than using an approximate index.
- **Sparse search** (BM25 over stopword-filtered tokens) is used only to
  **rescue** chunks dense search missed entirely (e.g. a bare citation
  number like "164.514", which embeds poorly but matches exactly on
  keywords) — appended after the dense results, never reordering them.
- Equal-weight rank fusion (RRF) was tried first and rejected: a
  keyword-dense-but-shallow section (164.304, Definitions, which briefly
  touches dozens of terms) could out-rank the single best dense match on
  almost any query.
- Rescued chunks still have to clear a relaxed secondary similarity floor
  (`similarity_threshold * 0.5`) — raw BM25 score magnitude alone isn't
  comparable across different queries (an off-topic query sharing a common
  regulatory word like "change" can outscore a genuinely relevant one), so
  rescue candidates are still sanity-checked against the dense signal.
- Results are deduplicated by citation (best chunk per section) so the
  top-k spans distinct sections instead of several slots going to the same
  long section.

## Guardrails

Everything below is a mechanism specifically aimed at reducing hallucination
or out-of-scope answers. "Level" describes how much is actually
implemented vs. planned.

| # | Guardrail | Where | Level | What it does |
|---|---|---|---|---|
| 1 | **Relevance threshold** | `retriever.py` | **Fully implemented** | Chunks below `similarity_threshold` (0.35 cosine) are dropped before generation ever sees them. |
| 2 | **Decline on empty context** | `generate.py` (`NO_CONTEXT_MESSAGE`) | **Fully implemented** | If retrieval returns zero chunks, a fixed decline message is returned *without calling the LLM at all* — no chance to hallucinate from general knowledge. |
| 3 | **Grounding instruction (prompt-level)** | `generate.py` (`SYSTEM_PROMPT`) | **Fully implemented, soft guardrail** | Instructs the model to answer only from the provided excerpts, cite the specific CFR section per claim, and say explicitly when the excerpts don't fully answer the question. This relies on model compliance — it isn't verified, which is what guardrail #4 is for. |
| 4 | **Faithfulness / groundedness check** | `guardrails.py` (`check_faithfulness`) | **Fully implemented, "warn" level** | A *second*, independent LLM call (`guardrail_model`, separate from `generation_model`) re-checks the generated answer against the same source excerpts and flags any claim not actually supported, via structured output (`is_faithful`, `unsupported_claims`, `explanation`). Verified against both a real answer (passed) and a deliberately fabricated one (correctly flagged, with the specific invented claim identified). |
| 5 | **Rescue-candidate sanity floor** | `retriever.py` (`RESCUE_SIMILARITY_THRESHOLD`) | **Fully implemented** | Prevents the BM25 rescue mechanism from pulling in chunks with no real semantic relationship to the query, just because of incidental keyword overlap with common regulatory vocabulary. |

**What guardrail #4 currently does *not* do** (a deliberate choice, not a gap
that was missed): when an answer is flagged unfaithful, it is **not**
blocked or auto-retried — `qa.answer_question()` returns the answer *with*
the faithfulness result attached, so a caller (e.g. the future UI) can show
a warning banner. Two alternatives were considered and explicitly declined
for now:
- *Retry once with a stricter prompt* (feed the unsupported claims back in
  and regenerate) — an extra LLM call, more likely to self-correct, not yet
  built.
- *Block and replace with a safe decline* — safest for compliance, but more
  likely to unhelpfully refuse borderline-fine answers.

**Not implemented / not yet planned in detail:**
- PII/PHI redaction of user input or model output.
- Rate limiting or abuse controls.
- Persistent audit logging of guardrail triggers (faithfulness failures,
  declined answers) for later review.

## Project status

| Piece | Status |
|---|---|
| Ingestion pipeline (fetch/parse/chunk/embed/store) | Done |
| Hybrid retrieval | Done |
| Generation with citations | Done |
| Faithfulness guardrail | Done (warn-only) |
| RQ worker / job queue | Not started |
| Streamlit UI | Not started |
