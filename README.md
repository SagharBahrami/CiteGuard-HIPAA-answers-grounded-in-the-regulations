# CitedGuard

A retrieval-augmented Q&A system over 45 CFR Parts 160, 162, and 164 (the HIPAA
Administrative Simplification regulations — Privacy Rule, Security Rule,
transaction/code set standards, and identifiers), built as a learning project
covering RAG, hybrid retrieval, LLM guardrails, and a Redis/RQ task queue.

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

Run the test suite (mocks the OpenAI API and, where relevant, Chroma/network
calls — no API costs or external requests):

```bash
pytest
```

Config lives in `config.py` (via `pydantic-settings`, reading from `.env`):

| Setting | Default | Purpose |
|---|---|---|
| `openai_api_key` | *(required)* | OpenAI API key |
| `generation_model` | `gpt-5.6-terra` | Answers the question (only supports default temperature) |
| `guardrail_model` | `gpt-5.6-luna` | Runs the faithfulness check |
| `embedding_model` | `text-embedding-3-small` | Embeds chunks and queries |
| `redis_url` | `redis://localhost:6379/0` | Where the RQ job queue connects (needs a running Redis; see below) |
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

**Ask a question** (calls `qa.answer_question` directly — synchronous, no
Redis/worker needed):

```bash
python -m qa
```

**Or launch the chat UI** (goes through the Redis/RQ job queue instead of
calling `answer_question` directly — needs Redis and a worker running; see
[Background jobs](#background-jobs-redisrq) below):

```bash
streamlit run app.py
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

**Or expose it to an MCP client** (Claude Desktop, Claude Code, etc.) via
`mcp_server.py`, which serves two tools over stdio:

- `ask_hipaa_question(query, top_k=3)` — full pipeline (retrieval +
  generation + faithfulness guardrail), returns the answer, faithfulness
  result, and sources. Goes through the same Redis/RQ job queue as the
  Streamlit UI (see [Background jobs](#background-jobs-redisrq) below) —
  requires Redis and a worker running.
- `search_hipaa_regulations(query, top_k=5)` — retrieval only, for when the
  caller would rather read the source excerpts itself. No generation or
  guardrail LLM calls, so it's cheaper.

Add it to Claude Desktop's config (`claude_desktop_config.json`) or Claude
Code's (`claude mcp add`):

```json
{
  "mcpServers": {
    "citedguard": {
      "command": "/absolute/path/to/CitedGuard/.venv/Scripts/python.exe",
      "args": ["mcp_server.py"],
      "cwd": "/absolute/path/to/CitedGuard"
    }
  }
}
```

or run it directly:

```bash
python mcp_server.py
```

## Background jobs (Redis/RQ)

`jobs.py` defines an RQ queue that two things get submitted to instead of
running inline:

- `enqueue_question(query, top_k)` — used by both the Streamlit UI (`app.py`)
  and the MCP server's `ask_hipaa_question` tool (`mcp_server.py`) for every
  question. This is what lets either one handle multiple concurrent
  callers without one blocking another; it also naturally caps how many
  OpenAI calls run at once to however many workers are running, which
  doubles as a crude form of the rate limiting the guardrails table above
  flags as otherwise unimplemented. `search_hipaa_regulations` (the other
  MCP tool) deliberately stays direct, not queued — it's cheap enough
  (retrieval only, no generation) that queuing it would only add latency.
- `enqueue_ingestion(force)` — used by `python -m ingest --queue`, for
  running ingestion as a background job instead of a blocking CLI command.

Either way, a **separate worker process** has to actually run the jobs —
enqueuing one without a worker running just leaves it queued forever.

**Concurrency is capped by the number of worker processes, not the number of
users.** A single `rq worker` processes jobs strictly one at a time — extra
submissions queue up and wait their turn, they don't fail. `docker-compose.yml`
runs the `worker` service at `deploy.replicas: 4`, tested by enqueuing 4
different questions simultaneously and confirming all 4 started in the same
second on 4 separate workers and finished within a few seconds of each other
(~18s total), instead of the ~60-70s it would take running one after another
on a single worker. Raise or lower `replicas` to match expected concurrent
users — but note this only removes *our own* bottleneck; OpenAI's own
per-account rate limits (requests/tokens per minute) are a harder ceiling
above that, and scaling workers past what your account tier allows just means
more workers competing for the same capped throughput, not more real
throughput.

**Windows note:** RQ's default `Worker` class calls `os.fork()` to isolate
each job in a child process, which doesn't exist on Windows — the worker
starts fine but crashes the moment a job actually runs
(`AttributeError: module 'os' has no attribute 'fork'`). Use RQ's fork-free
`SimpleWorker` instead (this is what `docker-compose.yml`'s `worker` service
also uses, for consistency across platforms):

```bash
rq worker citedguard --url redis://localhost:6379/0 --worker-class rq.SimpleWorker
```

(Requires a Redis instance reachable at that URL — `docker run -p 6379:6379 redis:7-alpine` is the fastest way to get one locally.)

## Docker deployment

`docker-compose.yml` runs three services from the one `Dockerfile` (which
packages the Streamlit UI, `app.py`) — not the MCP server, which uses stdio
transport and is meant to be launched locally by an MCP client rather than
run standalone in a container:

- **`redis`** — the job queue backend (`redis:7-alpine`, unmodified).
- **`app`** — the Streamlit UI, submitting questions to the queue via
  `REDIS_URL=redis://redis:6379/0` (Docker's internal networking — service
  names resolve as hostnames between containers, unlike the `localhost` URL
  used for local, non-Docker runs).
- **`worker`** — runs `rq worker` with `--worker-class rq.SimpleWorker`
  (needed so the same compose file also works when `docker compose` itself
  is invoked from a Windows host).

The image does **not** bake in `chroma_db/` (your ingested regulation data)
or `.env` (secrets) — see `.dockerignore`. Both are supplied at run time
instead: `chroma_db/` as a mounted volume, shared by both `app` and `worker`
(so re-ingesting locally doesn't require rebuilding any image), and `.env` as
environment variables (so the API key never ends up baked into an image
layer).

```bash
docker compose up --build
```

Then open http://localhost:8501. Requires `chroma_db/` to already exist
locally (run ingestion first, outside Docker, if it doesn't) — the volume
mount surfaces whatever's already there, it doesn't create it.

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

| 6 | **Audit logging** | `audit.py` | **Fully implemented** | Every empty-context decline and every faithfulness failure is appended as a JSON line to `logs/guardrail_audit.jsonl`, including the query, generated answer, unsupported claims, and citations used — so guardrail triggers can be reviewed later instead of only existing for the life of the request. Logs full text verbatim (a deliberate choice for reviewability); `logs/` is gitignored since these entries carry the same sensitivity as whatever a user asked. |

**Not implemented / not yet planned in detail:**
- PII/PHI redaction of user input or model output.
- Rate limiting or abuse controls.

## Token usage

`generate_answer()` and `check_faithfulness()` both return a `TokenUsage`
(`usage.py`) alongside their result, and `qa.answer_question()` logs every
question's usage to `logs/token_usage.jsonl` via `audit.log_usage()` —
independent of whether a guardrail was triggered, so cost can be measured
before anything gets tuned to reduce it.

A real run against "What are the technical safeguards for encryption?" at
`top_k=5` logged:

| Call | Prompt tokens | Completion tokens |
|---|---|---|
| Generation | 1231 | 333 |
| Faithfulness guardrail | 1609 | 364 |

The guardrail call costs *more* prompt tokens than generation, despite doing
less work per token — it re-sends the same retrieved excerpts *and* the
generated answer. This is the actual cost of guardrail #4 being an
independent check rather than the model grading itself (a deliberate
trade-off, not an oversight — see guardrail #4 above).

Measured, this pointed straight at `top_k` as the lever: dropping the default
from 5 to 3 (`qa.answer_question`) cut the same question's total tokens from
3537 to 2287 (-35%), since both calls' context scales with it directly, and
the answer was still faithful with 3 sources instead of 5.

## Project status

| Piece | Status |
|---|---|
| Ingestion pipeline (fetch/parse/chunk/embed/store) | Done |
| Hybrid retrieval | Done |
| Generation with citations | Done |
| Faithfulness guardrail | Done (warn-only) |
| RQ worker / job queue | Done (`jobs.py`; UI questions + `--queue` ingestion) |
| Streamlit UI | Done |
| Test suite | Done (52 tests: chunk/parse/retriever/generate/guardrails/qa/audit/embed_store/fetch/usage/mcp_server/jobs) |
| Token usage instrumentation | Done (`logs/token_usage.jsonl`, every question) |
| MCP server | Done (`mcp_server.py`; `ask_hipaa_question` + `search_hipaa_regulations` tools) |
| Docker deployment (Streamlit UI + Redis + worker) | Done (`Dockerfile`, `docker-compose.yml`) |
