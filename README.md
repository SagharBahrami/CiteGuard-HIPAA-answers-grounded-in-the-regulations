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

1. **Ingestion** (`src/citedguard/ingest/`) pulls the regulation text, splits it into
   citable chunks, embeds them, and stores them in a local Chroma
   collection. Run once (and again whenever you want to refresh the data).
2. **Retrieval** (`src/citedguard/retriever.py`) finds the chunks relevant to a question.
3. **Generation** (`src/citedguard/generate.py`) answers the question using only those
   chunks, with citations.
4. **Guardrails** (`src/citedguard/guardrails.py`) independently checks the answer against
   its sources before it's returned.
5. **`src/citedguard/qa.py`** wires 2-4 together into a single `answer_question(query)` call.

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

Config lives in `src/citedguard/config.py` (via `pydantic-settings`, reading from `.env`):

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
python -m citedguard.ingest          # skips parts already cached in data/raw/
python -m citedguard.ingest --force  # re-fetches everything
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

**Or expose it to an MCP client** via
`mcp_server.py`, which serves two tools over stdio:

- `ask_hipaa_question(query, top_k=3)` — full pipeline (retrieval +
  generation + faithfulness guardrail), returns the answer, faithfulness
  result, and sources. Goes through the same Redis/RQ job queue as the
  Streamlit UI (see [Background jobs](#background-jobs-redisrq) below) —
  requires Redis and a worker running.
- `search_hipaa_regulations(query, top_k=5)` — retrieval only, for when the
  caller would rather read the source excerpts itself. No generation or
  guardrail LLM calls, so it's cheaper.

Add it to your MCP client's server configuration. Most clients read a JSON
config file with an `mcpServers` object; consult your client's docs for its
location and whether it offers a CLI command to register a server instead:

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

`src/citedguard/jobs.py` defines an RQ queue that two things get submitted to instead of
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
- `enqueue_ingestion(force)` — used by `python -m citedguard.ingest --queue`, for
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

- **`src/citedguard/ingest/fetch.py`** — Downloads raw XML for Parts 160/162/164 from
  eCFR's Content Versioner API (`api.ecfr.gov`), not by scraping
  `www.ecfr.gov` directly — the rendered site actively blocks scraping with
  a CAPTCHA wall. Caches to `data/raw/<issue_date>/`, one subdirectory per
  eCFR issue date, so a newer revision is fetched alongside (not over) older
  ones — `data/raw/` doubles as the historical snapshot archive of every raw
  XML eCFR has published. Skips parts already cached for that issue date
  unless `force=True`.
- **`src/citedguard/ingest/parse.py`** — Parses each part's XML into `Section` records
  (citation, heading, part, subpart, issue date, ordered paragraphs). The
  `(a)/(1)/(i)` legal outline within a section is flat text in the XML, not
  real nesting, so it isn't reconstructed here.
- **`src/citedguard/ingest/chunk.py`** — Turns each `Section` into one or more `Chunk`s,
  carrying its issue date along. A section under ~1500 characters stays
  whole; longer ones are split by greedily packing paragraphs up to that
  size, **splitting only at paragraph boundaries** (never mid-sentence).
  Split chunks repeat their previous chunk's last paragraph as lead-in
  context, so a chunk retrieved on its own isn't missing the
  requirement/standard it's an implementation detail of.
- **`src/citedguard/ingest/embed_store.py`** — Embeds chunks in batches of 100 via the
  OpenAI embeddings API and upserts them into Chroma with deterministic IDs
  (`<section>_<chunk_index>`) and an `issue_date` metadata field, so
  re-running ingestion updates existing rows instead of duplicating them.
  Before any upsert or delete, whatever version it's about to overwrite is
  archived to `src/citedguard/ingest/history.py`'s SQLite table first (and that archive
  write is committed before the Chroma call runs) — Chroma only ever holds
  the current version of each chunk, but nothing superseded or removed is
  lost. The Chroma collection is explicitly created with `hnsw:space=cosine`
  so similarity scores have a well-defined meaning (`1 - distance`).
- **`src/citedguard/ingest/update_check.py`** — Compares eCFR's latest published issue date
  against what's currently in Chroma and re-ingests only when they differ.
  Enqueue it on a schedule via `jobs.enqueue_update_check()` to pick up
  regulatory changes automatically instead of relying on someone running
  `python -m citedguard.ingest` by hand.
- **`src/citedguard/locks.py`** — A Redis lock serializing ingestion runs. Chroma's local
  persistent client and the SQLite archive both assume a single writer, but
  docker-compose runs four worker replicas and a scheduled update check can
  fire while a manual `python -m citedguard.ingest` is still going. The lock is
  **skip-if-held, not wait-your-turn**: a second run started mid-flight has
  nothing new to do, and queueing behind the lock only to re-embed the whole
  corpus would cost real money for no change — so `run()` returns `False`
  and exits instead. A 30-minute TTL keeps a killed worker from wedging
  ingestion. Redis is optional here (only the RQ queue requires it), so if
  it's unreachable ingestion logs a warning and proceeds unlocked — with
  Redis down the workers can't run either, leaving a manual CLI run as the
  only caller, with nothing to race.

Current corpus: 148 sections -> 427 chunks.

## Retrieval details (`src/citedguard/retriever.py`)

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

## Historical version queries (`src/citedguard/version_query.py`)

Chroma holds only the current version of each chunk, so retrieval always
answers from what applies today — unless the question is specifically asking
about a past version, in which case the retrieved chunks are re-resolved
against the SQLite history archive before generation.

- **Detection is a deterministic pattern match, not another LLM call.** The
  two failure directions aren't symmetric: missing a historical question just
  means the user gets today's rule and can re-ask, but a false positive would
  silently answer an ordinary compliance question from a no-longer-binding
  rule. So the patterns are narrow — "previous version", "used to require",
  "repealed", "before the 2023 amendment" trigger it; a bare "before" or
  "change" (as in "what must happen *before* disclosing PHI") does not.
- **Resolution is point-in-time.** Retrieval still runs normally first — it's
  what finds *which* sections are relevant — and only then does
  `resolve_as_of` swap each chunk's text for whichever version (current or
  archived) was in effect at the requested date. With no parseable date, it
  falls back to the most recently superseded version.
- **The answer says so.** A historical resolution switches generation to a
  system prompt that requires stating up front that the answer describes a
  superseded version, cites each claim with its issue date, and sets
  `Answer.historical` — which the Streamlit UI renders as a banner and the
  MCP `ask_hipaa_question` tool returns as a field.
- **Asking for a past version doesn't mean one exists.** The archive only
  reaches back to the first ingestion that superseded something, so until
  eCFR publishes a second issue date there is nothing older to return. When
  resolution can't reach the requested date it keeps the current text and
  sets `Answer.history_unavailable` instead of `historical` — the answer is
  generated with the normal current-version prompt and the UI says the
  archive doesn't go back that far. Framing today's rule as "a superseded
  version" would be precisely the wrong error for a compliance answer.
- **The faithfulness guardrail gets a matching carve-out.** That required
  disclaimer is by definition not supported by any excerpt, so without a
  carve-out the guardrail would flag it on *every* historical answer —
  spuriously warning the user and burning a corrective retry that's
  instructed to keep the disclaimer anyway. In historical mode the guardrail
  is told to judge only regulatory substance, not the version framing.

## Guardrails

Everything below is a mechanism specifically aimed at reducing hallucination
or out-of-scope answers. "Level" describes how much is actually
implemented vs. planned.

| # | Guardrail | Where | Level | What it does |
|---|---|---|---|---|
| 1 | **Relevance threshold** | `src/citedguard/retriever.py` | **Fully implemented** | Chunks below `similarity_threshold` (0.35 cosine) are dropped before generation ever sees them. |
| 2 | **Decline on empty context** | `src/citedguard/generate.py` (`NO_CONTEXT_MESSAGE`) | **Fully implemented** | If retrieval returns zero chunks, a fixed decline message is returned *without calling the LLM at all* — no chance to hallucinate from general knowledge. |
| 3 | **Grounding instruction (prompt-level)** | `src/citedguard/generate.py` (`SYSTEM_PROMPT`) | **Fully implemented, soft guardrail** | Instructs the model to answer only from the provided excerpts, cite the specific CFR section per claim, and say explicitly when the excerpts don't fully answer the question. This relies on model compliance — it isn't verified, which is what guardrail #4 is for. |
| 4 | **Faithfulness / groundedness check** | `src/citedguard/guardrails.py` (`check_faithfulness`) | **Fully implemented** | A *second*, independent LLM call (`guardrail_model`, separate from `generation_model`) re-checks the generated answer against the same source excerpts and flags any claim not actually supported, via structured output (`is_faithful`, `unsupported_claims`, `explanation`). See [Evaluation](#evaluation) for measured accuracy: 100% recall, 55.6% precision. |
| 5 | **Rescue-candidate sanity floor** | `src/citedguard/retriever.py` (`RESCUE_SIMILARITY_THRESHOLD`) | **Fully implemented** | Prevents the BM25 rescue mechanism from pulling in chunks with no real semantic relationship to the query, just because of incidental keyword overlap with common regulatory vocabulary. |
| 6 | **Corrective retry** | `src/citedguard/qa.py` (`answer_question`), `src/citedguard/generate.py` (`regenerate_answer`) | **Fully implemented** | If guardrail #4 flags an answer, it's regenerated once with the specific unsupported claims fed back as feedback ("this claim wasn't supported: X — remove it or ground it properly"), then re-checked. If the retry passes, the corrected answer is returned with **no warning shown** — the guardrail did its job silently. If it's still flagged, the retried answer is returned with the warning banner, same as before — not blocked. Verified against a real deliberately-fabricated answer (AES-256/24-hour-checksum claims that don't exist in the source): the retry removed both fabrications and replaced them with the actual regulatory text. |
| 7 | **Audit logging** | `src/citedguard/audit.py` | **Fully implemented** | Every empty-context decline, every *still*-unfaithful answer (post-retry), and every silent correction is appended as a JSON line to `logs/guardrail_audit.jsonl` (`log_decline` / `log_unfaithful` / `log_corrected`), including the query, answer text, unsupported claims, and citations — so guardrail activity can be reviewed later, including the corrections nobody saw a warning for. Logs full text verbatim (a deliberate choice for reviewability); `logs/` is gitignored since these entries carry the same sensitivity as whatever a user asked. |

**Why blocking was rejected instead of just falling back to warn:**
[Evaluation](#evaluation) measured the guardrail's precision at only
55.6% — over a third of genuinely faithful answers get flagged. Hard-
blocking (replacing any flagged answer, retried or not, with a safe decline)
would incorrectly refuse a lot of fine answers on top of the real
hallucinations it catches. Retry-then-warn targets the real fabrications
(which the retry fixes, silently) without inheriting that false-block rate
for the borderline cases that remain flagged after a retry.

**Not implemented / not yet planned in detail:**
- PII/PHI redaction of user input or model output.
- Rate limiting or abuse controls.

## Evaluation

`evals/` measures the faithfulness guardrail's actual accuracy, rather than
trusting it works from the single-example demo in `src/citedguard/guardrails.py`'s
`__main__` block. Separate from `tests/`: it makes real (paid, slightly
non-deterministic) calls to `guardrail_model`, so it isn't run as part of
`pytest` — run it deliberately, e.g. after changing the guardrail prompt or
swapping models:

```bash
python -m evals.eval_guardrail
```

`evals/cases.py` hand-pairs 10 cases (5 genuinely faithful, 5 deliberately
fabricated) with real chunks captured once via `retriever.retrieve` and
frozen there, so the eval isolates the guardrail itself rather than also
depending on retrieval. The script reports two different things, because
they answer two different questions:

**1. Direct check** (`check_faithfulness` alone, no retry) — is the
guardrail model itself accurate? Measured: 100% recall (every fabricated
claim was flagged — zero hallucinations slipped through unflagged), but only
~55-62% precision (a third or more of genuinely faithful answers were
flagged too, varying slightly run to run). Inspecting the false positives
showed they were mostly the guardrail correctly catching minor paraphrase
drift in the eval's own hand-written "faithful" answers (e.g. "contract" vs.
the source's "contract or other arrangement") — not the guardrail
misfiring. **The guardrail is calibrated stricter than a careful human
paraphrase**, which also explains an earlier observation in this project:
the same real question flipping between faithful/unfaithful across two
separate runs, because normal generation-model paraphrasing sits close
enough to that strict boundary to cross it either way.

**2. Full pipeline** (mirrors `qa.answer_question`: retry once if flagged) —
what does the user actually experience? This is *not* reported as
precision/recall — when the corrective retry rewrites a fabricated answer
into a genuinely accurate one (which it does; see guardrail #6), the case's
"expected faithful" label was written for the *original* text and no longer
describes what the user sees, so scoring that as a "false negative" would
equate it with a real hallucination slipping through unflagged, when it's
actually the best possible outcome. Instead each case is classified as
`passed_clean` (never flagged), `corrected` (flagged, retry fixed it —
accurate answer, no warning), `caught` (fabricated, still flagged — warning
correctly shown), `false_alarm` (faithful, still wrongly flagged after
retry), or `missed` (fabricated, never flagged at all — the only genuinely
dangerous outcome). A representative run: 0/5 fabricated cases were
`missed`, 0/5 faithful cases ended up a `false_alarm` — every case ended up
either accurate-with-no-warning or accurately-flagged-with-a-warning.

## Token usage

`generate_answer()` and `check_faithfulness()` both return a `TokenUsage`
(`src/citedguard/usage.py`) alongside their result, and `qa.answer_question()` logs every
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
| Faithfulness guardrail | Done (corrective retry, falls back to warn) |
| RQ worker / job queue | Done (`src/citedguard/jobs.py`; UI questions + `--queue` ingestion) |
| Streamlit UI | Done |
| Test suite | Done (60 tests: chunk/parse/retriever/generate/guardrails/qa/audit/embed_store/fetch/usage/mcp_server/jobs) |
| Token usage instrumentation | Done (`logs/token_usage.jsonl`, every question) |
| MCP server | Done (`mcp_server.py`; `ask_hipaa_question` + `search_hipaa_regulations` tools) |
| Docker deployment (Streamlit UI + Redis + worker) | Done (`Dockerfile`, `docker-compose.yml`) |
| Guardrail eval harness | Done (`evals/`; direct-check + retry-aware reports, see Evaluation) |
