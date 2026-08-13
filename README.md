# Legixo Q&A

A small Q&A HTTP API over documents: Python, LangGraph, Pinecone. Ask
questions about a document corpus and get back an answer with citations
verified against real retrieved chunks — out-of-corpus questions are
explicitly refused, never hallucinated.

---

## Table of contents

1. [Folder name / structure](#1-folder-name--where-to-put-this)
2. [Prerequisites](#2-prerequisites)
3. [Install](#3-install)
4. [Configure environment variables](#4-configure-environment-variables)
5. [Pinecone index — how it's created](#5-pinecone-index--how-its-created)
6. [Run ingest](#6-run-ingest)
7. [Idempotency](#7-idempotency--what-happens-if-you-run-ingest-twice)
8. [Start the API server](#8-start-the-api-server)
9. [LangGraph flow](#9-langgraph-flow)
10. [Self-test](#10-self-test)
11. [Run the tests](#11-run-the-tests)
12. [Troubleshooting](#12-troubleshooting)
13. [Extra: LangSmith tracing](#13-extra-langsmith-tracing-optional)
14. [What I'd do with more time](#14-what-id-do-with-more-time)

---

## 1. Folder name / where to put this

Name the project folder **`legixo-qa`** (matches the package imports used
throughout the code, e.g. `from src.config import settings`). If you clone
or unzip this under a different name, nothing breaks — Python imports here
are relative to wherever you run commands from, not the folder name itself
— but keep the **internal structure exactly as below**, since paths and
imports depend on it.

```
legixo-qa/                     <- run all commands from here
├── README.md                  <- this file
├── .env.example                <- copy to .env, fill in real keys
├── requirements.txt             <- runtime dependencies
├── requirements-dev.txt         <- + pytest, for running the test suite
├── pytest.ini
├── corpus/                     <- your source documents (.md / .txt)
│   ├── 01_matter_memo_arvind_v_northfield.md
│   ├── 02_employment_agreement_excerpt.md
│   ├── 03_hearing_notice_template.md
│   ├── 04_statute_style_excerpt_fictional.md
│   ├── 05_counsel_notes_settlement.md
│   └── 06_property_lease_clause.md
├── docs/
│   └── langgraph.md             <- node map, diagram, branch/loop-limit rationale
├── src/
│   ├── __init__.py
│   ├── config.py               <- all env vars, loaded once, validated
│   ├── chunking.py              <- token-aware text splitter
│   ├── embeddings.py            <- OpenAI embeddings wrapper
│   ├── pinecone_client.py       <- Pinecone index create/upsert/query
│   ├── ingest.py                <- ingest orchestrator
│   ├── llm.py                   <- Anthropic wrapper (tool-calling for structured output)
│   ├── graph.py                 <- LangGraph flow: retrieve/grade/branch/generate/validate
│   └── api.py                   <- FastAPI app: POST /ask, GET /health
├── scripts/
│   ├── __init__.py
│   ├── ingest_cli.py            <- `python -m scripts.ingest_cli`
│   └── run_server.py            <- `python -m scripts.run_server`
├── eval/
│   ├── __init__.py
│   ├── self_test.json           <- 15 questions with expected citations + notes
│   └── run_self_test.py         <- hits the live API and checks results automatically
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_chunking.py
    ├── test_ingest.py
    ├── test_graph.py
    └── test_api.py
```

---

## 2. Prerequisites

- Python 3.10+
- A Pinecone account + API key ([pinecone.io](https://www.pinecone.io))
- An OpenAI API key (used for embeddings only)
- An Anthropic API key (not used yet in Day 1 — needed later for the
  grading/answer LLM calls in Day 2, but required in `.env` now since
  `config.py` validates all keys up front)

## 3. Install

```bash
cd legixo-qa
python -m venv .venv
source .venv/Scripts/activate    # Windows Git Bash: .venv/Scripts/activate
                                   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

For running the test suite, also install dev dependencies:

```bash
pip install -r requirements-dev.txt
```

> **Windows/Git Bash tip:** always confirm the venv is actually active
> before installing — run `python -c "import sys; print(sys.executable)"`
> and check the path points inside `.venv`. If `pip install` runs before
> activation (or `pip` resolves to a different Python than `python` does),
> packages silently land in the wrong place and imports fail later with no
> obvious cause. When in doubt, use `python -m pip install ...` instead of
> bare `pip install ...` — it guarantees pip runs as a module of that exact
> Python interpreter.

## 4. Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env`:

| Variable | Required | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | Used in Day 2 (grading + answer generation). Must be set now — config validation checks it. |
| `ANTHROPIC_MODEL` | no | Default `claude-sonnet-4-5`. |
| `OPENAI_API_KEY` | yes | Used for embeddings. |
| `OPENAI_EMBEDDING_MODEL` | no | Default `text-embedding-3-small` (1536-dim). If you change this, also add its dimension to `EMBEDDING_DIMENSIONS` in `src/pinecone_client.py`. |
| `PINECONE_API_KEY` | yes | From your Pinecone dashboard. |
| `PINECONE_CLOUD` / `PINECONE_REGION` | no | Default `aws` / `us-east-1`. Used only if the index doesn't exist yet — see §5. |
| `PINECONE_INDEX_NAME` | no | Default `legixo-qa-docs`. |
| `PINECONE_NAMESPACE` | no | Default `default`. Lets you keep multiple corpora in one index. |
| `CORPUS_DIR` | no | Default `corpus`. Relative paths resolve from the repo root. |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | no | Default `500` / `50`. Overlap must be smaller than size. |
| `TOP_K` | no | Default `5`. Used in Day 2 retrieval. |
| `MAX_LOOPS` | no | Default `2`. Used in Day 2's loop guard. |
| `MIN_SIMILARITY` | no | Default `0.15`. Used in Day 2 grading. |

**Never commit `.env`.** Only `.env.example` (dummy values) belongs in git.

## 5. Pinecone index — how it's created

You don't need to manually create the index in the Pinecone console. The
first time you run ingest, `ensure_index_exists()` in
`src/pinecone_client.py`:

1. Lists your existing indexes.
2. If `PINECONE_INDEX_NAME` isn't among them, creates a **serverless**
   index with `metric="cosine"`, dimension matched to your embedding model
   (1536 for `text-embedding-3-small`), in `PINECONE_CLOUD` /
   `PINECONE_REGION`.
3. Polls until Pinecone reports the index ready (serverless creation is
   async), or raises after 60s if it never becomes ready.

If you'd rather create it yourself first: Pinecone console → Create Index
→ name matching `PINECONE_INDEX_NAME`, dimension `1536`, metric `cosine`,
serverless, same cloud/region as your `.env`. Ingest will detect it exists
and skip creation.

## 6. Run ingest

```bash
python -m scripts.ingest_cli --verbose
```

Expected output (counts will match your corpus):

```
Ingest complete.
  Files processed:   6
  Chunks created:     <N>
  Vectors upserted:   <N>
```

## 7. Idempotency — what happens if you run ingest twice

**Chosen strategy: deterministic point IDs.** Each chunk's Pinecone vector
ID is `chunk_<sha256(source_file_path : chunk_index)[:24]>`. Running
ingest again on an unchanged corpus produces the exact same IDs, so
Pinecone's `upsert` **overwrites** those vectors in place — no duplicates,
no growing vector count.

Why this over "delete-namespace-first": deterministic IDs let you
re-ingest a single changed file without wiping and re-embedding the whole
corpus, and there's no window where the index is partially empty mid-run.
The tradeoff: if a file's *content* changes enough to shift chunk
boundaries, old chunks at higher indices than the new chunk count can be
left behind under stale IDs. (Not handled in Day 1 — noted here for
transparency; a `MAX(chunk_index)` cleanup pass would be one fix.)

**To verify it yourself:**

```bash
python -m scripts.ingest_cli   # run 1
python -m scripts.ingest_cli   # run 2, same corpus
```

Then check vector count is unchanged either via the Pinecone console, or:

```bash
python3 -c "
from src.pinecone_client import PineconeClient
print(PineconeClient().stats())
"
```

`total_vector_count` should be identical after both runs.

**Verified live (not simulated):** ran ingest twice against a real
Pinecone serverless index (`legixo-qa-docs`, `aws`/`us-east-1`) with the
6-file sample corpus. Both runs reported `Vectors upserted: 6`, and
`describe_index_stats()` after both runs confirmed:

```
DescribeIndexStatsResponse(dimension=1536, total_vector_count=6, metric='cosine', namespaces=1)
```

`total_vector_count` stayed at `6` after the second run — proof the
deterministic-ID strategy overwrites in place rather than duplicating.

## 8. Start the API server

```bash
python -m scripts.run_server
```

Starts on `http://localhost:8000` by default (configurable via `API_HOST`/`API_PORT`). Interactive docs at `http://localhost:8000/docs`.

### `GET /health`

```bash
curl http://localhost:8000/health
```
Basic liveness + redacted config check. Doesn't touch Pinecone/Anthropic.

### `POST /ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What notice period does Priya Nambiar have in her employment agreement?"}'
```

Example response:
```json
{
  "answer": "Priya Nambiar has a notice period of 60 days written notice...",
  "citations": [
    {"chunk_id": "chunk_343a94501cc46b188e0c4e10", "source_file": "02_employment_agreement_excerpt.md"}
  ],
  "trace": [
    "retrieve (loop 1): query='...' -> 5/5 chunk(s) above min_similarity=0.15",
    "grade_chunks: sufficient=True -- ...",
    "generate_answer: proposed 1 citation(s): [...]",
    "validate_citations: all 1 citation(s) verified against retrieved chunks"
  ],
  "loop_count": 1
}
```

**Trace is included by default** — a deliberate choice so the graph's
reasoning is visible from a plain curl call, not hidden behind a flag a
reviewer would need to discover first. Set `?include_trace=false` for a
lighter response:

```bash
curl -X POST "http://localhost:8000/ask?include_trace=false" \
  -H "Content-Type: application/json" \
  -d '{"question": "..."}'
```

### Out-of-corpus example

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```
```json
{
  "answer": "I can't find information about this in the provided documents.",
  "citations": [],
  "trace": [
    "retrieve (loop 1): query='What is the capital of France?' -> 0/5 chunk(s) above min_similarity=0.15",
    "grade_chunks: sufficient=False -- No chunks were retrieved above the similarity threshold.",
    "rewrite_query: 'What is the capital of France?' -> 'France capital city Paris'",
    "retrieve (loop 2): query='France capital city Paris' -> 0/5 chunk(s) above min_similarity=0.15",
    "grade_chunks: sufficient=False -- No chunks were retrieved above the similarity threshold.",
    "no_answer: could not produce a grounded, cited answer from the corpus"
  ],
  "loop_count": 2
}
```

### Error responses

| Status | When | Body shape |
|---|---|---|
| `400` | Empty/whitespace-only question | `{"error": "invalid_request", "detail": "..."}` |
| `422` | Malformed request body (missing `question` field) | FastAPI's standard Pydantic validation error |
| `502` | Pinecone unreachable, or index not yet ingested | `{"error": "pinecone_error", "detail": "..."}` |
| `502` | Anthropic call failed (rate limit, auth, etc.) | `{"error": "llm_error", "detail": "..."}` |
| `500` | Anything unexpected | `{"error": "internal_error", "detail": "..."}` |

No bare 500s with an empty body — every failure path returns useful JSON.

## 9. LangGraph flow

See [`docs/langgraph.md`](docs/langgraph.md) for the full node map, diagram,
and the reasoning behind the two conditional branches and the loop limit.
Short version:

```
retrieve → grade_chunks ─(good)──────────────→ generate_answer → validate_citations ─(valid citations)→ END
                         │                                                            └─(no valid citations)→ no_answer → END
                         ├─(bad, loop < max_loops)→ rewrite_query → retrieve  (loops back)
                         └─(bad, loop ≥ max_loops)──────────────────────────→ no_answer → END
```

## 10. Self-test

```bash
# server must be running (see §8)
python -m eval.run_self_test
```

Runs `eval/self_test.json` (15 questions covering all 6 corpus files, plus
out-of-corpus and "plausible-sounding trap" cases) against the live API,
checks whether the expected source files appear in citations, and writes
`eval/self_test_results.json`. My own pass/fail notes and honest
self-critique are inline in `eval/self_test.json` — see especially case 14
(the trap question) and case 15 (below) for the most interesting results.

**Automated citation-checking has a known blind spot, and I hit it
for real:** case 15 asks about a trial outcome the corpus doesn't contain.
One run produced a correct, honest, *cited* answer — citing real chunks
not to fabricate a verdict, but to explain the real context it does have
(case still at evidence stage) while explicitly saying the outcome itself
isn't in the documents. My eval script originally treated any non-empty
citation on an out-of-corpus question as an automatic FAIL, which
flagged this as a failure even though the system never hallucinated.
Re-running the identical question moments later via the graph directly
produced a *different* result — a hard refusal with zero citations after
2 loops. Same code, same corpus, two different (but both
non-hallucinating) behaviors on this one borderline question. Fixed the
eval script to flag known-ambiguous cases like this for manual review
instead of a blind auto-fail (see `KNOWN_AMBIGUOUS_OUT_OF_CORPUS_IDS` in
`eval/run_self_test.py`), and documented the underlying non-determinism
in `eval/self_test.json`'s notes for case 15 rather than smoothing it over.
The one thing that held across both runs: it never invented a trial
outcome that wasn't in the documents.

## 11. Run the tests

```bash
pip install -r requirements-dev.txt   # if not already installed
pytest -v
```

14 tests cover chunking (windowing, overlap, edge cases) and ingest
(deterministic IDs, idempotent re-run, empty-file/corpus handling), all
mocked against Pinecone/OpenAI so they run offline. One test
(`test_real_tiktoken_encoding_used_by_default_when_available`) is skipped
automatically if your environment can't reach tiktoken's vocab CDN on
first use — it isn't a code issue, just a network check.

## 12. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ConfigError: Missing required environment variable` | `.env` not created, or missing a required key. Copy `.env.example` → `.env`. |
| `PineconeIndexError: ... did not become ready within 60.0s` | Rare; re-run the ingest command — index creation usually finishes within a few seconds. |
| `IngestError: No files with extensions {'.md', '.txt'} found` | `CORPUS_DIR` points somewhere without `.md`/`.txt` files. |
| `EmbeddingError: OpenAI embeddings call failed` | Check `OPENAI_API_KEY` is valid and has quota. |
| Vector count grows on repeat ingest | Shouldn't happen — if it does, confirm you didn't change `CORPUS_DIR`/file paths between runs (IDs are derived from relative file path). |

## 13. Extra: LangSmith tracing (optional)

Off by default — the assignment's required `trace` field in `/ask`
responses already satisfies the "optional trace" requirement on its own.
This is the one extra chosen from their list ("LangSmith, hybrid search,
reranker") since it directly extends something already built, rather than
adding an unrelated capability.

**What it adds over the in-response `trace`:** a full dashboard view per
question — every node's exact input/output, every LLM call's prompt and
response with token usage, and wall-clock latency per step. Genuinely
useful for debugging *why* a grading judgment went a certain way in a way
a printed trace string can't show (e.g., seeing the exact chunk text
Claude was grading against).

**To enable:**

1. Get a free API key at [smith.langchain.com](https://smith.langchain.com).
2. In `.env`:
   ```
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=<your real key>
   LANGSMITH_PROJECT=legixo-qa
   ```
3. Restart the server (`python -m scripts.run_server`) — tracing config is
   read once at process start via `src/config.py`, which propagates it to
   the env vars LangGraph's LangChain-based runtime and the `langsmith`
   SDK actually read (`LANGSMITH_TRACING`/`LANGCHAIN_TRACING_V2` etc. —
   both old and new naming set, for compatibility across LangChain
   versions).
4. Call `/ask` as normal. Traces appear at
   [smith.langchain.com](https://smith.langchain.com) under the
   `legixo-qa` project — each call shows the full node sequence
   (`retrieve` → `grade_chunks` → ...), with `grade_relevance`,
   `generate_answer`, and `rewrite_query` traced individually as LLM spans
   (via `@traceable` in `src/llm.py`), nested under their parent node.

**Verify it's off when you don't want it:** `GET /health` reports
`"langsmith_tracing": false` and redacts the API key entirely (returns
`"(tracing disabled)"` rather than a partial key) whenever tracing is
off — confirmed by
`tests/test_api.py::test_health_endpoint_reports_langsmith_tracing_state`.

---

## 14. What I'd do with more time

Being explicit about what's scoped out rather than silent about it:

- **Stale-chunk cleanup on re-ingest.** If a source file's content shrinks
  (fewer chunks than before), old chunk IDs at higher indices are left
  behind in Pinecone under stale content. Fix: track the max chunk_index
  per file from the previous run (in a small metadata store) and delete
  any now-orphaned higher-index IDs after upsert.
- **Only similarity-threshold + LLM grading, no reranker.** A cross-encoder
  reranker between retrieve and grade would likely reduce the number of
  rewrite loops needed on ambiguous questions. Scoped out as a "one
  tasteful extra" candidate rather than mixed into the core flow.
- **No hybrid (keyword + vector) search.** For this legal-style corpus,
  exact term matching (case numbers, statute section numbers) might
  outperform pure embedding similarity on some questions — worth testing
  if the corpus grows.
- **LangSmith tracing is wired up (§13)**, but only exercised against the
  15-question self-test set. Would want to run it against a much larger,
  more adversarial question set to actually use it for what it's good at
  — spotting patterns in *why* grading judgments go wrong across many
  examples, not just confirming the 15 already-known-good cases.
- **Chunking barely exercised.** The sample corpus's files are short
  enough that each became exactly one chunk — confirmed across all 15
  self-test questions in `eval/self_test.json` (every `loop_count` and
  citation traces to a single whole-document chunk, never a partial
  passage). The overlap/windowing logic is unit-tested in isolation
  (`tests/test_chunking.py`) but never proven end-to-end against a real
  multi-chunk document. Would want a longer real document in the corpus
  to validate this properly.
- **One terminology-precision gap found in self-testing** (case 12,
  `eval/self_test.json`): the source text says settlement talks are
  "without prejudice" (a specific legal term about admissibility in
  court), and the system's answer said they're "confidential" — factually
  adjacent but not the same claim. Citation and underlying facts were
  correct; the paraphrase lost precision. Fix: tighten the
  answer-generation system prompt to preserve specific legal/technical
  terms verbatim rather than substituting more common language.

---
