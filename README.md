# Legixo Q&A

A small Q&A HTTP API over documents: Python, LangGraph, Pinecone. Ask
questions about a document corpus and get back an answer with citations
verified against real retrieved chunks — out-of-corpus questions are
explicitly refused, never hallucinated.

**📹 Demo video (5–10 min):** https://youtu.be/YmZrpRZW41g
Walks through install, ingest, starting the API, calling `/ask` (good
answers with citations + one out-of-corpus refusal), and the LangGraph
layout.

---

## Quickstart

```bash
git clone <repo-url> legixo-qa && cd legixo-qa
python -m venv .venv && source .venv/Scripts/activate   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then add your 3 API keys (Anthropic, OpenAI, Pinecone)
python -m scripts.ingest_cli    # load the corpus into Pinecone
python -m scripts.run_server    # starts http://localhost:8000
```

Then, in another terminal:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What notice period does Priya Nambiar have?"}'
```

That's the whole happy path. The sections below cover each step in detail,
the Pinecone/idempotency specifics, the LangGraph flow, and honest notes
on what works, what's imperfect, and what I'd do next.

---

## Table of contents

1. [Folder structure](#1-folder-structure)
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

## 1. Folder structure

Run all commands from the repo root. Keep the internal structure intact —
imports are relative to it (e.g. `from src.config import settings`).

```
legixo-qa/
├── corpus/            # source documents (.md / .txt)
├── docs/
│   └── langgraph.md   # node map, diagram, branch/loop-limit rationale
├── src/
│   ├── config.py      # all env vars, loaded once, validated
│   ├── chunking.py    # token-aware text splitter
│   ├── embeddings.py  # OpenAI embeddings wrapper
│   ├── pinecone_client.py
│   ├── ingest.py      # ingest orchestrator (deterministic IDs)
│   ├── llm.py         # Anthropic wrapper (tool-calling for structured output)
│   ├── graph.py       # LangGraph flow: retrieve/grade/branch/generate/validate
│   └── api.py         # FastAPI app: POST /ask, GET /health
├── scripts/
│   ├── ingest_cli.py  # python -m scripts.ingest_cli
│   └── run_server.py  # python -m scripts.run_server
├── eval/
│   ├── self_test.json     # 15 questions, expected citations, pass/fail notes
│   └── run_self_test.py   # hits the live API and checks results
├── tests/             # 37 tests (chunking, ingest, graph, api)
├── requirements.txt / requirements-dev.txt
└── .env.example
```

---

## 2. Prerequisites

- Python 3.10+
- A Pinecone account + API key ([pinecone.io](https://www.pinecone.io))
- An OpenAI API key (used for embeddings only)
- An Anthropic API key (grading + answer generation)

## 3. Install

```bash
cd legixo-qa
python -m venv .venv
source .venv/Scripts/activate    # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # only needed to run the test suite
```

> **Windows/Git Bash tip:** if imports fail after install, your venv
> probably wasn't active when `pip` ran. Use `python -m pip install ...`
> to guarantee pip installs into the same interpreter `python` uses. See
> §12.

## 4. Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env`:

| Variable | Required | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | Grading + answer generation. Config validation checks it at startup. |
| `ANTHROPIC_MODEL` | no | Default `claude-sonnet-4-5`. |
| `OPENAI_API_KEY` | yes | Used for embeddings. |
| `OPENAI_EMBEDDING_MODEL` | no | Default `text-embedding-3-small` (1536-dim). If you change this, also update `EMBEDDING_DIMENSIONS` in `src/pinecone_client.py`. |
| `PINECONE_API_KEY` | yes | From your Pinecone dashboard. |
| `PINECONE_CLOUD` / `PINECONE_REGION` | no | Default `aws` / `us-east-1`. Used only if the index doesn't exist yet — see §5. |
| `PINECONE_INDEX_NAME` | no | Default `legixo-qa-docs`. |
| `PINECONE_NAMESPACE` | no | Default `default`. Lets you keep multiple corpora in one index. |
| `CORPUS_DIR` | no | Default `corpus`. Relative paths resolve from the repo root. |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | no | Default `500` / `50`. Overlap must be smaller than size. |
| `TOP_K` | no | Default `5`. |
| `MAX_LOOPS` | no | Default `2`. |
| `MIN_SIMILARITY` | no | Default `0.15`. |

**Never commit `.env`.** Only `.env.example` (dummy values) belongs in git.

## 5. Pinecone index — how it's created

You don't need to manually create the index. The first time you run
ingest, `ensure_index_exists()` in `src/pinecone_client.py`:

1. Lists your existing indexes.
2. If `PINECONE_INDEX_NAME` isn't among them, creates a **serverless**
   index with `metric="cosine"`, dimension matched to your embedding model
   (1536 for `text-embedding-3-small`), in `PINECONE_CLOUD` /
   `PINECONE_REGION`.
3. Polls until Pinecone reports the index ready (serverless creation is
   async), or raises after 60s if it never becomes ready.

To create it yourself instead: Pinecone console → Create Index → name
matching `PINECONE_INDEX_NAME`, dimension `1536`, metric `cosine`,
serverless, same cloud/region as your `.env`. Ingest detects it and skips
creation.

## 6. Run ingest

```bash
python -m scripts.ingest_cli --verbose
```

Expected output:

```
Ingest complete.
  Files processed:   6
  Chunks created:     <N>
  Vectors upserted:   <N>
```

## 7. Idempotency — what happens if you run ingest twice

**Chosen strategy: deterministic point IDs.** Each chunk's Pinecone vector
ID is `chunk_<sha256(source_file_path : chunk_index)[:24]>`. Running
ingest again on an unchanged corpus produces the same IDs, so Pinecone's
`upsert` **overwrites** in place — no duplicates, no growing vector count.

Why this over "delete-namespace-first": deterministic IDs let you
re-ingest a single changed file without wiping and re-embedding the whole
corpus, and there's no window where the index is partially empty mid-run.
Tradeoff: if a file's content changes enough to shift chunk boundaries,
old chunks at higher indices than the new chunk count can be left behind
under stale IDs (see §14).

**To verify:**

```bash
python -m scripts.ingest_cli   # run 1
python -m scripts.ingest_cli   # run 2, same corpus
python -c "from src.pinecone_client import PineconeClient; print(PineconeClient().stats())"
```

`total_vector_count` should be identical after both runs.

**Verified live (not simulated):** ran ingest twice against a real
Pinecone serverless index with the 6-file sample corpus. Both runs
reported `Vectors upserted: 6`, and `describe_index_stats()` confirmed
`total_vector_count=6` after both — proof the deterministic-ID strategy
overwrites in place rather than duplicating.

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
lighter response.

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
(the trap question) and case 15 (below).

**Automated citation-checking has a known blind spot, and I hit it for
real:** case 15 asks about a trial outcome the corpus doesn't contain. One
run produced a correct, honest, *cited* answer — citing real chunks not to
fabricate a verdict, but to explain the real context it does have (case
still at evidence stage) while explicitly saying the outcome itself isn't
in the documents. My eval script originally treated any non-empty citation
on an out-of-corpus question as an automatic FAIL, which flagged this even
though the system never hallucinated. Re-running the identical question
moments later produced a *different* result — a hard refusal with zero
citations. Same code, same corpus, two different (both non-hallucinating)
behaviors on this one borderline question. Fixed the eval script to flag
known-ambiguous cases for manual review instead of a blind auto-fail (see
`KNOWN_AMBIGUOUS_OUT_OF_CORPUS_IDS` in `eval/run_self_test.py`), and
documented the non-determinism in case 15's notes rather than smoothing it
over. The invariant that held across both runs: it never invented a trial
outcome that wasn't in the documents.

## 11. Run the tests

```bash
pip install -r requirements-dev.txt   # if not already installed
pytest -v
```

37 tests cover chunking (windowing, overlap, edge cases), ingest
(deterministic IDs, idempotent re-run, empty-file/corpus handling), the
LangGraph flow (both branches, loop limit, citation validation), and the
API (happy path, trace toggle, every error status code) — all mocked
against Pinecone/OpenAI/Anthropic so they run offline. One test
(`test_real_tiktoken_encoding_used_by_default_when_available`) is skipped
automatically if your environment can't reach tiktoken's vocab CDN — a
network check, not a code issue.

## 12. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ModuleNotFoundError` after install | venv wasn't active when `pip` ran, or `pip`/`python` point to different interpreters. Activate the venv and reinstall with `python -m pip install -r requirements.txt`. |
| `ConfigError: Missing required environment variable` | `.env` not created, or missing a required key. Copy `.env.example` → `.env`. |
| `PineconeIndexError: ... did not become ready within 60.0s` | Rare; re-run the ingest command — index creation usually finishes within a few seconds. |
| `IngestError: No files with extensions {'.md', '.txt'} found` | `CORPUS_DIR` points somewhere without `.md`/`.txt` files. |
| `EmbeddingError: OpenAI embeddings call failed` | Check `OPENAI_API_KEY` is valid and has quota. |
| Vector count grows on repeat ingest | Shouldn't happen — confirm you didn't change `CORPUS_DIR`/file paths between runs (IDs derive from relative file path). |

## 13. Extra: LangSmith tracing (optional)

Off by default — the required `trace` field in `/ask` responses already
satisfies the "optional trace" requirement. This is the one extra chosen
from the assignment's list (LangSmith, hybrid search, reranker), picked
because it directly extends something already built.

**What it adds over the in-response `trace`:** a full dashboard view per
question — every node's exact input/output, every LLM call's prompt and
response with token usage, and per-step latency. Useful for debugging
*why* a grading judgment went a certain way (e.g., seeing the exact chunk
text Claude was grading against).

**To enable:**

1. Get a free API key at [smith.langchain.com](https://smith.langchain.com).
2. In `.env`:
   ```
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=<your real key>
   LANGSMITH_PROJECT=legixo-qa
   ```
3. Restart the server. Tracing config is read once at startup via
   `src/config.py`, which sets both current (`LANGSMITH_*`) and legacy
   (`LANGCHAIN_*`) env var names for cross-version compatibility.
4. Call `/ask`. Traces appear at
   [smith.langchain.com](https://smith.langchain.com) under the
   `legixo-qa` project — each shows the full node sequence, with
   `grade_relevance`/`generate_answer`/`rewrite_query` as nested LLM spans.

Verified live end-to-end. `GET /health` reports `"langsmith_tracing"` and
redacts the key entirely (`"(tracing disabled)"`) when off — confirmed by
`tests/test_api.py::test_health_endpoint_reports_langsmith_tracing_state`.

---

## 14. What I'd do with more time

Scoped out explicitly rather than left silent:

- **Stale-chunk cleanup on re-ingest.** If a file shrinks (fewer chunks),
  old higher-index chunk IDs are left behind. Fix: track max chunk_index
  per file and delete orphaned IDs after upsert.
- **A reranker** between retrieve and grade would likely cut the number of
  rewrite loops on ambiguous questions.
- **Hybrid (keyword + vector) search** — exact term matching (case/statute
  numbers) might beat pure embedding similarity on this legal corpus.
- **Chunking is barely exercised** — each sample file is short enough to
  become exactly one chunk, so citations are effectively whole-document.
  The windowing/overlap logic is unit-tested in isolation but not proven
  end-to-end against a real multi-chunk document.
- **Grading non-determinism** (§10, case 15) — the grade step can judge
  relevant-but-incomplete chunks either way. Both outcomes are safe, but
  a larger adversarial question set (and the LangSmith traces to study it)
  would help tune the threshold.

---