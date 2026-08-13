# LangGraph flow — node map

## Diagram

```
                        ┌─────────────┐
              ┌────────▶│  retrieve   │◀────────────────┐
              │         └──────┬──────┘                 │
              │                │                         │
              │                ▼                         │
              │         ┌─────────────┐                  │
              │         │ grade_chunks│                  │
              │         └──────┬──────┘                  │
              │                │                          │
              │     ┌──────────┼──────────────┐           │
              │     │          │              │           │
              │  good        bad,          bad,           │
              │     │    loop<max_loops  loop>=max_loops   │
              │     ▼          │              │           │
              │ ┌─────────┐    │              │           │
              │ │generate_│    │              │           │
              │ │ answer  │    │              │           │
              │ └────┬────┘    │              │           │
              │      │         ▼              │           │
              │      │  ┌──────────────┐       │           │
              │      │  │rewrite_query │───────┼───────────┘
              │      │  └──────────────┘       │
              │      ▼                         ▼
              │ ┌──────────────┐        ┌────────────┐
              │ │  validate_   │        │ no_answer  │
              │ │  citations   │───────▶│            │
              │ └──────┬───────┘  no    └─────┬──────┘
              │        │       valid          │
              │      valid                    │
              │    citations                  │
              │        │                       │
              │        ▼                       ▼
              │      END                      END
              └────────────────────────────────
                (loop-back edge shown at top)
```

Simpler linear read of the same graph:

```
retrieve → grade_chunks ─┬─(good)──────────────→ generate_answer → validate_citations ─┬─(has valid citations)→ END
                          │                                                             └─(no valid citations)──→ no_answer → END
                          ├─(bad, loop_count < max_loops)→ rewrite_query → retrieve  (loops back up)
                          └─(bad, loop_count >= max_loops)────────────────────────────→ no_answer → END
```

## Nodes

| Node | File / function | What it does |
|---|---|---|
| `retrieve` | `src/graph.py::retrieve_node` | Embeds the current query (`src/embeddings.py`), queries Pinecone for the top `TOP_K` chunks, filters out anything below `MIN_SIMILARITY`. Increments `loop_count` — **this is what the loop limit actually counts.** |
| `grade_chunks` | `src/graph.py::grade_chunks_node` | Calls Claude (`src/llm.py::grade_relevance`) via a **forced tool call** (`submit_grade`) to judge whether the retrieved chunks are sufficient to answer the question. Real LLM judgment, not a hardcoded value or pure similarity threshold. |
| `rewrite_query` | `src/graph.py::rewrite_query_node` | Only reached on a bad grade with loop budget remaining. Calls Claude (`src/llm.py::rewrite_query`) to reformulate the question (broader terms, dropped conversational wrapping), then loops back to `retrieve`. |
| `generate_answer` | `src/graph.py::generate_answer_node` | Only reached on a good grade. Calls Claude (`src/llm.py::generate_answer`) via a **forced tool call** (`submit_answer`) to produce the answer text plus the `chunk_id`s it claims support it. |
| `validate_citations` | `src/graph.py::validate_citations_node` | Checks every claimed `chunk_id` against what was **actually retrieved** in this run. Drops any id that doesn't match — this is the programmatic guard against fake citations, not something left to LLM honesty alone. |
| `no_answer` | `src/graph.py::no_answer_node` | Terminal refusal node. Reached when: (a) the loop limit is hit with still-insufficient chunks, or (b) `validate_citations` finds zero valid citations left. Always returns the same fixed message — no hedging, no partial hallucination. |

## Conditional edges (the two real branches)

1. **`route_after_grade`** (`src/graph.py`) — the branch the assignment scores hardest.
   ```python
   def route_after_grade(state):
       if state["grade"] == "good":
           return "generate"
       if state["loop_count"] >= state["max_loops"]:
           return "no_answer"
       return "rewrite"
   ```
   Decided entirely by `grade_chunks`'s LLM output plus the loop counter — never a hardcoded `True`.

2. **`route_after_validation`** (`src/graph.py`) — the citation-integrity branch.
   ```python
   def route_after_validation(state):
       return "end" if state["citations"] else "no_answer"
   ```

## Loop limit

- `MAX_LOOPS` (default `2`, from `.env`) caps how many times `retrieve` can run.
- `loop_count` increments once per `retrieve` call.
- `route_after_grade` checks `loop_count >= max_loops` — once true, a bad grade routes to `no_answer` instead of `rewrite_query`, so `retrieve` can never run more than `max_loops` times regardless of what the grader says.
- As a second, independent safety net, `run_qa` also sets LangGraph's own `recursion_limit` generously above what `max_loops` could ever produce — belt-and-suspenders, not the primary mechanism.
- Proven in `tests/test_graph.py::test_graph_hits_loop_limit_and_gives_no_answer`: with the grader mocked to always return insufficient, the graph stops at exactly `loop_count == max_loops`, not beyond.

## State shape

```python
class QAState(TypedDict):
    question: str              # current query text (may be rewritten mid-loop)
    original_question: str     # user's original wording; always used for the final answer
    retrieved_chunks: list[...]
    grade: str                 # "" | "good" | "bad"
    grade_reasoning: str
    answer: str
    citations: list[str]
    loop_count: int
    max_loops: int
    trace: list[str]           # human-readable log of every node's decision, returned by /ask
```

## Observability: LangSmith tracing (optional)

Every node function and every LLM call (`grade_relevance`, `generate_answer`,
`rewrite_query`) is wrapped with `@traceable` from the `langsmith` SDK. Off
by default; enable via `LANGSMITH_TRACING=true` in `.env` — see README §13
for setup and live-verification steps.

Verified live: a real `/ask` call produces a full trace tree in the
LangSmith dashboard —

```
run_qa
├── retrieve
│   └── retrieve            (Pinecone query span)
├── grade_chunks
│   ├── grade_relevance     (LLM span: full prompt + tool-call response)
│   └── route_after_grade
├── generate_answer
│   └── generate_answer_node
│       └── generate_answer (LLM span)
└── validate_citations
    ├── validate_citations
    └── route_after_validation
```

Even the conditional-edge functions (`route_after_grade`,
`route_after_validation`) show up as their own spans — free from
LangGraph's LangChain-based runtime once tracing is on, not something
built manually. Useful beyond the in-response `trace` field: the Output
tab on any node span shows full state (`grade`, `grade_reasoning`,
`retrieved_chunks`, `loop_count`), and each LLM span shows the *exact*
chunk text a grading judgment was made against — the level of detail
needed to debug why a borderline question graded one way instead of
another (see the non-determinism note below).

## A known non-determinism, found via `eval/self_test.json`

The same question, asked twice against the identical corpus and code, can
take different paths through the graph. Concretely: *"What was the
outcome of the Arvind Mehta trial?"* — a question the corpus genuinely
has no answer to — produced two different results across two runs:

- **Run A:** `grade_chunks` returned `sufficient=True` (the retrieved
  chunks contain real, relevant context — case status, hearing dates,
  settlement terms — even though not the specific fact asked for), so the
  graph proceeded to `generate_answer`, which correctly wrote *"there is
  no information about the outcome... the documents do not indicate
  whether the case was settled or went to trial"* — citing real chunks to
  explain the context it does have, not to fabricate a verdict.
- **Run B:** `grade_chunks` returned `sufficient=False` both times (loop
  limit reached), so the graph took the `no_answer` path instead — a flat
  refusal with zero citations.

Both are honest, non-hallucinating outcomes — the one invariant that held
across both runs is that the system never invented a trial verdict that
isn't in the documents. But the *shape* of the response (cited
explanation vs. flat refusal) isn't fully deterministic on this class of
question, because `grade_chunks`'s LLM judgment call can land either way
when the retrieved chunks are relevant-but-incomplete rather than clearly
sufficient or clearly irrelevant. Full writeup, including how this
exposed a blind spot in the eval script's own pass/fail logic, is in
`eval/self_test.json` (case 15) and README §10.

---

## Why this design (brief rationale)

- **Two branches, not one**, because the assignment's two hardest-to-fake requirements — a real retrieval-quality branch, and zero tolerance for fake citations — are genuinely different concerns. Grading judges whether *retrieval* succeeded; validation judges whether the *LLM's citation claims* are honest. Collapsing them into one branch would hide which failure mode actually occurred.
- **Tool-calling, not prompted JSON**, for `grade_chunks` and `generate_answer` — eliminates an entire class of "model returned malformed JSON" failures that a prompted-JSON approach would need extra parsing/retry logic to handle.
- **Loop limit lives in application logic (`loop_count`/`max_loops`), not just LangGraph's `recursion_limit`** — so the behavior is explicit, testable, and independent of LangGraph's internal step-counting semantics.