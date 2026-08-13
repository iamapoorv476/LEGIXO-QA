"""The Q&A LangGraph flow.

    retrieve -> grade_chunks --[good]--> generate_answer -> validate_citations --[has valid citations]--> END
                    |                                              |
                    |--[bad, loop_count < max_loops]--> rewrite_query -> retrieve (loop)
                    |
                    |--[bad, loop_count >= max_loops]--> no_answer -> END

                                              validate_citations --[no valid citations]--> no_answer -> END

Two real conditional branches:
  1. grade_chunks -> {generate_answer | rewrite_query | no_answer}, decided by
     an LLM-graded relevance judgment (src/llm.grade_relevance), not a
     hardcoded value.
  2. validate_citations -> {END | no_answer}, decided by checking every
     cited chunk_id actually exists in what was retrieved.

Loop limit: `loop_count` increments once per `retrieve` call and is checked
in `route_after_grade`. Once loop_count reaches max_loops, the bad-grade
path is forced to `no_answer` instead of `rewrite_query`, so retrieve can
run at most `max_loops` times regardless of how the LLM grades things.
"""

from __future__ import annotations

import logging
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph
from langsmith import traceable

from src.config import settings
from src.embeddings import embed_query
from src.llm import generate_answer, grade_relevance, rewrite_query
from src.pinecone_client import PineconeClient

logger = logging.getLogger(__name__)

NO_ANSWER_MESSAGE = "I can't find information about this in the provided documents."


class RetrievedChunkDict(TypedDict):
    chunk_id: str
    text: str
    source_file: str
    score: float


class QAState(TypedDict):
    question: str  # current query text (may be rewritten mid-loop)
    original_question: str  # always the user's original wording, used for the final answer
    retrieved_chunks: list[RetrievedChunkDict]
    grade: str  # "" | "good" | "bad"
    grade_reasoning: str
    answer: str
    citations: list[str]
    loop_count: int
    max_loops: int
    trace: list[str]


# --- Nodes --------------------------------------------------------------------


@traceable(run_type="retriever", name="retrieve")
def retrieve_node(state: QAState, pinecone_client: PineconeClient | None = None) -> dict:
    """Embed the current query, search Pinecone, keep chunks above the
    similarity floor. Increments loop_count -- this is what the loop
    limit actually counts."""

    client = pinecone_client or PineconeClient()
    embedding = embed_query(state["question"])
    scored = client.query(embedding, top_k=settings.top_k)
    filtered = [c for c in scored if c.score >= settings.min_similarity]

    chunks: list[RetrievedChunkDict] = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source_file": c.source_file,
            "score": c.score,
        }
        for c in filtered
    ]

    new_loop_count = state.get("loop_count", 0) + 1
    trace = state.get("trace", []) + [
        f"retrieve (loop {new_loop_count}): query={state['question']!r} "
        f"-> {len(chunks)}/{len(scored)} chunk(s) above min_similarity={settings.min_similarity}"
    ]
    return {"retrieved_chunks": chunks, "loop_count": new_loop_count, "trace": trace}


@traceable(run_type="chain", name="grade_chunks")
def grade_chunks_node(state: QAState) -> dict:
    """Real branch condition: an LLM judges whether retrieved chunks are
    sufficient. No hardcoded True/False."""

    chunks = state["retrieved_chunks"]
    if not chunks:
        is_sufficient, reasoning = False, "No chunks were retrieved above the similarity threshold."
    else:
        result = grade_relevance(state["question"], chunks)
        is_sufficient, reasoning = result.is_sufficient, result.reasoning

    trace = state["trace"] + [f"grade_chunks: sufficient={is_sufficient} -- {reasoning}"]
    return {
        "grade": "good" if is_sufficient else "bad",
        "grade_reasoning": reasoning,
        "trace": trace,
    }


@traceable(run_type="chain", name="rewrite_query_node")
def rewrite_query_node(state: QAState) -> dict:
    new_query = rewrite_query(state["original_question"], state["grade_reasoning"])
    trace = state["trace"] + [f"rewrite_query: {state['question']!r} -> {new_query!r}"]
    return {"question": new_query, "trace": trace}


@traceable(run_type="chain", name="generate_answer_node")
def generate_answer_node(state: QAState) -> dict:
    result = generate_answer(state["original_question"], state["retrieved_chunks"])
    trace = state["trace"] + [
        f"generate_answer: proposed {len(result.cited_chunk_ids)} citation(s): {result.cited_chunk_ids}"
    ]
    return {"answer": result.answer, "citations": result.cited_chunk_ids, "trace": trace}


@traceable(run_type="chain", name="validate_citations")
def validate_citations_node(state: QAState) -> dict:
    """Programmatic guarantee against fake citations: every cited chunk_id
    must exist in what was actually retrieved, or it's dropped."""

    valid_ids = {c["chunk_id"] for c in state["retrieved_chunks"]}
    validated = [cid for cid in state["citations"] if cid in valid_ids]
    dropped = [cid for cid in state["citations"] if cid not in valid_ids]

    if dropped:
        trace_line = f"validate_citations: dropped {len(dropped)} unverifiable id(s): {dropped}"
        logger.warning("Dropped unverifiable citation ids: %s", dropped)
    else:
        trace_line = f"validate_citations: all {len(validated)} citation(s) verified against retrieved chunks"

    return {"citations": validated, "trace": state["trace"] + [trace_line]}


@traceable(run_type="chain", name="no_answer")
def no_answer_node(state: QAState) -> dict:
    trace = state["trace"] + ["no_answer: could not produce a grounded, cited answer from the corpus"]
    return {"answer": NO_ANSWER_MESSAGE, "citations": [], "trace": trace}


# --- Conditional edges ----------------------------------------------------------


def route_after_grade(state: QAState) -> Literal["generate", "rewrite", "no_answer"]:
    if state["grade"] == "good":
        return "generate"
    if state["loop_count"] >= state["max_loops"]:
        return "no_answer"
    return "rewrite"


def route_after_validation(state: QAState) -> Literal["end", "no_answer"]:
    return "end" if state["citations"] else "no_answer"


# --- Graph assembly --------------------------------------------------------------


def build_graph():
    builder = StateGraph(QAState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade_chunks", grade_chunks_node)
    builder.add_node("rewrite_query", rewrite_query_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("validate_citations", validate_citations_node)
    builder.add_node("no_answer", no_answer_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "grade_chunks")
    builder.add_conditional_edges(
        "grade_chunks",
        route_after_grade,
        {
            "generate": "generate_answer",
            "rewrite": "rewrite_query",
            "no_answer": "no_answer",
        },
    )
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("generate_answer", "validate_citations")
    builder.add_conditional_edges(
        "validate_citations",
        route_after_validation,
        {
            "end": END,
            "no_answer": "no_answer",
        },
    )
    builder.add_edge("no_answer", END)

    return builder.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


@traceable(run_type="chain", name="run_qa")
def run_qa(question: str) -> QAState:
    """Entry point used by the API layer: run the full graph for one question."""
    graph = get_graph()
    initial_state: QAState = {
        "question": question,
        "original_question": question,
        "retrieved_chunks": [],
        "grade": "",
        "grade_reasoning": "",
        "answer": "",
        "citations": [],
        "loop_count": 0,
        "max_loops": settings.max_loops,
        "trace": [],
    }
    # Generous recursion_limit relative to max_loops: each loop iteration is
    # retrieve -> grade_chunks -> rewrite_query (3 node visits), plus a
    # fixed tail of generate_answer -> validate_citations (-> no_answer).
    # This bounds LangGraph's own step counter well above what max_loops
    # can ever produce, so OUR loop limit is what actually stops execution.
    recursion_limit = (settings.max_loops * 3) + 10
    final_state = graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
    return final_state