"""Anthropic LLM calls, isolated behind typed functions.

Two of the three calls (grading, answer generation) use forced tool-calling
so the model MUST return a schema-conformant structure -- no free-text
parsing, no "hope it's valid JSON." The third (query rewriting) returns
plain text since a single rewritten question string doesn't benefit from
a forced schema.
"""

from __future__ import annotations

from dataclasses import dataclass

from anthropic import Anthropic

from src.config import settings
from src.pinecone_client import ScoredChunk

_client = Anthropic(api_key=settings.anthropic_api_key)

_MAX_TOKENS = 1024


class LLMError(RuntimeError):
    """Raised when an Anthropic call fails or returns an unusable response."""


@dataclass
class GradeResult:
    is_sufficient: bool
    reasoning: str


@dataclass
class AnswerResult:
    answer: str
    cited_chunk_ids: list[str]


# --- Tool schemas -----------------------------------------------------------

_GRADE_TOOL = {
    "name": "submit_grade",
    "description": (
        "Submit a judgment on whether the retrieved document chunks contain "
        "enough information to accurately answer the user's question."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "is_sufficient": {
                "type": "boolean",
                "description": (
                    "True only if the chunks directly and specifically answer "
                    "the question. False if the chunks are off-topic, only "
                    "tangentially related, or missing the key fact asked for."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "One or two sentences explaining the judgment.",
            },
        },
        "required": ["is_sufficient", "reasoning"],
    },
}

_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": (
        "Submit the final answer to the user's question, grounded strictly "
        "in the provided chunks, along with the IDs of the chunks that "
        "actually support it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "The answer, using only information present in the "
                    "provided chunks. Do not use outside knowledge."
                ),
            },
            "cited_chunk_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "chunk_id values, copied exactly from the provided chunks, "
                    "for every chunk actually used to support the answer. "
                    "Must be a subset of the chunk_ids given to you -- never "
                    "invent an id."
                ),
            },
        },
        "required": ["answer", "cited_chunk_ids"],
    },
}


def _format_chunks(chunks: list[ScoredChunk] | list[dict]) -> str:
    """Render chunks as a numbered block the model can cite by chunk_id."""
    lines = []
    for c in chunks:
        chunk_id = c.chunk_id if hasattr(c, "chunk_id") else c["chunk_id"]
        source_file = c.source_file if hasattr(c, "source_file") else c["source_file"]
        text = c.text if hasattr(c, "text") else c["text"]
        lines.append(f'[chunk_id="{chunk_id}" source="{source_file}"]\n{text}')
    return "\n\n".join(lines)


def _extract_tool_input(response, tool_name: str) -> dict:
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            return block.input
    raise LLMError(
        f"Expected a '{tool_name}' tool_use block in the response but found none. "
        f"stop_reason={response.stop_reason!r}"
    )


# --- Public functions --------------------------------------------------------


def grade_relevance(question: str, chunks: list[ScoredChunk] | list[dict]) -> GradeResult:
    """Ask Claude whether the retrieved chunks are sufficient to answer
    `question`. This is the real decision behind the graph's branch --
    not a heuristic, not a hardcoded True."""

    system = (
        "You are a strict relevance grader for a document Q&A system. "
        "Judge only whether the given chunks contain enough specific "
        "information to answer the question. Be skeptical: partial or "
        "vaguely-related content should be marked insufficient."
    )
    user = (
        f"Question: {question}\n\n"
        f"Retrieved chunks:\n{_format_chunks(chunks)}\n\n"
        "Do these chunks contain enough information to answer the question "
        "accurately? Call submit_grade with your judgment."
    )

    try:
        response = _client.messages.create(
            model=settings.anthropic_model,
            max_tokens=_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[_GRADE_TOOL],
            tool_choice={"type": "tool", "name": "submit_grade"},
        )
    except Exception as exc:
        raise LLMError(f"grade_relevance call failed: {exc}") from exc

    data = _extract_tool_input(response, "submit_grade")
    return GradeResult(
        is_sufficient=bool(data["is_sufficient"]),
        reasoning=str(data["reasoning"]),
    )


def generate_answer(question: str, chunks: list[ScoredChunk] | list[dict]) -> AnswerResult:
    """Ask Claude to answer `question` using only the given chunks, with
    forced structured citations."""

    system = (
        "You answer questions using ONLY the provided document chunks. "
        "Never use outside knowledge, even if you know the answer. "
        "Every claim in your answer must be traceable to a specific chunk. "
        "Cite every chunk_id you actually relied on -- never cite a chunk "
        "you didn't use, and never invent a chunk_id that wasn't provided."
    )
    user = (
        f"Question: {question}\n\n"
        f"Chunks:\n{_format_chunks(chunks)}\n\n"
        "Call submit_answer with your answer and the chunk_ids that support it."
    )

    try:
        response = _client.messages.create(
            model=settings.anthropic_model,
            max_tokens=_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[_ANSWER_TOOL],
            tool_choice={"type": "tool", "name": "submit_answer"},
        )
    except Exception as exc:
        raise LLMError(f"generate_answer call failed: {exc}") from exc

    data = _extract_tool_input(response, "submit_answer")
    cited = data.get("cited_chunk_ids", [])
    if not isinstance(cited, list):
        raise LLMError(f"cited_chunk_ids was not a list: {cited!r}")
    return AnswerResult(
        answer=str(data["answer"]),
        cited_chunk_ids=[str(c) for c in cited],
    )


def rewrite_query(original_question: str, grade_reasoning: str) -> str:
    """Reformulate the question when the first retrieval attempt came back
    insufficient. Plain text, not tool-called -- a single string doesn't
    need a forced schema, and this keeps the call cheap."""

    system = (
        "You rewrite search queries for a document retrieval system. "
        "Given a question that didn't retrieve enough information, produce "
        "ONE alternative phrasing more likely to match relevant document "
        "text -- broader terms, synonyms, or a rephrasing that drops "
        "conversational wrapping. Reply with ONLY the rewritten question, "
        "nothing else."
    )
    user = (
        f"Original question: {original_question}\n"
        f"Why the first retrieval was insufficient: {grade_reasoning}\n\n"
        "Rewritten question:"
    )

    try:
        response = _client.messages.create(
            model=settings.anthropic_model,
            max_tokens=200,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as exc:
        raise LLMError(f"rewrite_query call failed: {exc}") from exc

    text_blocks = [b.text for b in response.content if b.type == "text"]
    if not text_blocks:
        raise LLMError("rewrite_query response had no text content")
    return text_blocks[0].strip().strip('"')