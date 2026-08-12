"""Tests for src.graph -- the LangGraph flow.

Everything that would hit a real network (Pinecone queries, Anthropic
calls, embeddings) is mocked so these run offline and deterministically.
What's under test is the graph's *control flow*: does the branch actually
route based on the grade, does the loop limit actually stop it, does
citation validation actually drop bad ids.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.graph import (
    NO_ANSWER_MESSAGE,
    QAState,
    build_graph,
    route_after_grade,
    route_after_validation,
)
from src.llm import AnswerResult, GradeResult
from src.pinecone_client import ScoredChunk


# --- Unit tests for the routing functions themselves -------------------------


def test_route_after_grade_good_goes_to_generate():
    state: QAState = {"grade": "good", "loop_count": 1, "max_loops": 2}  # type: ignore[typeddict-item]
    assert route_after_grade(state) == "generate"


def test_route_after_grade_bad_under_limit_goes_to_rewrite():
    state: QAState = {"grade": "bad", "loop_count": 1, "max_loops": 2}  # type: ignore[typeddict-item]
    assert route_after_grade(state) == "rewrite"


def test_route_after_grade_bad_at_limit_goes_to_no_answer():
    state: QAState = {"grade": "bad", "loop_count": 2, "max_loops": 2}  # type: ignore[typeddict-item]
    assert route_after_grade(state) == "no_answer"


def test_route_after_grade_bad_over_limit_goes_to_no_answer():
    """Defensive: even if loop_count somehow exceeds max_loops, still stop."""
    state: QAState = {"grade": "bad", "loop_count": 5, "max_loops": 2}  # type: ignore[typeddict-item]
    assert route_after_grade(state) == "no_answer"


def test_route_after_validation_with_citations_ends():
    state: QAState = {"citations": ["chunk_abc"]}  # type: ignore[typeddict-item]
    assert route_after_validation(state) == "end"


def test_route_after_validation_without_citations_goes_to_no_answer():
    state: QAState = {"citations": []}  # type: ignore[typeddict-item]
    assert route_after_validation(state) == "no_answer"


# --- Full graph integration tests, with retrieve/grade/generate mocked -------


def _fake_scored_chunk(chunk_id: str, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}",
        source_file="fake.md",
        chunk_index=0,
        score=score,
    )


@pytest.fixture
def mock_pinecone_query():
    with patch("src.graph.PineconeClient") as MockClient:
        instance = MockClient.return_value
        instance.query.return_value = [_fake_scored_chunk("chunk_1"), _fake_scored_chunk("chunk_2")]
        yield instance


@pytest.fixture
def mock_embed():
    with patch("src.graph.embed_query", return_value=[0.1, 0.2, 0.3]):
        yield


def test_graph_happy_path_good_grade_valid_citations(mock_pinecone_query, mock_embed):
    """grade=good -> generate_answer -> validate_citations (all valid) -> END.
    Never touches rewrite_query or no_answer."""
    with patch(
        "src.graph.grade_relevance",
        return_value=GradeResult(is_sufficient=True, reasoning="Chunks directly answer it."),
    ) as mock_grade, patch(
        "src.graph.generate_answer",
        return_value=AnswerResult(answer="The answer is X.", cited_chunk_ids=["chunk_1"]),
    ) as mock_gen, patch("src.graph.rewrite_query") as mock_rewrite:
        graph = build_graph()
        final = graph.invoke(
            {
                "question": "What is X?",
                "original_question": "What is X?",
                "retrieved_chunks": [],
                "grade": "",
                "grade_reasoning": "",
                "answer": "",
                "citations": [],
                "loop_count": 0,
                "max_loops": 2,
                "trace": [],
            }
        )

    assert final["answer"] == "The answer is X."
    assert final["citations"] == ["chunk_1"]
    assert final["loop_count"] == 1  # retrieve only ran once
    mock_grade.assert_called_once()
    mock_gen.assert_called_once()
    mock_rewrite.assert_not_called()


def test_graph_bad_grade_loops_then_succeeds(mock_pinecone_query, mock_embed):
    """First grade is bad -> rewrite -> retrieve again -> second grade good
    -> generate. Proves the real branch + loop-back actually execute."""
    grade_results = [
        GradeResult(is_sufficient=False, reasoning="Not enough detail."),
        GradeResult(is_sufficient=True, reasoning="Now sufficient."),
    ]
    with patch("src.graph.grade_relevance", side_effect=grade_results) as mock_grade, patch(
        "src.graph.rewrite_query", return_value="a broader version of the question"
    ) as mock_rewrite, patch(
        "src.graph.generate_answer",
        return_value=AnswerResult(answer="Found it.", cited_chunk_ids=["chunk_2"]),
    ):
        graph = build_graph()
        final = graph.invoke(
            {
                "question": "narrow question",
                "original_question": "narrow question",
                "retrieved_chunks": [],
                "grade": "",
                "grade_reasoning": "",
                "answer": "",
                "citations": [],
                "loop_count": 0,
                "max_loops": 3,
                "trace": [],
            }
        )

    assert final["answer"] == "Found it."
    assert final["loop_count"] == 2  # retrieve ran twice
    assert mock_grade.call_count == 2
    mock_rewrite.assert_called_once()
    # second retrieve should have used the rewritten query
    assert final["question"] == "a broader version of the question"


def test_graph_hits_loop_limit_and_gives_no_answer(mock_pinecone_query, mock_embed):
    """Grade is bad every time -> loop limit must stop it, not spin forever."""
    with patch(
        "src.graph.grade_relevance",
        return_value=GradeResult(is_sufficient=False, reasoning="Still not enough."),
    ) as mock_grade, patch(
        "src.graph.rewrite_query", return_value="still not helping"
    ) as mock_rewrite, patch("src.graph.generate_answer") as mock_gen:
        graph = build_graph()
        final = graph.invoke(
            {
                "question": "unanswerable question",
                "original_question": "unanswerable question",
                "retrieved_chunks": [],
                "grade": "",
                "grade_reasoning": "",
                "answer": "",
                "citations": [],
                "loop_count": 0,
                "max_loops": 2,
                "trace": [],
            }
        )

    assert final["answer"] == NO_ANSWER_MESSAGE
    assert final["citations"] == []
    assert final["loop_count"] == 2  # stopped exactly at max_loops, not beyond
    assert mock_grade.call_count == 2
    mock_rewrite.assert_called_once()  # loop 1 -> bad -> rewrite; loop 2 -> bad -> no_answer (no 2nd rewrite)
    mock_gen.assert_not_called()  # never reached generate_answer


def test_graph_drops_fake_citations_and_falls_back_to_no_answer(mock_pinecone_query, mock_embed):
    """The LLM claims a citation to a chunk_id that was never retrieved --
    validate_citations must strip it, and with nothing valid left, the
    graph must fall back to no_answer rather than returning a fake cite."""
    with patch(
        "src.graph.grade_relevance",
        return_value=GradeResult(is_sufficient=True, reasoning="Looks sufficient."),
    ), patch(
        "src.graph.generate_answer",
        return_value=AnswerResult(
            answer="A plausible-sounding but ungrounded answer.",
            cited_chunk_ids=["chunk_does_not_exist"],
        ),
    ):
        graph = build_graph()
        final = graph.invoke(
            {
                "question": "question",
                "original_question": "question",
                "retrieved_chunks": [],
                "grade": "",
                "grade_reasoning": "",
                "answer": "",
                "citations": [],
                "loop_count": 0,
                "max_loops": 2,
                "trace": [],
            }
        )

    assert final["answer"] == NO_ANSWER_MESSAGE
    assert final["citations"] == []


def test_graph_drops_only_invalid_citations_keeps_valid_ones(mock_pinecone_query, mock_embed):
    """Mixed case: one real citation, one fake -- fake gets dropped, real
    one survives, and since something valid remains we do NOT fall back."""
    with patch(
        "src.graph.grade_relevance",
        return_value=GradeResult(is_sufficient=True, reasoning="Sufficient."),
    ), patch(
        "src.graph.generate_answer",
        return_value=AnswerResult(
            answer="Partly grounded answer.",
            cited_chunk_ids=["chunk_1", "chunk_fabricated"],
        ),
    ):
        graph = build_graph()
        final = graph.invoke(
            {
                "question": "question",
                "original_question": "question",
                "retrieved_chunks": [],
                "grade": "",
                "grade_reasoning": "",
                "answer": "",
                "citations": [],
                "loop_count": 0,
                "max_loops": 2,
                "trace": [],
            }
        )

    assert final["answer"] == "Partly grounded answer."
    assert final["citations"] == ["chunk_1"]


def test_retrieve_node_filters_below_min_similarity(mock_embed):
    from dataclasses import replace

    from src.config import settings as real_settings
    from src.graph import retrieve_node

    mock_client = MagicMock()
    mock_client.query.return_value = [
        _fake_scored_chunk("chunk_high", score=0.9),
        _fake_scored_chunk("chunk_low", score=0.01),
    ]
    patched_settings = replace(real_settings, min_similarity=0.15)
    with patch("src.graph.settings", patched_settings):
        result = retrieve_node(
            {
                "question": "q",
                "trace": [],
                "loop_count": 0,
            },  # type: ignore[typeddict-item]
            pinecone_client=mock_client,
        )

    ids = [c["chunk_id"] for c in result["retrieved_chunks"]]
    assert "chunk_high" in ids
    assert "chunk_low" not in ids
    assert result["loop_count"] == 1