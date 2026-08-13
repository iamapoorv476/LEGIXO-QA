"""Tests for src.api using FastAPI's TestClient.

run_qa (the graph entry point) is mocked so these test the HTTP layer
itself -- request validation, response shape, error mapping -- without
depending on real Pinecone/Anthropic calls or on graph internals (those
are covered by tests/test_graph.py).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.llm import LLMError
from src.pinecone_client import PineconeIndexError

client = TestClient(app)


def _fake_graph_result(**overrides) -> dict:
    base = {
        "answer": "The notice period is 60 days.",
        "citations": ["chunk_abc123"],
        "retrieved_chunks": [
            {
                "chunk_id": "chunk_abc123",
                "text": "...",
                "source_file": "02_employment_agreement_excerpt.md",
                "score": 0.87,
            }
        ],
        "trace": [
            "retrieve (loop 1): query='...' -> 1/1 chunk(s) above min_similarity=0.15",
            "grade_chunks: sufficient=True -- clear match",
            "generate_answer: proposed 1 citation(s): ['chunk_abc123']",
            "validate_citations: all 1 citation(s) verified against retrieved chunks",
        ],
        "loop_count": 1,
    }
    base.update(overrides)
    return base


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "config" in body
    # secrets must be redacted, not echoed in full
    assert "…redacted" in body["config"]["anthropic_api_key"]


def test_ask_happy_path_includes_trace_by_default():
    with patch("src.api.run_qa", return_value=_fake_graph_result()) as mock_run:
        response = client.post("/ask", json={"question": "What is the notice period?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The notice period is 60 days."
    assert body["citations"] == [
        {"chunk_id": "chunk_abc123", "source_file": "02_employment_agreement_excerpt.md"}
    ]
    assert body["trace"] is not None
    assert len(body["trace"]) == 4
    assert body["loop_count"] == 1
    mock_run.assert_called_once_with("What is the notice period?")


def test_ask_with_include_trace_false_omits_trace():
    with patch("src.api.run_qa", return_value=_fake_graph_result()):
        response = client.post(
            "/ask?include_trace=false", json={"question": "What is the notice period?"}
        )

    assert response.status_code == 200
    assert response.json()["trace"] is None


def test_ask_out_of_corpus_returns_empty_citations_not_error():
    no_answer_result = _fake_graph_result(
        answer="I can't find information about this in the provided documents.",
        citations=[],
        retrieved_chunks=[],
        loop_count=2,
    )
    with patch("src.api.run_qa", return_value=no_answer_result):
        response = client.post("/ask", json={"question": "What is the capital of France?"})

    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []
    assert "can't find" in body["answer"].lower()


def test_ask_empty_question_returns_400():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"


def test_ask_missing_question_field_returns_422():
    """Pydantic validation error for a malformed request body -- FastAPI's
    default 422, not a raw 500."""
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_pinecone_failure_returns_502_not_500():
    with patch("src.api.run_qa", side_effect=PineconeIndexError("index not found")):
        response = client.post("/ask", json={"question": "anything"})

    assert response.status_code == 502
    body = response.json()
    assert body["error"] == "pinecone_error"
    assert "index not found" in body["detail"]


def test_ask_llm_failure_returns_502_not_500():
    with patch("src.api.run_qa", side_effect=LLMError("rate limited")):
        response = client.post("/ask", json={"question": "anything"})

    assert response.status_code == 502
    body = response.json()
    assert body["error"] == "llm_error"


def test_ask_unexpected_exception_returns_500_with_json_not_a_crash():
    with patch("src.api.run_qa", side_effect=RuntimeError("something unforeseen")):
        response = client.post("/ask", json={"question": "anything"})

    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "internal_error"
    assert "something unforeseen" in body["detail"]


def test_ask_citations_missing_from_retrieved_chunks_are_silently_excluded():
    """Defense in depth at the API layer too: if a citation id somehow
    isn't in retrieved_chunks (shouldn't happen after graph validation,
    but the API shouldn't crash trying to look up source_file for it)."""
    result = _fake_graph_result(citations=["chunk_abc123", "chunk_not_in_retrieved"])
    with patch("src.api.run_qa", return_value=result):
        response = client.post("/ask", json={"question": "anything"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["citations"]) == 1
    assert body["citations"][0]["chunk_id"] == "chunk_abc123"