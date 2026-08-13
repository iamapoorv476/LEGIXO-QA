"""FastAPI application exposing the Q&A system over HTTP.

Per the assignment spec: HTTP API only for asking questions. Ingestion is
a separate CLI command (scripts/ingest_cli.py), not exposed here.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import ConfigError, settings
from src.graph import run_qa
from src.llm import LLMError
from src.pinecone_client import PineconeIndexError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Legixo Q&A API",
    description=(
        "Ask questions over a document corpus. Answers are grounded in "
        "retrieved chunks with verified citations; out-of-corpus questions "
        "are explicitly refused rather than hallucinated."
    ),
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask.")


class Citation(BaseModel):
    chunk_id: str
    source_file: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    trace: list[str] | None = None
    loop_count: int


class ErrorResponse(BaseModel):
    error: str
    detail: str


@app.get("/health")
def health() -> dict:
    """Basic liveness + config sanity check. Does not hit Pinecone/Anthropic."""
    return {"status": "ok", "config": settings.as_safe_dict()}


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        502: {"model": ErrorResponse, "description": "Upstream (Pinecone/Anthropic) failure"},
        500: {"model": ErrorResponse, "description": "Unexpected server error"},
    },
)
def ask(
    request: AskRequest,
    include_trace: bool = Query(
        default=True,
        description=(
            "Include the step-by-step LangGraph trace in the response. "
            "Defaults to true so the graph's reasoning is visible without "
            "extra configuration; set to false for a lighter response."
        ),
    ),
):
    """Answer a question using the ingested corpus.

    Runs the LangGraph flow: retrieve -> grade -> (rewrite-and-retry |
    generate-and-validate | refuse). Citations returned here have already
    been verified against actually-retrieved chunks -- see
    src/graph.py:validate_citations_node.
    """
    question = request.question.strip()
    if not question:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="invalid_request", detail="question must not be empty"
            ).model_dump(),
        )

    try:
        result = run_qa(question)
    except (PineconeIndexError,) as exc:
        logger.error("Pinecone error while answering %r: %s", question, exc)
        return JSONResponse(
            status_code=502,
            content=ErrorResponse(
                error="pinecone_error",
                detail=(
                    "The vector database is unavailable or the corpus hasn't "
                    "been ingested yet. Run `python -m scripts.ingest_cli` first. "
                    f"Details: {exc}"
                ),
            ).model_dump(),
        )
    except (LLMError,) as exc:
        logger.error("LLM error while answering %r: %s", question, exc)
        return JSONResponse(
            status_code=502,
            content=ErrorResponse(
                error="llm_error",
                detail=f"The language model call failed: {exc}",
            ).model_dump(),
        )
    except ConfigError as exc:
        logger.error("Config error while answering %r: %s", question, exc)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="config_error", detail=str(exc)).model_dump(),
        )
    except Exception as exc:  # last-resort guard: never a silent bare 500
        logger.exception("Unexpected error while answering %r", question)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error", detail=f"Unexpected error: {exc}"
            ).model_dump(),
        )

    chunk_by_id = {c["chunk_id"]: c for c in result["retrieved_chunks"]}
    citations = [
        Citation(chunk_id=cid, source_file=chunk_by_id[cid]["source_file"])
        for cid in result["citations"]
        if cid in chunk_by_id
    ]

    return AskResponse(
        answer=result["answer"],
        citations=citations,
        trace=result["trace"] if include_trace else None,
        loop_count=result["loop_count"],
    )