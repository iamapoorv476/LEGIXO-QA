"""Unit tests for src.chunking.

Uses a fake whitespace-based TokenEncoding (injected via chunk_text's
`encoding=` param) so these tests run offline and deterministically,
without depending on tiktoken's network-fetched vocab file.
"""

from __future__ import annotations

import pytest

from src.chunking import chunk_text


class FakeWhitespaceEncoding:
    """Treats each whitespace-separated word as one 'token'."""

    def encode(self, text: str) -> list[str]:
        return text.split(" ")

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


@pytest.fixture
def fake_encoding():
    return FakeWhitespaceEncoding()


def test_chunk_text_splits_into_expected_count(fake_encoding):
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size_tokens=20, chunk_overlap_tokens=5, encoding=fake_encoding)
    assert len(chunks) == 7
    assert [c.chunk_index for c in chunks] == list(range(7))


def test_chunk_overlap_repeats_tail_words(fake_encoding):
    text = " ".join(f"word{i}" for i in range(30))
    chunks = chunk_text(text, chunk_size_tokens=10, chunk_overlap_tokens=3, encoding=fake_encoding)
    tail_of_first = chunks[0].text.split(" ")[-3:]
    head_of_second = chunks[1].text.split(" ")[:3]
    assert tail_of_first == head_of_second


def test_empty_text_returns_no_chunks(fake_encoding):
    assert chunk_text("", 20, 5, encoding=fake_encoding) == []


def test_short_text_returns_single_chunk(fake_encoding):
    chunks = chunk_text("only three words", 20, 5, encoding=fake_encoding)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "only three words"


@pytest.mark.parametrize(
    "size,overlap",
    [(0, 0), (-5, 0), (10, 10), (10, 15)],
)
def test_invalid_size_overlap_raises(fake_encoding, size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text here", size, overlap, encoding=fake_encoding)


def test_real_tiktoken_encoding_used_by_default_when_available():
    """If tiktoken's vocab is reachable/cached, the real default path works
    too -- this is skipped (not failed) in network-restricted environments."""
    pytest.importorskip("tiktoken")
    try:
        chunks = chunk_text("A short sentence for real tokenization.", 10, 2)
    except Exception as exc:
        pytest.skip(f"tiktoken vocab not reachable in this environment: {exc}")
    assert len(chunks) >= 1