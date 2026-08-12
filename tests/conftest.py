"""Shared test fixtures.

Sets dummy environment variables before any test imports src.config, since
Settings loading fails fast on missing keys (by design — see config.py).
None of these values are real; tests that touch Pinecone/OpenAI/Anthropic
mock the network calls rather than using these credentials for real requests.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
os.environ.setdefault("PINECONE_API_KEY", "pcsk-test-dummy")
os.environ.setdefault("PINECONE_INDEX_NAME", "legixo-qa-test")
os.environ.setdefault("CORPUS_DIR", str(Path(__file__).resolve().parent.parent / "corpus"))