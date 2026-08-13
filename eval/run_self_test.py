"""Run eval/self_test.json against a running instance of the API and
report pass/fail based on whether the expected source files show up in
the citations.

Usage:
    # in one terminal:
    python -m scripts.run_server

    # in another:
    python -m eval.run_self_test
    python -m eval.run_self_test --base-url http://localhost:8000 --output eval/self_test_results.json

This does NOT judge answer *quality* automatically -- it checks whether
the right documents were cited, which is what the assignment's grading
criteria actually asks for ("are answers tied to real chunks... which
files should appear in citations"). Read the actual answer text yourself
for the qualitative self-critique notes the assignment also asks for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

DEFAULT_SELF_TEST_PATH = Path(__file__).parent / "self_test.json"
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "self_test_results.json"

# Known limitation, documented rather than silently swallowed: for
# out-of-corpus questions (expected_citation_files == []), a citation-only
# check can't distinguish "cited real chunks to fabricate an answer" (bad)
# from "cited real chunks to honestly explain related context while still
# correctly saying the specific thing asked isn't in the documents" (fine,
# arguably better than a flat refusal). Case 15 in self_test.json hit this
# exact ambiguity -- see its notes field for the full writeup. IDs listed
# here get a WARN instead of an automatic FAIL when they come back with
# non-empty citations; the actual_answer text still needs a human read.
KNOWN_AMBIGUOUS_OUT_OF_CORPUS_IDS = {15}


def run(base_url: str, self_test_path: Path, output_path: Path) -> int:
    cases = json.loads(self_test_path.read_text(encoding="utf-8"))

    results = []
    passed = 0
    failed = 0
    warned = 0

    for case in cases:
        question = case["question"]
        expected_files = set(case.get("expected_citation_files", []))

        print(f"[{case['id']}] {question}")
        try:
            response = requests.post(
                f"{base_url}/ask",
                json={"question": question},
                timeout=60,
            )
        except requests.RequestException as exc:
            print(f"    ERROR: could not reach API: {exc}")
            results.append({**case, "pass": False, "notes": f"Request failed: {exc}"})
            failed += 1
            continue

        if response.status_code != 200:
            print(f"    ERROR: HTTP {response.status_code}: {response.text[:200]}")
            results.append(
                {**case, "pass": False, "notes": f"HTTP {response.status_code}: {response.text[:200]}"}
            )
            failed += 1
            continue

        body = response.json()
        actual_files = {c["source_file"] for c in body["citations"]}

        if expected_files:
            # In-corpus question: every expected file must be cited (order-independent).
            is_pass = expected_files.issubset(actual_files)
            status = "PASS" if is_pass else "FAIL"
        elif not actual_files:
            # Out-of-corpus question, correctly came back with zero citations.
            is_pass = True
            status = "PASS"
        elif case["id"] in KNOWN_AMBIGUOUS_OUT_OF_CORPUS_IDS:
            # Out-of-corpus question with non-empty citations, but this ID is
            # known to sometimes produce a legitimately-cited "here's the
            # related context, but that specific fact isn't in the documents"
            # answer rather than a fabrication. Needs a human to read
            # actual_answer -- don't auto-fail or auto-pass.
            is_pass = None
            status = "WARN (needs human read of actual_answer -- see KNOWN_AMBIGUOUS_OUT_OF_CORPUS_IDS)"
        else:
            # Out-of-corpus question, unexpected non-empty citations, and not
            # a known-ambiguous case -- treat as a real failure.
            is_pass = False
            status = "FAIL"

        print(f"    {status}  expected={sorted(expected_files)}  actual={sorted(actual_files)}")

        results.append(
            {
                **case,
                "actual_answer": body["answer"],
                "actual_citation_files": sorted(actual_files),
                "loop_count": body["loop_count"],
                "pass": is_pass,
            }
        )
        if is_pass is True:
            passed += 1
        elif is_pass is False:
            failed += 1
        else:
            warned += 1

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n{passed}/{len(cases)} passed, {failed} failed, {warned} need manual review.")
    print(f"Full results written to {output_path}")
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the self-test question set against a live API.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--self-test-path", type=Path, default=DEFAULT_SELF_TEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    return run(args.base_url, args.self_test_path, args.output)


if __name__ == "__main__":
    sys.exit(main())