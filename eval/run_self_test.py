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


def run(base_url: str, self_test_path: Path, output_path: Path) -> int:
    cases = json.loads(self_test_path.read_text(encoding="utf-8"))

    results = []
    passed = 0
    failed = 0

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
        else:
            # Out-of-corpus question: must come back with zero citations.
            is_pass = len(actual_files) == 0

        status = "PASS" if is_pass else "FAIL"
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
        if is_pass:
            passed += 1
        else:
            failed += 1

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n{passed}/{len(cases)} passed, {failed} failed.")
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