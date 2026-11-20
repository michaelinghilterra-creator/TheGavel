"""Two-minute demo, no GPU and no model server.

Run from the repository root:  python examples/demo.py

It scores the example reviews against an in-process mock of an OpenAI-compatible server
(the same mock the test suite uses), so you can see TheGavel's output format right away.
The mock answers with fixed, made-up probabilities, so the numbers are not a judgment;
point --base-url at a real server for that (see examples/README.md).
"""
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from mock_server import MockLLM  # noqa: E402

from gavel.backends.openai_compat import OpenAICompatBackend  # noqa: E402
from gavel.cli import main  # noqa: E402

mock = MockLLM(ratings={"": {str(n): 1 / 6 for n in range(6)}})


def backend(url, model, api_key):
    return OpenAICompatBackend(url, model, api_key=api_key, client=httpx.AsyncClient(transport=mock.transport))


if __name__ == "__main__":
    rubric = ROOT / "examples" / "reviews" / "rubric.yaml"
    items = ROOT / "examples" / "reviews" / "items.jsonl"
    raise SystemExit(main(["score", str(rubric), str(items)], backend_factory=backend))
