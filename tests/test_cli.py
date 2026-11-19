import json
import math
from pathlib import Path

import httpx

from gavel.backends.openai_compat import OpenAICompatBackend
from gavel.cli import main
from mock_server import MockLLM


FIXTURES = Path(__file__).parent / "fixtures"
RUBRIC = FIXTURES / "demo_rubric.yaml"


def factory_for(mock: MockLLM):
    def factory(base_url, model, api_key):
        client = httpx.AsyncClient(transport=mock.transport)
        return OpenAICompatBackend(base_url, model, api_key=api_key, client=client)

    return factory


def checked_mock() -> MockLLM:
    labels = {str(value): 1 / 6 for value in range(6)}
    return MockLLM(ratings={"": labels})


def test_score_writes_five_results_in_order(capsys) -> None:
    code = main(
        ["score", str(RUBRIC), str(FIXTURES / "demo_items.jsonl")],
        backend_factory=factory_for(checked_mock()),
    )
    captured = capsys.readouterr()
    results = [json.loads(line) for line in captured.out.splitlines()]
    assert code == 0
    assert [result["id"] for result in results] == [f"zw-{i}" for i in range(1, 6)]
    assert "items 5 ok 5 errors 0" in captured.err


def test_missing_input_file_returns_two(capsys) -> None:
    code = main(
        ["score", str(RUBRIC), str(FIXTURES / "missing.jsonl")],
        backend_factory=factory_for(checked_mock()),
    )
    assert code == 2
    assert "input error" in capsys.readouterr().err


def test_item_error_returns_one(capsys) -> None:
    mock = checked_mock()
    mock.ratings.update({"clarity": {"x": 1.0}})
    code = main(
        ["score", str(RUBRIC), str(FIXTURES / "demo_items.jsonl")],
        backend_factory=factory_for(mock),
    )
    results = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 1
    assert all("error" in result for result in results)


def test_bench_two_runs_json_has_noise(capsys) -> None:
    code = main(
        [
            "bench",
            str(RUBRIC),
            str(FIXTURES / "demo_labelled.jsonl"),
            "--runs",
            "2",
            "--json",
        ],
        backend_factory=factory_for(checked_mock()),
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert len(report["runs"]) == 2
    assert report["noise"]["n"] == 8


def test_no_label_check_skips_probe_requests(capsys) -> None:
    mock = MockLLM()
    code = main(
        [
            "score",
            str(RUBRIC),
            str(FIXTURES / "demo_items.jsonl"),
            "--no-label-check",
        ],
        backend_factory=factory_for(mock),
    )
    capsys.readouterr()
    assert code == 0
    assert len(mock.requests) == 20
    assert not any(request["messages"][0]["content"].startswith("Reply with") for request in mock.requests)
