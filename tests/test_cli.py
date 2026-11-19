import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import httpx
from fastapi import FastAPI

import gavel.cli as cli
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


class StatsBackend:
    def __init__(self, backend):
        self.backend = backend
        self.stats = SimpleNamespace(
            prefill_tokens=0,
            reused_tokens=0,
            prefill_s=0.0,
            decode_tokens=0,
            decode_s=0.0,
        )

    def _increment_stats(self) -> None:
        self.stats.prefill_tokens += 10
        self.stats.reused_tokens += 2
        self.stats.prefill_s += 0.5
        self.stats.decode_tokens += 3
        self.stats.decode_s += 0.25

    async def next_token_logprobs(self, *args, **kwargs):
        self._increment_stats()
        return await self.backend.next_token_logprobs(*args, **kwargs)

    async def generate(self, *args, **kwargs):
        self._increment_stats()
        return await self.backend.generate(*args, **kwargs)

    async def aclose(self) -> None:
        await self.backend.aclose()


def stats_factory_for(mock: MockLLM):
    base_factory = factory_for(mock)

    def factory(base_url, model, api_key):
        return StatsBackend(base_factory(base_url, model, api_key))

    return factory


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


def test_native_requires_model_path(capsys) -> None:
    code = main(
        [
            "score",
            str(RUBRIC),
            str(FIXTURES / "demo_items.jsonl"),
            "--backend",
            "native",
        ]
    )
    assert code == 2
    assert (
        "input error: --model-path is required for --backend native"
        in capsys.readouterr().err
    )


def test_completions_requires_one_prompt_renderer(capsys) -> None:
    code = main(
        [
            "score",
            str(RUBRIC),
            str(FIXTURES / "demo_items.jsonl"),
            "--backend",
            "completions",
        ]
    )
    assert code == 2
    assert (
        "input error: --backend completions needs --tokenizer or --prompt-format"
        in capsys.readouterr().err
    )


def test_completions_rejects_bad_template_kwargs(capsys) -> None:
    code = main(
        [
            "score",
            str(RUBRIC),
            str(FIXTURES / "demo_items.jsonl"),
            "--backend",
            "completions",
            "--prompt-format",
            "chatml",
            "--template-kwargs",
            "{bad",
        ]
    )
    assert code == 2
    assert "input error: invalid --template-kwargs JSON" in capsys.readouterr().err


def test_score_prints_native_stats(capsys) -> None:
    code = main(
        ["score", str(RUBRIC), str(FIXTURES / "demo_items.jsonl")],
        backend_factory=stats_factory_for(checked_mock()),
    )
    captured = capsys.readouterr()
    assert code == 0
    assert (
        "prefill_tokens 260 reused_tokens 52 prefill_seconds 13.000 "
        "prefill_tok_per_s 20.0 decode_tokens 78 decode_seconds 6.500"
        in captured.err
    )


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


def test_bench_json_native_stats_are_per_run(capsys) -> None:
    code = main(
        [
            "bench",
            str(RUBRIC),
            str(FIXTURES / "demo_labelled.jsonl"),
            "--runs",
            "2",
            "--json",
            "--no-label-check",
        ],
        backend_factory=stats_factory_for(checked_mock()),
    )
    report = json.loads(capsys.readouterr().out)
    expected = {
        "prefill_tokens": 320,
        "reused_tokens": 64,
        "prefill_seconds": 16.0,
        "decode_tokens": 96,
        "decode_seconds": 8.0,
        "prefill_tok_per_s": 20.0,
    }
    assert code == 0
    assert report["runs"][0]["native"] == expected
    assert report["runs"][1]["native"] == expected


def test_cli_has_no_module_level_native_import() -> None:
    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name.endswith("backends.native") for name in imported)


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


def test_serve_builds_app_and_runs_uvicorn(monkeypatch) -> None:
    called = {}

    def fake_run(app, *, host, port):
        called.update(app=app, host=host, port=port)

    monkeypatch.setattr("gavel.cli.uvicorn.run", fake_run)
    code = main(
        [
            "serve",
            "--rubric",
            str(RUBRIC),
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
        ],
        backend_factory=factory_for(checked_mock()),
    )
    assert code == 0
    assert called["host"] == "0.0.0.0"
    assert called["port"] == 9000
    assert isinstance(called["app"], FastAPI)
