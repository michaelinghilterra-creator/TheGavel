from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from gavel.backends.openai_compat import OpenAICompatBackend
from gavel.rubric import load_rubric
from gavel.server import create_app
from mock_server import MockLLM


FIXTURES = Path(__file__).parent / "fixtures"
RUBRIC = load_rubric(FIXTURES / "demo_rubric.yaml")


def checked_mock() -> MockLLM:
    labels = {str(value): 1 / 6 for value in range(6)}
    return MockLLM(ratings={"": labels})


def app_for(
    mock: MockLLM,
    *,
    rubrics=None,
    max_items: int = 256,
    label_check: bool = True,
):
    def backend_factory():
        client = httpx.AsyncClient(transport=mock.transport)
        return OpenAICompatBackend("http://mock", "tiny", client=client)

    return create_app(
        backend_factory,
        rubrics,
        max_items=max_items,
        label_check=label_check,
    )


def request_body(rubric="demo-reviews", items=None) -> dict:
    return {
        "rubric": rubric,
        "mode": "lean",
        "concurrency": 2,
        "items": items if items is not None else [{"id": "a1", "text": "Clear"}],
    }


def inline_rubric() -> dict:
    return {
        "name": "inline",
        "preamble": "Score the item.",
        "dimensions": [{"key": "quality", "label": "Quality"}],
    }


def test_healthz_lists_preloaded_rubrics() -> None:
    with TestClient(app_for(checked_mock(), rubrics={RUBRIC.name: RUBRIC})) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "rubrics": ["demo-reviews"]}


def test_judge_with_preloaded_rubric() -> None:
    with TestClient(app_for(checked_mock(), rubrics={RUBRIC.name: RUBRIC})) as client:
        response = client.post("/v1/judge", json=request_body())
    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "a1"


def test_judge_with_inline_rubric() -> None:
    with TestClient(app_for(checked_mock())) as client:
        response = client.post("/v1/judge", json=request_body(inline_rubric()))
    assert response.status_code == 200
    assert response.json()["results"][0]["ratings"]["quality"]["argmax"] == 3


def test_inline_preamble_file_is_rejected() -> None:
    rubric = inline_rubric()
    rubric.pop("preamble")
    rubric["preamble_file"] = "secret.txt"
    with TestClient(app_for(checked_mock())) as client:
        response = client.post("/v1/judge", json=request_body(rubric))
    assert response.status_code == 400
    assert "preamble_file" in response.json()["detail"]


def test_unknown_rubric_name_returns_not_found() -> None:
    with TestClient(app_for(checked_mock())) as client:
        response = client.post("/v1/judge", json=request_body("missing"))
    assert response.status_code == 404


def test_empty_items_are_rejected() -> None:
    with TestClient(app_for(checked_mock(), rubrics={RUBRIC.name: RUBRIC})) as client:
        response = client.post("/v1/judge", json=request_body(items=[]))
    assert response.status_code == 400


def test_too_many_items_are_rejected() -> None:
    items = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]
    with TestClient(
        app_for(checked_mock(), rubrics={RUBRIC.name: RUBRIC}, max_items=1)
    ) as client:
        response = client.post("/v1/judge", json=request_body(items=items))
    assert response.status_code == 400


def test_duplicate_item_ids_are_rejected() -> None:
    items = [{"id": "same", "text": "A"}, {"id": "same", "text": "B"}]
    with TestClient(app_for(checked_mock(), rubrics={RUBRIC.name: RUBRIC})) as client:
        response = client.post("/v1/judge", json=request_body(items=items))
    assert response.status_code == 400


def test_bad_mode_is_rejected() -> None:
    body = request_body()
    body["mode"] = "fast"
    with TestClient(app_for(checked_mock(), rubrics={RUBRIC.name: RUBRIC})) as client:
        response = client.post("/v1/judge", json=body)
    assert response.status_code == 400


def test_results_preserve_input_order() -> None:
    items = [{"id": str(value), "text": str(value)} for value in range(5)]
    with TestClient(app_for(checked_mock(), rubrics={RUBRIC.name: RUBRIC})) as client:
        response = client.post("/v1/judge", json=request_body(items=items))
    assert [result["id"] for result in response.json()["results"]] == [
        str(value) for value in range(5)
    ]


def test_response_includes_stats() -> None:
    with TestClient(app_for(checked_mock(), rubrics={RUBRIC.name: RUBRIC})) as client:
        response = client.post("/v1/judge", json=request_body())
    assert set(response.json()["stats"]) == {
        "calls",
        "generated_tokens",
        "wall_s",
        "low_mass",
    }


def test_label_check_runs_once_for_two_requests() -> None:
    mock = checked_mock()
    with TestClient(app_for(mock, rubrics={RUBRIC.name: RUBRIC})) as client:
        first = client.post("/v1/judge", json=request_body())
        second = client.post("/v1/judge", json=request_body())
    assert first.status_code == second.status_code == 200
    probe_requests = [
        request
        for request in mock.requests
        if request["messages"][0]["content"].startswith("Reply with")
    ]
    assert len(probe_requests) == 6
