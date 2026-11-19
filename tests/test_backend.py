import asyncio

import httpx
import pytest

from gavel.backends.openai_compat import BackendError, OpenAICompatBackend
from mock_server import MockLLM


async def no_sleep(_: float) -> None:
    pass


def run(coro):
    return asyncio.run(coro)


def backend_for(mock: MockLLM, **kwargs) -> OpenAICompatBackend:
    client = httpx.AsyncClient(transport=mock.transport)
    backend = OpenAICompatBackend("http://mock", "tiny", client=client, **kwargs)
    backend._sleep = no_sleep
    return backend


def test_logprob_body_shape_and_result() -> None:
    mock = MockLLM()
    backend = backend_for(mock)
    values, tokens = run(backend.next_token_logprobs("user", "prefix", top=7))
    body = mock.requests[0]
    assert body["continue_final_message"] is True
    assert body["temperature"] == 0
    assert body["model"] == "tiny"
    assert body["messages"] == [
        {"role": "user", "content": "user"},
        {"role": "assistant", "content": "prefix"},
    ]
    assert body["max_tokens"] == 1
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 7
    assert [value["token"] for value in values] == ["3", "4"]
    assert tokens == 1


def test_generate_body_and_result() -> None:
    mock = MockLLM(field_text="Fine")
    backend = backend_for(mock)
    value, tokens = run(backend.generate("u", "p", 12, ['"']))
    assert value == 'Fine"'
    assert tokens == 5
    assert mock.requests[0]["max_tokens"] == 12
    assert mock.requests[0]["stop"] == ['"']


@pytest.mark.parametrize("api_key,present", [("secret", True), (None, False)])
def test_auth_header_only_when_api_key_is_set(api_key, present) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {},
            },
        )

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = OpenAICompatBackend(
            "http://mock", "tiny", api_key=api_key, client=client
        )
        await backend.generate("u", "p", 2, [])
        await client.aclose()

    run(scenario())
    assert (seen[0].get("Authorization") == "Bearer secret") is present


@pytest.mark.parametrize("status", [500, 429])
def test_retryable_status_then_success(status: int) -> None:
    mock = MockLLM(fail_first=1, status=status)
    backend = backend_for(mock)
    value, _ = run(backend.generate("u", "p", 2, []))
    assert value == 'Solid gadget"'
    assert len(mock.requests) == 2


def test_400_raises_at_once() -> None:
    mock = MockLLM(fail_first=5, status=400)
    backend = backend_for(mock)
    with pytest.raises(BackendError, match="HTTP 400: boom"):
        run(backend.generate("u", "p", 2, []))
    assert len(mock.requests) == 1


def test_retries_exhausted_raises() -> None:
    mock = MockLLM(fail_first=5)
    backend = backend_for(mock, retries=2)
    with pytest.raises(BackendError, match="HTTP 500"):
        run(backend.generate("u", "p", 2, []))
    assert len(mock.requests) == 3


def test_no_logprob_content_and_missing_usage() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"logprobs": {}}]})
    )
    client = httpx.AsyncClient(transport=transport)
    backend = OpenAICompatBackend("http://mock", "tiny", client=client)
    assert run(backend.next_token_logprobs("u", "p")) == ([], 0)
