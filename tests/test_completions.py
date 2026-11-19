import ast
import asyncio
import json
import math
import re
import string
from pathlib import Path

import httpx
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

import gavel.backends.completions as completions
from gavel.backends.completions import (
    CompletionsBackend,
    chatml_renderer,
    tokenizer_renderer,
)
from gavel.backends.openai_compat import BackendError
from gavel.judge import judge
from gavel.rubric import load_rubric


FIXTURE = Path(__file__).parent / "fixtures" / "demo_rubric.yaml"


def run(coro):
    return asyncio.run(coro)


async def no_sleep(_: float) -> None:
    pass


def test_chatml_renderer_exact_text() -> None:
    assert chatml_renderer("hello", "pre") == (
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\npre"
    )


def test_tokenizer_renderer_matches_template_and_passes_kwargs(tmp_path) -> None:
    characters = list(dict.fromkeys(string.printable + "\n"))
    vocabulary = {character: index for index, character in enumerate(characters)}
    vocabulary["<unk>"] = len(vocabulary)
    raw = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    raw.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>")
    tokenizer.chat_template = (
        "{{ marker }}"
        "{% for message in messages %}"
        "<{{ message['role'] }}>{{ message['content'] }}"
        "{% if not (loop.last and continue_final_message) %}</{{ message['role'] }}>"
        "{% endif %}{% endfor %}"
    )
    tokenizer.save_pretrained(tmp_path)
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "pre"},
    ]
    expected = tokenizer.apply_chat_template(
        messages,
        continue_final_message=True,
        tokenize=False,
        marker="MARK",
    )

    actual = tokenizer_renderer(str(tmp_path), {"marker": "MARK"})("hello", "pre")

    assert actual == expected
    assert actual.endswith("pre")


def test_no_module_level_transformers_import() -> None:
    tree = ast.parse(Path(completions.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert "transformers" not in imports


def test_backend_request_shape_sorted_result_and_generate() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if body["max_tokens"] == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "logprobs": {
                                "top_logprobs": [{"b": -1.0, "a": -1.0, "c": -0.5}]
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 1},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"text": "done"}],
                "usage": {"completion_tokens": 4},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = CompletionsBackend(
        "http://mock", "tiny", render=lambda user, prefix: user + prefix, client=client
    )
    values, tokens = run(backend.next_token_logprobs("u", "p", top=7))
    text, generated = run(backend.generate("u", "p", 12, ['"']))

    assert requests[0] == (
        "/v1/completions",
        {
            "model": "tiny",
            "prompt": "up",
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 7,
        },
    )
    assert values == [
        {"token": "c", "logprob": -0.5},
        {"token": "a", "logprob": -1.0},
        {"token": "b", "logprob": -1.0},
    ]
    assert tokens == 1
    assert text == "done"
    assert generated == 4
    assert requests[1][0] == "/v1/completions"
    assert requests[1][1]["stop"] == ['"']
    assert "logprobs" not in requests[1][1]


@pytest.mark.parametrize("logprobs", [None, {}])
def test_null_or_missing_logprobs(logprobs) -> None:
    choice = {} if logprobs == {} else {"logprobs": None}
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [choice]})
    )
    backend = CompletionsBackend(
        "http://mock",
        "tiny",
        render=chatml_renderer,
        client=httpx.AsyncClient(transport=transport),
    )
    assert run(backend.next_token_logprobs("u", "p")) == ([], 0)


@pytest.mark.parametrize("status", [500, 429])
def test_retryable_status_then_success(status) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status, text="boom")
        return httpx.Response(200, json={"choices": [{"text": "ok"}]})

    backend = CompletionsBackend(
        "http://mock",
        "tiny",
        render=chatml_renderer,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    backend._sleep = no_sleep
    assert run(backend.generate("u", "p", 2, [])) == ("ok", 0)
    assert calls == 2


def test_400_raises_at_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="boom")

    backend = CompletionsBackend(
        "http://mock",
        "tiny",
        render=chatml_renderer,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(BackendError, match="HTTP 400: boom"):
        run(backend.generate("u", "p", 2, []))
    assert calls == 1


def test_judge_uses_only_completions_and_returns_ratings() -> None:
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        body = json.loads(request.content)
        if body["max_tokens"] != 1:
            return httpx.Response(
                200,
                json={
                    "choices": [{"text": 'A concise summary"'}],
                    "usage": {"completion_tokens": 3},
                },
            )
        prefix = body["prompt"].split("<|im_start|>assistant\n", 1)[1]
        match = re.search(r'"([A-Za-z][A-Za-z0-9_]*)":\s*$', prefix)
        key = match.group(1) if match else ""
        rating = {"clarity": "4", "accuracy": "5", "tone": "3"}.get(key, "3")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "logprobs": {
                            "top_logprobs": [{rating: math.log(0.8), "2": math.log(0.2)}]
                        }
                    }
                ],
                "usage": {"completion_tokens": 1},
            },
        )

    backend = CompletionsBackend(
        "http://mock",
        "tiny",
        render=chatml_renderer,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    results, _ = run(
        judge(
            backend,
            load_rubric(FIXTURE),
            [{"id": "one", "text": "A clear answer."}],
            concurrency=1,
        )
    )

    assert "error" not in results[0]
    assert set(results[0]["ratings"]) == {"clarity", "accuracy", "tone"}
    assert paths
    assert set(paths) == {"/v1/completions"}
