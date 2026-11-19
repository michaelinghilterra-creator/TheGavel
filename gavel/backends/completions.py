"""Client for OpenAI-compatible legacy completion endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx

from .openai_compat import BackendError


Renderer = Callable[[str, str], str]


def chatml_renderer(user: str, prefix: str) -> str:
    """Render a user message and assistant prefix as ChatML."""

    return (
        "<|im_start|>user\n"
        + user
        + "<|im_end|>\n<|im_start|>assistant\n"
        + prefix
    )


def tokenizer_renderer(
    path: str, template_kwargs: dict | None = None
) -> Renderer:
    """Load a tokenizer and return its continuing chat prompt renderer."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    kwargs = {} if template_kwargs is None else template_kwargs

    def render(user: str, prefix: str) -> str:
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": prefix},
            ],
            continue_final_message=True,
            tokenize=False,
            **kwargs,
        )

    return render


class CompletionsBackend:
    """Read log probabilities and generate text through a completion endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        render: Renderer,
        api_key: str | None = None,
        timeout: float = 120.0,
        retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.render = render
        self.retries = retries
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout, headers=headers)
        self._headers = headers if client is not None else None
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    async def _post(self, body: dict) -> dict:
        url = f"{self.base_url}/v1/completions"
        for attempt in range(self.retries + 1):
            try:
                response = await self._client.post(
                    url, json=body, headers=self._headers
                )
            except httpx.TransportError as exc:
                if attempt == self.retries:
                    raise BackendError(f"transport error: {exc}") from exc
            else:
                if response.status_code < 400:
                    return response.json()
                detail = response.text[:300]
                retryable = response.status_code == 429 or response.status_code >= 500
                if not retryable or attempt == self.retries:
                    raise BackendError(f"HTTP {response.status_code}: {detail}")
            await self._sleep(min(0.5 * (2**attempt), 2.0))
        raise AssertionError("unreachable")

    async def next_token_logprobs(
        self, user: str, prefix: str, top: int = 20
    ) -> tuple[list[dict], int]:
        """Return the next token candidates and completion token usage."""

        data = await self._post(
            {
                "model": self.model,
                "prompt": self.render(user, prefix),
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": top,
            }
        )
        logprobs = data["choices"][0].get("logprobs")
        top_logprobs = (logprobs or {}).get("top_logprobs") or []
        values = top_logprobs[0] if top_logprobs else {}
        ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
        result = [{"token": token, "logprob": logprob} for token, logprob in ordered]
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return result, tokens

    async def generate(
        self, user: str, prefix: str, max_tokens: int, stop: list[str]
    ) -> tuple[str, int]:
        """Generate a continuation and return its completion token usage."""

        data = await self._post(
            {
                "model": self.model,
                "prompt": self.render(user, prefix),
                "max_tokens": max_tokens,
                "stop": stop,
                "temperature": 0,
            }
        )
        text = data["choices"][0].get("text") or ""
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return text, tokens

    async def aclose(self) -> None:
        """Close the internally created HTTP client."""

        if self._owns_client:
            await self._client.aclose()
