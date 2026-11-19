"""Client for OpenAI-compatible chat completion endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx


class BackendError(Exception):
    """An OpenAI-compatible backend request failed."""


class OpenAICompatBackend:
    """Read log probabilities and generate text through a chat endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.retries = retries
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout, headers=headers)
        self._headers = headers if client is not None else None
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    def _body(self, user: str, prefix: str) -> dict:
        return {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": prefix},
            ],
            "continue_final_message": True,
            "temperature": 0,
            "model": self.model,
        }

    async def _post(self, body: dict) -> dict:
        url = f"{self.base_url}/v1/chat/completions"
        for attempt in range(self.retries + 1):
            try:
                response = await self._client.post(url, json=body, headers=self._headers)
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

        body = self._body(user, prefix)
        body.update({"max_tokens": 1, "logprobs": True, "top_logprobs": top})
        data = await self._post(body)
        content = data["choices"][0].get("logprobs", {}).get("content") or []
        top_logprobs = content[0].get("top_logprobs", []) if content else []
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return top_logprobs, tokens

    async def generate(
        self, user: str, prefix: str, max_tokens: int, stop: list[str]
    ) -> tuple[str, int]:
        """Generate a continuation and return its completion token usage."""

        body = self._body(user, prefix)
        body.update({"max_tokens": max_tokens, "stop": stop})
        data = await self._post(body)
        content = data["choices"][0].get("message", {}).get("content") or ""
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return content, tokens

    async def aclose(self) -> None:
        """Close the internally created HTTP client."""

        if self._owns_client:
            await self._client.aclose()
