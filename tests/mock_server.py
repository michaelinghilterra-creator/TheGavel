"""Deterministic in-process fake for OpenAI-compatible requests."""

from __future__ import annotations

import json
import math
import re

import httpx


class MockLLM:
    def __init__(
        self,
        ratings: dict[str, dict[str, float]] | None = None,
        choices: dict[str, dict[str, float]] | None = None,
        choice_retries: dict[str, dict[str, float]] | None = None,
        no_choice_mass_text: str | None = None,
        field_text: str = "Solid gadget",
        fail_first: int = 0,
        status: int = 500,
    ):
        self.ratings = ratings or {}
        self.choices = choices or {}
        self.choice_retries = choice_retries or {}
        self.no_choice_mass_text = no_choice_mass_text
        self.field_text = field_text
        self.fail_first = fail_first
        self.status = status
        self.requests: list[dict] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if len(self.requests) <= self.fail_first:
            return httpx.Response(self.status, text="boom")
        if body["max_tokens"] == 1:
            prefix = body["messages"][-1]["content"]
            evidence_keys = re.findall(
                r'"([A-Za-z][A-Za-z0-9_]*)":\{"evidence":', prefix
            )
            plain_match = re.search(
                r'"([A-Za-z][A-Za-z0-9_]*)":\s*$', prefix
            )
            choice_match = re.search(
                r'"([A-Za-z][A-Za-z0-9_]*)":"\s*$', prefix
            )
            key = (
                evidence_keys[-1]
                if evidence_keys and prefix.rstrip().endswith('"val":')
                else choice_match.group(1) if choice_match is not None
                else plain_match.group(1) if plain_match is not None else ""
            )
            user = body["messages"][0]["content"]
            if key in self.choices and (
                self.no_choice_mass_text is None
                or self.no_choice_mass_text not in user
            ):
                probabilities = (
                    self.choice_retries.get(key, self.choices[key])
                    if prefix.endswith(" ")
                    else self.choices[key]
                )
            elif key in self.choices:
                probabilities = {"x": 1.0}
            else:
                probabilities = self.ratings.get(key, {"3": 0.6, "4": 0.4})
            top = [
                {"token": label, "logprob": math.log(probability)}
                for label, probability in probabilities.items()
            ]
            payload = {
                "choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}],
                "usage": {"completion_tokens": 1},
            }
        else:
            payload = {
                "choices": [{"message": {"content": self.field_text + '"'}}],
                "usage": {"completion_tokens": 5},
            }
        return httpx.Response(200, json=payload)
