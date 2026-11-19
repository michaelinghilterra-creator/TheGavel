import asyncio
import json
import math
from pathlib import Path

import httpx
import pytest

from gavel.backends.openai_compat import OpenAICompatBackend
from gavel.judge import (
    build_user_message,
    check_single_token_labels,
    judge,
    judge_item,
)
from gavel.rubric import Dimension, Rubric, load_rubric
from mock_server import MockLLM


FIXTURE = Path(__file__).parent / "fixtures" / "demo_rubric.yaml"


def run(coro):
    return asyncio.run(coro)


def backend_for(mock: MockLLM) -> OpenAICompatBackend:
    client = httpx.AsyncClient(transport=mock.transport)
    return OpenAICompatBackend("http://mock", "tiny", client=client)


def test_lean_expected_values_score_order_and_stats() -> None:
    rubric = load_rubric(FIXTURE)
    ratings = {
        "clarity": {"4": 0.75, "2": 0.25},
        "accuracy": {"5": 0.5, "3": 0.5},
        "tone": {"2": 0.2, "3": 0.8},
    }
    mock = MockLLM(ratings=ratings)
    results, stats = run(
        judge(backend_for(mock), rubric, [{"id": "a", "text": "Review"}])
    )
    result = results[0]
    assert result["ratings"]["clarity"]["ev"] == pytest.approx(3.5)
    assert result["ratings"]["accuracy"]["ev"] == pytest.approx(4)
    assert result["ratings"]["tone"]["ev"] == pytest.approx(2.8)
    assert result["score"] == pytest.approx((2 * 3.5 + 4 + 2.8) / 4)
    prefixes = [request["messages"][1]["content"] for request in mock.requests]
    assert '"summary":"' in prefixes[0]
    assert prefixes[1].endswith('"clarity":')
    assert prefixes[2].endswith('"accuracy":')
    assert prefixes[3].endswith('"tone":')
    assert stats.calls == 4
    assert stats.generated_tokens == 8
    assert stats.low_mass == 0
    assert stats.wall_s >= 0


def test_user_message_is_identical_for_equal_text_and_preamble_first() -> None:
    rubric = load_rubric(FIXTURE)
    mock = MockLLM()
    run(
        judge(
            backend_for(mock),
            rubric,
            [{"id": "a", "text": "Same"}, {"id": "b", "text": "Same"}],
        )
    )
    users = [request["messages"][0]["content"] for request in mock.requests]
    assert len(set(users)) == 1
    assert users[0].startswith(rubric.preamble)
    assert users[0] == build_user_message(rubric, "Same")


class SpaceRetryBackend:
    def __init__(self, never_mass: str | None = None):
        self.prefixes = []
        self.never_mass = never_mass

    async def generate(self, user, prefix, max_tokens, stop):
        return 'ok"', 5

    async def next_token_logprobs(self, user, prefix, top=20):
        self.prefixes.append(prefix)
        if self.never_mass and self.never_mass in user:
            return [{"token": "x", "logprob": 0.0}], 1
        if not prefix.endswith(" "):
            return [
                {"token": "3", "logprob": math.log(0.2)},
                {"token": "x", "logprob": math.log(0.8)},
            ], 1
        return [{"token": "4", "logprob": math.log(0.9)}], 1


def one_dimension() -> Rubric:
    return Rubric("one", "Preamble", (Dimension("quality", "Quality"),))


def test_space_retry_keeps_larger_mass() -> None:
    backend = SpaceRetryBackend()
    result, stats = run(
        judge(backend, one_dimension(), [{"id": "a", "text": "fine"}])
    )
    assert len(backend.prefixes) == 2
    assert backend.prefixes[1] == backend.prefixes[0] + " "
    assert result[0]["ratings"]["quality"]["argmax"] == 4
    assert stats.calls == 2
    assert stats.generated_tokens == 2


def test_bad_item_errors_and_other_item_succeeds() -> None:
    backend = SpaceRetryBackend(never_mass="bad")
    results, _ = run(
        judge(
            backend,
            one_dimension(),
            [{"id": "bad", "text": "bad"}, {"id": "good", "text": "good"}],
        )
    )
    assert results[0] == {"id": "bad", "error": "no rating mass for quality"}
    assert results[1]["id"] == "good"
    assert "error" not in results[1]


class DelayedBackend:
    async def next_token_logprobs(self, user, prefix, top=20):
        number = int(user.rsplit(" ", 1)[-1])
        await asyncio.sleep((4 - number) * 0.001)
        return [{"token": "3", "logprob": 0.0}], 1


def test_concurrency_preserves_five_item_input_order() -> None:
    items = [{"id": str(i), "text": f"item {i}"} for i in range(5)]
    results, stats = run(judge(DelayedBackend(), one_dimension(), items, concurrency=3))
    assert [result["id"] for result in results] == [str(i) for i in range(5)]
    assert stats.calls == 5
    assert stats.generated_tokens == 5


def test_explain_generates_evidence_before_each_rating() -> None:
    rubric = Rubric(
        "two",
        "Preamble",
        (Dimension("first", "First"), Dimension("second", "Second")),
    )
    mock = MockLLM()
    result = run(judge_item(backend_for(mock), rubric, "a", "text", mode="explain"))
    prefixes = [request["messages"][1]["content"] for request in mock.requests]
    assert '"first":{"evidence":"' in prefixes[0]
    assert prefixes[1].endswith('"val":')
    assert '"second":{"evidence":"' in prefixes[2]
    assert prefixes[3].endswith('"val":')
    assert result["ratings"]["first"]["evidence"] == "Solid gadget"


def test_quoted_field_value_round_trips_through_json() -> None:
    rubric = load_rubric(FIXTURE)
    mock = MockLLM(field_text='A "quoted" value')
    result = run(judge_item(backend_for(mock), rubric, "a", "text"))
    assert json.loads(json.dumps(result))["fields"]["summary"] == 'A "quoted" value'


class LabelBackend:
    def __init__(self, valid: bool = True):
        self.valid = valid

    async def next_token_logprobs(self, user, prefix, top=20):
        label = user[-1]
        token = label if self.valid else label + "x"
        return [{"token": token, "logprob": 0.0}], 1


def test_single_token_label_check_passes_and_fails() -> None:
    rubric = Rubric("labels", "P", (Dimension("small", "Small", 3, 4),))
    run(check_single_token_labels(backend_for(MockLLM()), rubric))
    with pytest.raises(ValueError, match="label '3'"):
        run(check_single_token_labels(LabelBackend(valid=False), rubric))
