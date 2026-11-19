"""Asynchronous rubric judging loop."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from .assemble import evidence_prefix, field_prefix, final_json, rating_prefix
from .backends.openai_compat import BackendError
from .readout import ChoiceReadout, Readout, read_choice, read_distribution
from .rubric import Field, Rubric
from .score import item_score


@dataclass
class JudgeStats:
    calls: int = 0
    generated_tokens: int = 0
    wall_s: float = 0.0
    low_mass: int = 0


def format_section(rubric: Rubric, mode: str = "lean") -> str:
    """Describe the exact output layout and rating scale for a rubric."""

    before_fields = tuple(field for field in rubric.fields if not field.after)
    after_fields = tuple(field for field in rubric.fields if field.after)

    def field_layout(field: Field) -> str:
        value = (
            "..."
            if field.type == "text"
            else f"<one of {', '.join(field.choices)}>"
        )
        return f'"{field.key}":"{value}"'

    fields = ",".join(field_layout(field) for field in before_fields)
    if mode == "explain":
        ratings = ",".join(
            f'"{dimension.key}":{{"evidence":"<one short sentence>",'
            f'"val":<integer {dimension.min} to {dimension.max}>}}'
            for dimension in rubric.dimensions
        )
    else:
        ratings = ",".join(
            f'"{dimension.key}":<integer {dimension.min} to {dimension.max}>'
            for dimension in rubric.dimensions
        )
    lines = [
        "# Output format",
        "",
        "Your reply is one JSON object written by the scoring engine:",
        f'{{"fields":{{{fields}}},"ratings":{{{ratings}}}'
        + (
            ',"after":{'
            + ",".join(field_layout(field) for field in after_fields)
            + "}"
            if after_fields
            else ""
        )
        + "}",
        "",
        "Each rating is a single digit. Meaning of each rating:",
        *(
            f"- {dimension.key} ({dimension.label}): "
            f"{dimension.min} = worst, {dimension.max} = best"
            for dimension in rubric.dimensions
        ),
    ]
    if rubric.fields:
        lines.extend(["", "Fields:"])
        for field in rubric.fields:
            description = (
                f"free text, at most {field.max_tokens} tokens"
                if field.type == "text"
                else f"one of {', '.join(field.choices)}"
            )
            if field.after:
                description += " (written after the ratings)"
            if field.meaning:
                description += f". {field.meaning}"
            lines.append(f"- {field.key}: {description}")
    if any(field.type == "text" for field in rubric.fields):
        lines.extend(["", "Write each text field as plain prose with no quote characters."])
    return "\n".join(lines)


def build_user_message(rubric: Rubric, text: str, mode: str = "lean") -> str:
    """Build the cache-friendly user message for an item."""

    return (
        rubric.preamble
        + "\n\n"
        + format_section(rubric, mode)
        + "\n\n---\n\n# Item to score\n\n"
        + text
    )


def _clean_generated(value: str) -> str:
    value = value.strip()
    if value.endswith('"'):
        value = value[:-1]
    return value.strip()


async def _judge_item(
    backend, rubric: Rubric, item_id: str, text: str, mode: str
) -> tuple[dict, JudgeStats]:
    stats = JudgeStats()
    fields: dict[str, str] = {}
    ratings: dict[str, int] = {}
    readouts: dict[str, Readout] = {}
    choice_readouts: dict[str, ChoiceReadout] = {}
    evidence: dict[str, str] | None = {} if mode == "explain" else None
    user = build_user_message(rubric, text, mode)

    async def generate(prefix: str, max_tokens: int) -> str:
        stats.calls += 1
        value, tokens = await backend.generate(user, prefix, max_tokens, stop=['"'])
        stats.generated_tokens += tokens
        return _clean_generated(value)

    async def logprobs(prefix: str) -> list[dict]:
        stats.calls += 1
        values, tokens = await backend.next_token_logprobs(user, prefix)
        stats.generated_tokens += tokens
        return values

    try:
        if mode not in {"lean", "explain"}:
            raise ValueError(f"unknown judge mode: {mode}")
        before_fields = tuple(field for field in rubric.fields if not field.after)
        after_fields = tuple(field for field in rubric.fields if field.after)
        before_done: dict[str, str] = {}
        after_done: dict[str, str] = {}

        async def produce_field(field: Field, prefix: str) -> str | None:
            if field.type == "text":
                return await generate(prefix, field.max_tokens)
            first = read_choice(await logprobs(prefix), field.choices)
            kept = first
            if first is None or first.mass < 0.5:
                second = read_choice(await logprobs(prefix + " "), field.choices)
                first_mass = first.mass if first is not None else 0.0
                second_mass = second.mass if second is not None else 0.0
                if second_mass > first_mass:
                    kept = second
            if kept is None:
                return None
            if kept.mass < 0.5:
                stats.low_mass += 1
            choice_readouts[field.key] = kept
            return kept.value

        for field in before_fields:
            value = await produce_field(field, field_prefix(before_done, field.key))
            if value is None:
                return {"id": item_id, "error": f"no choice mass for {field.key}"}, stats
            before_done[field.key] = value
            fields[field.key] = value

        for dimension in rubric.dimensions:
            if evidence is not None:
                evidence[dimension.key] = await generate(
                    evidence_prefix(fields, ratings, dimension.key, evidence), 40
                )
            prefix = rating_prefix(fields, ratings, dimension.key, evidence)
            first = read_distribution(await logprobs(prefix), dimension)
            kept = first
            if first is None or first.mass < 0.5:
                second = read_distribution(await logprobs(prefix + " "), dimension)
                first_mass = first.mass if first is not None else 0.0
                second_mass = second.mass if second is not None else 0.0
                if second_mass > first_mass:
                    kept = second
            if kept is None:
                return {"id": item_id, "error": f"no rating mass for {dimension.key}"}, stats
            if kept.mass < 0.5:
                stats.low_mass += 1
            readouts[dimension.key] = kept
            ratings[dimension.key] = kept.argmax

        for field in after_fields:
            prefix = field_prefix(
                before_done,
                field.key,
                ratings=ratings,
                evidence=evidence,
                after_done=after_done,
            )
            value = await produce_field(field, prefix)
            if value is None:
                return {"id": item_id, "error": f"no choice mass for {field.key}"}, stats
            after_done[field.key] = value
            fields[field.key] = value
    except BackendError as exc:
        return {"id": item_id, "error": str(exc)}, stats

    score = item_score(readouts, rubric)
    return {
        "id": item_id,
        **final_json(
            rubric, fields, readouts, evidence, score, choices=choice_readouts
        ),
    }, stats


async def judge_item(
    backend, rubric: Rubric, item_id: str, text: str, mode: str = "lean"
) -> dict:
    """Judge one item, returning an error result for backend failures."""

    result, _ = await _judge_item(backend, rubric, item_id, text, mode)
    return result


async def judge(
    backend,
    rubric: Rubric,
    items: list[dict],
    mode: str = "lean",
    concurrency: int = 8,
) -> tuple[list[dict], JudgeStats]:
    """Judge items concurrently while preserving their input order."""

    semaphore = asyncio.Semaphore(concurrency)
    started = time.perf_counter()

    async def run(item: dict) -> tuple[dict, JudgeStats]:
        async with semaphore:
            return await _judge_item(backend, rubric, item["id"], item["text"], mode)

    completed = await asyncio.gather(*(run(item) for item in items))
    stats = JudgeStats(wall_s=time.perf_counter() - started)
    results = []
    for result, item_stats in completed:
        results.append(result)
        stats.calls += item_stats.calls
        stats.generated_tokens += item_stats.generated_tokens
        stats.low_mass += item_stats.low_mass
    return results, stats


async def check_single_token_labels(backend, rubric: Rubric) -> None:
    """Heuristically probe whether each distinct label is a single top token."""

    labels = dict.fromkeys(
        [label for dimension in rubric.dimensions for label in dimension.labels()]
        + [
            label
            for field in rubric.fields
            if field.type == "choice"
            for label in field.choices
        ]
    )
    for label in labels:
        top_logprobs, _ = await backend.next_token_logprobs(
            user="Reply with the single character " + label, prefix=""
        )
        if not any(item.get("token", "").strip() == label for item in top_logprobs):
            raise ValueError(f"label {label!r} is not a single token")
