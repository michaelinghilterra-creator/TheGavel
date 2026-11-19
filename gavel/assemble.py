"""Build deterministic assistant prefixes and final results."""

from __future__ import annotations

import json

from .readout import Readout
from .rubric import Rubric


def json_str(s: str) -> str:
    """Encode a string as compact JSON."""

    return json.dumps(s, ensure_ascii=False)


def _fields_text(fields: dict[str, str]) -> str:
    return ",".join(f"{json_str(key)}:{json_str(value)}" for key, value in fields.items())


def _rating_text(
    key: str, value: int, evidence: dict[str, str] | None
) -> str:
    if evidence is not None and key in evidence:
        return (
            f"{json_str(key)}:{{\"evidence\":{json_str(evidence[key])},"
            f"\"val\":{value}}}"
        )
    return f"{json_str(key)}:{value}"


def _ratings_start(
    fields: dict[str, str], ratings: dict[str, int], evidence: dict[str, str] | None
) -> str:
    rating_items = ",".join(
        _rating_text(key, value, evidence) for key, value in ratings.items()
    )
    return f'{{"fields":{{{_fields_text(fields)}}},"ratings":{{{rating_items}'


def field_prefix(done: dict[str, str], key: str) -> str:
    """Build a prefix ready for a free text field value."""

    separator = "," if done else ""
    return f'{{"fields":{{{_fields_text(done)}{separator}{json_str(key)}:"'


def rating_prefix(
    fields: dict[str, str],
    ratings: dict[str, int],
    key: str,
    evidence: dict[str, str] | None = None,
) -> str:
    """Build a prefix ready for a rating digit."""

    separator = "," if ratings else ""
    start = _ratings_start(fields, ratings, evidence)
    if evidence is not None and key in evidence:
        return (
            f"{start}{separator}{json_str(key)}:"
            f'{{"evidence":{json_str(evidence[key])},"val":'
        )
    return f"{start}{separator}{json_str(key)}:"


def evidence_prefix(
    fields: dict[str, str],
    ratings: dict[str, int],
    key: str,
    evidence: dict[str, str],
) -> str:
    """Build a prefix ready for evidence text."""

    separator = "," if ratings else ""
    start = _ratings_start(fields, ratings, evidence)
    return f'{start}{separator}{json_str(key)}:{{"evidence":"'


def final_json(
    rubric: Rubric,
    fields: dict[str, str],
    readouts: dict[str, Readout],
    evidence: dict[str, str] | None,
    score: dict,
) -> dict:
    """Assemble the completed evaluation as a plain dictionary."""

    ratings = {}
    for dimension in rubric.dimensions:
        if dimension.key not in readouts:
            continue
        readout = readouts[dimension.key]
        result = {
            "ev": readout.ev,
            "argmax": readout.argmax,
            "sd": readout.sd,
            "mass": readout.mass,
            "probs": {str(key): value for key, value in readout.probs.items()},
        }
        if evidence is not None and dimension.key in evidence:
            result["evidence"] = evidence[dimension.key]
        ratings[dimension.key] = result

    numeric_score = score["score"]
    passed = (
        numeric_score >= rubric.threshold
        if numeric_score is not None and rubric.threshold is not None
        else None
    )
    return {
        "rubric": rubric.name,
        "fields": dict(fields),
        "ratings": ratings,
        "score": numeric_score,
        "capped_by": list(score["capped_by"]),
        "pass": passed,
    }
