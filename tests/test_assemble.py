import json

from gavel.assemble import (
    evidence_prefix,
    field_prefix,
    final_json,
    rating_prefix,
)
from gavel.readout import Readout
from gavel.rubric import Dimension, Field, Rubric


def test_first_field_prefix() -> None:
    assert field_prefix({}, "summary") == '{"fields":{"summary":"'


def test_second_field_prefix() -> None:
    assert field_prefix({"first": "done"}, "second") == (
        '{"fields":{"first":"done","second":"'
    )


def test_first_rating_with_no_fields() -> None:
    assert rating_prefix({}, {}, "clarity") == (
        '{"fields":{},"ratings":{"clarity":'
    )


def test_rating_after_two_completed_ratings() -> None:
    assert rating_prefix(
        {"summary": "Fine"}, {"clarity": 4, "tone": 3}, "accuracy"
    ) == (
        '{"fields":{"summary":"Fine"},"ratings":{"clarity":4,"tone":3,'
        '"accuracy":'
    )


def test_rating_with_evidence() -> None:
    evidence = {"clarity": "Direct", "tone": "Kind"}
    assert rating_prefix(
        {}, {"clarity": 4}, "tone", evidence
    ) == (
        '{"fields":{},"ratings":{"clarity":{"evidence":"Direct","val":4},'
        '"tone":{"evidence":"Kind","val":'
    )


def test_evidence_prefix() -> None:
    assert evidence_prefix(
        {"summary": "Fine"}, {"clarity": 4}, "tone", {"clarity": "Direct"}
    ) == (
        '{"fields":{"summary":"Fine"},"ratings":{"clarity":'
        '{"evidence":"Direct","val":4},"tone":{"evidence":"'
    )


def test_escaped_value_round_trips() -> None:
    value = 'A "quote", a \\ slash, and a\nline'
    prefix = field_prefix({"first": value}, "second")
    completed = prefix + 'done"},"ratings":{}}'
    parsed = json.loads(completed)
    assert parsed["fields"] == {"first": value, "second": "done"}


def sample_rubric(threshold: float | None = 3.5) -> Rubric:
    return Rubric(
        "demo",
        "Judge it.",
        (Dimension("clarity", "Clarity"),),
        (Field("summary"),),
        threshold,
    )


def test_final_json_passes_and_stringifies_probs() -> None:
    readout = Readout(4.0, 4, 0.5, 0.9, {0: 0.0, 4: 1.0})
    result = final_json(
        sample_rubric(),
        {"summary": "Fine"},
        {"clarity": readout},
        {"clarity": "Direct"},
        {"score": 4.0, "capped_by": []},
    )
    assert result == {
        "rubric": "demo",
        "fields": {"summary": "Fine"},
        "ratings": {
            "clarity": {
                "ev": 4.0,
                "argmax": 4,
                "sd": 0.5,
                "mass": 0.9,
                "probs": {"0": 0.0, "4": 1.0},
                "evidence": "Direct",
            }
        },
        "score": 4.0,
        "capped_by": [],
        "pass": True,
    }


def test_final_json_fail_and_none_pass_logic() -> None:
    failed = final_json(
        sample_rubric(), {}, {}, None, {"score": 3.0, "capped_by": ["clarity"]}
    )
    no_score = final_json(
        sample_rubric(), {}, {}, None, {"score": None, "capped_by": []}
    )
    no_threshold = final_json(
        sample_rubric(None), {}, {}, None, {"score": 5.0, "capped_by": []}
    )
    assert failed["pass"] is False
    assert no_score["pass"] is None
    assert no_threshold["pass"] is None
