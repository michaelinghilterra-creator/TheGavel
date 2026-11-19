from pathlib import Path

import pytest

from gavel.rubric import Dimension, load_rubric, rubric_from_dict


FIXTURES = Path(__file__).parent / "fixtures"


def base_data() -> dict:
    return {"name": "test", "preamble": "Judge it.", "dimensions": [{"key": "quality"}]}


def test_loads_demo_fixture() -> None:
    rubric = load_rubric(FIXTURES / "demo_rubric.yaml")
    assert rubric.name == "demo-reviews"
    assert "Zorblax Widgetry" in rubric.preamble
    assert rubric.threshold == 3.5
    assert rubric.fields[0].key == "summary"
    assert [dimension.key for dimension in rubric.dimensions] == [
        "clarity",
        "accuracy",
        "tone",
    ]
    assert rubric.dimensions[0].weight == 2
    assert rubric.dimensions[1].cap_to == 2.5


def test_labels() -> None:
    assert Dimension("stars", "Stars", 1, 3).labels() == ["1", "2", "3"]


def test_no_dimensions_is_invalid() -> None:
    data = base_data()
    data["dimensions"] = []
    with pytest.raises(ValueError, match="dimension"):
        rubric_from_dict(data)


@pytest.mark.parametrize(
    "changes",
    [
        {"preamble_file": "unused.md"},
        {"preamble": None, "preamble_file": None},
    ],
)
def test_exactly_one_preamble_source_is_required(changes: dict) -> None:
    data = base_data()
    data.update(changes)
    with pytest.raises(ValueError, match="exactly one"):
        rubric_from_dict(data)


@pytest.mark.parametrize(
    "fields,dimensions",
    [
        ([{"key": "same"}], [{"key": "same"}]),
        ([], [{"key": "same"}, {"key": "same"}]),
        ([{"key": "same"}, {"key": "same"}], [{"key": "other"}]),
    ],
)
def test_duplicate_keys_are_invalid(fields: list, dimensions: list) -> None:
    data = base_data()
    data["fields"] = fields
    data["dimensions"] = dimensions
    with pytest.raises(ValueError, match="duplicate"):
        rubric_from_dict(data)


@pytest.mark.parametrize("key", ["", "2bad", "bad-key", "has space"])
def test_invalid_identifier_is_rejected(key: str) -> None:
    data = base_data()
    data["dimensions"] = [{"key": key}]
    with pytest.raises(ValueError, match="invalid key"):
        rubric_from_dict(data)


@pytest.mark.parametrize("minimum,maximum", [(2, 2), (3, 2)])
def test_min_must_be_less_than_max(minimum: int, maximum: int) -> None:
    data = base_data()
    data["dimensions"] = [{"key": "quality", "min": minimum, "max": maximum}]
    with pytest.raises(ValueError, match="min less than max"):
        rubric_from_dict(data)


def test_range_cannot_exceed_nine() -> None:
    data = base_data()
    data["dimensions"] = [{"key": "quality", "min": 0, "max": 10}]
    with pytest.raises(ValueError, match="range"):
        rubric_from_dict(data)


def test_weight_cannot_be_negative() -> None:
    data = base_data()
    data["dimensions"] = [{"key": "quality", "weight": -0.1}]
    with pytest.raises(ValueError, match="weight"):
        rubric_from_dict(data)


@pytest.mark.parametrize(
    "cap_values", [{"cap_below": 2}, {"cap_to": 2}]
)
def test_cap_values_must_be_paired(cap_values: dict) -> None:
    data = base_data()
    data["dimensions"] = [{"key": "quality", **cap_values}]
    with pytest.raises(ValueError, match="both cap"):
        rubric_from_dict(data)


@pytest.mark.parametrize(
    "cap_below,cap_to", [(-1, 2), (6, 2), (2, -1), (2, 6)]
)
def test_cap_values_must_be_in_range(cap_below: float, cap_to: float) -> None:
    data = base_data()
    data["dimensions"] = [
        {"key": "quality", "cap_below": cap_below, "cap_to": cap_to}
    ]
    with pytest.raises(ValueError, match="outside"):
        rubric_from_dict(data)
