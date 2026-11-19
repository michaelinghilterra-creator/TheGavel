import pytest

from gavel.readout import Readout
from gavel.rubric import Dimension, Rubric
from gavel.score import item_score


def readout(ev: float) -> Readout:
    return Readout(ev=ev, argmax=round(ev), sd=0, mass=1, probs={})


def rubric(*dimensions: Dimension) -> Rubric:
    return Rubric("test", "Judge it.", dimensions)


def test_equal_weights() -> None:
    result = item_score(
        {"a": readout(2), "b": readout(4)},
        rubric(Dimension("a", "A"), Dimension("b", "B")),
    )
    assert result["score"] == pytest.approx(3)
    assert result["weights_used"] == {"a": 0.5, "b": 0.5}


def test_unequal_weights() -> None:
    result = item_score(
        {"a": readout(2), "b": readout(5)},
        rubric(Dimension("a", "A", weight=2), Dimension("b", "B", weight=1)),
    )
    assert result["score"] == pytest.approx(3)
    assert result["weights_used"] == pytest.approx({"a": 2 / 3, "b": 1 / 3})


def test_missing_dimension_renormalizes() -> None:
    result = item_score(
        {"b": readout(4)},
        rubric(Dimension("a", "A", weight=3), Dimension("b", "B")),
    )
    assert result["score"] == 4
    assert result["weights_used"] == {"b": 1}


def test_zero_weight_dimension_is_ignored() -> None:
    result = item_score(
        {"a": readout(0), "b": readout(4)},
        rubric(Dimension("a", "A", weight=0), Dimension("b", "B")),
    )
    assert result["score"] == 4
    assert result["weights_used"] == {"b": 1}


def test_no_usable_dimension_returns_none() -> None:
    result = item_score({}, rubric(Dimension("a", "A")))
    assert result == {"score": None, "capped_by": [], "weights_used": {}}


def test_cap_fires_and_is_recorded() -> None:
    result = item_score(
        {"a": readout(5), "risk": readout(1)},
        rubric(
            Dimension("a", "A"),
            Dimension("risk", "Risk", weight=0, cap_below=2, cap_to=2.5),
        ),
    )
    assert result["score"] == 2.5
    assert result["capped_by"] == ["risk"]


def test_two_caps_use_the_lower_one() -> None:
    result = item_score(
        {"a": readout(5), "first": readout(1), "second": readout(1)},
        rubric(
            Dimension("a", "A"),
            Dimension("first", "First", weight=0, cap_below=2, cap_to=3),
            Dimension("second", "Second", weight=0, cap_below=2, cap_to=2),
        ),
    )
    assert result["score"] == 2
    assert result["capped_by"] == ["first", "second"]


def test_nonstandard_scale_is_normalized() -> None:
    result = item_score(
        {"small": readout(2)}, rubric(Dimension("small", "Small", 1, 3))
    )
    assert result["score"] == 2.5
