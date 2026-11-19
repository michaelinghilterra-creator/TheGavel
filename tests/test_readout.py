import math

import pytest

from gavel.readout import read_distribution
from gavel.rubric import Dimension


DIMENSION = Dimension("rating", "Rating")


def test_single_certain_label() -> None:
    result = read_distribution([{"token": "3", "logprob": 0.0}], DIMENSION)
    assert result is not None
    assert result.ev == 3
    assert result.argmax == 3
    assert result.sd == 0
    assert result.mass == 1


def test_two_label_split() -> None:
    result = read_distribution(
        [
            {"token": "1", "logprob": math.log(0.25)},
            {"token": "3", "logprob": math.log(0.75)},
        ],
        DIMENSION,
    )
    assert result is not None
    assert result.ev == pytest.approx(2.5)
    assert result.sd == pytest.approx(math.sqrt(0.75))
    assert result.probs[1] == pytest.approx(0.25)
    assert result.probs[3] == pytest.approx(0.75)


def test_tokens_for_same_label_are_merged() -> None:
    result = read_distribution(
        [
            {"token": " 4", "logprob": math.log(0.2)},
            {"token": "4", "logprob": math.log(0.3)},
            {"token": "2", "logprob": math.log(0.5)},
        ],
        DIMENSION,
    )
    assert result is not None
    assert result.probs[4] == pytest.approx(0.5)
    assert result.argmax == 2


def test_non_labels_are_ignored_and_mass_is_raw() -> None:
    result = read_distribution(
        [
            {"token": "5", "logprob": math.log(0.4)},
            {"token": "five", "logprob": math.log(0.6)},
        ],
        DIMENSION,
    )
    assert result is not None
    assert result.mass == pytest.approx(0.4)
    assert result.probs[5] == 1


def test_no_valid_label_returns_none() -> None:
    assert read_distribution([{"token": "x", "logprob": 0.0}], DIMENSION) is None


def test_argmax_tie_uses_lower_label() -> None:
    result = read_distribution(
        [
            {"token": "4", "logprob": math.log(0.5)},
            {"token": "2", "logprob": math.log(0.5)},
        ],
        DIMENSION,
    )
    assert result is not None
    assert result.argmax == 2


def test_one_to_three_scale() -> None:
    dimension = Dimension("small", "Small", 1, 3)
    result = read_distribution(
        [
            {"token": "0", "logprob": math.log(0.4)},
            {"token": "1", "logprob": math.log(0.2)},
            {"token": "3", "logprob": math.log(0.4)},
        ],
        dimension,
    )
    assert result is not None
    assert set(result.probs) == {1, 2, 3}
    assert result.ev == pytest.approx(7 / 3)
    assert result.mass == pytest.approx(0.6)
