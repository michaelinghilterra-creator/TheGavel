import pytest

from gavel.metrics import pearson, recall_fp, run_noise, spearman


def test_spearman_ties_checked_by_hand() -> None:
    assert spearman([1, 2, 2, 4], [1, 2, 3, 4]) == pytest.approx(0.9486832981)


def test_perfect_and_inverse_order() -> None:
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1)
    assert spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1)
    assert pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1)


def test_too_few_or_zero_variance_returns_none() -> None:
    assert spearman([1, 2], [2, 1]) is None
    assert pearson([1, 2], [2, 1]) is None
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert pearson([1, 1, 1], [1, 2, 3]) is None


def test_recall_and_false_positive_counts() -> None:
    assert recall_fp([5, 4, 2, 1], [4, 2, 4, 1], 3.5) == {
        "strong": 2,
        "kept": 1,
        "weak": 2,
        "fp": 1,
        "recall": 0.5,
    }


def test_run_noise_uses_only_shared_ids() -> None:
    assert run_noise({"a": 1, "b": 2, "left": 9}, {"a": 2, "b": 2.5, "right": 1}) == {
        "n": 2,
        "mean_abs_delta": 0.75,
        "max_abs_delta": 1,
        "moved_ge_1": 1,
    }
