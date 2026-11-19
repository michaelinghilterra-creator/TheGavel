"""Small dependency-free metrics for benchmark reports."""

from __future__ import annotations

import math


def pearson(x: list[float], y: list[float]) -> float | None:
    """Return Pearson correlation, or None when it is not defined."""

    if len(x) != len(y):
        raise ValueError("inputs must have equal length")
    if len(x) < 3:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]
    sum_sq_x = sum(value * value for value in centered_x)
    sum_sq_y = sum(value * value for value in centered_y)
    if sum_sq_x == 0 or sum_sq_y == 0:
        return None
    return sum(a * b for a, b in zip(centered_x, centered_y)) / math.sqrt(
        sum_sq_x * sum_sq_y
    )


def _ranks(values: list[float]) -> list[float]:
    ranked = [0.0] * len(values)
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for index, _ in ordered[start:end]:
            ranked[index] = average_rank
        start = end
    return ranked


def spearman(x: list[float], y: list[float]) -> float | None:
    """Return tie-aware Spearman correlation."""

    if len(x) != len(y):
        raise ValueError("inputs must have equal length")
    if len(x) < 3:
        return None
    return pearson(_ranks(x), _ranks(y))


def recall_fp(truth: list[float], pred: list[float], threshold: float) -> dict:
    """Count recall and false positives at a score threshold."""

    if len(truth) != len(pred):
        raise ValueError("inputs must have equal length")
    strong = sum(value >= threshold for value in truth)
    kept = sum(
        actual >= threshold and predicted >= threshold
        for actual, predicted in zip(truth, pred)
    )
    weak = sum(value < threshold for value in truth)
    fp = sum(
        actual < threshold and predicted >= threshold
        for actual, predicted in zip(truth, pred)
    )
    return {
        "strong": strong,
        "kept": kept,
        "weak": weak,
        "fp": fp,
        "recall": kept / strong if strong else None,
    }


def run_noise(a: dict[str, float], b: dict[str, float]) -> dict:
    """Summarize absolute score changes for ids shared by two runs."""

    shared = a.keys() & b.keys()
    deltas = [abs(a[item_id] - b[item_id]) for item_id in shared]
    return {
        "n": len(deltas),
        "mean_abs_delta": sum(deltas) / len(deltas) if deltas else None,
        "max_abs_delta": max(deltas) if deltas else None,
        "moved_ge_1": sum(delta >= 1.0 for delta in deltas),
    }
