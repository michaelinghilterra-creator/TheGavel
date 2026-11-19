"""Combine dimension readouts into an item score."""

from __future__ import annotations

from .readout import Readout
from .rubric import Rubric


def item_score(readouts: dict[str, Readout], rubric: Rubric) -> dict:
    """Calculate a weighted score and apply rubric caps."""

    included = [
        dimension
        for dimension in rubric.dimensions
        if dimension.key in readouts and dimension.weight > 0
    ]
    weight_total = sum(dimension.weight for dimension in included)
    if weight_total == 0:
        return {"score": None, "capped_by": [], "weights_used": {}}

    weights_used = {
        dimension.key: dimension.weight / weight_total for dimension in included
    }
    score = 5 * sum(
        weights_used[dimension.key]
        * (readouts[dimension.key].ev - dimension.min)
        / (dimension.max - dimension.min)
        for dimension in included
    )

    capped_by: list[str] = []
    for dimension in rubric.dimensions:
        readout = readouts.get(dimension.key)
        if (
            readout is not None
            and dimension.cap_below is not None
            and readout.ev < dimension.cap_below
        ):
            cap = 5 * (dimension.cap_to - dimension.min) / (
                dimension.max - dimension.min
            )
            score = min(score, cap)
            capped_by.append(dimension.key)

    return {"score": score, "capped_by": capped_by, "weights_used": weights_used}
