"""Convert token probabilities into rating distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .rubric import Dimension


@dataclass(frozen=True)
class Readout:
    """Statistics for a rating label distribution."""

    ev: float
    argmax: int
    sd: float
    mass: float
    probs: dict[int, float]


def read_distribution(
    top_logprobs: list[dict], dim: Dimension
) -> Readout | None:
    """Read a normalized distribution from next token log probabilities."""

    raw = {value: 0.0 for value in range(dim.min, dim.max + 1)}
    valid_labels = set(dim.labels())
    for item in top_logprobs:
        token = item["token"].strip()
        if token in valid_labels:
            raw[int(token)] += math.exp(item["logprob"])

    mass = sum(raw.values())
    if mass == 0.0:
        return None

    probs = {value: probability / mass for value, probability in raw.items()}
    ev = sum(value * probability for value, probability in probs.items())
    variance = sum(
        probability * (value - ev) ** 2 for value, probability in probs.items()
    )
    argmax = max(probs, key=lambda value: (probs[value], -value))
    return Readout(ev=ev, argmax=argmax, sd=math.sqrt(variance), mass=mass, probs=probs)
