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


@dataclass(frozen=True)
class ChoiceReadout:
    """Statistics for a choice label distribution."""

    value: str
    mass: float
    probs: dict[str, float]


def _prob(item: dict) -> float:
    """Probability of one top_logprobs entry. Some servers send a logprob of
    negative infinity as JSON null; that counts as zero probability."""

    logprob = item.get("logprob")
    return 0.0 if logprob is None else math.exp(logprob)


def read_distribution(
    top_logprobs: list[dict], dim: Dimension
) -> Readout | None:
    """Read a normalized distribution from next token log probabilities."""

    raw = {value: 0.0 for value in range(dim.min, dim.max + 1)}
    valid_labels = set(dim.labels())
    for item in top_logprobs:
        token = item["token"].strip()
        if token in valid_labels:
            raw[int(token)] += _prob(item)

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


def read_choice(
    top_logprobs: list[dict], choices: tuple[str, ...]
) -> ChoiceReadout | None:
    """Read a normalized distribution over choice labels."""

    raw = {choice: 0.0 for choice in choices}
    for item in top_logprobs:
        token = item["token"].strip()
        if token in raw:
            raw[token] += _prob(item)

    mass = sum(raw.values())
    if mass == 0.0:
        return None

    probs = {choice: probability / mass for choice, probability in raw.items()}
    value = max(choices, key=lambda choice: probs[choice])
    return ChoiceReadout(value=value, mass=mass, probs=probs)
