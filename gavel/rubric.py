"""Rubric definitions and YAML loading."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_KEY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


@dataclass(frozen=True)
class Dimension:
    """A numeric rubric dimension."""

    key: str
    label: str
    min: int = 0
    max: int = 5
    weight: float = 1.0
    cap_below: float | None = None
    cap_to: float | None = None

    def labels(self) -> list[str]:
        """Return the allowed single digit labels."""

        return [str(value) for value in range(self.min, self.max + 1)]


@dataclass(frozen=True)
class Field:
    """A field produced during evaluation."""

    key: str
    max_tokens: int = 60
    type: str = "text"
    choices: tuple[str, ...] = ()
    meaning: str | None = None
    after: bool = False


@dataclass(frozen=True)
class Rubric:
    """A complete rubric definition."""

    name: str
    preamble: str
    dimensions: tuple[Dimension, ...]
    fields: tuple[Field, ...] = ()
    threshold: float | None = None


def _validate_key(key: str) -> None:
    if not isinstance(key, str) or _KEY_PATTERN.fullmatch(key) is None:
        raise ValueError(f"invalid key: {key!r}")


def _dimension_from_dict(item: dict[str, Any]) -> Dimension:
    key = item.get("key")
    _validate_key(key)
    minimum = item.get("min", 0)
    maximum = item.get("max", 5)
    weight = item.get("weight", 1.0)
    cap_below = item.get("cap_below")
    cap_to = item.get("cap_to")

    if minimum >= maximum:
        raise ValueError(f"dimension {key!r} must have min less than max")
    if minimum < 0:
        raise ValueError(f"dimension {key!r} range min cannot be negative")
    if maximum > 9:
        raise ValueError(f"dimension {key!r} range max cannot exceed 9")
    if maximum - minimum > 9:
        raise ValueError(f"dimension {key!r} range cannot exceed 9")
    if weight < 0:
        raise ValueError(f"dimension {key!r} weight cannot be negative")
    if (cap_below is None) != (cap_to is None):
        raise ValueError(f"dimension {key!r} requires both cap values")
    if cap_below is not None:
        if not minimum <= cap_below <= maximum:
            raise ValueError(f"dimension {key!r} cap_below is outside its range")
        if not minimum <= cap_to <= maximum:
            raise ValueError(f"dimension {key!r} cap_to is outside its range")

    return Dimension(
        key=key,
        label=item.get("label", key),
        min=minimum,
        max=maximum,
        weight=weight,
        cap_below=cap_below,
        cap_to=cap_to,
    )


def _field_from_dict(item: dict[str, Any]) -> Field:
    key = item.get("key")
    _validate_key(key)
    field_type = item.get("type", "text")
    choices = item.get("choices", [])
    meaning = item.get("meaning")
    after = item.get("after", False)

    if not isinstance(field_type, str) or field_type not in {"text", "choice"}:
        raise ValueError(f"field {key!r} has invalid type")
    if not isinstance(choices, list):
        raise ValueError(f"field {key!r} has invalid choices")
    if field_type == "choice":
        if (
            not choices
            or any(
                not isinstance(choice, str)
                or not choice
                or any(character.isspace() for character in choice)
                or '"' in choice
                for choice in choices
            )
            or len(set(choices)) != len(choices)
        ):
            raise ValueError(f"field {key!r} has invalid choices")
    elif choices:
        raise ValueError(f"field {key!r} is text and must not have choices")
    if not isinstance(after, bool):
        raise ValueError(f"field {key!r} after must be a bool")
    if meaning is not None and not isinstance(meaning, str):
        raise ValueError(f"field {key!r} meaning must be a string")

    return Field(
        key=key,
        max_tokens=item.get("max_tokens", 60),
        type=field_type,
        choices=tuple(choices),
        meaning=meaning,
        after=after,
    )


def rubric_from_dict(
    d: dict[str, Any], base_dir: str | os.PathLike[str] = "."
) -> Rubric:
    """Build and validate a rubric from parsed YAML data."""

    has_preamble = "preamble" in d
    has_preamble_file = "preamble_file" in d
    if has_preamble == has_preamble_file:
        raise ValueError("exactly one of preamble or preamble_file is required")

    if has_preamble:
        preamble = d["preamble"]
    else:
        preamble_path = Path(base_dir) / d["preamble_file"]
        preamble = preamble_path.read_text(encoding="utf-8")

    dimension_items = d.get("dimensions", [])
    if not dimension_items:
        raise ValueError("rubric requires at least one dimension")

    dimensions = tuple(_dimension_from_dict(item) for item in dimension_items)
    fields = tuple(_field_from_dict(item) for item in d.get("fields", []))

    seen: set[str] = set()
    for entry in (*dimensions, *fields):
        _validate_key(entry.key)
        if entry.key in seen:
            raise ValueError(f"duplicate key: {entry.key!r}")
        seen.add(entry.key)

    return Rubric(
        name=d["name"],
        preamble=preamble,
        dimensions=dimensions,
        fields=fields,
        threshold=d.get("threshold"),
    )


def load_rubric(path: str | os.PathLike[str]) -> Rubric:
    """Load and validate a rubric from a YAML file."""

    rubric_path = Path(path)
    with rubric_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("rubric YAML must contain a mapping")
    return rubric_from_dict(data, rubric_path.parent)
