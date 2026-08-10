"""Synthetic record contract for MAGI corpus v0.1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


GENERATOR_LICENSE = "NULLXES_SYNTHETIC"


@dataclass(frozen=True)
class SyntheticRecord:
    id: str
    text: str
    domain: str
    language: str
    prompt_family: str
    semantic_pins: dict[str, str]
    generator_id: str
    license: str = GENERATOR_LICENSE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.id:
            raise ValueError("SyntheticRecord.id must be non-empty")
        if not self.text.strip():
            raise ValueError(f"{self.id}: text must be non-empty")
        if self.license != GENERATOR_LICENSE:
            raise ValueError(f"{self.id}: license must be {GENERATOR_LICENSE}")
        if not self.semantic_pins:
            raise ValueError(f"{self.id}: semantic_pins required")
        validate_pins(self.text, self.semantic_pins, record_id=self.id)


def validate_pins(text: str, pins: dict[str, str], *, record_id: str = "?") -> None:
    for key, value in pins.items():
        if value is None or str(value) == "":
            raise ValueError(f"{record_id}: pin {key!r} is empty")
        if str(value) not in text:
            raise ValueError(f"{record_id}: pin {key}={value!r} missing from text")


@dataclass
class DomainStats:
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, domain: str) -> None:
        self.counts[domain] = self.counts.get(domain, 0) + 1
