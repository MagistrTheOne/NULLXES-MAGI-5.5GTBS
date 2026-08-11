"""Synthetic record contract for MAGI bring-up corpus (v0.2).

semantic_pins are required text anchors + ground-truth fields.
Lexical presence is NOT full semantic truth — use domain validators for that.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any, Sequence


GENERATOR_LICENSE = "NULLXES_SYNTHETIC"
GENERATOR_VERSION = "v0.2"


@dataclass(frozen=True)
class SyntheticRecord:
    id: str
    text: str
    domain: str
    language: str
    prompt_family: str
    semantic_pins: dict[str, str]
    semantic_hash: str
    generator_id: str
    generator_version: str = GENERATOR_VERSION
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
        if not self.semantic_hash:
            raise ValueError(f"{self.id}: semantic_hash required")
        expected = compute_semantic_hash(
            domain=self.domain,
            prompt_family=self.prompt_family,
            pins=self.semantic_pins,
        )
        if self.semantic_hash != expected:
            raise ValueError(
                f"{self.id}: semantic_hash mismatch "
                f"(got {self.semantic_hash}, expected {expected})"
            )
        validate_pin_presence(self.text, self.semantic_pins, record_id=self.id)


def compute_semantic_hash(
    *,
    domain: str,
    prompt_family: str,
    pins: dict[str, str],
) -> str:
    payload = {
        "domain": domain,
        "prompt_family": prompt_family,
        "pins": {str(k): str(v) for k, v in sorted(pins.items())},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return sha256(blob).hexdigest()


def validate_pin_presence(text: str, pins: dict[str, str], *, record_id: str = "?") -> None:
    """Lexical anchor check — not a semantic solver.

    Numeric pins require digit-boundary match so pin '2' does not match inside '128'.
    """
    for key, value in pins.items():
        if value is None or str(value) == "":
            raise ValueError(f"{record_id}: pin {key!r} is empty")
        token = str(value)
        if _is_numeric_token(token):
            pattern = rf"(?<!\d){re.escape(token)}(?!\d)"
            if not re.search(pattern, text):
                raise ValueError(
                    f"{record_id}: numeric pin {key}={token!r} missing as whole token in text"
                )
        elif token not in text:
            raise ValueError(f"{record_id}: pin {key}={token!r} missing from text")


def validate_pins(text: str, pins: dict[str, str], *, record_id: str = "?") -> None:
    """Alias kept for callers — pin presence only."""
    validate_pin_presence(text, pins, record_id=record_id)


def _is_numeric_token(token: str) -> bool:
    if token.isdigit():
        return True
    if token.startswith("-") and token[1:].isdigit():
        return True
    return False


@dataclass
class CorpusStats:
    domains: Counter[str] = field(default_factory=Counter)
    languages: Counter[str] = field(default_factory=Counter)
    families: Counter[str] = field(default_factory=Counter)
    generators: Counter[str] = field(default_factory=Counter)
    total_records: int = 0
    unique_semantics: int = 0
    duplicate_semantics: int = 0

    @classmethod
    def from_records(cls, records: Sequence[SyntheticRecord]) -> CorpusStats:
        stats = cls()
        seen: set[str] = set()
        for rec in records:
            stats.total_records += 1
            stats.domains[rec.domain] += 1
            stats.languages[rec.language] += 1
            stats.families[rec.prompt_family] += 1
            stats.generators[rec.generator_id] += 1
            if rec.semantic_hash in seen:
                stats.duplicate_semantics += 1
            else:
                seen.add(rec.semantic_hash)
        stats.unique_semantics = len(seen)
        return stats

    @property
    def semantic_uniqueness_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return self.unique_semantics / self.total_records

    def to_dict(self) -> dict[str, Any]:
        return {
            "domains": dict(self.domains),
            "languages": dict(self.languages),
            "families": dict(self.families),
            "generators": dict(self.generators),
            "total_records": self.total_records,
            "unique_semantics": self.unique_semantics,
            "duplicate_semantics": self.duplicate_semantics,
            "semantic_uniqueness_rate": self.semantic_uniqueness_rate,
        }


# Backward-compatible name used by older call sites.
@dataclass
class DomainStats:
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, domain: str) -> None:
        self.counts[domain] = self.counts.get(domain, 0) + 1
