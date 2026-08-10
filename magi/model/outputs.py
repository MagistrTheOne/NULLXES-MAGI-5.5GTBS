"""Native MAGI model outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MagiModelOutput:
    logits: Any
    past_key_values: tuple[tuple[Any, Any], ...] | None = None
    hidden_states: tuple[Any, ...] | None = None
    attentions: tuple[Any, ...] | None = None
