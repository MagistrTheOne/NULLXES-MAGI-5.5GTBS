"""Deterministic NULLXES synthetic corpus generators (no external LLM)."""

from magi.data.synthetic.build import build_synthetic_dataset
from magi.data.synthetic.generators import GENERATOR_ID, generate_records
from magi.data.synthetic.record import SyntheticRecord, validate_pins, validate_pin_presence

__all__ = [
    "GENERATOR_ID",
    "SyntheticRecord",
    "build_synthetic_dataset",
    "generate_records",
    "validate_pin_presence",
    "validate_pins",
]
