"""Independent domain/family semantic validators for synthetic records.

Generator is not allowed to be the sole source of truth for pin correctness.
"""

from __future__ import annotations

from typing import Callable

from magi.data.synthetic.record import SyntheticRecord

Validator = Callable[[SyntheticRecord], None]


def validate_arithmetic(rec: SyntheticRecord) -> None:
    operands = rec.semantic_pins.get("operands", "")
    if "," not in operands:
        raise ValueError(f"{rec.id}: operands pin missing for arithmetic")
    left_s, right_s = operands.split(",", 1)
    a = int(left_s)
    b = int(right_s)
    answer = int(rec.semantic_pins["answer"])
    for op, fn in (
        ("+", lambda x, y: x + y),
        ("-", lambda x, y: x - y),
        ("*", lambda x, y: x * y),
    ):
        if f"{a} {op} {b}" not in rec.text:
            continue
        expected = fn(a, b)
        if expected != answer:
            raise ValueError(
                f"{rec.id}: arithmetic answer {answer} != solver {expected} for {a}{op}{b}"
            )
        return
    raise ValueError(f"{rec.id}: arithmetic expression {a} ? {b} not found in text")


def validate_programming(rec: SyntheticRecord) -> None:
    x = int(rec.semantic_pins["x"])
    y = int(rec.semantic_pins["y"])
    fn = rec.semantic_pins["fn_name"]
    expected = int(rec.semantic_pins["return_value"])
    if fn == "add":
        got = x + y
    elif fn == "scale":
        got = x * y
    elif fn == "clamp_unit":
        got = min(1, max(0, x - y))
    elif fn == "square":
        got = x * x
    else:
        raise ValueError(f"{rec.id}: unknown fn_name={fn}")
    if got != expected:
        raise ValueError(f"{rec.id}: programming return_value mismatch ({got} != {expected})")


def validate_measurement(rec: SyntheticRecord) -> None:
    value = rec.semantic_pins.get("value")
    unit = rec.semantic_pins.get("unit")
    quantity = rec.semantic_pins.get("quantity")
    if value is None or unit is None or quantity is None:
        raise ValueError(f"{rec.id}: measurement pins incomplete")
    if f"{value} {unit}" not in rec.text:
        raise ValueError(f"{rec.id}: measurement value/unit not rendered together")


def validate_json_record(rec: SyntheticRecord) -> None:
    key = rec.semantic_pins["key"]
    value = rec.semantic_pins["value"]
    needle = f'"{key}":{value}'
    if needle not in rec.text:
        raise ValueError(f"{rec.id}: json key/value pair missing as {needle!r}")


VALIDATORS: dict[str, Validator] = {
    "math_arithmetic_direct_v2": validate_arithmetic,
    "math_arithmetic_reverse_v2": validate_arithmetic,
    "math_arithmetic_compare_v2": validate_arithmetic,
    "math_arithmetic_verify_v2": validate_arithmetic,
    "math_arithmetic_expression_v2": validate_arithmetic,
    "code_io_v2": validate_programming,
    "unit_measure_v2": validate_measurement,
    "json_kv_v2": validate_json_record,
}


def validate_record_semantics(rec: SyntheticRecord) -> None:
    validator = VALIDATORS.get(rec.prompt_family)
    if validator is None:
        return
    validator(rec)
