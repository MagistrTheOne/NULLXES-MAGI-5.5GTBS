"""Deterministic template generators for MAGI synthetic corpus."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Mapping, Sequence

from magi.data.synthetic.record import SyntheticRecord

GENERATOR_ID = "magi_synth_templates_v0.1"

DEFAULT_DOMAIN_WEIGHTS: dict[str, float] = {
    "mathematics": 0.16,
    "programming": 0.16,
    "reasoning": 0.14,
    "science": 0.12,
    "systems": 0.12,
    "multilingual": 0.12,
    "dialogue": 0.10,
    "structured_data": 0.08,
}


def _record(
    *,
    index: int,
    domain: str,
    language: str,
    prompt_family: str,
    text: str,
    pins: dict[str, str],
) -> SyntheticRecord:
    digest = hashlib.sha256(f"{GENERATOR_ID}:{domain}:{index}:{text}".encode("utf-8")).hexdigest()[:12]
    rec = SyntheticRecord(
        id=f"synth_{domain}_{index:06d}_{digest}",
        text=text,
        domain=domain,
        language=language,
        prompt_family=prompt_family,
        semantic_pins=pins,
        generator_id=GENERATOR_ID,
    )
    rec.validate()
    return rec


def _gen_mathematics(rng: random.Random, index: int) -> SyntheticRecord:
    a = rng.randint(2, 97)
    b = rng.randint(2, 97)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        answer = a + b
    elif op == "-":
        answer = a - b
    else:
        answer = a * b
    operands = f"{a},{b}"
    text = (
        f"Mathematics drill {index}: compute {a} {op} {b}. "
        f"operands={operands}. The exact answer is {answer}."
    )
    return _record(
        index=index,
        domain="mathematics",
        language="en",
        prompt_family="math_arithmetic_v1",
        text=text,
        pins={"answer": str(answer), "operands": operands},
    )


def _gen_programming(rng: random.Random, index: int) -> SyntheticRecord:
    fn_name = rng.choice(["add", "scale", "clamp_unit", "square"])
    x = rng.randint(1, 20)
    y = rng.randint(1, 20)
    if fn_name == "add":
        return_value = x + y
        body = f"return {x} + {y}"
    elif fn_name == "scale":
        return_value = x * y
        body = f"return {x} * {y}"
    elif fn_name == "clamp_unit":
        return_value = min(1, max(0, x - y))
        body = f"return min(1, max(0, {x} - {y}))"
    else:
        return_value = x * x
        body = f"return {x} * {x}"
    lang = rng.choice(["python", "typescript"])
    if lang == "python":
        snippet = f"def {fn_name}():\n    {body}"
    else:
        snippet = f"function {fn_name}() {{ {body}; }}"
    text = (
        f"Programming sample {index} ({lang}):\n{snippet}\n"
        f"fn_name={fn_name}; expected return_value={return_value}."
    )
    return _record(
        index=index,
        domain="programming",
        language="en",
        prompt_family="code_snippet_v1",
        text=text,
        pins={"fn_name": fn_name, "return_value": str(return_value)},
    )


def _gen_reasoning(rng: random.Random, index: int) -> SyntheticRecord:
    subject = rng.choice(["routers", "experts", "tensors", "shards"])
    property_ = rng.choice(["finite", "normalized", "deterministic", "auditable"])
    conclusion = f"all {subject} are {property_}"
    text = (
        f"Reasoning item {index}: If every MAGI {subject[:-1] if subject.endswith('s') else subject} "
        f"is {property_}, and the batch only contains {subject}, then {conclusion}."
    )
    return _record(
        index=index,
        domain="reasoning",
        language="en",
        prompt_family="syllogism_v1",
        text=text,
        pins={"conclusion": conclusion},
    )


def _gen_science(rng: random.Random, index: int) -> SyntheticRecord:
    quantity = rng.choice(["latency", "throughput", "temperature", "pressure"])
    value = rng.randint(1, 500)
    unit = {"latency": "ms", "throughput": "tok/s", "temperature": "K", "pressure": "kPa"}[quantity]
    text = (
        f"Science note {index}: measured {quantity} equals {value} {unit}. "
        f"quantity={quantity}; unit={unit}; keep the numeric pin {value}."
    )
    return _record(
        index=index,
        domain="science",
        language="en",
        prompt_family="unit_measure_v1",
        text=text,
        pins={"quantity": quantity, "unit": unit},
    )


def _gen_systems(rng: random.Random, index: int) -> SyntheticRecord:
    component = rng.choice(
        ["MoERouter", "GQAAttention", "SequencePacker", "ShardWriter", "GradScaler"]
    )
    text = (
        f"Systems note {index}: MAGI runtime validates component={component}. "
        f"The {component} path must remain deterministic under seed control."
    )
    return _record(
        index=index,
        domain="systems",
        language="en",
        prompt_family="systems_component_v1",
        text=text,
        pins={"component": component},
    )


def _gen_multilingual(rng: random.Random, index: int) -> SyntheticRecord:
    pairs = [
        ("en", "ru", "router", "маршрутизатор"),
        ("en", "ru", "expert", "эксперт"),
        ("en", "ru", "checkpoint", "чекпоинт"),
        ("en", "ru", "tokenizer", "токенайзер"),
    ]
    src_lang, tgt_lang, en, ru = rng.choice(pairs)
    lang_pair = f"{src_lang}-{tgt_lang}"
    text = (
        f"Multilingual pair {index} lang_pair={lang_pair}: "
        f"EN '{en}' ↔ RU '{ru}'. Preserve both pins: {en} and {ru}."
    )
    return _record(
        index=index,
        domain="multilingual",
        language="ru_en",
        prompt_family="bilingual_term_v1",
        text=text,
        pins={"lang_pair": lang_pair, "en": en, "ru": ru},
    )


def _gen_dialogue(rng: random.Random, index: int) -> SyntheticRecord:
    intent = rng.choice(["clarify", "summarize", "validate", "continue"])
    text = (
        f"Dialogue {index}: User asks MAGI to {intent} the MoE routing report. "
        f"intent={intent}. MAGI replies with a short factual acknowledgement and next step."
    )
    return _record(
        index=index,
        domain="dialogue",
        language="en",
        prompt_family="qa_intent_v1",
        text=text,
        pins={"intent": intent},
    )


def _gen_structured_data(rng: random.Random, index: int) -> SyntheticRecord:
    key = rng.choice(["top_k", "n_experts", "seq_len", "batch_size"])
    value = str(rng.choice([2, 4, 8, 16, 32, 64, 128]))
    text = (
        f"Structured record {index}: "
        f'{{"dataset":"magi_synth","{key}":{value},"license":"NULLXES_SYNTHETIC"}}. '
        f"key={key}; value={value}."
    )
    return _record(
        index=index,
        domain="structured_data",
        language="en",
        prompt_family="json_kv_v1",
        text=text,
        pins={"key": key, "value": value},
    )


_GENERATORS: dict[str, Callable[[random.Random, int], SyntheticRecord]] = {
    "mathematics": _gen_mathematics,
    "programming": _gen_programming,
    "reasoning": _gen_reasoning,
    "science": _gen_science,
    "systems": _gen_systems,
    "multilingual": _gen_multilingual,
    "dialogue": _gen_dialogue,
    "structured_data": _gen_structured_data,
}


def _weighted_choice(rng: random.Random, weights: Mapping[str, float]) -> str:
    domains = list(weights.keys())
    totals = [float(weights[d]) for d in domains]
    return rng.choices(domains, weights=totals, k=1)[0]


def generate_records(
    *,
    n_docs: int,
    seed: int = 42,
    domain_weights: Mapping[str, float] | None = None,
) -> list[SyntheticRecord]:
    if n_docs < 1:
        raise ValueError("n_docs must be >= 1")
    weights = dict(domain_weights or DEFAULT_DOMAIN_WEIGHTS)
    unknown = set(weights) - set(_GENERATORS)
    if unknown:
        raise ValueError(f"unknown domains in weights: {sorted(unknown)}")
    rng = random.Random(seed)
    records: list[SyntheticRecord] = []
    seen_text: set[str] = set()
    attempts = 0
    max_attempts = n_docs * 20
    while len(records) < n_docs and attempts < max_attempts:
        attempts += 1
        domain = _weighted_choice(rng, weights)
        rec = _GENERATORS[domain](rng, len(records))
        if rec.text in seen_text:
            continue
        seen_text.add(rec.text)
        records.append(rec)
    if len(records) < n_docs:
        raise RuntimeError(f"exact-dedup exhausted pool: got {len(records)} < {n_docs}")
    return records


def domain_histogram(records: Sequence[SyntheticRecord]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for rec in records:
        hist[rec.domain] = hist.get(rec.domain, 0) + 1
    return hist
