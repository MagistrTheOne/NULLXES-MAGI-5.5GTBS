"""Deterministic template generators for MAGI synthetic bring-up (v0.2).

Dedup is on semantic_hash (domain + family + pins), not rendered text.
`index` is a sample slot only — never part of semantic content.
This is a bring-up generator, not production pretraining mixture.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Mapping, Sequence

from magi.data.synthetic.record import (
    GENERATOR_VERSION,
    SyntheticRecord,
    compute_semantic_hash,
)

GENERATOR_ID = "magi_synth_templates_v0.2"

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


def _sample_rng(*, seed: int, index: int, domain: str) -> random.Random:
    material = f"{GENERATOR_ID}:{seed}:{index}:{domain}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _make_record(
    *,
    domain: str,
    language: str,
    prompt_family: str,
    text: str,
    pins: dict[str, str],
) -> SyntheticRecord:
    sem = compute_semantic_hash(domain=domain, prompt_family=prompt_family, pins=pins)
    rec = SyntheticRecord(
        id=f"synth_{domain}_{sem[:16]}",
        text=text,
        domain=domain,
        language=language,
        prompt_family=prompt_family,
        semantic_pins=pins,
        semantic_hash=sem,
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
    )
    rec.validate()
    return rec


def _gen_mathematics(rng: random.Random, _index: int) -> SyntheticRecord:
    family = rng.choice(
        [
            "direct",
            "reverse",
            "compare",
            "verify",
            "expression",
        ]
    )
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
    if family == "direct":
        text = (
            f"Compute {a} {op} {b}. family={family}. operands={operands}. "
            f"The exact answer is {answer}."
        )
        prompt_family = "math_arithmetic_direct_v2"
    elif family == "reverse":
        text = (
            f"Find the result of the reverse check: known answer {answer} for "
            f"{a} {op} {b}. family={family}. operands={operands}. The exact answer is {answer}."
        )
        prompt_family = "math_arithmetic_reverse_v2"
    elif family == "compare":
        other = answer + rng.choice([-3, -1, 1, 3])
        relation = "greater" if answer > other else "less" if answer < other else "equal"
        text = (
            f"Compare values: {a} {op} {b} versus {other}. family={family}. "
            f"operands={operands}. relation={relation}. The exact answer is {answer}."
        )
        prompt_family = "math_arithmetic_compare_v2"
        pins = {
            "answer": str(answer),
            "operands": operands,
            "relation": relation,
            "family": family,
        }
        return _make_record(
            domain="mathematics",
            language="en",
            prompt_family=prompt_family,
            text=text,
            pins=pins,
        )
    elif family == "verify":
        claimed = answer if rng.random() < 0.7 else answer + rng.choice([-2, -1, 1, 2])
        verdict = "correct" if claimed == answer else "incorrect"
        text = (
            f"Verify claim: {a} {op} {b} = {claimed}. family={family}. operands={operands}. "
            f"verdict={verdict}. The exact answer is {answer}."
        )
        prompt_family = "math_arithmetic_verify_v2"
        return _make_record(
            domain="mathematics",
            language="en",
            prompt_family=prompt_family,
            text=text,
            pins={
                "answer": str(answer),
                "operands": operands,
                "verdict": verdict,
                "family": family,
            },
        )
    else:
        text = (
            f"Evaluate expression ({a} {op} {b}). family={family}. operands={operands}. "
            f"The exact answer is {answer}."
        )
        prompt_family = "math_arithmetic_expression_v2"
    return _make_record(
        domain="mathematics",
        language="en",
        prompt_family=prompt_family,
        text=text,
        pins={"answer": str(answer), "operands": operands, "family": family},
    )


def _gen_programming(rng: random.Random, _index: int) -> SyntheticRecord:
    fn_name = rng.choice(["add", "scale", "clamp_unit", "square"])
    x = rng.randint(1, 20)
    y = rng.randint(1, 20)
    lang = rng.choice(["python", "typescript"])
    if fn_name == "add":
        return_value = x + y
        py_body = f"return x + y"
        ts_body = f"return x + y;"
    elif fn_name == "scale":
        return_value = x * y
        py_body = f"return x * y"
        ts_body = f"return x * y;"
    elif fn_name == "clamp_unit":
        return_value = min(1, max(0, x - y))
        py_body = f"return min(1, max(0, x - y))"
        ts_body = f"return Math.min(1, Math.max(0, x - y));"
    else:
        return_value = x * x
        py_body = f"return x * x"
        ts_body = f"return x * x;"
        y = 0
    if lang == "python":
        snippet = f"def {fn_name}(x: int, y: int) -> int:\n    {py_body}"
    else:
        snippet = f"function {fn_name}(x: number, y: number): number {{ {ts_body} }}"
    text = (
        f"Programming sample ({lang}):\n{snippet}\n"
        f"fn_name={fn_name}; input x={x} y={y}; expected return_value={return_value}."
    )
    return _make_record(
        domain="programming",
        language="en",
        prompt_family="code_io_v2",
        text=text,
        pins={
            "fn_name": fn_name,
            "return_value": str(return_value),
            "x": str(x),
            "y": str(y),
            "lang": lang,
        },
    )


def _gen_reasoning(rng: random.Random, _index: int) -> SyntheticRecord:
    pattern = rng.choice(
        [
            "modus_ponens",
            "modus_tollens",
            "transitivity",
            "insufficient",
            "counterexample",
        ]
    )
    subject = rng.choice(["router", "expert", "tensor", "shard"])
    prop = rng.choice(["finite", "normalized", "deterministic", "auditable"])
    if pattern == "modus_ponens":
        conclusion = f"this {subject} is {prop}"
        text = (
            f"pattern={pattern}. If every MAGI {subject} is {prop}, and this object is a {subject}, "
            f"then conclusion={conclusion}."
        )
        pins = {"conclusion": conclusion, "pattern": pattern}
    elif pattern == "modus_tollens":
        conclusion = f"this object is not a {subject}"
        text = (
            f"pattern={pattern}. If every MAGI {subject} is {prop}, and this object is not {prop}, "
            f"then conclusion={conclusion}."
        )
        pins = {"conclusion": conclusion, "pattern": pattern}
    elif pattern == "transitivity":
        mid = rng.choice(["stable", "bounded", "routable"])
        conclusion = f"every {subject} is {mid}"
        text = (
            f"pattern={pattern}. If every MAGI {subject} is {prop}, and every {prop} item is {mid}, "
            f"then conclusion={conclusion}. mid={mid}."
        )
        pins = {"conclusion": conclusion, "pattern": pattern, "mid": mid}
    elif pattern == "insufficient":
        conclusion = "insufficient_information"
        text = (
            f"pattern={pattern}. Given only that some MAGI objects are {prop}, decide whether every "
            f"{subject} is {prop}. conclusion={conclusion}."
        )
        pins = {"conclusion": conclusion, "pattern": pattern}
    else:
        conclusion = f"not every {subject} is {prop}"
        text = (
            f"pattern={pattern}. Counterexample task: one {subject} fails to be {prop}. "
            f"conclusion={conclusion}."
        )
        pins = {"conclusion": conclusion, "pattern": pattern}
    return _make_record(
        domain="reasoning",
        language="en",
        prompt_family="logic_pattern_v2",
        text=text,
        pins=pins,
    )


def _gen_science(rng: random.Random, _index: int) -> SyntheticRecord:
    quantity = rng.choice(["latency", "throughput", "temperature", "pressure"])
    value = rng.randint(1, 500)
    unit = {"latency": "ms", "throughput": "tok/s", "temperature": "K", "pressure": "kPa"}[quantity]
    text = (
        f"Science note: measured {quantity} equals {value} {unit}. "
        f"quantity={quantity}; unit={unit}; value={value}."
    )
    return _make_record(
        domain="science",
        language="en",
        prompt_family="unit_measure_v2",
        text=text,
        pins={"quantity": quantity, "unit": unit, "value": str(value)},
    )


def _gen_systems(rng: random.Random, _index: int) -> SyntheticRecord:
    component = rng.choice(
        ["MoERouter", "GQAAttention", "SequencePacker", "ShardWriter", "GradScaler"]
    )
    property_ = rng.choice(["deterministic", "seeded", "auditable", "idempotent"])
    text = (
        f"Systems note: MAGI runtime validates component={component}. "
        f"The {component} path must remain {property_} under seed control. "
        f"property={property_}."
    )
    return _make_record(
        domain="systems",
        language="en",
        prompt_family="systems_component_v2",
        text=text,
        pins={"component": component, "property": property_},
    )


def _gen_multilingual(rng: random.Random, _index: int) -> SyntheticRecord:
    pairs = [
        ("en", "ru", "router", "маршрутизатор"),
        ("en", "ru", "expert", "эксперт"),
        ("en", "ru", "checkpoint", "чекпоинт"),
        ("en", "ru", "tokenizer", "токенайзер"),
        ("en", "ru", "mixture", "смесь"),
        ("en", "ru", "shard", "шард"),
    ]
    src_lang, tgt_lang, en, ru = rng.choice(pairs)
    lang_pair = f"{src_lang}-{tgt_lang}"
    text = (
        f"Multilingual pair lang_pair={lang_pair}: "
        f"EN '{en}' ↔ RU '{ru}'. Preserve both pins: {en} and {ru}."
    )
    return _make_record(
        domain="multilingual",
        language="ru_en",
        prompt_family="bilingual_term_v2",
        text=text,
        pins={"lang_pair": lang_pair, "en": en, "ru": ru},
    )


def _gen_dialogue(rng: random.Random, _index: int) -> SyntheticRecord:
    intent = rng.choice(["clarify", "summarize", "validate", "continue", "refuse", "escalate"])
    topic = rng.choice(["routing report", "loss spike", "tokenizer fertility", "license gate"])
    text = (
        f"Dialogue: User asks MAGI to {intent} the {topic}. "
        f"intent={intent}; topic={topic}. "
        f"MAGI replies with a short factual acknowledgement and next step."
    )
    return _make_record(
        domain="dialogue",
        language="en",
        prompt_family="qa_intent_v2",
        text=text,
        pins={"intent": intent, "topic": topic},
    )


def _gen_structured_data(rng: random.Random, _index: int) -> SyntheticRecord:
    key = rng.choice(["top_k", "n_experts", "seq_len", "batch_size", "world_size"])
    value = str(rng.choice([2, 4, 8, 16, 32, 64, 128, 256]))
    text = (
        f"Structured record: "
        f'{{"dataset":"magi_synth","{key}":{value},"license":"NULLXES_SYNTHETIC"}}. '
        f"key={key}; value={value}."
    )
    return _make_record(
        domain="structured_data",
        language="en",
        prompt_family="json_kv_v2",
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


def validate_domain_weights(weights: Mapping[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError("domain_weights must not be empty")
    cleaned: dict[str, float] = {}
    for domain, weight in weights.items():
        w = float(weight)
        if w < 0:
            raise ValueError(f"negative weight for {domain}: {weight}")
        if w == 0:
            continue
        cleaned[str(domain)] = w
    if not cleaned:
        raise ValueError("domain weights must have positive total mass")
    if sum(cleaned.values()) <= 0:
        raise ValueError("domain weights must have positive total mass")
    unknown = set(cleaned) - set(_GENERATORS)
    if unknown:
        raise ValueError(f"unknown domains in weights: {sorted(unknown)}")
    return cleaned


def _weighted_choice(rng: random.Random, weights: Mapping[str, float]) -> str:
    cleaned = validate_domain_weights(weights)
    domains = list(cleaned.keys())
    totals = [cleaned[d] for d in domains]
    return rng.choices(domains, weights=totals, k=1)[0]


def generate_records(
    *,
    n_docs: int,
    seed: int = 42,
    domain_weights: Mapping[str, float] | None = None,
) -> list[SyntheticRecord]:
    if n_docs < 1:
        raise ValueError("n_docs must be >= 1")
    weights = validate_domain_weights(domain_weights or DEFAULT_DOMAIN_WEIGHTS)
    domain_rng = random.Random(seed)
    records: list[SyntheticRecord] = []
    seen_semantics: set[str] = set()
    attempts = 0
    max_attempts = max(n_docs * 50, 1000)
    slot = 0
    while len(records) < n_docs and attempts < max_attempts:
        attempts += 1
        domain = _weighted_choice(domain_rng, weights)
        sample_rng = _sample_rng(seed=seed, index=slot, domain=domain)
        slot += 1
        rec = _GENERATORS[domain](sample_rng, slot)
        if rec.semantic_hash in seen_semantics:
            continue
        seen_semantics.add(rec.semantic_hash)
        records.append(rec)
    if len(records) < n_docs:
        raise RuntimeError(
            f"semantic-dedup exhausted pool: got {len(records)} < {n_docs} "
            f"(unique semantics available under current generator families)"
        )
    return records


def domain_histogram(records: Sequence[SyntheticRecord]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for rec in records:
        hist[rec.domain] = hist.get(rec.domain, 0) + 1
    return hist
