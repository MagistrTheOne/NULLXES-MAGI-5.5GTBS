"""Build MAGI synthetic dataset artifacts (jsonl, manifests, shards)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from magi.checkpoint.manifest import file_sha256
from magi.config import load_simple_yaml
from magi.data.synthetic.generators import GENERATOR_ID, domain_histogram, generate_records
from magi.data.synthetic.pack_shards import (
    pack_token_windows,
    tokenize_records,
    write_training_shards,
)
from magi.data.synthetic.record import (
    GENERATOR_LICENSE,
    GENERATOR_VERSION,
    CorpusStats,
    SyntheticRecord,
    compute_semantic_hash,
)
from magi.data.synthetic.validators import validate_record_semantics
from magi.tokenizer import load_tokenizer


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stable_json_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256(blob).hexdigest()


def write_jsonl(path: Path, records: list[SyntheticRecord]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for rec in records:
            line = json.dumps(rec.to_dict(), ensure_ascii=False, sort_keys=True)
            handle.write(line + "\n")
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def load_records_jsonl(path: Path) -> list[SyntheticRecord]:
    records: list[SyntheticRecord] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        pins = {str(k): str(v) for k, v in dict(raw["semantic_pins"]).items()}
        domain = str(raw["domain"])
        prompt_family = str(raw["prompt_family"])
        semantic_hash = str(raw.get("semantic_hash") or "")
        if not semantic_hash:
            semantic_hash = compute_semantic_hash(
                domain=domain, prompt_family=prompt_family, pins=pins
            )
        rec = SyntheticRecord(
            id=str(raw["id"]),
            text=str(raw["text"]),
            domain=domain,
            language=str(raw["language"]),
            prompt_family=prompt_family,
            semantic_pins=pins,
            semantic_hash=semantic_hash,
            generator_id=str(raw["generator_id"]),
            generator_version=str(raw.get("generator_version", GENERATOR_VERSION)),
            license=str(raw.get("license", GENERATOR_LICENSE)),
        )
        try:
            rec.validate()
            validate_record_semantics(rec)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
        records.append(rec)
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def build_dataset_manifest(
    *,
    dataset_id: str,
    version: str,
    records: list[SyntheticRecord],
    records_sha256: str,
) -> dict[str, Any]:
    hist = domain_histogram(records)
    total = max(len(records), 1)
    lang_counts: dict[str, int] = {}
    for rec in records:
        lang_counts[rec.language] = lang_counts.get(rec.language, 0) + 1
    language_distribution = {k: v / total for k, v in sorted(lang_counts.items())}
    captured_at = _utc_now()
    content_identity = {
        "dataset_id": dataset_id,
        "version": version,
        "generator_id": GENERATOR_ID,
        "records_sha256": records_sha256,
        "document_count": len(records),
        "domain_histogram": hist,
        "language_distribution": language_distribution,
    }
    content_hash = _stable_json_hash(content_identity)
    manifest = {
        "dataset_id": dataset_id,
        "version": version,
        "source_type": "nullxes_synthetic",
        "license_status": GENERATOR_LICENSE,
        "license_reference": GENERATOR_LICENSE,
        "domains": sorted(hist.keys()),
        "language_distribution": language_distribution,
        "provenance": {
            "owner": "NULLXES",
            "capture_method": "deterministic_template_generator",
            "captured_at": captured_at,
            "source_uri": None,
        },
        "synthetic": {
            "is_synthetic": True,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "semantic_pins_required": True,
            "regime": "bring_up_not_production_pretrain",
        },
        "hashes": {
            "records_sha256": records_sha256,
            "content_sha256": content_hash,
            "manifest_sha256": "",
        },
        "document_count": len(records),
        "domain_histogram": hist,
    }
    # Identity excludes wall-clock metadata and the final manifest hash itself.
    manifest_identity = {
        k: v
        for k, v in manifest.items()
        if k not in {"hashes", "provenance"}
    }
    manifest_identity["provenance"] = {
        k: v for k, v in manifest["provenance"].items() if k != "captured_at"
    }
    manifest["hashes"]["manifest_sha256"] = _stable_json_hash(manifest_identity)
    return manifest


def build_generator_manifest(*, seed: int, n_docs: int, domain_weights: Mapping[str, float]) -> dict[str, Any]:
    return {
        "generator_id": GENERATOR_ID,
        "version": GENERATOR_VERSION,
        "seed": int(seed),
        "n_docs": int(n_docs),
        "domain_weights": dict(domain_weights),
        "external_llm_runtime": False,
        "license_status": GENERATOR_LICENSE,
        "semantic_pins_required": True,
        "dedup": "semantic_hash",
        "regime": "bring_up_not_production_pretrain",
        "prompt_families": [
            "math_arithmetic_*_v2",
            "code_io_v2",
            "logic_pattern_v2",
            "unit_measure_v2",
            "systems_component_v2",
            "bilingual_term_v2",
            "qa_intent_v2",
            "json_kv_v2",
        ],
    }


def build_contamination_report(*, dataset_id: str) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "version": GENERATOR_VERSION,
        "eval_sets_checked": [],
        "overlap_count": None,
        "status": "not_checked",
        "notes": "No external evaluation corpora were available for contamination analysis.",
    }


def build_synthetic_dataset(
    *,
    output_dir: str | Path,
    n_docs: int = 5000,
    seed: int = 42,
    seq_len: int = 128,
    tokenizer_vocab: int = 8192,
    domain_weights: Mapping[str, float] | None = None,
    dataset_id: str = "magi_synth_v0.2",
    version: str = "v0.2",
    write_shards: bool = True,
    config_path: str | Path | None = None,
    target_tokens_per_shard: int = 65_536,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weights = dict(domain_weights) if domain_weights is not None else None
    if weights is None and config_path is not None:
        cfg = load_simple_yaml(config_path)
        weights = {str(k): float(v) for k, v in dict(cfg.get("domain_weights", {})).items()} or None

    records = generate_records(n_docs=n_docs, seed=seed, domain_weights=weights)
    pin_pass = 0
    for rec in records:
        rec.validate()
        validate_record_semantics(rec)
        pin_pass += 1
    pin_pass_rate = pin_pass / max(len(records), 1)
    corpus_stats = CorpusStats.from_records(records)

    records_path = output / "records.jsonl"
    records_sha256 = write_jsonl(records_path, records)

    dataset_manifest = build_dataset_manifest(
        dataset_id=dataset_id,
        version=version,
        records=records,
        records_sha256=records_sha256,
    )
    (output / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    generator_manifest = build_generator_manifest(
        seed=seed,
        n_docs=n_docs,
        domain_weights=weights
        or {k: 1.0 for k in sorted({r.domain for r in records})},
    )
    (output / "generator_manifest.json").write_text(
        json.dumps(generator_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    contamination = build_contamination_report(dataset_id=dataset_id)
    (output / "contamination_report.json").write_text(
        json.dumps(contamination, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    root = Path(__file__).resolve().parents[3]
    tok_path = root / "tokenizer" / "artifacts" / "magi_bringup_8k_v0.1.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"bringup tokenizer missing: {tok_path}. "
            "Production synthetic build does not call T4 smoke builders."
        )
    tokenizer = load_tokenizer(tok_path)
    if tokenizer.vocab_size != tokenizer_vocab:
        raise ValueError(
            f"tokenizer vocab {tokenizer.vocab_size} != requested {tokenizer_vocab}"
        )
    tokenizer_hash = file_sha256(tok_path)
    token_ids = tokenize_records(tokenizer, records)
    raw_token_count = len(token_ids)
    shard_manifest = None
    if write_shards:
        windows = pack_token_windows(token_ids, seq_len=seq_len)
        shards_dir = output / "shards"
        config_hash = file_sha256(config_path) if config_path is not None else None
        shard_manifest = write_training_shards(
            shards_dir,
            windows=windows,
            tokenizer=tokenizer,
            tokenizer_hash=tokenizer_hash,
            document_count=len(records),
            dataset_id=dataset_id,
            config_hash=config_hash,
            target_tokens_per_shard=target_tokens_per_shard,
            raw_token_count=raw_token_count,
        )

    report = {
        "dataset_id": dataset_id,
        "version": version,
        "generator_id": GENERATOR_ID,
        "n_docs": len(records),
        "seed": seed,
        "domain_histogram": domain_histogram(records),
        "corpus_stats": corpus_stats.to_dict(),
        "pin_pass_rate": pin_pass_rate,
        "raw_token_count": raw_token_count,
        "packed_token_count": None
        if shard_manifest is None
        else shard_manifest["packed_token_count"],
        "training_token_count": None
        if shard_manifest is None
        else shard_manifest["training_token_count"],
        "seq_len": seq_len,
        "records_path": str(records_path),
        "records_sha256": records_sha256,
        "tokenizer_sha256": tokenizer_hash,
        "shard_count": None if shard_manifest is None else shard_manifest["shard_count"],
        "status": "OK",
    }
    (output / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
