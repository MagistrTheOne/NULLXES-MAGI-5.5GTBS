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
from magi.data.synthetic.pack_shards import pack_token_windows, tokenize_records, write_training_shard
from magi.data.synthetic.record import GENERATOR_LICENSE, SyntheticRecord
from magi.tokenizer import build_t4_smoke_tokenizer


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stable_json_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
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
        rec = SyntheticRecord(
            id=str(raw["id"]),
            text=str(raw["text"]),
            domain=str(raw["domain"]),
            language=str(raw["language"]),
            prompt_family=str(raw["prompt_family"]),
            semantic_pins={str(k): str(v) for k, v in dict(raw["semantic_pins"]).items()},
            generator_id=str(raw["generator_id"]),
            license=str(raw.get("license", GENERATOR_LICENSE)),
        )
        try:
            rec.validate()
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
    records_hash: str,
) -> dict[str, Any]:
    hist = domain_histogram(records)
    total = max(len(records), 1)
    lang_counts: dict[str, int] = {}
    for rec in records:
        lang_counts[rec.language] = lang_counts.get(rec.language, 0) + 1
    language_distribution = {k: v / total for k, v in sorted(lang_counts.items())}
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
            "captured_at": _utc_now(),
            "source_uri": None,
        },
        "synthetic": {
            "is_synthetic": True,
            "generator_id": GENERATOR_ID,
            "semantic_pins_required": True,
        },
        "hashes": {
            "raw_manifest_hash": records_hash,
            "normalized_hash": records_hash,
        },
    }
    # Stable hash over schema fields excluding hashes themselves.
    manifest["hashes"]["raw_manifest_hash"] = _stable_json_hash(
        {k: v for k, v in manifest.items() if k != "hashes"}
    )
    # Keep histogram only for callers; strip before schema-strict write if needed.
    manifest["_domain_histogram"] = hist
    manifest["_document_count"] = len(records)
    return manifest


def build_generator_manifest(*, seed: int, n_docs: int, domain_weights: Mapping[str, float]) -> dict[str, Any]:
    return {
        "generator_id": GENERATOR_ID,
        "version": "v0.1",
        "seed": int(seed),
        "n_docs": int(n_docs),
        "domain_weights": dict(domain_weights),
        "external_llm_runtime": False,
        "license_status": GENERATOR_LICENSE,
        "semantic_pins_required": True,
        "prompt_families": [
            "math_arithmetic_v1",
            "code_snippet_v1",
            "syllogism_v1",
            "unit_measure_v1",
            "systems_component_v1",
            "bilingual_term_v1",
            "qa_intent_v1",
            "json_kv_v1",
        ],
    }


def build_contamination_report(*, dataset_id: str) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "version": "v0.1",
        "eval_sets_checked": [],
        "overlap_count": 0,
        "status": "clean_synthetic_v0.1",
        "notes": "No external eval corpora linked in v0.1 synthetic bring-up.",
    }


def build_synthetic_dataset(
    *,
    output_dir: str | Path,
    n_docs: int = 5000,
    seed: int = 42,
    seq_len: int = 128,
    tokenizer_vocab: int = 8192,
    domain_weights: Mapping[str, float] | None = None,
    dataset_id: str = "magi_synth_v0.1",
    version: str = "v0.1",
    write_shards: bool = True,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weights = dict(domain_weights) if domain_weights is not None else None
    if weights is None and config_path is not None:
        cfg = load_simple_yaml(config_path)
        weights = {str(k): float(v) for k, v in dict(cfg.get("domain_weights", {})).items()} or None

    records = generate_records(n_docs=n_docs, seed=seed, domain_weights=weights)
    for rec in records:
        rec.validate()

    records_path = output / "records.jsonl"
    records_hash = write_jsonl(records_path, records)

    dataset_manifest = build_dataset_manifest(
        dataset_id=dataset_id,
        version=version,
        records=records,
        records_hash=records_hash,
    )
    dataset_manifest_public = {
        k: v for k, v in dataset_manifest.items() if not k.startswith("_")
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest_public, indent=2, sort_keys=True) + "\n",
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

    tokenizer = build_t4_smoke_tokenizer(vocab_size=tokenizer_vocab)
    token_ids = tokenize_records(tokenizer, records)
    shard_manifest = None
    if write_shards:
        windows = pack_token_windows(token_ids, seq_len=seq_len)
        shards_dir = output / "shards"
        config_hash = file_sha256(config_path) if config_path is not None else None
        # tokenizer artifact hash proxy: hash of vocab size + id + first/last merges count
        tokenizer_hash = sha256(
            f"{tokenizer.tokenizer_id}:{tokenizer.vocab_size}:{len(tokenizer.merges)}".encode("utf-8")
        ).hexdigest()
        shard_manifest = write_training_shard(
            shards_dir,
            shard_id="train-00000",
            windows=windows,
            tokenizer=tokenizer,
            tokenizer_hash=tokenizer_hash,
            document_count=len(records),
            dataset_id=dataset_id,
            config_hash=config_hash,
        )

    report = {
        "dataset_id": dataset_id,
        "version": version,
        "generator_id": GENERATOR_ID,
        "n_docs": len(records),
        "seed": seed,
        "domain_histogram": domain_histogram(records),
        "pin_pass_rate": 1.0,
        "token_count": len(token_ids),
        "seq_len": seq_len,
        "records_path": str(records_path),
        "shard_token_count": None if shard_manifest is None else shard_manifest["token_count"],
        "status": "OK",
    }
    (output / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
