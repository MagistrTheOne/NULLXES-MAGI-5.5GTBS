#!/usr/bin/env python3
"""MAGI_DATA_PILOT_v0.1 — TEXT NORMALIZATION (+ lang tag, quality gates light).

Reads RAW_LOCK + SCHEMA_INSPECT. Writes:
  filtered/normalized/<source>/*.jsonl.gz
  reports/TEXT_NORMALIZE_v0.1.json

Stack v2 (bigcode/the-stack-v2) is METADATA_ONLY — no file body in parquet.
Those buckets are quarantined here; code text requires SWH fetch or Stack v1 content.

Requires: MAGI_DATA_PILOT_APPROVED=1, pyarrow
Usage:
  python scripts/pilot_stage_normalize.py --data-root /workspace/magi_data
  python scripts/pilot_stage_normalize.py --data-root /workspace/magi_data --max-docs-per-source 50000
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_NL = re.compile(r"\n{4,}")
MULTI_SPACE = re.compile(r"[ \t]{3,}")

# Exact text columns only — never *_id
SOURCE_TEXT_COL: dict[str, str] = {
    "finemath": "text",
    "fineweb2_ru": "text",
    "fineweb_en": "text",
    "wikipedia_ru": "text",
    "wikipedia_en": "text",
}

# Metadata-only / no body in parquet
QUARANTINE_NO_TEXT: dict[str, str] = {
    "stack_v2": "the-stack-v2 stores SWH ids only; content lives on softwareheritage S3",
}

MIN_CHARS = 64
MAX_CHARS = 200_000


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_pilot() -> None:
    if os.environ.get("MAGI_DATA_PRODUCTION_APPROVED") == "1":
        raise SystemExit("REFUSE: PRODUCTION approval is not authorized")
    if os.environ.get("MAGI_DATA_PILOT_APPROVED") != "1":
        if os.environ.get("MAGI_DATA_MANIFEST_APPROVED") == "1":
            print("WARN: legacy MAGI_DATA_MANIFEST_APPROVED=1 mapped to PILOT only")
        else:
            raise SystemExit("REFUSE: set MAGI_DATA_PILOT_APPROVED=1")


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = CTRL_RE.sub("", t)
    t = MULTI_NL.sub("\n\n\n", t)
    t = MULTI_SPACE.sub("  ", t)
    return t.strip()


def quality_gate(text: str) -> str | None:
    n = len(text)
    if n < MIN_CHARS:
        return "too_short"
    if n > MAX_CHARS:
        return "too_long"
    # printable / letter ratio
    letters = sum(1 for c in text if c.isalpha())
    if letters / max(n, 1) < 0.15:
        return "low_alpha"
    return None


def lang_hint(source: str, row: dict[str, Any]) -> str:
    if source.endswith("_ru") or source == "fineweb2_ru":
        return "ru"
    if source.endswith("_en") or source == "fineweb_en":
        return "en"
    if source == "finemath":
        return "en"
    lang = row.get("language") or row.get("lang")
    if isinstance(lang, str) and lang:
        return lang
    return "und"


def iter_rows(parquet_path: Path, columns: list[str], batch_size: int = 1024) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    present = [c for c in columns if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=batch_size, columns=present):
        cols = {name: batch.column(name) for name in batch.schema.names}
        n = batch.num_rows
        for i in range(n):
            yield {name: cols[name][i].as_py() for name in cols}


def process_source(
    source: str,
    raw_bucket: Path,
    out_dir: Path,
    text_col: str,
    max_docs: int | None,
) -> dict[str, Any]:
    files = sorted(p for p in raw_bucket.rglob("*.parquet") if ".cache" not in p.parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "source": source,
        "text_column": text_col,
        "files_in": len(files),
        "rows_seen": 0,
        "accepted": 0,
        "rejected": {},
        "out_files": [],
        "bytes_out": 0,
    }

    def bump(reason: str) -> None:
        stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1

    doc_i = 0
    shard_i = 0
    fh = None
    out_path = None

    def open_shard() -> None:
        nonlocal fh, out_path, shard_i
        if fh is not None:
            fh.close()
        out_path = out_dir / f"norm-{shard_i:05d}.jsonl.gz"
        fh = gzip.open(out_path, "wt", encoding="utf-8")
        stats["out_files"].append(str(out_path.name))
        shard_i += 1

    open_shard()
    assert fh is not None

    extra_cols = ["id", "url", "dump", "language", "lang", "title", "path", "license_type", "detected_licenses"]
    try:
        for fp in files:
            if max_docs is not None and stats["accepted"] >= max_docs:
                break
            rel = str(fp.relative_to(raw_bucket))
            for row in iter_rows(fp, [text_col] + extra_cols):
                if max_docs is not None and stats["accepted"] >= max_docs:
                    break
                stats["rows_seen"] += 1
                raw = row.get(text_col)
                if raw is None:
                    bump("null_text")
                    continue
                if not isinstance(raw, str):
                    bump("non_str_text")
                    continue
                text = normalize_text(raw)
                reason = quality_gate(text)
                if reason:
                    bump(reason)
                    continue
                doc_id = row.get("id") or row.get("url") or f"{source}:{rel}:{stats['rows_seen']}"
                rec = {
                    "doc_id": str(doc_id),
                    "source": source,
                    "lang": lang_hint(source, row),
                    "text": text,
                    "n_chars": len(text),
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "raw_relpath": rel,
                    "stage": "TEXT_NORMALIZE_v0.1",
                }
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                fh.write(line)
                stats["accepted"] += 1
                doc_i += 1
                if doc_i % 50_000 == 0:
                    open_shard()
                    assert fh is not None
                    print(f"[{source}] accepted={stats['accepted']} seen={stats['rows_seen']}")
    finally:
        if fh is not None:
            fh.close()

    for name in stats["out_files"]:
        p = out_dir / name
        if p.exists():
            stats["bytes_out"] += p.stat().st_size
    stats["bytes_out_gb"] = round(stats["bytes_out"] / 1024**3, 4)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("MAGI_DATA_ROOT", "/workspace/magi_data"))
    ap.add_argument("--max-docs-per-source", type=int, default=None, help="cap for timed runs")
    ap.add_argument("--sources", nargs="*", default=None, help="subset of sources")
    args = ap.parse_args()
    _require_pilot()

    data_root = Path(args.data_root)
    raw = data_root / "raw"
    reports = data_root / "reports"
    out_root = data_root / "filtered" / "normalized"
    reports.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "scope": "MAGI_DATA_PILOT_v0.1",
        "stage": "TEXT_NORMALIZE",
        "started_at": _utc(),
        "sources": {},
        "quarantined": {},
    }

    sources = args.sources or sorted(
        p.name for p in raw.iterdir() if p.is_dir() and (args.sources is None or p.name in args.sources)
    )
    if args.sources:
        sources = list(args.sources)

    for source in sources:
        bucket = raw / source
        if not bucket.is_dir():
            print(f"[skip] missing {source}")
            continue
        if source in QUARANTINE_NO_TEXT:
            q = {
                "status": "QUARANTINE_METADATA_ONLY",
                "reason": QUARANTINE_NO_TEXT[source],
                "action": "do_not_normalize; fetch body via SWH or replace with content-bearing code corpus",
            }
            report["quarantined"][source] = q
            (bucket / "QUARANTINE.json").write_text(json.dumps(q, indent=2) + "\n")
            print(f"[QUARANTINE] {source}: {q['reason']}")
            continue
        text_col = SOURCE_TEXT_COL.get(source)
        if not text_col:
            report["quarantined"][source] = {"status": "NO_TEXT_MAP", "reason": "unknown source text column"}
            print(f"[skip] no text map for {source}")
            continue
        print(f"[norm] {source} col={text_col}")
        stats = process_source(
            source=source,
            raw_bucket=bucket,
            out_dir=out_root / source,
            text_col=text_col,
            max_docs=args.max_docs_per_source,
        )
        report["sources"][source] = stats
        print(
            f"[done] {source}: accepted={stats['accepted']} seen={stats['rows_seen']} "
            f"out_gb={stats['bytes_out_gb']} rejected={stats['rejected']}"
        )

    report["finished_at"] = _utc()
    out = reports / "TEXT_NORMALIZE_v0.1.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("->", out)


if __name__ == "__main__":
    main()
