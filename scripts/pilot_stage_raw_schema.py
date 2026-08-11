#!/usr/bin/env python3
"""MAGI_DATA_PILOT_v0.1 — RAW LOCK + SCHEMA + corruption/null probe.

Stages:
  1 raw_lock
  2 schema_inspect
  3 corruption_null_check

Requires: MAGI_DATA_PILOT_APPROVED=1, pyarrow
Usage:
  python scripts/pilot_stage_raw_schema.py --data-root /workspace/magi_data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEXT_ALIASES = ("text", "content", "raw_content", "code", "body", "document", "article")
# Never treat identifier columns as text body
TEXT_REJECT_SUFFIXES = ("_id", "_hash", "_key", "_uuid")
REQUIRED_RAW_BUCKETS = (
    "fineweb2_ru",
    "fineweb_en",
    "finemath",
    "stack_v2",
    "wikipedia_ru",
    "wikipedia_en",
)
ROW_PROBE = 2048


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_pilot() -> None:
    if os.environ.get("MAGI_DATA_PRODUCTION_APPROVED") == "1":
        raise SystemExit("REFUSE: PRODUCTION approval is not authorized")
    if os.environ.get("MAGI_DATA_PILOT_APPROVED") != "1":
        legacy = os.environ.get("MAGI_DATA_MANIFEST_APPROVED")
        if legacy == "1":
            print("WARN: legacy MAGI_DATA_MANIFEST_APPROVED=1 mapped to PILOT only")
        else:
            raise SystemExit("REFUSE: set MAGI_DATA_PILOT_APPROVED=1")


def _parquet_files(bucket: Path) -> list[Path]:
    return sorted(p for p in bucket.rglob("*.parquet") if ".cache" not in p.parts)


def _sha256_file(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def _pick_text_col(names: list[str]) -> str | None:
    lower = {n.lower(): n for n in names}
    for alias in TEXT_ALIASES:
        if alias in lower:
            return lower[alias]
    for n in names:
        ln = n.lower()
        if ln.endswith(TEXT_REJECT_SUFFIXES):
            continue
        if "text" in ln or ln == "content" or ln.endswith("_content") or ln == "code":
            return n
    return None


def raw_lock(data_root: Path) -> dict[str, Any]:
    raw = data_root / "raw"
    reports = data_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, Any] = {}
    for bucket in sorted(p for p in raw.iterdir() if p.is_dir()):
        files = _parquet_files(bucket)
        entries = []
        total = 0
        for fp in files:
            size = fp.stat().st_size
            total += size
            entries.append(
                {
                    "path": str(fp.relative_to(bucket)),
                    "bytes": size,
                    "sha256_head64mib": _sha256_file(fp),
                }
            )
        buckets[bucket.name] = {
            "n_parquet": len(files),
            "downloaded_bytes": total,
            "downloaded_gb": round(total / 1024**3, 4),
            "files": entries,
        }
        prov = {
            "scope": "MAGI_DATA_PILOT_v0.1",
            "source_id": bucket.name,
            "locked_at": _utc(),
            "policy": "raw_lock_no_further_download_into_bucket",
            **buckets[bucket.name],
        }
        (bucket / "RAW_LOCK.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")
    lock = {
        "scope": "MAGI_DATA_PILOT_v0.1",
        "stage": "RAW_LOCK",
        "locked_at": _utc(),
        "buckets": buckets,
        "missing_required": [
            k for k in REQUIRED_RAW_BUCKETS if k not in buckets or buckets[k]["n_parquet"] == 0
        ],
        "notes": (
            "nullxes_domain optional until author slice; "
            "fineweb_en = HuggingFaceFW/fineweb (FineWeb-2 has no English); "
            "stack_v2 parquet is metadata/SWH ids only — body not in files"
        ),
    }
    out = reports / "RAW_LOCK_v0.1.json"
    out.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"raw_lock": str(out), "missing": lock["missing_required"]}, indent=2))
    return lock


def schema_inspect(data_root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    raw = data_root / "raw"
    reports = data_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"scope": "MAGI_DATA_PILOT_v0.1", "stage": "SCHEMA_INSPECT", "inspected_at": _utc(), "sources": {}}
    for bucket in sorted(p for p in raw.iterdir() if p.is_dir()):
        files = _parquet_files(bucket)
        src: dict[str, Any] = {"n_files": len(files), "schemas": []}
        for fp in files:
            try:
                pf = pq.ParquetFile(fp)
                schema = pf.schema_arrow
                names = list(schema.names)
                text_col = _pick_text_col(names)
                md = pf.metadata
                src["schemas"].append(
                    {
                        "path": str(fp.relative_to(bucket)),
                        "ok": True,
                        "num_row_groups": pf.num_row_groups,
                        "num_rows": md.num_rows if md is not None else None,
                        "columns": [{"name": f.name, "type": str(f.type)} for f in schema],
                        "text_column": text_col,
                        "column_names": names,
                    }
                )
            except Exception as exc:  # noqa: BLE001 — record per-file failure
                src["schemas"].append({"path": str(fp.relative_to(bucket)), "ok": False, "error": repr(exc)})
        # unify text column across files
        texts = {s.get("text_column") for s in src["schemas"] if s.get("ok")}
        src["text_column_consensus"] = next(iter(texts)) if len(texts) == 1 else sorted(t for t in texts if t)
        src["schema_ok"] = all(s.get("ok") for s in src["schemas"]) and len(files) > 0
        result["sources"][bucket.name] = src
        print(f"[schema] {bucket.name}: files={len(files)} text={src['text_column_consensus']} ok={src['schema_ok']}")
    out = reports / "SCHEMA_INSPECT_v0.1.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("->", out)
    return result


def corruption_null_check(data_root: Path, row_probe: int = ROW_PROBE) -> dict[str, Any]:
    import pyarrow.parquet as pq

    raw = data_root / "raw"
    reports = data_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    schema_path = reports / "SCHEMA_INSPECT_v0.1.json"
    schemas = json.loads(schema_path.read_text()) if schema_path.exists() else {"sources": {}}
    result: dict[str, Any] = {
        "scope": "MAGI_DATA_PILOT_v0.1",
        "stage": "CORRUPTION_NULL_CHECK",
        "checked_at": _utc(),
        "row_probe": row_probe,
        "sources": {},
    }
    for bucket in sorted(p for p in raw.iterdir() if p.is_dir()):
        files = _parquet_files(bucket)
        src_schema = schemas.get("sources", {}).get(bucket.name, {})
        consensus = src_schema.get("text_column_consensus")
        text_col = consensus if isinstance(consensus, str) else None
        file_reports = []
        for fp in files:
            entry: dict[str, Any] = {"path": str(fp.relative_to(bucket))}
            try:
                pf = pq.ParquetFile(fp)
                # open first row group — corruption probe
                table = pf.read_row_group(0)
                if table.num_rows > row_probe:
                    table = table.slice(0, row_probe)
                cols = table.column_names
                tc = text_col or _pick_text_col(cols)
                entry["text_column"] = tc
                entry["rows_probed"] = table.num_rows
                entry["open_ok"] = True
                if tc and tc in cols:
                    col = table.column(tc)
                    nulls = col.null_count
                    empty = 0
                    non_str = 0
                    for i in range(col.length()):
                        v = col[i].as_py()
                        if v is None:
                            continue
                        if not isinstance(v, str):
                            non_str += 1
                        elif not v.strip():
                            empty += 1
                    entry["null_text"] = int(nulls)
                    entry["empty_text"] = empty
                    entry["non_str_text"] = non_str
                    entry["null_or_empty_rate"] = round((nulls + empty) / max(table.num_rows, 1), 6)
                else:
                    entry["warning"] = "no_text_column"
                # remaining row groups: metadata-only open
                for rg in range(1, pf.num_row_groups):
                    _ = pf.metadata.row_group(rg).num_rows
                entry["all_row_groups_meta_ok"] = True
            except Exception as exc:  # noqa: BLE001
                entry["open_ok"] = False
                entry["error"] = repr(exc)
            file_reports.append(entry)
            status = "OK" if entry.get("open_ok") else "FAIL"
            print(f"[corrupt] {bucket.name}/{entry['path']}: {status} null_empty={entry.get('null_or_empty_rate')}")
        result["sources"][bucket.name] = {
            "n_files": len(files),
            "all_open_ok": all(f.get("open_ok") for f in file_reports) and len(files) > 0,
            "files": file_reports,
        }
    out = reports / "CORRUPTION_NULL_CHECK_v0.1.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("->", out)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("MAGI_DATA_ROOT", "/workspace/magi_data"))
    ap.add_argument("--stage", choices=("all", "raw_lock", "schema", "corrupt"), default="all")
    ap.add_argument("--row-probe", type=int, default=ROW_PROBE)
    args = ap.parse_args()
    _require_pilot()
    data_root = Path(args.data_root)
    if not (data_root / "raw").is_dir():
        raise SystemExit(f"missing raw dir: {data_root / 'raw'}")
    if args.stage in ("all", "raw_lock"):
        raw_lock(data_root)
    if args.stage in ("all", "schema"):
        schema_inspect(data_root)
    if args.stage in ("all", "corrupt"):
        corruption_null_check(data_root, row_probe=args.row_probe)


if __name__ == "__main__":
    main()
