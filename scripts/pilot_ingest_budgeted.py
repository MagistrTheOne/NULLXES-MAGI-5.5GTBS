#!/usr/bin/env python3
"""Budgeted MAGI_DATA_PILOT_v0.1 ingest — GB caps, never vanity full dumps.

Requires:
  MAGI_DATA_PILOT_APPROVED=1
  HF token for gated repos (huggingface-cli login or HF_TOKEN)

Usage:
  python scripts/pilot_ingest_budgeted.py --source fineweb2_ru --max-gb 6
  python scripts/pilot_ingest_budgeted.py --source all --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Pilot source table — budgets are HARD caps on downloaded compressed bytes.
PILOT_SOURCES: dict[str, dict] = {
    "fineweb2_ru": {
        "candidate": "fineweb2",
        "hub_id": "HuggingFaceFW/fineweb-2",
        "repo_type": "dataset",
        "prefixes": ["data/rus_Cyrl/"],
        "max_gb": 6.0,
        "license": "odc-by",
    },
    "fineweb2_en": {
        "candidate": "fineweb2",
        "hub_id": "HuggingFaceFW/fineweb-2",
        "repo_type": "dataset",
        "prefixes": ["data/eng_Latn/"],
        "max_gb": 6.0,
        "license": "odc-by",
    },
    "stack_python": {
        "candidate": "the_stack_v2",
        "hub_id": "bigcode/the-stack-v2",
        "repo_type": "dataset",
        "prefixes": ["data/Python/"],
        "max_gb": 4.0,
        "license": "other",
        "notes": "per-repo license gate REQUIRED before BASE mix; pilot raw only",
    },
    "stack_typescript": {
        "candidate": "the_stack_v2",
        "hub_id": "bigcode/the-stack-v2",
        "repo_type": "dataset",
        "prefixes": ["data/TypeScript/"],
        "max_gb": 2.5,
        "license": "other",
        "notes": "per-repo license gate REQUIRED before BASE mix; pilot raw only",
    },
    "finemath": {
        "candidate": "finemath",
        "hub_id": "HuggingFaceTB/finemath",
        "repo_type": "dataset",
        "prefixes": ["data/", "finemath/", ""],
        "max_gb": 2.5,
        "license": "odc-by",
        "prefer_substrings": ["3plus", "4plus", "train"],
    },
    "wikipedia_ru": {
        "candidate": "wikimedia_wikipedia",
        "hub_id": "wikimedia/wikipedia",
        "repo_type": "dataset",
        "prefixes": ["20231101.ru/", "data/20231101.ru/", "ru/"],
        "max_gb": 1.5,
        "license": "VERIFY_PER_DUMP",
    },
    "wikipedia_en": {
        "candidate": "wikimedia_wikipedia",
        "hub_id": "wikimedia/wikipedia",
        "repo_type": "dataset",
        "prefixes": ["20231101.en/", "data/20231101.en/", "en/"],
        "max_gb": 1.5,
        "license": "VERIFY_PER_DUMP",
    },
    "nullxes_domain": {
        "candidate": "nullxes_domain",
        "hub_id": "NULLXES/private_or_local",
        "repo_type": "local",
        "prefixes": [],
        "max_gb": 0.0,
        "license": "NULLXES_INTERNAL",
        "notes": "author-supplied approved slice only — no Hub download",
    },
}


def _require_pilot() -> None:
    if os.environ.get("MAGI_DATA_PRODUCTION_APPROVED") == "1":
        raise SystemExit("REFUSE: PRODUCTION approval is not authorized")
    if os.environ.get("MAGI_DATA_PILOT_APPROVED") != "1":
        raise SystemExit("REFUSE: set MAGI_DATA_PILOT_APPROVED=1")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_provenance(out_dir: Path, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "PROVENANCE.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _match_prefix(path: str, prefixes: list[str]) -> bool:
    if not prefixes or prefixes == [""]:
        return True
    return any(path.startswith(p) or (p == "" and True) for p in prefixes if p != "")


def ingest_hub_source(
    *,
    source_id: str,
    spec: dict,
    root_raw: Path,
    dry_run: bool,
    max_gb: float | None,
) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    hub_id = spec["hub_id"]
    repo_type = spec["repo_type"]
    prefixes = list(spec.get("prefixes") or [])
    budget_gb = float(max_gb if max_gb is not None else spec["max_gb"])
    budget = int(budget_gb * (1024**3))
    out_dir = root_raw / source_id
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    print(f"listing={hub_id}")
    files = api.list_repo_files(repo_id=hub_id, repo_type=repo_type)
    candidates = [f for f in files if _match_prefix(f, prefixes)]
    prefer = list(spec.get("prefer_substrings") or [])
    if prefer:
        preferred = [f for f in candidates if any(s in f for s in prefer)]
        if preferred:
            candidates = preferred + [f for f in candidates if f not in preferred]

    # Prefer parquet/jsonl/zst/txt over random blobs.
    def rank(path: str) -> tuple[int, str]:
        ext_rank = 0
        lower = path.lower()
        if lower.endswith((".parquet", ".jsonl", ".jsonl.gz", ".zst", ".txt", ".json")):
            ext_rank = 0
        elif "/data/" in lower or lower.startswith("data/"):
            ext_rank = 1
        else:
            ext_rank = 2
        return (ext_rank, path)

    candidates = sorted(set(candidates), key=rank)
    if not candidates:
        # Fallback: take any data-looking files if prefixes missed.
        candidates = sorted(
            [f for f in files if any(x in f.lower() for x in (".parquet", ".jsonl", "train"))],
            key=rank,
        )[:200]

    print(f"matched_files={len(candidates)} budget_gb={budget_gb}")
    downloaded: list[dict] = []
    total = 0
    for rel in candidates:
        if total >= budget:
            break
        if dry_run:
            print(f"DRY would_fetch={rel}")
            downloaded.append({"path": rel, "bytes": None, "dry_run": True})
            if len(downloaded) >= 5:
                break
            continue
        try:
            local = hf_hub_download(
                repo_id=hub_id,
                repo_type=repo_type,
                filename=rel,
                local_dir=str(out_dir),
                local_dir_use_symlinks=False,
            )
            size = Path(local).stat().st_size
        except Exception as exc:  # noqa: BLE001 — continue other shards
            print(f"skip={rel} err={exc}")
            continue
        total += size
        downloaded.append({"path": rel, "bytes": size, "local": str(local)})
        print(f"fetched={rel} bytes={size} total_gb={total / 1024**3:.3f}")

    payload = {
        "scope": "MAGI_DATA_PILOT_v0.1",
        "source_id": source_id,
        "candidate": spec["candidate"],
        "hub_id": hub_id,
        "license": spec.get("license"),
        "notes": spec.get("notes"),
        "captured_at": _utc_now(),
        "budget_gb": budget_gb,
        "downloaded_bytes": total,
        "downloaded_gb": round(total / 1024**3, 4),
        "file_count": len(downloaded),
        "files": downloaded,
        "dry_run": dry_run,
        "policy": "budgeted_pilot_never_full_dump",
    }
    prov = _write_provenance(out_dir, payload)
    print(f"provenance={prov}")
    print(f"status={'DRY_OK' if dry_run else 'INGEST_OK'} source={source_id}")
    return payload


def ingest_nullxes(*, root_raw: Path) -> dict:
    out_dir = root_raw / "nullxes_domain"
    payload = {
        "scope": "MAGI_DATA_PILOT_v0.1",
        "source_id": "nullxes_domain",
        "candidate": "nullxes_domain",
        "hub_id": "NULLXES/private_or_local",
        "license": "NULLXES_INTERNAL",
        "captured_at": _utc_now(),
        "status": "AWAITING_AUTHOR_SLICE",
        "notes": "Place approved NULLXES pretraining files under this directory, then re-run provenance update.",
        "expected_path": str(out_dir),
        "downloaded_bytes": 0,
        "file_count": 0,
    }
    prov = _write_provenance(out_dir, payload)
    print(f"provenance={prov}")
    print("status=NULLXES_SLOT_READY")
    return payload


def main() -> int:
    _require_pilot()
    parser = argparse.ArgumentParser(description="Budgeted MAGI pilot ingest")
    parser.add_argument(
        "--source",
        default="all",
        help="one of: all | " + " | ".join(PILOT_SOURCES),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/workspace/magi_data/raw"),
    )
    parser.add_argument("--max-gb", type=float, default=None, help="override budget for one source")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources = list(PILOT_SOURCES) if args.source == "all" else [args.source]
    for sid in sources:
        if sid not in PILOT_SOURCES:
            raise SystemExit(f"unknown source={sid}")
        print()
        print("=" * 60)
        print(f"SOURCE {sid}")
        print("=" * 60)
        spec = PILOT_SOURCES[sid]
        if spec["repo_type"] == "local":
            ingest_nullxes(root_raw=args.raw_root)
            continue
        ingest_hub_source(
            source_id=sid,
            spec=spec,
            root_raw=args.raw_root,
            dry_run=args.dry_run,
            max_gb=args.max_gb,
        )
    print()
    print("status=PILOT_BATCH_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
