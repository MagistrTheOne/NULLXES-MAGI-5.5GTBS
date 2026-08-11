#!/usr/bin/env python3
"""MAGI_DATA_PILOT_v0.1 — LANGUAGE PURITY + light QUALITY + EXACT DEDUP.

Input:  filtered/normalized/<source>/*.jsonl.gz
Output: filtered/accepted/<source>/*.jsonl.gz
Report: reports/LANG_QUALITY_EXACT_DEDUP_v0.1.json

Lang gate (heuristic, no external model dependency):
  fineweb2_ru / wikipedia_ru → require Cyrillic ratio
  fineweb_en / wikipedia_en / finemath → require Latin letter ratio
  stack_v1 → skip lang (code), keep license/content gates only

Exact dedup: sha256(text) global across sources (first wins).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CYR = re.compile(r"[А-Яа-яЁёІіЇїЄєҐґ]")
LAT = re.compile(r"[A-Za-z]")

SOURCE_LANG_POLICY: dict[str, str] = {
    "fineweb2_ru": "ru",
    "wikipedia_ru": "ru",
    "fineweb_en": "en",
    "wikipedia_en": "en",
    "finemath": "en",
    "stack_v1": "code",
}


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


def iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def lang_ok(policy: str, text: str, declared: str | None) -> tuple[bool, str]:
    if policy == "code":
        return True, "code_skip"
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False, "no_letters"
    n = len(letters)
    cyr = sum(1 for c in letters if CYR.match(c))
    lat = sum(1 for c in letters if LAT.match(c))
    cyr_r = cyr / n
    lat_r = lat / n
    if policy == "ru":
        if cyr_r < 0.55:
            return False, f"ru_cyr_low:{cyr_r:.3f}"
        return True, f"ru_cyr:{cyr_r:.3f}"
    if policy == "en":
        if lat_r < 0.70:
            return False, f"en_lat_low:{lat_r:.3f}"
        # reject heavy cyrillic contamination in EN buckets
        if cyr_r > 0.15:
            return False, f"en_cyr_contam:{cyr_r:.3f}"
        return True, f"en_lat:{lat_r:.3f}"
    return False, f"unknown_policy:{policy}"


def quality_ok(text: str, source: str) -> tuple[bool, str]:
    n = len(text)
    if n < 64:
        return False, "too_short"
    if n > 200_000:
        return False, "too_long"
    # repeated-char spam
    if re.search(r"(.)\1{40,}", text):
        return False, "char_spam"
    # URL-only / boilerplate heavy
    urls = len(re.findall(r"https?://", text))
    if urls > 30 and urls * 40 > n:
        return False, "url_heavy"
    if source == "stack_v1":
        # reject tiny stubs
        if text.count("\n") < 1 and n < 120:
            return False, "code_too_flat"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("MAGI_DATA_ROOT", "/workspace/magi_data"))
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--max-docs-per-source", type=int, default=None)
    args = ap.parse_args()
    _require_pilot()

    data_root = Path(args.data_root)
    norm_root = data_root / "filtered" / "normalized"
    acc_root = data_root / "filtered" / "accepted"
    reports = data_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    acc_root.mkdir(parents=True, exist_ok=True)

    sources = args.sources or sorted(p.name for p in norm_root.iterdir() if p.is_dir())
    seen_hash: set[str] = set()
    report: dict[str, Any] = {
        "scope": "MAGI_DATA_PILOT_v0.1",
        "stage": "LANG_QUALITY_EXACT_DEDUP",
        "started_at": _utc(),
        "sources": {},
        "global_exact_dedup_dropped": 0,
    }

    for source in sources:
        src_dir = norm_root / source
        if not src_dir.is_dir():
            print(f"[skip] {source}")
            continue
        policy = SOURCE_LANG_POLICY.get(source, "und")
        files = sorted(src_dir.glob("norm-*.jsonl.gz"))
        out_dir = acc_root / source
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "accepted-00000.jsonl.gz"
        stats = {
            "source": source,
            "lang_policy": policy,
            "files_in": len(files),
            "seen": 0,
            "accepted": 0,
            "rejected": {},
            "bytes_out": 0,
        }

        def bump(r: str) -> None:
            stats["rejected"][r] = stats["rejected"].get(r, 0) + 1

        print(f"[gate] {source} policy={policy} files={len(files)}")
        with gzip.open(out_path, "wt", encoding="utf-8") as out:
            for fp in files:
                if args.max_docs_per_source is not None and stats["accepted"] >= args.max_docs_per_source:
                    break
                for rec in iter_jsonl_gz(fp):
                    if args.max_docs_per_source is not None and stats["accepted"] >= args.max_docs_per_source:
                        break
                    stats["seen"] += 1
                    text = rec.get("text")
                    if not isinstance(text, str):
                        bump("bad_text")
                        continue
                    ok_q, why_q = quality_ok(text, source)
                    if not ok_q:
                        bump(why_q)
                        continue
                    ok_l, why_l = lang_ok(policy, text, rec.get("lang"))
                    if not ok_l:
                        bump(why_l)
                        continue
                    h = rec.get("text_sha256")
                    if not isinstance(h, str) or len(h) != 64:
                        bump("bad_hash")
                        continue
                    if h in seen_hash:
                        bump("exact_dedup")
                        report["global_exact_dedup_dropped"] += 1
                        continue
                    seen_hash.add(h)
                    out_rec = {
                        **rec,
                        "lang_gate": why_l,
                        "quality_gate": why_q,
                        "stage": "LANG_QUALITY_EXACT_DEDUP_v0.1",
                    }
                    out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    stats["accepted"] += 1
                    if stats["accepted"] % 50_000 == 0:
                        print(f"  [{source}] accepted={stats['accepted']} seen={stats['seen']}")
        stats["bytes_out"] = out_path.stat().st_size if out_path.exists() else 0
        stats["bytes_out_gb"] = round(stats["bytes_out"] / 1024**3, 4)
        report["sources"][source] = stats
        print(
            f"[done] {source}: accepted={stats['accepted']} seen={stats['seen']} "
            f"rejected={stats['rejected']}"
        )

    report["finished_at"] = _utc()
    report["unique_hashes"] = len(seen_hash)
    outp = reports / "LANG_QUALITY_EXACT_DEDUP_v0.1.json"
    outp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("->", outp)


if __name__ == "__main__":
    main()
