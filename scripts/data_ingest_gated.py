#!/usr/bin/env python3
"""Gated cloud ingest entry — PILOT_V0.1 only. Never production full dump.

Usage (cloud, after author signs PILOT_V0.1):
  MAGI_DATA_PILOT_APPROVED=1 python scripts/data_ingest_gated.py --candidate fineweb2 --dry-run

Legacy alias MAGI_DATA_MANIFEST_APPROVED=1 is treated as PILOT only.
MAGI_DATA_PRODUCTION_APPROVED is refused — production path does not exist yet.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magi.config import load_simple_yaml

PILOT_BLOCKED_STATUS = {
    "PLACEHOLDER_NEEDS_SOURCE_SELECTION",
    "TO_BE_BUILT_IN_REPO",
    "SPEC_PLUS_GOLDEN_SEEDS_ONLY",
    "NOT_APPROVED_FOR_BASE",
    "REFERENCE_NOT_IN_PILOT",
    "CANDIDATE_PENDING_APPROVAL",
}


def _approval_mode() -> str:
    if os.environ.get("MAGI_DATA_PRODUCTION_APPROVED") == "1":
        raise SystemExit(
            "REFUSE: MAGI_DATA_PRODUCTION_APPROVED=1 is not authorized. "
            "Production ingest does not exist in this program version. "
            "Use MAGI_DATA_PILOT_APPROVED=1 for PILOT_V0.1 only."
        )
    if os.environ.get("MAGI_DATA_PILOT_APPROVED") == "1":
        return "PILOT_V0.1"
    if os.environ.get("MAGI_DATA_MANIFEST_APPROVED") == "1":
        # Legacy alias maps ONLY to pilot — never full FineWeb / Stack dump.
        return "PILOT_V0.1_VIA_LEGACY_ALIAS"
    raise SystemExit(
        "REFUSE: set MAGI_DATA_PILOT_APPROVED=1 on approved cloud machine "
        "after author signs data/program/MAGI_DATA_MANIFEST_v0.1.yaml "
        "approval_checklist for PILOT_V0.1. "
        "This is NOT permission to download FineWeb2 / Stack v2 in full."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="MAGI gated PILOT ingest (cloud)")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "program" / "MAGI_DATA_MANIFEST_v0.1.yaml",
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="print cloud hf command template (still no local bulk)",
    )
    args = parser.parse_args()

    mode = _approval_mode()
    raw = load_simple_yaml(args.manifest)
    pilot = raw.get("pilot_v0_1") or {}
    allowed = pilot.get("allowed_candidates") or {}
    cand = (raw.get("candidates") or {}).get(args.candidate)
    if cand is None:
        raise SystemExit(f"unknown candidate: {args.candidate}")

    if args.candidate not in allowed or allowed.get(args.candidate) not in (True, "true", 1, "1"):
        raise SystemExit(
            f"REFUSE: {args.candidate} is outside PILOT_V0.1 allowed_candidates. "
            f"Character/SFT/distillation/reference mixes are not pilot ingest."
        )

    status = str(cand.get("status", ""))
    if status in PILOT_BLOCKED_STATUS:
        raise SystemExit(f"candidate not ready for pilot ingest: {args.candidate} status={status}")
    if status != "PILOT_CANDIDATE":
        raise SystemExit(
            f"candidate status must be PILOT_CANDIDATE for pilot ingest; "
            f"got {args.candidate} status={status}"
        )
    if cand.get("layer") == "CHARACTER" or cand.get("base_ingest") == "NOT_APPROVED_FOR_BASE":
        raise SystemExit(f"REFUSE: {args.candidate} is NOT_APPROVED_FOR_BASE")

    hub_id = cand["hub_id"]
    print(f"approval_mode={mode}")
    print("approval_scope=PILOT_V0.1")
    print("production_ingest=false")
    print(f"candidate={args.candidate}")
    print(f"hub_id={hub_id}")
    print(f"layer={cand.get('layer')}")
    print(f"license={cand.get('license')}")
    print(f"pilot_accepted_gb_target={pilot.get('expected_accepted_text_gb_min')}-{pilot.get('expected_accepted_text_gb_max')}")
    print("policy=pilot_subset_only_never_vanity_full_dump")
    print(
        "next=run hf download on CLOUD with language/subset filters "
        "(see data/program/CLOUD_INGEST_PLAYBOOK_v0.1.md)"
    )
    if args.execute and not args.dry_run:
        raise SystemExit(
            "Execute path not implemented in-repo by design. "
            "Run hf download manually on cloud per playbook after pilot sizing."
        )
    print("status=GATED_OK_DRY_RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
