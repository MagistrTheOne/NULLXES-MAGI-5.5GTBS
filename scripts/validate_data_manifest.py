"""Validate MAGI_DATA_MANIFEST_v0.1 — stdlib only. Never downloads corpora."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magi.config import load_simple_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MAGI data manifest (no download)")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "program" / "MAGI_DATA_MANIFEST_v0.1.yaml",
    )
    args = parser.parse_args()

    raw = load_simple_yaml(args.manifest)
    meta = raw.get("meta", {})
    if meta.get("local_workstation_download") != "FORBIDDEN":
        raise SystemExit("meta.local_workstation_download must be FORBIDDEN")
    if meta.get("download_policy") != "cloud_only_after_approval":
        raise SystemExit("meta.download_policy must be cloud_only_after_approval")

    approval = raw.get("approval") or {}
    if approval.get("scope") not in {"PILOT_V0.1", "MAGI_DATA_PILOT_v0.1"}:
        raise SystemExit("approval.scope must be MAGI_DATA_PILOT_v0.1")
    if approval.get("production_ingest") not in (False, "false"):
        raise SystemExit("approval.production_ingest must be false")
    if approval.get("require_env_pilot") != "MAGI_DATA_PILOT_APPROVED":
        raise SystemExit("approval.require_env_pilot must be MAGI_DATA_PILOT_APPROVED")

    mix = raw.get("base_mixture_v0_1") or {}
    if mix.get("mixture_status") != "EXPERIMENTAL_HYPOTHESIS":
        raise SystemExit("base_mixture_v0_1.mixture_status must be EXPERIMENTAL_HYPOTHESIS")
    if mix.get("production_weighting_approved") not in (False, "false"):
        raise SystemExit("production_weighting_approved must be false")

    buckets = mix.get("buckets") or {}
    total = sum(float(v) for v in buckets.values())
    if abs(total - 1.0) > 1.0e-6:
        raise SystemExit(f"base_mixture buckets must sum to 1.0, got {total}")

    pilot = raw.get("pilot_v0_1") or {}
    if pilot.get("scope") not in {"PILOT_V0.1", "MAGI_DATA_PILOT_v0.1"}:
        raise SystemExit("pilot_v0_1.scope must be MAGI_DATA_PILOT_v0.1")
    for forbidden_key in (
        "character_in_base",
        "distillation_in_base",
        "sft_in_base",
    ):
        if pilot.get(forbidden_key) != "NOT_APPROVED_FOR_BASE":
            raise SystemExit(f"pilot_v0_1.{forbidden_key} must be NOT_APPROVED_FOR_BASE")

    allowed = pilot.get("allowed_candidates") or {}
    required_allowed = ("fineweb2", "the_stack_v2", "finemath", "wikimedia_wikipedia", "nullxes_domain")
    for cid in required_allowed:
        if allowed.get(cid) not in (True, "true", 1, "1"):
            raise SystemExit(f"pilot allowed_candidates missing {cid}")

    checklist = raw.get("approval_checklist") or {}
    for key in (
        "pilot_only",
        "production_ingest",
        "character_data_in_base",
        "sft_data_in_base",
        "distillation_data_in_base",
        "synthetic_bulk_generation",
    ):
        if key not in checklist:
            raise SystemExit(f"approval_checklist.{key} missing")
    if checklist.get("pilot_only") not in (True, "true"):
        raise SystemExit("approval_checklist.pilot_only must be true")
    if checklist.get("production_ingest") not in (False, "false"):
        raise SystemExit("approval_checklist.production_ingest must be false")
    for banned in (
        "character_data_in_base",
        "sft_data_in_base",
        "distillation_data_in_base",
        "synthetic_bulk_generation",
    ):
        if checklist.get(banned) != "forbidden":
            raise SystemExit(f"approval_checklist.{banned} must be forbidden")

    candidates = raw.get("candidates") or {}
    if not candidates:
        raise SystemExit("candidates missing")

    bad_download = []
    missing_fields = []
    for cid, spec in candidates.items():
        if not isinstance(spec, dict):
            raise SystemExit(f"candidate {cid} must be a map")
        for field in ("hub_id", "layer", "role", "license", "status", "download_now"):
            if field not in spec:
                missing_fields.append(f"{cid}.{field}")
        if spec.get("download_now") is True or str(spec.get("download_now")).lower() == "true":
            bad_download.append(cid)

    if missing_fields:
        raise SystemExit("missing fields: " + ", ".join(missing_fields))
    if bad_download:
        raise SystemExit(
            "download_now must be false until approval; offenders: " + ", ".join(bad_download)
        )

    character = candidates.get("magi_character_v1") or {}
    if character.get("mass_llm_generation") != "FORBIDDEN":
        raise SystemExit("magi_character_v1.mass_llm_generation must be FORBIDDEN")
    if character.get("status") != "SPEC_PLUS_GOLDEN_SEEDS_ONLY":
        raise SystemExit("magi_character_v1.status must be SPEC_PLUS_GOLDEN_SEEDS_ONLY")
    if character.get("base_ingest") != "NOT_APPROVED_FOR_BASE":
        raise SystemExit("magi_character_v1.base_ingest must be NOT_APPROVED_FOR_BASE")

    print(f"manifest={args.manifest}")
    print(f"candidates={len(candidates)}")
    print(f"base_mixture_sum={total}")
    print(f"approval_scope={approval.get('scope')}")
    print("production_weighting_approved=false")
    print("download_now_all_false=true")
    print("local_workstation_download=FORBIDDEN")
    print("status=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
